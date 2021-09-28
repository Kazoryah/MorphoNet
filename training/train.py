import argparse
import pathlib
from importlib import import_module

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import model_selection
from tqdm import tqdm

from preprocessing.ops import OPS, OPS_NOISE, OPS_MORPH
from utils.visualizer import ModuleVisualizer
from task.data import load_data, load_gt
from task.patch import make_patches
from preprocessing.sel import STRUCTURING_ELEMENTS


def make_ssim_loss():
    import pytorch_ssim

    ssim_loss = pytorch_ssim.SSIM(window_size=3)

    def loss_func(a, b):
        return 1.0 - ssim_loss(a, b)

    return loss_func


def make_mse_loss():
    return nn.MSELoss()


LOSSES = {
    "ssim": make_ssim_loss,
    "mse": make_mse_loss,
}

PRECISIONS = {"f32": torch.float32, "f64": torch.float64}
PRECISIONS_NP = {"f32": "float32", "f64": "float64"}

parser = argparse.ArgumentParser(description="Train a model.")
parser.add_argument("model", help="model to load")
parser.add_argument("loss", choices=LOSSES.keys(), help="loss to use")

parser.add_argument("op", choices=OPS.keys(), help="operation to perform")
parser.add_argument(
    "--sel", choices=STRUCTURING_ELEMENTS.keys(), help="structuring element to use"
)

parser.add_argument(
    "--percentage", type=int, default=5, help="percentage of noise"
)

parser.add_argument(
    "--epochs", type=int, default=100, help="number of epochs to train for"
)
parser.add_argument(
    "--patience",
    type=int,
    default=20,
    help="number of epochs to continue training for with no improvement to validation loss",
)
parser.add_argument(
    "--filter_size", type=int, default=9, help="size of the filter to apply and learn",
)
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument(
    "--validation_split", type=float, default=0.1, help="validation split"
)
parser.add_argument("--precision", choices=PRECISIONS.keys(), default="f64")
parser.add_argument(
    "--out_dir",
    default="output",
    help="directory where to store output (weights, visualization)",
)
parser.add_argument(
    "--gpu",
    type=int,
    default=0,
    help="which GPU to restrict the training session to",
)
parser.add_argument(
    "--layers",
    type=str,
    default=None,
    help="which layers to visualize (comma-separated)",
)
parser.add_argument(
    "--vis_freq",
    type=int,
    default=32,
    help="how often to save layers outputs and weights (in batches)",
)

subparsers = parser.add_subparsers(help="Dataset to train on", dest="dataset")

mnist_parser = subparsers.add_parser("mnist", help="Train on MNIST")
mnist_parser.add_argument("dataset_path", help="dataset to train on")

fashion_mnist_parser = subparsers.add_parser("fashion_mnist", help="Train on FashionMNIST")
fashion_mnist_parser.add_argument("dataset_path", help="dataset to train on")

sidd_parser = subparsers.add_parser("sidd", help="Train on sidd")

sidd_parser.add_argument("dataset_path", nargs=1, help="dataset to train on")
sidd_parser.add_argument(
    "--patch_size", type=int, default=64, help="width and height of the image patches",
)
sidd_parser.add_argument(
    "--smartphone_codes",
    default=None,
    help="which smartphone codes to use from the dataset (comma-separated)",
)
sidd_parser.add_argument(
    "--iso_levels",
    default=None,
    help="which iso levels to use from the dataset (comma-separated)",
)
sidd_parser.add_argument(
    "--shutter_speeds",
    default=None,
    help="which shutter speeds to use from the dataset (comma-separated)",
)
sidd_parser.add_argument(
    "--illuminants",
    default=None,
    help="which illuminants to use from the dataset (comma-separated)",
)
sidd_parser.add_argument(
    "--ibcs",
    default=None,
    help="which illuminant brightness codes to use from the dataset (comma-separated)",
)


def ensure_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_out_dir(path):
    i = 0
    while pathlib.Path(f"{path}_{i}").exists():
        i += 1
    return f"{path}_{i}"


def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


def get_model(model_path, model_args):
    model_module = import_module(model_path)
    get_actual_model = model_module.get_model
    model_name = model_module.MODEL_NAME
    return (model_name, *get_actual_model(model_args))


def loss_batch(model, loss_func, xb, yb, opt=None):
    if opt is None:
        batch_start = time.time_ns()
        out = model(xb)
        batch_elapsed = time.time_ns() - batch_start
        loss = loss_func(out, yb)
    else:

        def evaluate():
            nonlocal batch_elapsed
            opt.zero_grad()
            batch_start = time.time_ns()
            out = model(xb)
            batch_elapsed = time.time_ns() - batch_start
            loss = loss_func(out, yb)
            loss_start = time.time_ns()
            loss.backward()
            batch_elapsed += time.time_ns() - loss_start
            return loss

        loss = opt.step(evaluate)

        with torch.no_grad():
            for module in model.modules():
                if hasattr(module, "after_batch"):
                    module.after_batch()

    return loss.item(), batch_elapsed


import math


def format_duration(d):
    if d < 1e4:
        unit = "ns"
    elif d < 1e7:
        unit = "µs"
        d /= 1e3
    elif d < 1e10:
        unit = "ms"
        d /= 1e6
    else:
        unit = "s"
        d /= 1e9
    return f"{round(d)}{unit}"


def format_stats(stats, prefix=""):
    loss, elapsed = stats["loss"], stats["elapsed"]
    loss_fmt = f"{loss:6g}".rjust(10, " ")
    elapsed_fmt = format_duration(elapsed).rjust(8, " ")
    return f"{prefix}loss: {loss_fmt}, {prefix}elapsed: {elapsed_fmt}"


def run_epoch(desc, model, loss_func, dl, opt=None, visualizer=None):
    with tqdm(
        total=len(dl),
        leave=False,
        bar_format="{postfix[0][left]} {bar:16} {postfix[0][right]}",
        postfix=[{"left": desc, "right": "N/A"}],
    ) as pbar:
        epoch_loss = 0.0
        epoch_elapsed = 0.0
        epoch_seen = 0
        for batch, (xb, yb) in enumerate(dl):
            loss_item, batch_elapsed = loss_batch(model, loss_func, xb, yb, opt)

            # The loss is already divided by the number of elements (reduction = 'mean').
            epoch_loss += loss_item * len(xb)
            epoch_elapsed += float(batch_elapsed)
            epoch_seen += len(xb)

            loss = epoch_loss / float(epoch_seen)
            elapsed = epoch_elapsed / float(epoch_seen)
            batch_fmt = str(batch).rjust(math.ceil(math.log10(len(dl))), "0")
            pbar.postfix[0]["left"] = f"{desc}: {batch_fmt}/{len(dl)}"
            stats = {"loss": loss, "elapsed": elapsed}
            pbar.postfix[0]["right"] = format_stats(stats)
            pbar.update()

            if visualizer is not None:
                visualizer.step_batch({"loss": loss})

    return stats


def fit(model, epochs, patience, loss_func, opt, scheduler, train_dl, valid_dl,
        visualizer):
    total_elapsed = 0.0
    total_seen = 0
    params = map(lambda param_group: param_group["params"], opt.param_groups)

    best_valid_loss = float("inf")
    best_valid_loss_seen = 0
    for epoch in range(epochs):
        if visualizer is not None:
            visualizer.step_epoch()

        model.train()
        stats = run_epoch(
            f"Epoch {epoch} (train)",
            model,
            loss_func,
            train_dl,
            opt=opt,
            visualizer=visualizer,
        )

        model.eval()
        with torch.no_grad():
            valid_stats = run_epoch(
                f"Epoch {epoch} (valid)", model, loss_func, valid_dl
            )

        log = f"Epoch {epoch}: {format_stats(stats)}, {format_stats(valid_stats, 'valid_')}"

        if valid_stats["loss"] < best_valid_loss:
            # Only consider significant changes.
            if (best_valid_loss - valid_stats["loss"]) >= best_valid_loss * 1e-4:
                best_valid_loss = valid_stats["loss"]
                best_valid_loss_seen = epoch
                log += " (best)"
            else:
                log += " (~)"

        print(log)

        if epoch - best_valid_loss_seen > patience:
            message = f"No improvement to validation loss in {patience} epochs, terminating."
            print(message)
            visualizer.finish({"end_reason": message, "end_epoch": epoch})
            return

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(valid_stats["loss"])
            else:
                scheduler.step()

        with torch.no_grad():
            for module in model.modules():
                if hasattr(module, "after_epoch"):
                    module.after_epoch()

        for param_group, original_params in zip(opt.param_groups, params):
            param_group["params"] = list(
                filter(lambda p: p.requires_grad, original_params)
            )

    visualizer.finish({"end_reason": "Reached max epoch", "end_epoch": epochs})

def load_mnist(**kwargs):
    import torchvision

    dtype = PRECISIONS_NP[kwargs["precision"]]

    images_train = (
        torchvision.datasets.MNIST(kwargs["dataset_path"])
        .data.numpy()
        .astype(dtype)
    )
    images_test = (
        torchvision.datasets.MNIST(kwargs["dataset_path"], train=False)
        .data.numpy()
        .astype(dtype)
    )

    x_all = np.concatenate((images_train, images_test))

    # Load as NCHW.
    x_all = x_all[:, np.newaxis, :, :].astype(dtype)
    x_all /= 255.0

    filter_padding = kwargs["filter_size"] // 2
    if kwargs["op"] == "closing" or kwargs["op"] == "opening":
        filter_padding *= 2
    if kwargs["sel"] is not None:
        x_all = np.pad(
            x_all,
            (
                (0, 0),
                (0, 0),
                (filter_padding, filter_padding),
                (filter_padding, filter_padding),
            ),
            mode="minimum",
        )

    return x_all[: len(images_train)], x_all[len(images_train) :]

def load_fmnist(**kwargs):
    import torchvision

    dtype = PRECISIONS_NP[kwargs["precision"]]

    images_train = (
        torchvision.datasets.FashionMNIST(kwargs["dataset_path"])
        .data.numpy()
        .astype(dtype)
    )
    images_test = (
        torchvision.datasets.FashionMNIST(kwargs["dataset_path"], train=False)
        .data.numpy()
        .astype(dtype)
    )

    x_all = np.concatenate((images_train, images_test))

    # Load as NCHW.
    x_all = x_all[:, np.newaxis, :, :].astype(dtype)
    x_all /= 255.0

    filter_padding = kwargs["filter_size"] // 2
    if kwargs["op"] == "closing" or kwargs["op"] == "opening":
        filter_padding *= 2
    if kwargs["sel"] is not None:
        x_all = np.pad(
            x_all,
            (
                (0, 0),
                (0, 0),
                (filter_padding, filter_padding),
                (filter_padding, filter_padding),
            ),
            mode="minimum",
        )

    return x_all[: len(images_train)], x_all[len(images_train) :]

def load_sidd(
    *,
    patch_size,
    dataset_path,
    smartphone_codes,
    iso_levels,
    shutter_speeds,
    illuminants,
    ibcs,
    **kwargs,
):
    patch_shape = (patch_size, patch_size)
    dataset_path = dataset_path[0]
    filters = {
        "smartphone_code": split_arg(smartphone_codes),
        "iso_level": split_arg(iso_levels, int),
        "shutter_speed": split_arg(shutter_speeds, int),
        "illuminant": split_arg(illuminants),
        "illuminant_brightness_code": split_arg(ibcs),
    }

    def filter_data(inst):
        keep = True
        for key, values in filters.items():
            if values is not None:
                keep = keep and inst[key] in values
        return keep

    x_raw = y_raw = list(load_gt(dataset_path, filter_data))

    print(f"Raw X: {len(x_raw)}\nRaw Y: {len(y_raw)}")

    patches = []
    x_all = []
    y_all = []
    for x, y in zip(x_raw, y_raw):
        p = make_patches(x.shape, patch_shape)
        patches.append(p)
        for pp in p:
            x_all.append(x[pp])
            y_all.append(y[pp])
    x_all = np.array(x_all).astype("float32")[:, np.newaxis, :, :] / 255.0
    y_all = np.array(y_all).astype("float32")[:, np.newaxis, :, :] / 255.0

    print(f"Cut into {len(x_all)} patches of shape {patch_shape}")

    x_train, x_valid = model_selection.train_test_split(
        x_all, test_size=0.33, random_state=42
    )

    return x_train, x_valid

def pad_inputs(x, model_name, filter_padding, pad_value=0):
    if "double" in model_name:
        filter_padding *= 2
    elif "five" in model_name:
        filter_padding *= 5
    elif "four" in model_name:
        filter_padding *= 4

    padded = np.pad(
        x,
        (
            (0, 0),
            (0, 0),
            (filter_padding, filter_padding),
            (filter_padding, filter_padding),
        ),
        mode="constant",
        constant_values=(pad_value,)
    )

    return padded


LOADERS = {"sidd": load_sidd, "mnist": load_mnist, "fashion_mnist": load_fmnist}


if __name__ == "__main__":
    args = parser.parse_args()

    def split_arg(value, mapper=lambda a: a):
        if value is None:
            return None
        return [mapper(v) for v in value.split(",")]

    gpus = [ (i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count()) ]
    print(f"Available GPUs: {gpus}")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    print(f"Running on '{torch.cuda.get_device_name()}'")

    dtype = PRECISIONS[args.precision]
    torch.set_default_dtype(dtype)

    kwargs = {"dtype": dtype, "device": device}

    model_name, model, opt, scheduler = get_model(
        args.model, {"filter_size": args.filter_size, **kwargs}
    )


    x_train, x_valid = LOADERS[args.dataset](**vars(args))
    x_all = np.concatenate((x_train, x_valid))

    loss_func = LOSSES[args.loss]()
    op = OPS[args.op]
    if args.sel != None:
        sel = STRUCTURING_ELEMENTS[args.sel](
            filter_shape=(args.filter_size, args.filter_size),
            dtype=PRECISIONS_NP[args.precision],
        )

        out_dir = ensure_dir(
            get_out_dir(
                f"{args.out_dir}/{args.dataset}_{model_name}_{args.loss}_{args.op}_{args.sel}"
            )
        )

        print(f"Loaded model {model_name}, saving to {out_dir}")
        #plt.imsave(args.out_dir + "/sel.png", sel.squeeze(), cmap="plasma")
        print("Creating target images...", end="", flush=True)
        y_all = op(x_all, sel)
        if args.op == 'bdilation' or  args.op == 'berosion' or  args.op == 'bclosing' or  args.op == 'bopening':
            x_all = x_all > 0
    else:
        sel = None
        out_dir = ensure_dir(
            get_out_dir(
                f"{args.out_dir}/{args.dataset}_{model_name}_{args.loss}_{args.op}_{args.percentage}"
            )
        )

        print(f"Loaded model {model_name}, saving to {out_dir}")
        print("Creating target images...", end="", flush=True)
        pad_value = 0
        if (args.op == "saltpepper"):
            x_all = 0.5 + x_all / 2
            pad_value = 0.5
        y_all = op(x_all, args.percentage,
                (args.filter_size, args.filter_size))

        x_all, y_all = pad_inputs(y_all, model_name, args.filter_size // 2, pad_value), x_all

    print(" [Done]")

    # Normalization step.
    # x_all = (x_all - np.min(x_all)) / (np.max(x_all) - np.min(x_all)) - 0.5
    # y_all = (y_all - np.min(y_all)) / (np.max(y_all) - np.min(y_all)) - 0.5
#x_all = (x_all - np.mean(x_all)) / np.std(x_all)
    # y_all = (y_all - np.mean(y_all)) / np.std(y_all)

    print(f"X: {x_all.shape}\nY: {y_all.shape}")

    #plt.imsave(args.out_dir + "/x.png", x_all[0].squeeze(), cmap="plasma")
    #plt.imsave(args.out_dir + "/y.png", y_all[0].squeeze(), cmap="plasma")

    x_train, x_valid = x_all[: len(x_train)], x_all[len(x_train) :]
    y_train, y_valid = y_all[: len(x_train)], y_all[len(x_train) :]

    x_train, y_train, x_valid, y_valid = (
        torch.tensor(x_train, **kwargs),
        torch.tensor(y_train, **kwargs),
        torch.tensor(x_valid, **kwargs),
        torch.tensor(y_valid, **kwargs),
    )
    train_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_valid, y_valid)

    visualizer = ModuleVisualizer(
        x_valid[:10],
        y_valid[:10],
        sel,
        model,
        ensure_dir(f"{out_dir}/vis"),
        filter_children=args.layers,
        freq=args.vis_freq,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        sel_name=args.sel,
        module_name=args.model,
        loss=args.loss,
        dataset=args.dataset,
        percentage=args.percentage
    )

    fit(
        model,
        args.epochs,
        args.patience,
        loss_func,
        opt,
        scheduler,
        *get_data(train_ds, valid_ds, args.batch_size),
        visualizer,
    )
