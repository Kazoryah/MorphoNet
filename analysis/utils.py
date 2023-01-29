"""Utility functions to display training results."""

import os
import pickle
from collections import OrderedDict
from typing import Dict, Iterator, List, Optional, Tuple, Type

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib.axes._axes import Axes
from mlflow.entities import Run, RunData
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import AxesDivider

from misc.utils import snr, psnr
from models.base import BaseNetwork
from operations.structuring_elements.base import StructuringElement


def get_true_path(path: str) -> str:
    """Get system path from path obtained via MLflow."""
    if "file://" in path:
        return path[7:]

    return path


def get_visfile_path(run: Run) -> str:
    """Get last saved visualisation file path."""
    outputs_path = get_true_path(run.info.artifact_uri) + "/outputs/"

    paths = list(
        filter(
            lambda p: p != "meta.pickle"
            and not os.path.isdir(outputs_path + p),
            os.listdir(outputs_path),
        )
    )
    paths.sort()

    return outputs_path + paths[-1]


def get_metafile_path(run: Run) -> str:
    """Get meta file path."""
    return get_true_path(run.info.artifact_uri) + "/outputs/meta.pickle"


def get_keys(path: str) -> None:
    """Returns available keys in pickle file."""
    path = get_true_path(path)
    with open(path, "rb") as visfile:
        out = pickle.load(visfile)
        print(out.keys())


def plot_sel(name: str, filter_size: int = 7) -> None:
    """Plot desired structuring element."""
    _, axis = plt.subplots(1, 1, figsize=(6, 6))

    structuring_element = StructuringElement.select(
        name, filter_size=filter_size, precision="f64"
    )()

    axis.pcolormesh(structuring_element, cmap="plasma")
    axis.set_box_aspect(1)
    axis.invert_yaxis()
    axis.axis("off")
    axis.set_title(name, fontsize=20, y=-0.15)

    divider = make_axes_locatable(axis)
    clb_ax1 = divider.append_axes("right", size="40%", pad=0.05)
    clb_ax2 = divider.append_axes("left", size="25%", pad=0.05)
    clb_ax1.axis("off")
    clb_ax2.axis("off")


def recreate_target_selem(
    run: RunData, structuring_element: str
) -> Optional[np.ndarray]:
    """Recreate target selem used in given run."""
    structuring_element_instance = StructuringElement.select(
        structuring_element,
        filter_size=int(run.params["filter_size"]),
        precision=run.params["precision"],
    )

    try:
        return structuring_element_instance()
    except NotImplementedError:
        return None


def filter_runs(
    runs: List[Run],
    parameters: Dict[str, str],
) -> List[Run]:
    """Filter the runs to get only the desired ones."""
    filtered = runs
    for key, value in parameters.items():
        filtered = list(
            filter(
                # pylint: disable=cell-var-from-loop
                lambda run: run.data.params[key] == value,
                filtered,
            )
        )

    return filtered


def forward(pl_module: pl.LightningModule, inputs: torch.Tensor) -> OrderedDict:
    """
    Execute a forward pass with the given model to get layers' `plot_` methods.
    """
    index, modules, handles = 0, {}, []

    def add_hook(module_: torch.nn.Module) -> None:
        def forward_hook(  # pylint: disable=unused-argument
            module__: torch.nn.Module,
            module_input: torch.Tensor,
            module_output: torch.Tensor,
        ) -> None:
            nonlocal index

            if hasattr(module__, "plot_"):
                plot_method = getattr(module__, "plot_")
            else:
                plot_method = None

            modules[index] = (
                module__.__class__.__name__,
                plot_method,
                module_output,
            )

            index += 1

        nonlocal handles
        handles.append(module_.register_forward_hook(forward_hook))

    pl_module.apply(add_hook)

    with torch.no_grad():
        pl_module.predict_step(inputs, -1)

    for handle in handles:
        handle.remove()

    return OrderedDict(sorted(modules.items()))


def plot_selem_row_(
    axes: Axes, structuring_elements: List[str], filter_size: int = 7
) -> Axes:
    """Plot the first row with structuring elements aspects."""
    for idx_s, name in enumerate(structuring_elements):
        structuring_element = StructuringElement.select(
            name, filter_size=filter_size, precision="f64"
        )()

        axis = axes[0, idx_s]
        axis.pcolormesh(structuring_element, cmap="plasma")
        axis.set_box_aspect(1)
        axis.invert_yaxis()
        axis.axis("off")
        axis.set_title(name, fontsize=20, y=-0.15)

        divider = make_axes_locatable(axis)
        clb_ax1 = divider.append_axes("right", size="40%", pad=0.05)
        clb_ax2 = divider.append_axes("left", size="25%", pad=0.05)
        clb_ax1.axis("off")
        clb_ax2.axis("off")

    return axes


def iterate_over_axes_(
    models: List[str],
    operations: List[str],
    structuring_elements: List[str],
    axes: Axes,
) -> Iterator[Tuple[int, str, int, str, int, str]]:
    """Iterate over the parameters."""
    for idx_m, model in enumerate(models):
        for idx_o, operation in enumerate(operations):
            axis = axes[1 + idx_m * len(operations) + idx_o, 0]
            axis.set_ylabel(
                f"{model}\n\n{operation}",
                fontsize=300 / len(model),
                rotation=0,
                labelpad=100,
                va="center",
            )

            for idx_s, structuring_element in enumerate(structuring_elements):
                yield idx_m, model, idx_o, operation, idx_s, structuring_element


def plot_submodules_(
    run: Run,
    axis: Axes,
    pl_module: pl.LightningModule,
    inputs: torch.Tensor,
    structuring_element: str,
    excluded: Optional[List[str]],
    limits: Optional[List[Tuple[float, float]]] = None,
) -> None:
    """Plot each modules of the network."""
    target_structuring_element = recreate_target_selem(
        run.data, structuring_element
    )

    divider = make_axes_locatable(axis)
    plot_index = 0

    comments = ""
    if "val_loss" in run.data.metrics:
        comments = f"Loss: {run.data.metrics['val_loss']:.3e}"

    for (
        _,
        (class_name, plot_method, _),
    ) in forward(pl_module, inputs[0][None, :, :, :]).items():
        if plot_method is None or (
            excluded is not None and class_name.lower() in excluded
        ):
            continue

        if plot_index > 0:
            axis = divider.append_axes("right", size="100%", pad=0.3)
            comments = ""

        try:
            plot_method(
                axis=axis,
                target=target_structuring_element,
                comments=comments,
                divider=divider,
                limits=limits,
            )
            plot_index += 1
        except NotImplementedError:
            pass


def plot(  # pylint: disable=too-many-locals,too-many-arguments,unused-argument
    uri: str,
    experiment_name: str,
    models: List[str],
    structuring_elements: List[str],
    operations: List[str],
    iterations: List[int],
    stretch_x: int = 1,
    excluded: Optional[List[str]] = None,
    limits: Optional[List[Tuple[float, float]]] = None,
) -> None:
    """
    Plot an entire figure with the results for desired models, structuring
    elements and operations.
    """
    client = mlflow.tracking.MlflowClient(uri)
    experiment = client.get_experiment_by_name(experiment_name)

    runs = client.search_runs(
        experiment.experiment_id, order_by=["attribute.start_time ASC"]
    )

    for iteration in iterations:
        y_len, x_len = len(models) * len(operations), len(structuring_elements)
        fig, axes = plt.subplots(
            1 + y_len, x_len, figsize=(stretch_x * 6 * x_len, 6 * (1 + y_len))
        )
        fig.tight_layout(h_pad=10.0, w_pad=5.0)
        if x_len == 1:
            axes = np.array([axes]).T

        axes = plot_selem_row_(axes, structuring_elements)

        for (
            idx_m,
            model,
            idx_o,
            operation,
            idx_s,
            structuring_element,
        ) in iterate_over_axes_(models, operations, structuring_elements, axes):
            filtered = filter_runs(
                runs,
                {
                    "model": model,
                    "structuring_element": structuring_element,
                    "operation": operation,
                },
            )
            if len(filtered) == 0:
                continue

            run = filtered[iteration]

            model_class = BaseNetwork.select_(model)
            if model_class is None:
                continue

            path = get_visfile_path(run)
            pl_module = model_class.load_from_checkpoint(path)

            with open(get_metafile_path(run), "rb") as metafile:
                inputs = pickle.load(metafile)["inputs"]

            axis = axes[1 + idx_m * len(operations) + idx_o, idx_s]

            plot_submodules_(
                run,
                axis,
                pl_module,
                inputs,
                structuring_element,
                excluded,
                limits,
            )

        plt.show()


def plot_image_(
    axis: Axes,
    image: torch.Tensor,
    percentage: Optional[str] = None,
    target: Optional[torch.Tensor] = None,
    divider: Optional[AxesDivider] = None,
    comments: str = "",
) -> Axes:
    """Plot (de)noised image and calculate its SNR with the target if given."""
    axis.invert_yaxis()
    axis.get_yaxis().set_ticks([])
    axis.get_xaxis().set_ticks([])
    axis.set_box_aspect(1)

    plot_ = axis.pcolormesh(image, cmap="plasma")
    if divider is None:
        divider = make_axes_locatable(axis)
    clb_ax = divider.append_axes("right", size="5%", pad=0.05)
    clb_ax.set_box_aspect(15)
    plt.colorbar(plot_, cax=clb_ax)

    if percentage is not None:
        axis.set_title(f"{percentage}%", fontsize=20)
    if target is not None:
        snr_ = snr(image.numpy(), target.numpy())
        psnr_ = psnr(image.numpy(), target.numpy())
        comments = f"SNR (dB): {snr_:.3f}\nPSNR (dB): {psnr_:.3f}\n{comments}"

    axis.set_xlabel(comments, fontsize=20)

    return axis


def plot_noise_(
    run: Run,
    axes: Axes,
    model_class: Type[BaseNetwork],
    idx_m: int,
    idx_p: int,
    input_: torch.Tensor,
    target: torch.Tensor,
) -> None:
    """Plot image prediction with noise related information."""
    comments = ""
    if "val_loss" in run.data.metrics:
        comments += f"Loss: {run.data.metrics['val_loss']:.3e}"
    if "mean_psnr" in run.data.metrics:
        comments += f"\nMean PSNR: {run.data.metrics['mean_psnr']:.3f}"

    path = get_visfile_path(run)
    pl_module = model_class.load_from_checkpoint(path)

    axis = axes[1 + idx_m, idx_p]
    divider = make_axes_locatable(axis)

    result = pl_module.predict_step(input_[None, None, :, :], -1).detach()
    plot_image_(
        axis, result[0, 0], target=target, divider=divider, comments=comments
    )


def plot_noise(  # pylint: disable=too-many-locals,too-many-arguments,unused-argument
    uri: str,
    experiment_name: str,
    models: List[str],
    percentages: List[str],
    operation: str,
    iterations: List[int],
) -> None:
    """
    Plot an entire figure with the results for desired models, percentage of
    noise and operation.
    """
    client = mlflow.tracking.MlflowClient(uri)
    experiment = client.get_experiment_by_name(experiment_name)

    runs = client.search_runs(
        experiment.experiment_id, order_by=["attribute.start_time ASC"]
    )

    for iteration in iterations:
        y_len, x_len = len(models), len(percentages)
        fig, axes = plt.subplots(
            1 + y_len, x_len, figsize=(6 * x_len, 6 * (1 + y_len))
        )
        fig.tight_layout(h_pad=10.0, w_pad=5.0)
        if x_len == 1:
            axes = np.array([axes]).T

        for idx_p, percentage in enumerate(percentages):
            for idx_m, model in enumerate(models):
                model_class = BaseNetwork.select_(model)
                if model_class is None:
                    continue

                filtered = filter_runs(
                    runs,
                    {
                        "model": model,
                        "operation": operation,
                        "percentage": percentage,
                    },
                )
                if len(filtered) == 0:
                    continue

                run = filtered[iteration]
                if idx_m == 0:
                    with open(get_metafile_path(run), "rb") as metafile:
                        data = pickle.load(metafile)
                        target = data["targets"][0, 0]
                        input_ = data["inputs"][0, 0]

                if idx_p == 0:
                    axes[1 + idx_m, 0].set_ylabel(
                        model,
                        fontsize=300 / len(model),
                        rotation=0,
                        labelpad=100,
                        va="center",
                    )

                if idx_m == 0:
                    # divider = make_axes_locatable(axes[0, idx_p])
                    plot_image_(
                        axes[0, idx_p],
                        input_,
                        percentage,
                        # divider=divider,
                        comments=operation,
                        target=target,
                    )

                plot_noise_(
                    run, axes, model_class, idx_m, idx_p, input_, target
                )

    plt.show()


def plot_noise_results_(  # pylint: disable=too-many-locals
    run: Run,
    axis: Axes,
    sub_axis: Axes,
    model_class: Type[BaseNetwork],
    input_: torch.Tensor,
    target: torch.Tensor,
    excluded: Optional[List[str]],
) -> None:
    """Plot filters and outputs."""
    comments = ""
    if "val_loss" in run.data.metrics:
        comments = f"Loss: {run.data.metrics['val_loss']:.3e}"

    path = get_visfile_path(run)
    pl_module = model_class.load_from_checkpoint(path)

    divider = make_axes_locatable(axis)
    sub_divider = make_axes_locatable(sub_axis)
    plot_index = 0

    modules = forward(pl_module, input_[None, None, :, :])

    for (_, (class_name, plot_method, layer_outputs)) in modules.items():
        if plot_method is None or (
            excluded is not None and class_name.lower() in excluded
        ):
            continue

        if plot_index > 0:
            axis = divider.append_axes("right", size="100%", pad=0.3)
            sub_axis = sub_divider.append_axes("right", size="100%", pad=0.3)
            comments = ""

        try:
            plot_method(
                axis=axis,
                comments=comments,
                divider=divider,
            )
            plot_image_(
                sub_axis,
                layer_outputs[0, 0],
                target=target,
                divider=sub_divider,
            )
            plot_index += 1
        except NotImplementedError:
            pass


def plot_noise_results(  # pylint: disable=too-many-locals,too-many-arguments,unused-argument
    uri: str,
    experiment_name: str,
    models: List[str],
    percentage: str,
    operation: str,
    iterations: List[int],
    stretch_x: int = 1,
    excluded: Optional[List[str]] = None,
    input_index: int = 0,
) -> None:
    """
    Plot filters for the desired denoising network with outputs of each layer.
    """
    client = mlflow.tracking.MlflowClient(uri)
    experiment = client.get_experiment_by_name(experiment_name)

    runs = client.search_runs(
        experiment.experiment_id, order_by=["attribute.start_time ASC"]
    )

    for iteration in iterations:
        y_len, x_len = 1 + 2 * len(models), 1
        fig, axes = plt.subplots(
            y_len, x_len, figsize=(stretch_x * 6 * x_len, 6 * (1 + y_len))
        )
        fig.tight_layout(h_pad=10.0, w_pad=5.0)

        for idx_m, model in enumerate(models):
            model_class = BaseNetwork.select_(model)
            if model_class is None:
                continue

            filtered = filter_runs(
                runs,
                {
                    "model": model,
                    "operation": operation,
                    "percentage": percentage,
                },
            )
            if len(filtered) == 0:
                continue

            run = filtered[iteration]

            axis = axes[1 + 2 * idx_m]
            sub_axis = axes[1 + 2 * idx_m + 1]
            axis.set_ylabel(
                model,
                fontsize=300 / len(model),
                rotation=0,
                labelpad=100,
                va="center",
            )

            sub_axis_title = "Layers Outputs"
            if "mean_psnr" in run.data.metrics:
                sub_axis_title += (
                    f"\nMean PSNR: {run.data.metrics['mean_psnr']:.3f}"
                )
            sub_axis.set_ylabel(
                sub_axis_title,
                fontsize=200 / max(map(len, sub_axis_title.split("\n"))),
                rotation=0,
                labelpad=100,
                va="center",
            )

            if idx_m == 0:
                with open(get_metafile_path(run), "rb") as metafile:
                    data = pickle.load(metafile)
                    target = data["targets"][input_index, 0]
                    input_ = data["inputs"][input_index, 0]

                divider = make_axes_locatable(axes[0])
                plot_image_(
                    axes[0],
                    input_,
                    percentage,
                    divider=divider,
                    comments=operation,
                    target=target,
                )

            plot_noise_results_(
                run, axis, sub_axis, model_class, input_, target, excluded
            )

    plt.show()
