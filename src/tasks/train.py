"""Root logic for trainings."""

import logging
import os
import re
from typing import Dict, Any, Union, Tuple
from argparse import Namespace

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
from pytorch_lightning import Trainer
import mlflow.pytorch
from mlflow import ActiveRun

from misc.context import RunContext, Task
from misc.parser import parser, LOSSES
from misc.visualizer import VisualizerCallback
from misc.utils import PRECISIONS_TORCH
from datasets.base import DataModule
from models.base import VAL_LOSS, BaseNetwork
from operations.base import Operation
from operations.structuring_elements import StructuringElement
from tasks import output_managment

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s.%(msecs)03d] %(levelname)s <%(module)s> %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# TODO filter size seems to be alsways int, but could be set to tuple
# TODO add param to change mlflow experiment
# TODO for lmorph, as inputs are in the range [1;2], the output should also be
# calculated from [1;2]: maybe harder to learn when starting in the wrong scale
# TODO normalisation des données
# TODO permettre de réduire la fréquence des print log (progress bar)
# TODO tester avec nombre de workers plus élever
# TODO check si les données sont bien en mémoire en permanance, peut-être la cause de la lenteur
# TODO select and select_ methods types like for models


def termination_logs(  # pylint: disable=too-many-arguments
    version: Union[str, int],
    state_dict: Dict[str, Any],
    data_module: DataModule,
    model: BaseNetwork,
    trainer: Trainer,
    args: Namespace,
    visualizer: VisualizerCallback,
) -> None:
    """Write logs after training."""
    # Avoiding a pytorch_lightning error reconstructing with `log_graph=True`
    tb_logger = TensorBoardLogger(
        f"{os.getcwd()}/lightning_logs/",
        name=args.experiment,
        version=version,
        log_graph=True,
    )
    tb_logger.log_graph(model, data_module.sample[0])
    tb_logger.log_hyperparams(vars(args))
    for idx, (input_image, target_image) in enumerate(
        zip(data_module.sample[0].cpu(), data_module.sample[1].cpu())
    ):
        tb_logger.experiment.add_image(f"Input Sample {idx}", input_image)
        tb_logger.experiment.add_image(f"Target Sample {idx}", target_image)

    mlflow.log_param("TB_folder", tb_logger.log_dir)
    mlflow.log_metrics(
        dict(
            map(
                lambda kv: (kv[0], kv[1].item()), trainer.logged_metrics.items()
            )
        )
    )
    mlflow.log_metric("last_epoch", trainer.current_epoch)

    ckpt_path = state_dict["best_model_path"]
    if ckpt_path != "":
        mlflow.log_metric(
            "best_epoch", int(re.split("-|=", ckpt_path.split("/")[-1])[1])
        )
        mlflow.log_metric("best_score", state_dict["best_model_score"].item())

        visualizer.plot_saved_model(
            model.__class__,
            ckpt_path,
            f"{visualizer.weigths_plots_path}/checkpoint/",
            visualizer.inputs.cpu(),
        )


def init_callbacks(
    run: ActiveRun, structuring_element: StructuringElement, args: Namespace
) -> Tuple[
    VisualizerCallback,
    ModelCheckpoint,
    EarlyStopping,
    TensorBoardLogger,
    TQDMProgressBar,
]:
    """Init callbacks used during training."""
    visualizer = VisualizerCallback(run, structuring_element, args.vis_freq)
    model_checkpoint = ModelCheckpoint(monitor="val_loss")
    early_stopping = EarlyStopping(
        monitor=VAL_LOSS,
        min_delta=0.00,
        patience=args.patience,
        verbose=False,
        mode="min",  # TODO change for classif
    )
    tb_logger = TensorBoardLogger(
        f"{os.getcwd()}/lightning_logs/", name=args.experiment
    )
    progress_bar = TQDMProgressBar(refresh_rate=100)

    return visualizer, model_checkpoint, early_stopping, tb_logger, progress_bar


def main(args: Namespace, run: ActiveRun) -> None:
    """Main loop preparing training data and starting fitting loop."""
    with Task("Logging parameters"):
        mlflow.log_params(vars(args))

    with Task("Loading structuring element"):
        structuring_element = StructuringElement.select(
            name=args.structuring_element,
            filter_size=args.filter_size,
            precision=args.precision,
        )

    with Task("Loading operation"):
        operation = Operation.select(
            name=args.operation,
            structuring_element=structuring_element,
            percentage=args.percentage,
        )

    # TODO model should have available keys in args
    with Task("Loading model"):
        model = BaseNetwork.select(
            name=args.model,
            filter_size=args.filter_size,
            loss_function=LOSSES[args.loss](),
            dtype=PRECISIONS_TORCH[args.precision],
            # device=device,
        )

    # Accuracy of best val loss comparison
    # TODO check dtype for the modle (should be same as arg)

    with Task("Loading dataset"):
        data_module = DataModule.select(
            name=args.dataset,
            dataset_path=args.dataset_path,
            batch_size=args.batch_size,
            precision=args.precision,
            operation=operation,
        )

    (
        visualizer,
        model_checkpoint,
        early_stopping,
        tb_logger,
        progress_bar,
    ) = init_callbacks(run, structuring_element, args)

    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=[
            progress_bar,
            early_stopping,
            visualizer,
            model_checkpoint,
        ],
        log_every_n_steps=1,
        accelerator="gpu",
        devices=[args.gpu],
        logger=[tb_logger],
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        fast_dev_run=False,
    )

    trainer.fit(model=model, datamodule=data_module)

    termination_logs(
        tb_logger.version,
        model_checkpoint.state_dict(),
        data_module,
        model,
        trainer,
        args,
        visualizer,
    )


if __name__ == "__main__":
    with Task("Parsing command line"):
        args_ = parser.parse_args()
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mlf_logger = MLFlowLogger(
        experiment_name=args_.experiment,
        tracking_uri=f"file://{os.getcwd()}/mlruns",
    )
    with mlflow.start_run(mlf_logger.run_id) as run_, RunContext(
        run_, output_managment
    ):
        main(args_, run_)
