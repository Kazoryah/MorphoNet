"""Argument parser definiton."""

import argparse

from operations import OPS
from operations.structuring_elements import STRUCTURING_ELEMENTS
from models import MODELS
from losses import LOSSES
from misc.utils import PRECISIONS_TORCH
from datasets.base import DataModule


class Parser(argparse.ArgumentParser):
    """Parser for the training arguments."""

    def __init__(self) -> None:
        super().__init__(description="Train a model.")

        self.add_argument("model", help="model to load", choices=MODELS)
        self.add_argument(
            "--loss", choices=LOSSES, help="loss to use", default="mse"
        )

        self.add_argument(
            "--op",
            default=None,
            choices=OPS,
            help="operation to perform",
            dest="operation",
        )
        self.add_argument(
            "--sel",
            choices=STRUCTURING_ELEMENTS,
            help="structuring element to use",
            default="",
            dest="structuring_element",
        )
        self.add_argument(
            "--filter_size",
            type=int,
            default=9,
            help="size of the filter to apply and learn",
        )
        self.add_argument(
            "--percentage", type=int, default=0, help="percentage of noise"
        )

        self.add_argument(
            "--epochs",
            type=int,
            default=100,
            help="number of epochs to train for",
        )
        self.add_argument(
            "--patience",
            type=int,
            default=20,
            help="number of epochs to continue training for with no improvement to validation loss",
        )

        self.add_argument(
            "--batch_size", type=int, default=32, help="batch size"
        )
        self.add_argument(  # TODO implement usage!
            "--validation_split",
            type=float,
            default=0.1,
            help="Validation split",
        )
        self.add_argument(
            "--no-std",
            action="store_false",
            default=True,
            dest="standardize",
            help="Do not standardize data.",
        )

        self.add_argument(
            "--precision", choices=PRECISIONS_TORCH.keys(), default="f64"
        )
        self.add_argument(
            "--gpu",
            type=int,
            default=0,
            help="which GPU to restrict the training session to",
        )
        self.add_argument(
            "--vis_freq",
            type=int,
            default=32,
            help="how often to save layers outputs and weights (in batches)",
        )
        self.add_argument(
            "--experiment",
            type=str,
            default="Default",
            help="Name of the MLflow experiment",
        )

        subparsers = self.add_subparsers(
            help="Dataset to train on",
            dest="dataset",
            parser_class=argparse.ArgumentParser,
        )

        for dataset_name in DataModule.listing():
            class_ = DataModule.select_(dataset_name)
            if class_ is None:
                continue

            dataset_parser = subparsers.add_parser(
                dataset_name, help=f"Train on {class_.__name__}"
            )
            dataset_parser.add_argument(
                "dataset_path", help="dataset to train on"
            )

            class_.add_specific_arguments(dataset_parser)
