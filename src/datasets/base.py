"""Loaders for the supported datasets."""

from abc import abstractmethod, ABCMeta
from typing import Any, List, Optional, Tuple, Type
import inspect

from argparse import ArgumentParser
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import mlflow

from misc.utils import PRECISIONS_NP, PRECISIONS_TORCH
from operations.base import Operation


NOISY_NAME = "NOISY_SRGB_010"
GT_NAME = "GT_SRGB_010"

# TODO add way to split data from args
# TODO with heavy data, dynamic loading needs to be implemented
# TODO ensure data is loaded as float and in [0,1] before targets computation


class Dataset(torch.utils.data.Dataset):
    """Implementation of torch.utils.Data.Dataset abstract class."""

    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index, :], self.targets[index, :]


class DataModule(
    pl.LightningDataModule, metaclass=ABCMeta
):  # pylint: disable=too-many-instance-attributes
    """Base abstract class for datasets."""

    class Standardizer:  # pylint: disable=too-few-public-methods
        """Classic ZScore standardizer needing mean and std from train dataset."""
        def __init__(self, data_module: "DataModule", train_data: torch.Tensor) -> None:
            self.mean = torch.mean(train_data)
            self.std = torch.std(train_data)
            self.active = data_module.standardize

            if mlflow.active_run() is not None:
                mlflow.log_param("train_dataset_mean", self.mean)
                mlflow.log_param("train_dataset_std", self.std)

        def __call__(self, data: torch.Tensor) -> torch.Tensor:
            """Standardize data if wanted."""
            if self.active:
                return (data - self.mean) / self.std

            return data


    def __init__(
        self,
        batch_size: int,
        dataset_path: str,
        precision: str,
        operation: Operation,
        standardize: bool,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.torch_precision = PRECISIONS_TORCH[precision]
        self.np_precision = PRECISIONS_NP[precision]
        self.num_workers = 8
        self.operation = operation
        self.standardize = standardize

        self.standardizer: DataModule.Standardizer
        self.train_dataset: Dataset
        self.val_dataset: Dataset

    def normalize(self, tensor: torch.Tensor, min_value: Optional[float] = 0.0, max_value: Optional[float] = 255.0) -> torch.Tensor:
        """
        Convert tensor to right type and precision, then nomalize it in [0; 1].
        """
        tensor = tensor.to(self.torch_precision)
        if min_value is None:
            min_value = torch.min(tensor).item()
        if max_value is None:
            max_value = torch.max(tensor).item()

        return (tensor - min_value) / (max_value - min_value)

    def remodel_data(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create adapted morphological dataset from given data."""
        inputs, targets = self.operation(inputs, targets)
        return inputs.to(self.torch_precision), targets.to(self.torch_precision)

    @property
    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a sample of the validation dataset."""
        # TODO change method when shuffling to always have same data
        return self.val_dataset.inputs[:10], self.val_dataset.targets[:10]

    def prepare_data(self) -> None:
        pass

    @abstractmethod
    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def _create_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(
            dataset=self.train_dataset,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader(
            dataset=self.val_dataset,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(Dataset(torch.empty(0), torch.empty(0)))

    @classmethod
    def select_(cls, name: str) -> Optional[Type["DataModule"]]:
        """
        Class method iterating over all subclasses to load the desired dataset.
        """
        if cls.__name__.lower() == name:
            return cls

        for subclass in cls.__subclasses__():
            selected = subclass.select_(name)
            if selected is not None:
                return selected

        return None

    @classmethod
    def select(cls, name: str, **kwargs: Any) -> "DataModule":
        """
        Class method iterating over all subclasses to instantiate the desired
        data module.
        """

        selected = cls.select_(name)
        if selected is None:
            raise ValueError("The selected dataset was not found.")

        return selected(**kwargs)

    @classmethod
    def listing(cls) -> List[str]:
        """List all the available dataset loaders."""
        subclasses = set()
        if not inspect.isabstract(cls):
            subclasses = {cls.__name__.lower()}

        for subclass in cls.__subclasses__():
            subclasses = subclasses.union(subclass.listing())

        return list(subclasses)

    @classmethod
    def add_specific_arguments(cls, parser: ArgumentParser) -> None:
        """Add specific arguments needed for the use of the dataset."""
