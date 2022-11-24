"""DataModule for the MNIST dataset."""

from typing import Optional
import torchvision

from .base import DataModule, Dataset


class FashionMNIST(DataModule):
    """FasionMNIST DataModule."""

    def prepare_data(self) -> None:
        torchvision.datasets.FashionMNIST(
            self.dataset_path, train=True, download=True
        )
        torchvision.datasets.FashionMNIST(
            self.dataset_path, train=False, download=True
        )

    def setup(self, stage: Optional[str] = None) -> None:
        train_dataset = torchvision.datasets.FashionMNIST(
            self.dataset_path,
            train=True,
        )
        normalized_inputs = self.normalize(train_dataset.data)
        inputs, targets = self.remodel_data(
            normalized_inputs, train_dataset.targets
        )

        self.standardizer = self.Standardizer(self, inputs)
        standardized_inputs = self.standardizer(inputs)

        self.train_dataset = Dataset(
            inputs=standardized_inputs,
            targets=targets,
        )

        val_dataset = torchvision.datasets.FashionMNIST(
            self.dataset_path, train=False
        )
        normalized_inputs = self.normalize(val_dataset.data)
        inputs, targets = self.remodel_data(normalized_inputs, val_dataset.targets)

        standardized_inputs = self.standardizer(inputs)

        self.val_dataset = Dataset(
            inputs=standardized_inputs,
            targets=targets,
        )
