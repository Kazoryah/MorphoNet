"""DataModule for the CIFAR10 dataset."""

from typing import Optional
import torchvision
import torch

from .base import DataModule, Dataset


class CIFAR10(DataModule):
    """CIFAR10 DataModule."""

    def prepare_data(self) -> None:
        torchvision.datasets.CIFAR10(self.dataset_path, train=True, download=True)
        torchvision.datasets.CIFAR10(
            self.dataset_path, train=False, download=True
        )

    def setup(self, stage: Optional[str] = None) -> None:
        train_dataset = torchvision.datasets.CIFAR10(
            self.dataset_path,
            train=True,
        )
        normalized_inputs = self.normalize(torch.tensor(train_dataset.data.mean(axis=3)))
        inputs, targets = self.remodel_data(
            normalized_inputs, train_dataset.targets
        )

        self.standardizer = self.Standardizer(self, inputs)
        standardized_inputs = self.standardizer(inputs)

        self.train_dataset = Dataset(
            inputs=standardized_inputs,
            targets=targets,
        )

        val_dataset = torchvision.datasets.CIFAR10(self.dataset_path, train=False)
        normalized_inputs = self.normalize(torch.tensor(val_dataset.data.mean(axis=3)))
        inputs, targets = self.remodel_data(normalized_inputs, val_dataset.targets)

        standardized_inputs = self.standardizer(inputs)

        self.val_dataset = Dataset(
            inputs=standardized_inputs,
            targets=targets,
        )
