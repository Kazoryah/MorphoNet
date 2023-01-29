"""Init datasets module."""

from .mnist import MNIST
from .fmnist import FashionMNIST
from .cifar10 import CIFAR10

from .base import DataModule

DATASETS = DataModule.listing()
