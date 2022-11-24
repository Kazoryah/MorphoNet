# pylint: disable=all
# type: ignore

from .base import DataModule

# TODO not reimplemented/generated
class BIWTOH(Datamodule):  # pragma: no cover
    def __init__(self, precision, dataset_path, train=False, **kwargs):
        dtype = PRECISIONS_NP[precision]

        if train:
            self.inputs = np.load(f"{dataset_path}/train-images.npy").astype(
                dtype
            )
            self.targets = np.load(f"{dataset_path}/train-labels.npy").astype(
                dtype
            )
        else:
            self.inputs = np.load(f"{dataset_path}/t10k-images.npy").astype(
                dtype
            )
            self.targets = np.load(f"{dataset_path}/t10k-labels.npy").astype(
                dtype
            )
