# pylint: disable=all
# type: ignore

from .base import DataModule

# TODO not reimplemented
class SIDD(DataModule):  # pragma: no cover
    # https://paperswithcode.com/dataset/sidd

    def __init__(
        self,
        precision,
        dataset_path,
        train=False,
        patch_size=None,
        smartphone_codes=None,
        iso_levels=None,
        shutter_speeds=None,
        illuminants=None,
        ibcs=None,
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

        def filter_data(instance: Dict[str, Any]):
            nonlocal filters
            keep = True
            for key, values in filters.items():
                if values is not None:
                    keep = keep and instance[key] in values
            return keep

        instances = SIDDDataset._get_instances(dataset_path, filter_data)
        self.inputs = SIDDDataset._load(instances, dataset_path, NOISY_NAME)
        self.targets = SIDDDataset._load(instances, dataset_path, GT_NAME)

        print(f"Raw X: {len(self.targets)}\nRaw Y: {len(self.inputs)}")

        patches = []
        x_all = []
        y_all = []
        for x, y in zip(self.inputs, self.targets):
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

    def _get_instances(dataset_path, filter_predicate=lambda inst: True):
        f = open(f"{dataset_path}/Scene_Instances.txt")
        instances_str = list(
            filter(
                lambda l: len(l) != 0,
                map(lambda l: l.strip(), f.read().split("\n")),
            )
        )

        def annotate_instance(l):
            parts = l.split("_")
            return {
                "path": l,
                "scene_number": parts[1],
                "smartphone_code": parts[2],
                "iso_level": int(parts[3]),
                "shutter_speed": int(parts[4]),
                "illuminant": parts[5],
                "illuminant_brightness_code": parts[6],
            }

        return list(
            filter(filter_predicate, map(annotate_instance, instances_str))
        )

    def _load(
        instances: List[Dict[str, Any]], dataset_path: str, image_name: str
    ):
        image_path = f"{dataset_path}/Data/{{}}/{image_name}.PNG"

        def load_instance(instance: Dict[str, Any]):
            nonlocal image_path
            return io.imread(image_path.format(instance["path"]))

        return [load_instance(instance) for instance in instances]

    def add_specific_arguments(cls, parser: ArgumentParser) -> None:
        parser.add_argument(
            "dataset_path", nargs=1, help="dataset to train on"
        )
        parser.add_argument(
            "--patch_size",
            type=int,
            default=64,
            help="width and height of the image patches",
        )
        parser.add_argument(
            "--smartphone_codes",
            default=None,
            help="which smartphone codes to use from the dataset (comma-separated)",
        )
        parser.add_argument(
            "--iso_levels",
            default=None,
            help="which iso levels to use from the dataset (comma-separated)",
        )
        parser.add_argument(
            "--shutter_speeds",
            default=None,
            help="which shutter speeds to use from the dataset (comma-separated)",
        )
        parser.add_argument(
            "--illuminants",
            default=None,
            help="which illuminants to use from the dataset (comma-separated)",
        )
        parser.add_argument(
            "--ibcs",
            default=None,
            help="which illuminant brightness codes to use from the dataset (comma-separated)",
        )
