"""Dataset wrappers with stable IDs and a torchvision-free MNIST reader."""

from __future__ import annotations

import gzip
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset

from .artifacts import hash_array, stable_hash

InputT = TypeVar("InputT")

_INPUT_KEYS = ("inputs", "input", "images", "image", "x", "features")
_LABEL_KEYS = ("labels", "label", "targets", "target", "y")
_ID_KEYS = ("sample_ids", "sample_id", "ids", "id")


def unpack_batch(batch: Any) -> tuple[Any, Any, Any | None]:
    """Normalize ``(input, label[, sample_id])`` tuples or mappings."""

    if isinstance(batch, Mapping):
        inputs = next((batch[key] for key in _INPUT_KEYS if key in batch), None)
        labels = next((batch[key] for key in _LABEL_KEYS if key in batch), None)
        if inputs is None or labels is None:
            raise ValueError("mapping batches must contain an input and a label")
        return inputs, labels, next((batch[key] for key in _ID_KEYS if key in batch), None)
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1], batch[2] if len(batch) > 2 else None
    raise TypeError("dataset items must be mappings or (input, label[, sample_id]) tuples")


def _open_idx(path: Path):
    if path.exists():
        return path.open("rb")
    compressed = Path(f"{path}.gz")
    if compressed.exists():
        return gzip.open(compressed, "rb")
    raise FileNotFoundError(f"missing IDX file {path} (or {compressed.name})")


def read_idx_images(path: str | Path) -> np.ndarray:
    """Read an IDX3 image file into a copied ``uint8 [N,H,W]`` array."""

    source = Path(path)
    with _open_idx(source) as handle:
        header = handle.read(16)
        if len(header) != 16:
            raise ValueError(f"truncated IDX image header: {source}")
        magic, count, rows, columns = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(f"invalid IDX image magic {magic} in {source}")
        payload = handle.read()
    expected = count * rows * columns
    if len(payload) != expected:
        raise ValueError(
            f"IDX image payload has {len(payload)} bytes, expected {expected}: {source}"
        )
    return np.frombuffer(payload, dtype=np.uint8).reshape(count, rows, columns).copy()


def read_idx_labels(path: str | Path) -> np.ndarray:
    """Read an IDX1 label file into a copied ``int64 [N]`` array."""

    source = Path(path)
    with _open_idx(source) as handle:
        header = handle.read(8)
        if len(header) != 8:
            raise ValueError(f"truncated IDX label header: {source}")
        magic, count = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"invalid IDX label magic {magic} in {source}")
        payload = handle.read()
    if len(payload) != count:
        raise ValueError(f"IDX label payload has {len(payload)} bytes, expected {count}: {source}")
    return np.frombuffer(payload, dtype=np.uint8).astype(np.int64, copy=True)


def _resolve_idx_root(root: str | Path, dataset_directory: str) -> Path:
    candidate = Path(root)
    options = (
        candidate,
        candidate / "raw",
        candidate / dataset_directory / "raw",
    )
    for option in options:
        if (option / "train-images-idx3-ubyte").exists() or (
            option / "train-images-idx3-ubyte.gz"
        ).exists():
            return option
    # Return the conventional path so the eventual error is actionable.
    return candidate / dataset_directory / "raw"


class MNISTIDXDataset(Dataset[tuple[Tensor, int, str]]):
    """MNIST/Fashion-MNIST reader that does not import torchvision.

    Each item follows the project contract ``(input, internal_label, sample_id)``.
    """

    def __init__(
        self,
        root: str | Path = "data",
        *,
        train: bool = True,
        transform: Callable[[Tensor], Any] | None = None,
        target_transform: Callable[[int], Any] | None = None,
        dataset_name: str = "mnist",
        dataset_directory: str | None = None,
        normalize: bool = True,
    ) -> None:
        directory_name = dataset_directory or (
            "FashionMNIST" if dataset_name.lower().replace("-", "_") == "fashion_mnist" else "MNIST"
        )
        raw_root = _resolve_idx_root(root, directory_name)
        image_name = "train-images-idx3-ubyte" if train else "t10k-images-idx3-ubyte"
        label_name = "train-labels-idx1-ubyte" if train else "t10k-labels-idx1-ubyte"
        self.images = read_idx_images(raw_root / image_name)
        self.external_targets = read_idx_labels(raw_root / label_name)
        if self.images.shape[0] != self.external_targets.shape[0]:
            raise ValueError("MNIST image and label counts differ")
        observed = np.unique(self.external_targets)
        self.class_ids = observed.astype(np.int64)
        self.class_names = tuple(str(value) for value in self.class_ids)
        external_to_internal = {int(value): index for index, value in enumerate(self.class_ids)}
        self.internal_targets = np.asarray(
            [external_to_internal[int(value)] for value in self.external_targets], dtype=np.int64
        )
        self.targets = self.internal_targets
        self.train = bool(train)
        self.split = "train" if self.train else "test"
        self.dataset_name = dataset_name.lower().replace("-", "_")
        width = max(8, len(str(len(self.targets) - 1)))
        self.sample_ids = np.asarray(
            [
                f"{self.dataset_name}:{self.split}:{index:0{width}d}"
                for index in range(len(self.targets))
            ],
            dtype=str,
        )
        self.transform = transform
        self.target_transform = target_transform
        self.normalize = bool(normalize)
        self.raw_root = raw_root

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> tuple[Any, Any, str]:
        image = torch.from_numpy(self.images[index].copy()).unsqueeze(0)
        if self.normalize:
            image = image.to(dtype=torch.float32).div_(255.0)
        label: Any = int(self.internal_targets[index])
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label, str(self.sample_ids[index])

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "dataset": self.dataset_name,
                "split": self.split,
                "images_hash": hash_array(self.images),
                "labels_hash": hash_array(self.external_targets),
            }
        )


class TensorSampleDataset(Dataset[tuple[Tensor, int, str]]):
    """In-memory dataset useful for deterministic synthetic integration tests."""

    def __init__(
        self,
        inputs: Any,
        labels: Any,
        *,
        sample_ids: Sequence[str] | None = None,
        dataset_name: str = "tensor",
        split: str = "train",
        transform: Callable[[Tensor], Any] | None = None,
    ) -> None:
        self.inputs = torch.as_tensor(inputs)
        external_labels = np.asarray(labels)
        if self.inputs.ndim < 1 or external_labels.ndim != 1:
            raise ValueError("inputs must be batched and labels must be one-dimensional")
        if self.inputs.shape[0] != external_labels.size:
            raise ValueError("inputs and labels must contain the same number of samples")
        self.class_ids = np.unique(external_labels)
        if self.class_ids.dtype.hasobject:
            raise TypeError("object-valued class IDs are not supported")
        mapping = {
            value.item() if isinstance(value, np.generic) else value: index
            for index, value in enumerate(self.class_ids)
        }
        self.internal_targets = np.asarray(
            [
                mapping[value.item() if isinstance(value, np.generic) else value]
                for value in external_labels
            ],
            dtype=np.int64,
        )
        self.external_targets = external_labels.copy()
        self.targets = self.internal_targets
        self.class_names = tuple(str(value) for value in self.class_ids)
        self.dataset_name = dataset_name
        self.split = split
        if sample_ids is None:
            width = max(8, len(str(external_labels.size - 1)))
            sample_ids = [
                f"{dataset_name}:{split}:{index:0{width}d}" for index in range(external_labels.size)
            ]
        self.sample_ids = np.asarray(sample_ids, dtype=str)
        if (
            self.sample_ids.shape != (external_labels.size,)
            or np.unique(self.sample_ids).size != external_labels.size
        ):
            raise ValueError("sample_ids must be unique with one entry per sample")
        self.transform = transform

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int) -> tuple[Any, int, str]:
        value = self.inputs[index]
        if self.transform is not None:
            value = self.transform(value)
        return value, int(self.internal_targets[index]), str(self.sample_ids[index])

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "dataset": self.dataset_name,
                "split": self.split,
                "inputs_hash": hash_array(self.inputs.detach().cpu().numpy()),
                "labels_hash": hash_array(self.internal_targets),
            }
        )


class SyntheticGaussianDataset(TensorSampleDataset):
    """Deterministic Gaussian classes with optional deliberately close pairs."""

    def __init__(
        self,
        *,
        split: str = "train",
        seed: int = 0,
        samples_per_class: int = 100,
        num_classes: int = 6,
        input_dimension: int = 8,
        class_separation: float = 4.0,
        noise_std: float = 0.55,
        confusing_pairs: Sequence[Sequence[int]] = ((0, 1),),
        confusing_pair_separation: float = 0.8,
        test_fraction: float = 0.2,
        transform: Callable[[Tensor], Any] | None = None,
    ) -> None:
        split = split.lower()
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        if samples_per_class < 2 or num_classes < 2 or input_dimension < 1:
            raise ValueError(
                "samples_per_class and num_classes must be at least 2; "
                "input_dimension must be positive"
            )
        if class_separation <= 0 or noise_std <= 0 or confusing_pair_separation <= 0:
            raise ValueError("synthetic separation and noise values must be positive")
        if not 0.0 < test_fraction < 1.0:
            raise ValueError("test_fraction must lie strictly between 0 and 1")

        test_count = int(round(samples_per_class * float(test_fraction)))
        test_count = min(max(test_count, 1), samples_per_class - 1)
        split_count = samples_per_class - test_count if split == "train" else test_count

        # Centers are generated from a split-independent stream; examples use a
        # split-specific stream, so loading one split never constructs the other.
        center_rng = np.random.default_rng(int(seed))
        centers = center_rng.normal(size=(num_classes, input_dimension))
        norms = np.linalg.norm(centers, axis=1, keepdims=True)
        tiny = np.finfo(np.float64).tiny
        centers = centers / np.maximum(norms, tiny) * float(class_separation)

        normalized_pairs: list[tuple[int, int]] = []
        used_classes: set[int] = set()
        for pair in confusing_pairs:
            if len(pair) != 2:
                raise ValueError(f"confusing pair must contain two class indices: {pair!r}")
            left, right = int(pair[0]), int(pair[1])
            if left == right or not (0 <= left < num_classes and 0 <= right < num_classes):
                raise ValueError(f"invalid confusing pair {(left, right)}")
            if left in used_classes or right in used_classes:
                raise ValueError("a class may occur in at most one confusing pair")
            used_classes.update((left, right))
            normalized_pairs.append((left, right))
            midpoint = 0.5 * (centers[left] + centers[right])
            direction = center_rng.normal(size=input_dimension)
            direction /= max(float(np.linalg.norm(direction)), tiny)
            half_gap = 0.5 * float(confusing_pair_separation)
            centers[left] = midpoint - half_gap * direction
            centers[right] = midpoint + half_gap * direction

        split_offset = 0x5A17 if split == "train" else 0xA51E
        sample_rng = np.random.default_rng(
            np.random.SeedSequence([int(seed) & 0xFFFFFFFF, split_offset])
        )
        inputs: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        sample_ids: list[str] = []
        for class_index in range(num_classes):
            class_inputs = sample_rng.normal(
                loc=centers[class_index],
                scale=float(noise_std),
                size=(split_count, input_dimension),
            ).astype(np.float32)
            inputs.append(class_inputs)
            labels.append(np.full(split_count, class_index, dtype=np.int64))
            sample_ids.extend(
                f"synthetic:{split}:c{class_index:04d}:{index:08d}" for index in range(split_count)
            )
        inputs_array = np.concatenate(inputs)
        labels_array = np.concatenate(labels)
        ids_array = np.asarray(sample_ids, dtype=str)
        order = sample_rng.permutation(labels_array.size)
        super().__init__(
            inputs_array[order],
            labels_array[order],
            sample_ids=ids_array[order],
            dataset_name="synthetic",
            split=split,
            transform=transform,
        )
        self.centers = torch.from_numpy(centers.astype(np.float32))
        self.generation_parameters = {
            "seed": int(seed),
            "samples_per_class": int(samples_per_class),
            "split_samples_per_class": int(split_count),
            "num_classes": int(num_classes),
            "input_dimension": int(input_dimension),
            "class_separation": float(class_separation),
            "noise_std": float(noise_std),
            "confusing_pairs": [list(pair) for pair in normalized_pairs],
            "confusing_pair_separation": float(confusing_pair_separation),
            "test_fraction": float(test_fraction),
        }


def generate_synthetic_dataset(
    *,
    split: str = "train",
    seed: int = 0,
    samples_per_class: int = 100,
    num_classes: int = 6,
    input_dimension: int = 8,
    class_separation: float = 4.0,
    noise_std: float = 0.55,
    confusing_pairs: Sequence[Sequence[int]] = ((0, 1),),
    confusing_pair_separation: float = 0.8,
    test_fraction: float = 0.2,
    transform: Callable[[Tensor], Any] | None = None,
) -> SyntheticGaussianDataset:
    return SyntheticGaussianDataset(
        split=split,
        seed=seed,
        samples_per_class=samples_per_class,
        num_classes=num_classes,
        input_dimension=input_dimension,
        class_separation=class_separation,
        noise_std=noise_std,
        confusing_pairs=confusing_pairs,
        confusing_pair_separation=confusing_pair_separation,
        test_fraction=test_fraction,
        transform=transform,
    )


class StableIdDataset(Dataset[tuple[Any, int, str]], Generic[InputT]):
    """Wrap a conventional ``(input, label)`` dataset with internal labels/IDs."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        dataset_name: str,
        split: str,
        labels: Sequence[Any] | None = None,
    ) -> None:
        self.dataset = dataset
        if labels is None:
            if not hasattr(dataset, "targets"):
                raise ValueError("labels are required when the wrapped dataset has no targets")
            labels = dataset.targets
        if isinstance(labels, Tensor):
            labels = labels.detach().cpu().numpy()
        labels_array = np.asarray(labels)
        if labels_array.shape != (len(dataset),):
            raise ValueError("wrapped labels must have one entry per dataset item")
        self.external_targets = labels_array.copy()
        self.class_ids = np.unique(labels_array)
        self.class_names = tuple(str(value) for value in self.class_ids)
        mapping = {
            value.item() if isinstance(value, np.generic) else value: index
            for index, value in enumerate(self.class_ids)
        }
        self.internal_targets = np.asarray(
            [
                mapping[value.item() if isinstance(value, np.generic) else value]
                for value in labels_array
            ],
            dtype=np.int64,
        )
        self.targets = self.internal_targets
        self.dataset_name = dataset_name
        self.split = split
        width = max(8, len(str(len(dataset) - 1)))
        self.sample_ids = np.asarray(
            [f"{dataset_name}:{split}:{index:0{width}d}" for index in range(len(dataset))],
            dtype=str,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Any, int, str]:
        item = self.dataset[index]
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise ValueError("wrapped dataset items must contain at least input and label")
        return item[0], int(self.internal_targets[index]), str(self.sample_ids[index])

    @property
    def fingerprint(self) -> str | None:
        """Content checksum of the wrapped raw array (e.g. torchvision CIFAR ``data``)."""

        data = getattr(self.dataset, "data", None)
        if data is None:
            return None
        return stable_hash(
            {
                "dataset": self.dataset_name,
                "split": self.split,
                "data_hash": hash_array(np.asarray(data)),
                "labels_hash": hash_array(self.external_targets),
            }
        )


class ImagePathDataset(Dataset[tuple[Any, int, str]]):
    """Small stable-ID wrapper for CUB-200 and TinyImageNet file manifests."""

    def __init__(
        self,
        paths: Sequence[Path],
        labels: Sequence[int],
        sample_ids: Sequence[str],
        *,
        dataset_name: str,
        split: str,
        class_names: Sequence[str],
        transform: Callable | None,
    ) -> None:
        self.paths = tuple(Path(path) for path in paths)
        self.internal_targets = np.asarray(labels, dtype=np.int64)
        self.targets = self.internal_targets
        self.sample_ids = np.asarray(sample_ids, dtype=str)
        if not (len(self.paths) == self.internal_targets.size == self.sample_ids.size):
            raise ValueError("image paths, labels, and sample IDs must align")
        self.class_ids = np.arange(len(class_names), dtype=np.int64)
        self.class_names = tuple(str(name) for name in class_names)
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Any, int, str]:
        from PIL import Image

        with Image.open(self.paths[index]) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.internal_targets[index]), str(self.sample_ids[index])

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "dataset": self.dataset_name,
                "split": self.split,
                "paths": [str(path) for path in self.paths],
                "labels": self.internal_targets.tolist(),
            }
        )


def _load_cub200(root: str | Path, split: str, transform: Callable | None) -> ImagePathDataset:
    base = Path(root)
    if (base / "CUB_200_2011").is_dir():
        base /= "CUB_200_2011"

    def indexed(path: Path) -> dict[int, str]:
        return {
            int(line.split(maxsplit=1)[0]): line.split(maxsplit=1)[1]
            for line in path.read_text(encoding="utf-8").splitlines()
        }

    images = indexed(base / "images.txt")
    labels = {
        key: int(value) - 1 for key, value in indexed(base / "image_class_labels.txt").items()
    }
    train_flags = {key: int(value) for key, value in indexed(base / "train_test_split.txt").items()}
    classes = indexed(base / "classes.txt")
    wanted = 1 if split == "train" else 0
    image_ids = sorted(key for key, flag in train_flags.items() if flag == wanted)
    return ImagePathDataset(
        [base / "images" / images[key] for key in image_ids],
        [labels[key] for key in image_ids],
        [f"cub200:{split}:{key:06d}" for key in image_ids],
        dataset_name="cub200",
        split=split,
        class_names=[classes[key] for key in sorted(classes)],
        transform=transform,
    )


def _load_tiny_imagenet(
    root: str | Path, split: str, transform: Callable | None
) -> ImagePathDataset:
    base = Path(root)
    if (base / "tiny-imagenet-200").is_dir():
        base /= "tiny-imagenet-200"
    class_names = tuple(
        line.strip()
        for line in (base / "wnids.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    class_index = {name: index for index, name in enumerate(class_names)}
    paths: list[Path] = []
    labels: list[int] = []
    ids: list[str] = []
    if split == "train":
        for name in class_names:
            for path in sorted((base / "train" / name / "images").glob("*.JPEG")):
                paths.append(path)
                labels.append(class_index[name])
                ids.append(f"tiny_imagenet:train:{name}:{path.name}")
    else:
        annotations = {}
        for line in (base / "val" / "val_annotations.txt").read_text(encoding="utf-8").splitlines():
            filename, name, *_ = line.split("\t")
            annotations[filename] = name
        for filename in sorted(annotations):
            paths.append(base / "val" / "images" / filename)
            labels.append(class_index[annotations[filename]])
            ids.append(f"tiny_imagenet:test:{filename}")
    return ImagePathDataset(
        paths,
        labels,
        ids,
        dataset_name="tiny_imagenet",
        split=split,
        class_names=class_names,
        transform=transform,
    )


def build_transforms(
    dataset: str,
    *,
    training: bool,
    vit: bool = False,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> Callable | None:
    """Return the locked train or deterministic evaluation preprocessing."""

    name = dataset.lower().replace("-", "_")
    if name in {"synthetic", "gaussian", "synthetic_gaussian"}:
        return None
    try:
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
    except Exception as exc:  # pragma: no cover - optional binary dependency
        raise RuntimeError("image preprocessing requires a compatible torchvision build") from exc
    if name == "mnist":
        return transforms.Normalize((0.1307,), (0.3081,))
    if name == "fashion_mnist":
        return transforms.Normalize((0.2860,), (0.3530,))
    if vit:
        if mean is None or std is None:
            raise ValueError("the pinned ViT checkpoint mean/std are required")
        spatial = (
            [
                transforms.RandomResizedCrop(
                    224, scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(),
            ]
            if training
            else [
                transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
            ]
        )
        return transforms.Compose(
            [*spatial, transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
    if name in {"cifar10", "cifar100"}:
        means = {"cifar10": (0.4914, 0.4822, 0.4465), "cifar100": (0.5071, 0.4867, 0.4408)}
        stds = {"cifar10": (0.2470, 0.2435, 0.2616), "cifar100": (0.2675, 0.2565, 0.2761)}
        augmentation = (
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
            if training
            else []
        )
        return transforms.Compose(
            [*augmentation, transforms.ToTensor(), transforms.Normalize(means[name], stds[name])]
        )
    if name == "tiny_imagenet":
        augmentation = (
            [transforms.RandomCrop(64, padding=8), transforms.RandomHorizontalFlip()]
            if training
            else []
        )
        return transforms.Compose(
            [
                *augmentation,
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    if name in {"cub200", "cub_200"}:
        spatial = (
            [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
            if training
            else [transforms.Resize(256), transforms.CenterCrop(224)]
        )
        return transforms.Compose(
            [
                *spatial,
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    raise ValueError(f"unsupported dataset preprocessing: {dataset!r}")


class SealedDataset(Dataset):
    """Sentinel that raises if development code touches the official test set."""

    def __init__(self, name: str = "official test set") -> None:
        self.name = name

    def __len__(self) -> int:
        raise RuntimeError(f"{self.name} is sealed and must not be accessed")

    def __getitem__(self, index: int) -> Any:
        del index
        raise RuntimeError(f"{self.name} is sealed and must not be accessed")


class IndexedDataset(Dataset):
    """Index view that preserves stable labels and IDs for split partitions."""

    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        if (
            self.indices.ndim != 1
            or np.any(self.indices < 0)
            or np.any(self.indices >= len(dataset))
        ):
            raise ValueError("indices must be a valid one-dimensional dataset selection")
        self.internal_targets = dataset_labels(dataset)[self.indices]
        self.targets = self.internal_targets
        self.sample_ids = dataset_sample_ids(dataset)[self.indices]
        self.class_ids = np.asarray(dataset.class_ids)
        self.class_names = tuple(dataset.class_names)
        self.dataset_name = str(dataset.dataset_name)
        self.split = str(getattr(dataset, "split", "partition"))

    def __len__(self) -> int:
        return self.indices.size

    def __getitem__(self, index: int) -> Any:
        return self.dataset[int(self.indices[index])]


@dataclass(frozen=True)
class DatasetMetadata:
    name: str
    split: str
    class_ids: tuple[Any, ...]
    class_names: tuple[str, ...]
    sample_count: int
    fingerprint: str | None


def _load_torchvision_dataset(
    name: str,
    root: str | Path,
    split: str,
    *,
    download: bool,
    transform: Callable | None,
) -> StableIdDataset:
    try:
        from torchvision import datasets
    except Exception as exc:  # catches binary-operator mismatches as well as ImportError
        raise RuntimeError(
            "torchvision could not be imported. Install a torchvision build matched "
            "to the active PyTorch build, or use the local MNIST IDX loader."
        ) from exc
    train = split == "train"
    normalized = name.lower().replace("-", "_")
    constructors = {
        "mnist": datasets.MNIST,
        "fashion_mnist": datasets.FashionMNIST,
        "cifar10": datasets.CIFAR10,
        "cifar100": datasets.CIFAR100,
    }
    if normalized not in constructors:
        raise ValueError(f"unsupported torchvision dataset: {name!r}")
    base = constructors[normalized](
        root=str(root), train=train, download=download, transform=transform
    )
    return StableIdDataset(base, dataset_name=normalized, split=split)


def load_dataset(
    name: str | Mapping[str, Any],
    root: str | Path = "data",
    *,
    split: str = "train",
    download: bool = False,
    transform: Callable | None = None,
    target_transform: Callable | None = None,
    prefer_torchvision: bool = False,
    seed: int = 0,
    samples_per_class: int = 100,
    num_classes: int = 6,
    input_dimension: int = 8,
    class_separation: float = 4.0,
    noise_std: float = 0.55,
    confusing_pairs: Sequence[Sequence[int]] = ((0, 1),),
    confusing_pair_separation: float = 0.8,
    test_fraction: float = 0.2,
) -> Dataset:
    """Load one explicitly requested official split.

    The function never constructs the opposite split, which lets upstream
    stages operate without touching the sealed official test set.
    """

    if isinstance(name, Mapping):
        data_config = dict(name.get("data", name))
        experiment_config = name.get("experiment", {})
        if not isinstance(experiment_config, Mapping):
            experiment_config = {}
        if "dataset" not in data_config:
            raise ValueError("dataset configuration must contain data.dataset")
        return load_dataset(
            str(data_config["dataset"]),
            data_config.get("root", root),
            split=str(data_config.get("split", split)),
            download=bool(data_config.get("download", download)),
            transform=transform,
            target_transform=target_transform,
            prefer_torchvision=bool(data_config.get("prefer_torchvision", prefer_torchvision)),
            seed=int(data_config.get("seed", experiment_config.get("seed", seed))),
            samples_per_class=int(data_config.get("samples_per_class", samples_per_class)),
            num_classes=int(data_config.get("num_classes", num_classes)),
            input_dimension=int(data_config.get("input_dimension", input_dimension)),
            class_separation=float(data_config.get("class_separation", class_separation)),
            noise_std=float(data_config.get("noise_std", noise_std)),
            confusing_pairs=data_config.get("confusing_pairs", confusing_pairs),
            confusing_pair_separation=float(
                data_config.get("confusing_pair_separation", confusing_pair_separation)
            ),
            test_fraction=float(data_config.get("test_fraction", test_fraction)),
        )
    normalized = name.lower().replace("-", "_")
    split = split.lower()
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    if normalized in {"synthetic", "gaussian", "synthetic_gaussian"}:
        if target_transform is not None:
            raise ValueError(
                "synthetic labels are fixed internal indices; target_transform is unsupported"
            )
        return generate_synthetic_dataset(
            split=split,
            seed=seed,
            samples_per_class=samples_per_class,
            num_classes=num_classes,
            input_dimension=input_dimension,
            class_separation=class_separation,
            noise_std=noise_std,
            confusing_pairs=confusing_pairs,
            confusing_pair_separation=confusing_pair_separation,
            test_fraction=test_fraction,
            transform=transform,
        )
    if normalized in {"cub200", "cub_200"}:
        if target_transform is not None:
            raise ValueError("CUB labels use fixed internal indices")
        return _load_cub200(root, split, transform)
    if normalized in {"tiny_imagenet", "tinyimagenet"}:
        if target_transform is not None:
            raise ValueError("TinyImageNet labels use fixed internal indices")
        return _load_tiny_imagenet(root, split, transform)
    if normalized in {"mnist", "fashion_mnist"} and not prefer_torchvision:
        try:
            return MNISTIDXDataset(
                root,
                train=split == "train",
                transform=transform,
                target_transform=target_transform,
                dataset_name=normalized,
            )
        except FileNotFoundError:
            if not download:
                raise
    dataset = _load_torchvision_dataset(
        normalized, root, split, download=download, transform=transform
    )
    if target_transform is not None:
        raise ValueError(
            "target_transform is unsupported by StableIdDataset; labels must remain stable internal IDs"
        )
    return dataset


build_dataset = load_dataset


def dataset_labels(dataset: Dataset) -> np.ndarray:
    """Return internal integer labels, preferring declared arrays over iteration."""

    if isinstance(dataset, Subset):
        return dataset_labels(dataset.dataset)[np.asarray(dataset.indices, dtype=np.int64)]
    for name in ("internal_targets", "targets", "labels"):
        values = getattr(dataset, name, None)
        if values is None:
            continue
        if isinstance(values, Tensor):
            values = values.detach().cpu().numpy()
        array = np.asarray(values, dtype=np.int64)
        if array.shape == (len(dataset),):
            return array.copy()
    return np.asarray(
        [int(unpack_batch(dataset[index])[1]) for index in range(len(dataset))], dtype=np.int64
    )


def dataset_sample_ids(dataset: Dataset) -> np.ndarray:
    """Return stable string sample IDs, falling back to positional IDs."""

    if isinstance(dataset, Subset):
        return dataset_sample_ids(dataset.dataset)[np.asarray(dataset.indices, dtype=np.int64)]
    values = getattr(dataset, "sample_ids", None)
    if values is not None:
        array = np.asarray(values).astype(str)
        if array.shape == (len(dataset),):
            return array.copy()
    found = [unpack_batch(dataset[index])[2] for index in range(len(dataset))]
    if any(value is None for value in found):
        return np.asarray([str(index) for index in range(len(dataset))], dtype=str)
    return np.asarray(
        [str(value.item() if isinstance(value, Tensor) else value) for value in found], dtype=str
    )


def dataset_metadata(dataset: Dataset) -> DatasetMetadata:
    class_ids = tuple(np.asarray(dataset.class_ids).tolist())
    return DatasetMetadata(
        name=str(getattr(dataset, "dataset_name", type(dataset).__name__)),
        split=str(getattr(dataset, "split", "unknown")),
        class_ids=class_ids,
        class_names=tuple(getattr(dataset, "class_names", map(str, class_ids))),
        sample_count=len(dataset),
        fingerprint=getattr(dataset, "fingerprint", None),
    )


__all__ = [
    "DatasetMetadata",
    "ImagePathDataset",
    "IndexedDataset",
    "MNISTIDXDataset",
    "SealedDataset",
    "StableIdDataset",
    "SyntheticGaussianDataset",
    "TensorSampleDataset",
    "build_dataset",
    "build_transforms",
    "dataset_labels",
    "dataset_metadata",
    "dataset_sample_ids",
    "generate_synthetic_dataset",
    "load_dataset",
    "read_idx_images",
    "read_idx_labels",
    "unpack_batch",
]
