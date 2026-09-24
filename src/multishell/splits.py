"""Immutable stratified train/validation/test manifests with OOF folds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import hash_class_order, load_json, save_json, stable_hash

TRAIN_PARTITION = "train"
VALIDATION_PARTITION = "validation"
TEST_PARTITION = "test"
SPLIT_SEED = 12345


def _readonly(value: Any, dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class SplitManifest:
    sample_ids: np.ndarray
    labels: np.ndarray
    partitions: np.ndarray
    fold_ids: np.ndarray
    class_ids: np.ndarray
    split_seed: int = SPLIT_SEED
    validation_fraction: float = 0.10
    n_folds: int = 3
    dataset_name: str = "dataset"
    dataset_version: str | None = None
    train_checksum: str | None = None
    test_checksum: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_ids", _readonly(self.sample_ids, str))
        object.__setattr__(self, "labels", _readonly(self.labels, np.int64))
        object.__setattr__(self, "partitions", _readonly(self.partitions, str))
        object.__setattr__(self, "fold_ids", _readonly(self.fold_ids, np.int64))
        object.__setattr__(self, "class_ids", _readonly(self.class_ids, np.int64))
        self.validate()

    def validate(self) -> None:
        count = self.sample_ids.size
        if count == 0 or any(
            array.ndim != 1
            for array in (self.sample_ids, self.labels, self.partitions, self.fold_ids, self.class_ids)
        ):
            raise ValueError("manifest fields must be non-empty vectors")
        if not (self.labels.size == self.partitions.size == self.fold_ids.size == count):
            raise ValueError("manifest row fields must have equal length")
        if np.unique(self.sample_ids).size != count:
            raise ValueError("sample IDs must be globally unique")
        if self.split_seed != SPLIT_SEED:
            raise ValueError(f"split_seed must be {SPLIT_SEED}")
        if self.n_folds != 3:
            raise ValueError("the pilot manifest must contain exactly 3 folds")
        if not np.isclose(self.validation_fraction, 0.10):
            raise ValueError("validation_fraction must be 0.10")
        if set(np.unique(self.partitions)) != {
            TRAIN_PARTITION,
            VALIDATION_PARTITION,
            TEST_PARTITION,
        }:
            raise ValueError("manifest must contain train, validation, and test rows")
        train = self.train_mask
        held_out = self.validation_mask | self.test_mask
        if np.any(self.fold_ids[held_out] != -1):
            raise ValueError("validation/test rows must have fold_id=-1")
        if np.any((self.fold_ids[train] < 0) | (self.fold_ids[train] >= self.n_folds)):
            raise ValueError("training rows must have a valid OOF fold")
        if set(np.unique(self.fold_ids[train])) != set(range(self.n_folds)):
            raise ValueError("every OOF fold must be represented")
        if self.class_ids.size < 2 or np.unique(self.class_ids).size != self.class_ids.size:
            raise ValueError("class_ids must contain at least two unique values")
        expected = np.arange(self.class_ids.size, dtype=np.int64)
        if np.any(self.labels < 0) or np.any(self.labels >= expected.size):
            raise ValueError("labels must be contiguous internal indices")
        for class_index in expected:
            if not np.any(self.labels[self.train_mask] == class_index):
                raise ValueError(f"class {class_index} is absent from train")
            if not np.any(self.labels[self.validation_mask] == class_index):
                raise ValueError(f"class {class_index} is absent from validation")

    @property
    def train_mask(self) -> np.ndarray:
        return self.partitions == TRAIN_PARTITION

    @property
    def validation_mask(self) -> np.ndarray:
        return self.partitions == VALIDATION_PARTITION

    @property
    def test_mask(self) -> np.ndarray:
        return self.partitions == TEST_PARTITION

    @property
    def train_indices(self) -> np.ndarray:
        return np.flatnonzero(self.train_mask)

    @property
    def validation_indices(self) -> np.ndarray:
        return np.flatnonzero(self.validation_mask)

    @property
    def test_indices(self) -> np.ndarray:
        return np.flatnonzero(self.test_mask)

    @property
    def class_order_hash(self) -> str:
        return hash_class_order(self.class_ids.tolist())

    @property
    def manifest_hash(self) -> str:
        return stable_hash(self.to_dict(include_hash=False))

    @property
    def planning_hash(self) -> str:
        """Hash only information permitted to influence pilot/confusion/planning."""

        train = self.train_mask
        return stable_hash(
            {
                "dataset_name": self.dataset_name,
                "dataset_version": self.dataset_version,
                "train_checksum": self.train_checksum,
                "train_sample_ids": self.sample_ids[train].tolist(),
                "train_labels": self.labels[train].tolist(),
                "train_fold_ids": self.fold_ids[train].tolist(),
                "class_ids": self.class_ids.tolist(),
                "split_seed": self.split_seed,
                "n_folds": self.n_folds,
            }
        )

    def fold_holdout_indices(self, fold_id: int) -> np.ndarray:
        self._validate_fold(fold_id)
        return np.flatnonzero(self.train_mask & (self.fold_ids == fold_id))

    def fold_train_indices(self, fold_id: int) -> np.ndarray:
        self._validate_fold(fold_id)
        return np.flatnonzero(self.train_mask & (self.fold_ids != fold_id))

    def _validate_fold(self, fold_id: int) -> None:
        if not 0 <= int(fold_id) < self.n_folds:
            raise ValueError(f"fold_id must lie in [0, {self.n_folds})")

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": 2,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "train_checksum": self.train_checksum,
            "test_checksum": self.test_checksum,
            "sample_ids": self.sample_ids.tolist(),
            "labels": self.labels.tolist(),
            "partitions": self.partitions.tolist(),
            "fold_ids": self.fold_ids.tolist(),
            "class_ids": self.class_ids.tolist(),
            "class_order_hash": self.class_order_hash,
            "planning_hash": self.planning_hash,
            "split_seed": self.split_seed,
            "validation_fraction": self.validation_fraction,
            "n_folds": self.n_folds,
        }
        if include_hash:
            payload["manifest_hash"] = stable_hash(payload)
        return payload


def make_split_manifest(
    labels: Any,
    *,
    sample_ids: Any,
    test_labels: Any,
    test_sample_ids: Any,
    validation_fraction: float = 0.10,
    n_folds: int = 3,
    seed: int = SPLIT_SEED,
    class_ids: Any | None = None,
    dataset_name: str = "dataset",
    dataset_version: str | None = None,
    train_checksum: str | None = None,
    test_checksum: str | None = None,
) -> SplitManifest:
    """Split official training rows and append the untouched official test rows."""

    train_labels = np.asarray(labels, dtype=np.int64)
    train_ids = np.asarray(sample_ids, dtype=str)
    sealed_labels = np.asarray(test_labels, dtype=np.int64)
    sealed_ids = np.asarray(test_sample_ids, dtype=str)
    if train_labels.ndim != 1 or sealed_labels.ndim != 1 or train_labels.size == 0 or sealed_labels.size == 0:
        raise ValueError("official train and test labels must be non-empty vectors")
    if train_ids.shape != train_labels.shape or sealed_ids.shape != sealed_labels.shape:
        raise ValueError("sample IDs must align with their official split")
    if np.intersect1d(train_ids, sealed_ids).size or np.unique(np.concatenate((train_ids, sealed_ids))).size != train_ids.size + sealed_ids.size:
        raise ValueError("official train/test sample IDs must be unique and disjoint")
    classes = np.arange(int(train_labels.max()) + 1, dtype=np.int64) if class_ids is None else np.asarray(class_ids, dtype=np.int64)
    if not np.array_equal(np.unique(train_labels), np.arange(classes.size)):
        raise ValueError("official training labels must use every contiguous internal class index")
    if np.any(sealed_labels < 0) or np.any(sealed_labels >= classes.size):
        raise ValueError("official test labels contain an invalid class index")
    if seed != SPLIT_SEED or n_folds != 3 or not np.isclose(validation_fraction, 0.10):
        raise ValueError("final protocol fixes seed=12345, validation_fraction=0.10, and folds=3")

    rng = np.random.default_rng(seed)
    train_partitions = np.full(train_labels.size, TRAIN_PARTITION, dtype="<U10")
    train_folds = np.full(train_labels.size, -1, dtype=np.int64)
    for class_index in range(classes.size):
        indices = np.flatnonzero(train_labels == class_index)
        validation_count = min(max(int(round(indices.size * validation_fraction)), 1), indices.size - 1)
        shuffled = rng.permutation(indices)
        validation_indices = shuffled[:validation_count]
        training_indices = rng.permutation(shuffled[validation_count:])
        if training_indices.size < n_folds:
            raise ValueError(f"class {class_index} has fewer than {n_folds} training examples")
        train_partitions[validation_indices] = VALIDATION_PARTITION
        train_folds[training_indices] = np.arange(training_indices.size) % n_folds

    return SplitManifest(
        sample_ids=np.concatenate((train_ids, sealed_ids)),
        labels=np.concatenate((train_labels, sealed_labels)),
        partitions=np.concatenate(
            (train_partitions, np.full(sealed_labels.size, TEST_PARTITION, dtype="<U10"))
        ),
        fold_ids=np.concatenate((train_folds, np.full(sealed_labels.size, -1, dtype=np.int64))),
        class_ids=classes,
        split_seed=seed,
        validation_fraction=validation_fraction,
        n_folds=n_folds,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        train_checksum=train_checksum,
        test_checksum=test_checksum,
    )


def save_split_manifest(manifest: SplitManifest, path: str | Path, *, overwrite: bool = False) -> Path:
    return save_json(path, manifest.to_dict(), overwrite=overwrite)


def load_split_manifest(path: str | Path) -> SplitManifest:
    payload = load_json(path)
    recorded_hash = payload.pop("manifest_hash", None)
    if recorded_hash is not None and stable_hash(payload) != recorded_hash:
        raise ValueError("split manifest hash does not match its content")
    payload.pop("schema_version", None)
    payload.pop("class_order_hash", None)
    recorded_planning_hash = payload.pop("planning_hash", None)
    manifest = SplitManifest(**payload)
    if recorded_hash is not None and manifest.manifest_hash != recorded_hash:
        raise ValueError("loaded split manifest does not reproduce its hash")
    if recorded_planning_hash is not None and manifest.planning_hash != recorded_planning_hash:
        raise ValueError("loaded split manifest does not reproduce its planning hash")
    return manifest


def validate_oof_coverage(sample_ids: Any, fold_ids: Any, manifest: SplitManifest) -> None:
    ids = np.asarray(sample_ids, dtype=str)
    folds = np.asarray(fold_ids, dtype=np.int64)
    if ids.ndim != 1 or folds.shape != ids.shape or np.unique(ids).size != ids.size:
        raise ValueError("OOF IDs/folds must be equal-length unique vectors")
    expected_ids = manifest.sample_ids[manifest.train_mask]
    if set(ids) != set(expected_ids):
        raise ValueError("OOF predictions must cover every training sample exactly once")
    manifest_row = {sample_id: index for index, sample_id in enumerate(manifest.sample_ids)}
    for sample_id, fold_id in zip(ids, folds, strict=True):
        if int(fold_id) != int(manifest.fold_ids[manifest_row[sample_id]]):
            raise ValueError(f"sample {sample_id!r} was not predicted by its held-out fold")


def validate_fold_training_ids(sample_ids: Any, fold_id: int, manifest: SplitManifest) -> None:
    observed = set(np.asarray(sample_ids, dtype=str))
    expected = set(manifest.sample_ids[manifest.fold_train_indices(fold_id)])
    if observed != expected:
        raise ValueError("pilot training IDs violate the immutable fold manifest")


def assert_disjoint_sample_ids(**groups: Any) -> None:
    normalized = {name: set(np.asarray(ids, dtype=str)) for name, ids in groups.items()}
    for index, left in enumerate(normalized):
        for right in list(normalized)[index + 1 :]:
            overlap = normalized[left] & normalized[right]
            if overlap:
                raise ValueError(f"sample ID overlap between {left} and {right}: {sorted(overlap)[:10]}")


__all__ = [
    "SPLIT_SEED",
    "TEST_PARTITION",
    "TRAIN_PARTITION",
    "VALIDATION_PARTITION",
    "SplitManifest",
    "assert_disjoint_sample_ids",
    "load_split_manifest",
    "make_split_manifest",
    "save_split_manifest",
    "validate_fold_training_ids",
    "validate_oof_coverage",
]
