"""Raw OOF soft-confusion artifacts for ShellMetric."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .artifacts import (
    hash_array,
    hash_class_order,
    load_array_artifact,
    save_array_artifact,
    stable_hash,
)
from .splits import SplitManifest, validate_oof_coverage

CONFUSION_EPSILON = 1.0e-12


def _as_numpy(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _readonly(value: Any, *, dtype: Any) -> np.ndarray:
    array = _as_numpy(value, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _validate_oof(
    labels: Any,
    probabilities: Any,
    *,
    class_count: int | None = None,
    probability_tolerance: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray, int]:
    labels_array = _as_numpy(labels, dtype=np.int64)
    probabilities_array = _as_numpy(probabilities, dtype=np.float64)
    if labels_array.ndim != 1 or labels_array.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    if probabilities_array.ndim != 2 or probabilities_array.shape[0] != labels_array.size:
        raise ValueError("probabilities must have shape [sample_count, class_count]")

    inferred_count = probabilities_array.shape[1]
    count = inferred_count if class_count is None else int(class_count)
    if count < 2 or inferred_count != count:
        raise ValueError("probabilities must contain at least two class columns")
    if np.any(labels_array < 0) or np.any(labels_array >= count):
        raise ValueError("labels contain an index outside [0, class_count)")
    missing = np.setdiff1d(np.arange(count, dtype=np.int64), np.unique(labels_array))
    if missing.size:
        raise ValueError(f"labels contain no samples for classes {missing.tolist()}")
    if not np.isfinite(probabilities_array).all() or np.any(probabilities_array < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(
        probabilities_array.sum(axis=1),
        1.0,
        atol=float(probability_tolerance),
        rtol=float(probability_tolerance),
    ):
        raise ValueError("each probability row must sum to one")
    return labels_array, probabilities_array, count


def soft_directed_confusion(
    labels: Any,
    probabilities: Any,
    *,
    class_count: int | None = None,
    probability_tolerance: float = 1.0e-6,
) -> np.ndarray:
    """Compute ``Q[i, j] = mean(p_j(x) | y=i)`` in float64."""

    labels_array, probabilities_array, count = _validate_oof(
        labels,
        probabilities,
        class_count=class_count,
        probability_tolerance=probability_tolerance,
    )
    totals = np.zeros((count, count), dtype=np.float64)
    np.add.at(totals, labels_array, probabilities_array)
    class_sizes = np.bincount(labels_array, minlength=count).astype(np.float64)
    return totals / class_sizes[:, None]


directed_soft_confusion = soft_directed_confusion


def symmetrize_confusion(directed: Any) -> np.ndarray:
    """Return the raw symmetric off-diagonal graph ``W`` from ``Q``."""

    Q = _as_numpy(directed, dtype=np.float64)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1] or Q.shape[0] < 2:
        raise ValueError("directed confusion must be a square matrix for at least two classes")
    if not np.isfinite(Q).all() or np.any(Q < 0.0):
        raise ValueError("directed confusion must be finite and non-negative")
    W = 0.5 * (Q + Q.T)
    np.fill_diagonal(W, 0.0)
    return W


def normalize_confusion(
    confusion: Any,
    *,
    epsilon: float = CONFUSION_EPSILON,
) -> np.ndarray:
    """Normalize raw ``W`` for margins without adding or redistributing mass."""

    W = _as_numpy(confusion, dtype=np.float64)
    if W.ndim != 2 or W.shape[0] != W.shape[1] or W.shape[0] < 2:
        raise ValueError("confusion must be a square matrix for at least two classes")
    if not np.isfinite(W).all() or np.any(W < 0.0):
        raise ValueError("confusion must be finite and non-negative")
    if not np.allclose(W, W.T, atol=1.0e-12, rtol=0.0):
        raise ValueError("confusion must be symmetric")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    clean = W.copy()
    np.fill_diagonal(clean, 0.0)
    maximum = float(clean[np.triu_indices(clean.shape[0], 1)].max(initial=0.0))
    if maximum <= float(epsilon):
        return np.zeros_like(clean)
    return clean / (maximum + float(epsilon))


def confusion_components(
    labels: Any,
    probabilities: Any,
    *,
    class_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the exact ``(Q, W, W_hat, h)`` ShellMetric quantities."""

    Q = soft_directed_confusion(labels, probabilities, class_count=class_count)
    W = symmetrize_confusion(Q)
    W_hat = normalize_confusion(W)
    h = W.sum(axis=1, dtype=np.float64)
    return Q, W, W_hat, h


@dataclass(frozen=True)
class ConfusionArtifact:
    """Immutable raw-confusion artifact used by ShellMetric planning."""

    class_ids: np.ndarray
    class_names: tuple[str, ...]
    Q: np.ndarray
    W: np.ndarray
    W_hat: np.ndarray
    h: np.ndarray
    sample_ids: np.ndarray
    fold_ids: np.ndarray
    pilot_seeds: tuple[int, ...]
    split_hash: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        class_ids = _readonly(self.class_ids, dtype=np.int64)
        if class_ids.ndim != 1 or class_ids.size < 2:
            raise ValueError("class_ids must contain at least two classes")
        if np.unique(class_ids).size != class_ids.size:
            raise ValueError("class_ids must be unique")
        if len(self.class_names) != class_ids.size:
            raise ValueError("class_names length must match class_ids")

        shape = (class_ids.size, class_ids.size)
        Q = _readonly(self.Q, dtype=np.float64)
        W = _readonly(self.W, dtype=np.float64)
        W_hat = _readonly(self.W_hat, dtype=np.float64)
        h = _readonly(self.h, dtype=np.float64)
        if Q.shape != shape or W.shape != shape or W_hat.shape != shape:
            raise ValueError(f"Q, W, and W_hat must have shape {shape}")
        if h.shape != (class_ids.size,):
            raise ValueError(f"h must have shape {(class_ids.size,)}")
        if not np.isfinite(Q).all() or np.any(Q < 0.0):
            raise ValueError("Q must be finite and non-negative")
        if not np.allclose(Q.sum(axis=1), 1.0, atol=1.0e-6, rtol=1.0e-6):
            raise ValueError("Q rows must sum to one")

        expected_W = symmetrize_confusion(Q)
        expected_W_hat = normalize_confusion(expected_W)
        expected_h = expected_W.sum(axis=1, dtype=np.float64)
        if not np.allclose(W, expected_W, atol=1.0e-12, rtol=1.0e-12):
            raise ValueError("W does not match the raw symmetric off-diagonal graph from Q")
        if not np.allclose(W_hat, expected_W_hat, atol=1.0e-12, rtol=1.0e-12):
            raise ValueError("W_hat does not match normalized raw W")
        if not np.allclose(h, expected_h, atol=1.0e-12, rtol=1.0e-12):
            raise ValueError("h must equal the row sums of raw W")

        sample_ids = _readonly(self.sample_ids, dtype=str)
        fold_ids = _readonly(self.fold_ids, dtype=np.int64)
        if sample_ids.ndim != 1 or sample_ids.size == 0:
            raise ValueError("sample_ids must be a non-empty vector")
        if fold_ids.shape != sample_ids.shape:
            raise ValueError("fold_ids must have one entry per sample ID")
        if np.unique(sample_ids).size != sample_ids.size:
            raise ValueError("sample_ids must be unique")

        seeds = tuple(int(seed) for seed in self.pilot_seeds)
        if len(set(seeds)) != len(seeds):
            raise ValueError("pilot_seeds must be unique")
        split_hash = None if self.split_hash is None else str(self.split_hash)

        object.__setattr__(self, "class_ids", class_ids)
        object.__setattr__(self, "class_names", tuple(str(name) for name in self.class_names))
        object.__setattr__(self, "Q", Q)
        object.__setattr__(self, "W", W)
        object.__setattr__(self, "W_hat", W_hat)
        object.__setattr__(self, "h", h)
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "fold_ids", fold_ids)
        object.__setattr__(self, "pilot_seeds", seeds)
        object.__setattr__(self, "split_hash", split_hash)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def class_order_hash(self) -> str:
        return hash_class_order(self.class_ids.tolist(), self.class_names)

    @property
    def artifact_hash(self) -> str:
        return stable_hash(
            {
                "schema_version": 2,
                "class_order_hash": self.class_order_hash,
                "arrays": {
                    "Q": hash_array(self.Q),
                    "W": hash_array(self.W),
                    "W_hat": hash_array(self.W_hat),
                    "h": hash_array(self.h),
                    "sample_ids": hash_array(self.sample_ids),
                    "fold_ids": hash_array(self.fold_ids),
                },
                "pilot_seeds": list(self.pilot_seeds),
                "split_hash": self.split_hash,
                "metadata": dict(self.metadata),
            }
        )

    @property
    def confusion_hash(self) -> str:
        return self.artifact_hash

def build_confusion_artifact(
    labels: Any,
    probabilities: Any,
    *,
    class_ids: Any | None = None,
    class_names: Sequence[str] | None = None,
    sample_ids: Any | None = None,
    fold_ids: Any | None = None,
    pilot_seeds: Sequence[int] | None = None,
    split_hash: str | None = None,
    split_manifest: SplitManifest | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ConfusionArtifact:
    """Build the final raw-confusion artifact from averaged OOF probabilities."""

    labels_array, probabilities_array, class_count = _validate_oof(labels, probabilities)
    if split_manifest is not None:
        if sample_ids is None or fold_ids is None:
            raise ValueError("sample_ids and fold_ids are required with split_manifest")
        validate_oof_coverage(sample_ids, fold_ids, split_manifest)
        row_by_id = {
            str(sample_id): index for index, sample_id in enumerate(split_manifest.sample_ids)
        }
        expected_labels = np.asarray(
            [split_manifest.labels[row_by_id[str(sample_id)]] for sample_id in sample_ids],
            dtype=np.int64,
        )
        if not np.array_equal(labels_array, expected_labels):
            raise ValueError("OOF labels do not match the split manifest")
        if split_hash is not None and str(split_hash) != split_manifest.planning_hash:
            raise ValueError("split_hash does not match split_manifest")
        split_hash = split_manifest.planning_hash
        if class_ids is None:
            class_ids = split_manifest.class_ids

    if sample_ids is None and fold_ids is None:
        width = max(1, len(str(labels_array.size - 1)))
        sample_ids = np.asarray(
            [f"oof:{index:0{width}d}" for index in range(labels_array.size)], dtype=str
        )
        fold_ids = np.full(labels_array.size, -1, dtype=np.int64)
    elif sample_ids is None or fold_ids is None:
        raise ValueError("sample_ids and fold_ids must be supplied together")

    sample_ids_array = _as_numpy(sample_ids, dtype=str)
    fold_ids_array = _as_numpy(fold_ids, dtype=np.int64)
    if sample_ids_array.shape != labels_array.shape or fold_ids_array.shape != labels_array.shape:
        raise ValueError("sample_ids and fold_ids must have one entry per OOF example")

    ids = (
        np.arange(class_count, dtype=np.int64)
        if class_ids is None
        else _as_numpy(class_ids, dtype=np.int64)
    )
    if ids.shape != (class_count,):
        raise ValueError(f"class_ids must have shape {(class_count,)}")
    names = (
        tuple(str(class_id) for class_id in ids)
        if class_names is None
        else tuple(str(name) for name in class_names)
    )
    Q, W, W_hat, h = confusion_components(
        labels_array, probabilities_array, class_count=class_count
    )
    return ConfusionArtifact(
        class_ids=ids,
        class_names=names,
        Q=Q,
        W=W,
        W_hat=W_hat,
        h=h,
        sample_ids=sample_ids_array,
        fold_ids=fold_ids_array,
        pilot_seeds=tuple(
            int(seed) for seed in (() if pilot_seeds is None else pilot_seeds)
        ),
        split_hash=split_hash,
        metadata=dict(metadata or {}),
    )


def save_confusion_artifact(
    artifact: ConfusionArtifact,
    directory: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    arrays = {
        "class_ids": artifact.class_ids,
        "Q": artifact.Q,
        "W": artifact.W,
        "W_hat": artifact.W_hat,
        "h": artifact.h,
        "sample_ids": artifact.sample_ids,
        "fold_ids": artifact.fold_ids,
        "pilot_seeds": np.asarray(artifact.pilot_seeds, dtype=np.int64),
    }
    metadata = {
        "schema_version": 2,
        "artifact_type": "shellmetric_confusion",
        "class_names": list(artifact.class_names),
        "class_order_hash": artifact.class_order_hash,
        "split_hash": artifact.split_hash,
        "confusion_hash": artifact.artifact_hash,
        "metadata": dict(artifact.metadata),
    }
    save_array_artifact(
        directory,
        arrays,
        metadata,
        basename="confusion",
        overwrite=overwrite,
    )


def load_confusion_artifact(directory: str | Path) -> ConfusionArtifact:
    arrays, metadata = load_array_artifact(directory, basename="confusion")
    if metadata.get("artifact_type") != "shellmetric_confusion":
        raise ValueError("not a ShellMetric confusion artifact")
    artifact = ConfusionArtifact(
        class_ids=arrays["class_ids"],
        class_names=tuple(metadata["class_names"]),
        Q=arrays["Q"],
        W=arrays["W"],
        W_hat=arrays["W_hat"],
        h=arrays["h"],
        sample_ids=arrays["sample_ids"],
        fold_ids=arrays["fold_ids"],
        pilot_seeds=tuple(int(seed) for seed in arrays["pilot_seeds"]),
        split_hash=metadata.get("split_hash"),
        metadata=metadata.get("metadata", {}),
    )
    if artifact.class_order_hash != metadata.get("class_order_hash"):
        raise ValueError("confusion artifact class-order hash mismatch")
    if artifact.artifact_hash != metadata.get("confusion_hash"):
        raise ValueError("confusion artifact content hash mismatch")
    return artifact


save_confusion = save_confusion_artifact
load_confusion = load_confusion_artifact


__all__ = [
    "CONFUSION_EPSILON",
    "ConfusionArtifact",
    "build_confusion_artifact",
    "confusion_components",
    "directed_soft_confusion",
    "load_confusion",
    "load_confusion_artifact",
    "normalize_confusion",
    "save_confusion",
    "save_confusion_artifact",
    "soft_directed_confusion",
    "symmetrize_confusion",
]
