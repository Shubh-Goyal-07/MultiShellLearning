"""Deterministic class-balanced P x K batch schedules.

Every epoch has its own seeded manifest, so the paired schedule shared by
ShellMetric and the metric baselines covers the training split instead of
replaying one fixed subset.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from math import ceil
from typing import Any

import numpy as np
from torch.utils.data import Sampler

from ..artifacts import hash_array, stable_hash
from ..reproducibility import derive_stage_seed

PK_ALGORITHM = "pk_cyclic_permutation_queues_v1"

Manifest = tuple[tuple[int, ...], ...]


def default_pk(class_count: int, target_batch_size: int = 128) -> tuple[int, int]:
    """Return the protocol's ``P=min(C,32)`` and ``K=max(4, floor(128/P))``."""

    class_count = int(class_count)
    if class_count < 2:
        raise ValueError("P x K sampling requires at least two classes")
    classes_per_batch = min(class_count, 32)
    return classes_per_batch, max(4, int(target_batch_size) // classes_per_batch)


def resolve_pk(
    class_count: int,
    classes_per_batch: int | None = None,
    samples_per_class: int | None = None,
    target_batch_size: int = 128,
) -> tuple[int, int]:
    """Fill unset P/K from :func:`default_pk` and validate explicit values."""

    default_p, default_k = default_pk(class_count, target_batch_size)
    p = default_p if classes_per_batch is None else int(classes_per_batch)
    k = default_k if samples_per_class is None else int(samples_per_class)
    if not 2 <= p <= class_count:
        raise ValueError(f"classes_per_batch must lie in [2, {class_count}]")
    if k < 2:
        raise ValueError("samples_per_class must be at least 2")
    return p, k


class _CyclicQueue:
    """Draw distinct items from successive seeded permutations of a pool."""

    def __init__(self, items: np.ndarray, rng: np.random.Generator) -> None:
        self.items = items
        self.rng = rng
        self.order = rng.permutation(items)
        self.position = 0

    def draw(self, count: int) -> np.ndarray:
        head = self.order[self.position : self.position + count]
        self.position += head.size
        missing = count - head.size
        if missing == 0:
            return head
        fresh = self.rng.permutation(self.items)
        tail = fresh[~np.isin(fresh, head)][:missing]
        self.order = np.concatenate((tail, fresh[~np.isin(fresh, tail)]))
        self.position = missing
        return np.concatenate((head, tail))


def _contiguous_labels(labels: Sequence[int] | np.ndarray) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("labels must be a non-empty vector")
    if not np.array_equal(np.unique(values), np.arange(int(values.max()) + 1)):
        raise ValueError("labels must use contiguous class indices beginning at zero")
    return values


def build_batch_manifest(
    labels: Sequence[int] | np.ndarray,
    *,
    seed: int,
    steps: int | None = None,
    classes_per_batch: int | None = None,
    samples_per_class: int | None = None,
    target_batch_size: int = 128,
) -> Manifest:
    """Build one epoch of P x K batches.

    Classes are taken from a cyclic queue of class permutations, so every class
    appears equally often; samples are taken from per-class permutation queues,
    so an epoch visits (nearly) every sample. Classes smaller than ``K`` are
    drawn with replacement. An epoch has ``ceil(N / target_batch_size)`` steps.
    """

    values = _contiguous_labels(labels)
    class_count = int(values.max()) + 1
    p, k = resolve_pk(class_count, classes_per_batch, samples_per_class, target_batch_size)
    batch_steps = ceil(values.size / int(target_batch_size)) if steps is None else int(steps)
    if batch_steps < 1:
        raise ValueError("steps must be positive")
    rng = np.random.default_rng(int(seed))
    class_queue = _CyclicQueue(np.arange(class_count), rng)
    members = [np.flatnonzero(values == label) for label in range(class_count)]
    sample_queues = [_CyclicQueue(pool, rng) if pool.size >= k else None for pool in members]
    batches: list[tuple[int, ...]] = []
    for _ in range(batch_steps):
        indices: list[int] = []
        for label in class_queue.draw(p):
            queue = sample_queues[int(label)]
            chosen = (
                rng.choice(members[int(label)], size=k, replace=True)
                if queue is None
                else queue.draw(k)
            )
            indices.extend(int(index) for index in chosen)
        batches.append(tuple(indices))
    return tuple(batches)


@dataclass(frozen=True)
class PKSchedule:
    """Seeded per-epoch P x K manifests over a fixed training label vector."""

    labels: np.ndarray = field(repr=False, compare=False)
    seed: int
    classes_per_batch: int
    samples_per_class: int
    steps_per_epoch: int

    @classmethod
    def create(
        cls,
        labels: Sequence[int] | np.ndarray,
        *,
        seed: int,
        classes_per_batch: int | None = None,
        samples_per_class: int | None = None,
        target_batch_size: int = 128,
    ) -> PKSchedule:
        values = _contiguous_labels(labels).copy()
        values.setflags(write=False)
        p, k = resolve_pk(
            int(values.max()) + 1, classes_per_batch, samples_per_class, target_batch_size
        )
        return cls(values, int(seed), p, k, ceil(values.size / int(target_batch_size)))

    def epoch(self, epoch: int) -> Manifest:
        """Return the manifest for a 1-based epoch number."""

        return build_batch_manifest(
            self.labels,
            seed=derive_stage_seed(self.seed, "pk_batches", int(epoch)),
            steps=self.steps_per_epoch,
            classes_per_batch=self.classes_per_batch,
            samples_per_class=self.samples_per_class,
        )

    @property
    def spec(self) -> dict[str, Any]:
        return {
            "algorithm": PK_ALGORITHM,
            "labels_hash": hash_array(self.labels),
            "seed": self.seed,
            "classes_per_batch": self.classes_per_batch,
            "samples_per_class": self.samples_per_class,
            "steps_per_epoch": self.steps_per_epoch,
        }

    @property
    def schedule_hash(self) -> str:
        return stable_hash(self.spec)


class ManifestBatchSampler(Sampler[list[int]]):
    """Replay an explicit batch manifest exactly."""

    def __init__(self, manifest: Sequence[Sequence[int]], *, dataset_size: int) -> None:
        self.manifest = tuple(tuple(int(index) for index in batch) for batch in manifest)
        if not self.manifest or any(not batch for batch in self.manifest):
            raise ValueError("manifest must contain non-empty batches")
        if any(not 0 <= index < dataset_size for batch in self.manifest for index in batch):
            raise ValueError("manifest contains an out-of-range dataset index")

    def __iter__(self) -> Iterator[list[int]]:
        return iter([list(batch) for batch in self.manifest])

    def __len__(self) -> int:
        return len(self.manifest)


__all__ = [
    "PK_ALGORITHM",
    "ManifestBatchSampler",
    "PKSchedule",
    "build_batch_manifest",
    "default_pk",
    "resolve_pk",
]
