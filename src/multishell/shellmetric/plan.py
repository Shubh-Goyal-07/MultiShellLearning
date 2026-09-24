"""Deterministic ShellMetric shell planning.

This module assigns only shell indices.  It deliberately has no notion of a
class direction, point, proxy, center, or prototype.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import ceil, sqrt
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import hash_array, load_array_artifact, save_array_artifact, stable_hash
from ..confusion import normalize_confusion

RADIUS_PARAMETERIZATION = "learned_simplex_gaps_ordered_rms"
RADIUS_INITIALIZATION = "zero_gap_logits"


def make_radius_settings(gap_floor_fraction: float = 0.10) -> dict[str, Any]:
    """Training-relevant radius settings recorded in every plan's semantic hash."""

    fraction = float(gap_floor_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("gap_floor_fraction must lie in (0, 1)")
    return {
        "parameterization": RADIUS_PARAMETERIZATION,
        "gap_floor_fraction": fraction,
        "initialization": RADIUS_INITIALIZATION,
    }


RADIUS_SETTINGS = make_radius_settings()


def smax(class_count: int) -> int:
    """Return the largest allowed shell count, ``ceil(sqrt(C))`` capped at C."""

    class_count = int(class_count)
    if class_count < 1:
        raise ValueError("class_count must be positive")
    return min(class_count, ceil(sqrt(class_count)))


maximum_shell_count = smax


def shell_capacities(class_count: int, shell_count: int) -> tuple[int, ...]:
    """Allocate classes with the exact outward-biased integer priority rule."""

    class_count = int(class_count)
    shell_count = int(shell_count)
    if class_count < 1:
        raise ValueError("class_count must be positive")
    if not 1 <= shell_count <= smax(class_count):
        raise ValueError(f"shell_count must lie in [1, {smax(class_count)}]")

    capacities = [1] * shell_count
    for _ in range(class_count - shell_count):
        best = 1
        for candidate in range(2, shell_count + 1):
            candidate_score = candidate * candidate * (capacities[best - 1] + 1)
            best_score = best * best * (capacities[candidate - 1] + 1)
            if candidate_score > best_score or (candidate_score == best_score and candidate > best):
                best = candidate
        capacities[best - 1] += 1

    if capacities != sorted(capacities):  # Defensive assertion for future rule edits.
        raise RuntimeError("capacity rule produced a decreasing allocation")
    return tuple(capacities)


def _validated_classes(
    difficulty: Sequence[float] | np.ndarray,
    class_ids: Sequence[int] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(difficulty, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("difficulty must be a non-empty finite vector")
    ids = (
        np.arange(values.size, dtype=np.int64)
        if class_ids is None
        else np.asarray(class_ids, dtype=np.int64)
    )
    if ids.shape != values.shape or np.unique(ids).size != ids.size:
        raise ValueError("class_ids must be a unique integer vector matching difficulty")
    return values, ids


def assign_classes_to_shells(
    difficulty: Sequence[float] | np.ndarray,
    shell_count: int,
    *,
    class_ids: Sequence[int] | np.ndarray | None = None,
    capacities: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 1-based shell assignments in class-id order and sorted class IDs."""

    values, ids = _validated_classes(difficulty, class_ids)
    expected = shell_capacities(values.size, shell_count)
    chosen = expected if capacities is None else tuple(int(value) for value in capacities)
    if chosen != expected:
        raise ValueError(f"capacities must equal the deterministic allocation {list(expected)}")

    order = np.lexsort((ids, values))
    assignment = np.empty(values.size, dtype=np.int64)
    offset = 0
    for shell_index, capacity in enumerate(chosen, start=1):
        members = order[offset : offset + capacity]
        assignment[members] = shell_index
        offset += capacity
    return assignment, ids[order]


def expand_fixed_shell_counts(
    requested: str | Sequence[int] | None,
    class_count: int,
    *,
    ensure_one_shell_control: bool = False,
) -> tuple[int, ...]:
    """Resolve ``compact``, ``all``, ``none``, or an explicit FixedS list."""

    maximum = smax(class_count)
    if requested is None or requested == "none" or requested == []:
        counts: list[int] = []
    elif requested == "all":
        counts = list(range(1, maximum + 1))
    elif requested == "compact":
        if maximum <= 6:
            counts = list(range(1, maximum + 1))
        else:
            counts = [1, ceil(maximum / 4), ceil(maximum / 2), ceil(3 * maximum / 4), maximum]
    elif isinstance(requested, Sequence) and not isinstance(requested, (str, bytes)):
        counts = [int(value) for value in requested]
    else:
        raise ValueError("fixed_shell_counts must be compact, all, none, or a sequence")
    invalid = sorted({value for value in counts if not 1 <= value <= maximum})
    if invalid:
        raise ValueError(f"invalid shell counts {invalid}; expected values in [1, {maximum}]")
    result = set(counts)
    if ensure_one_shell_control:
        result.add(1)
    return tuple(sorted(result))


@dataclass(frozen=True)
class ShellPlan:
    """Immutable class-to-shell plan with separate semantic/provenance hashes."""

    class_ids: tuple[int, ...]
    shell_count: int
    capacities: tuple[int, ...]
    assignment: tuple[int, ...]
    sorted_class_ids: tuple[int, ...]
    w_hat: np.ndarray = field(repr=False, compare=False)
    family: str = "fixed_s"
    confusion_hash: str | None = None
    selection: Mapping[str, Any] = field(default_factory=dict, compare=False)
    radius_settings: Mapping[str, Any] = field(default_factory=lambda: dict(RADIUS_SETTINGS))

    def __post_init__(self) -> None:
        class_count = len(self.class_ids)
        if class_count < 1 or len(set(self.class_ids)) != class_count:
            raise ValueError("class_ids must be unique and non-empty")
        expected = shell_capacities(class_count, self.shell_count)
        if tuple(self.capacities) != expected:
            raise ValueError(f"capacities must equal {list(expected)}")
        assignment = np.asarray(self.assignment, dtype=np.int64)
        if (
            assignment.shape != (class_count,)
            or np.any(assignment < 1)
            or np.any(assignment > self.shell_count)
        ):
            raise ValueError("assignment contains an invalid shell index")
        counts = tuple(
            int(np.count_nonzero(assignment == shell)) for shell in range(1, self.shell_count + 1)
        )
        if counts != expected:
            raise ValueError("assignment occupancy does not match capacities")
        if sorted(self.sorted_class_ids) != sorted(self.class_ids):
            raise ValueError("sorted_class_ids must be a permutation of class_ids")
        original = np.asarray(self.w_hat, dtype=np.float64)
        if original.shape != (class_count, class_count):
            raise ValueError("w_hat shape must match class count")
        if (
            not np.isfinite(original).all()
            or np.any(original < 0.0)
            or np.any(original > 1.0 + 1.0e-12)
            or not np.allclose(original, original.T, atol=1.0e-12, rtol=0.0)
            or not np.allclose(np.diag(original), 0.0, atol=1.0e-12, rtol=0.0)
        ):
            raise ValueError("w_hat must be finite, symmetric, diagonal-free, and in [0, 1]")
        immutable = original.copy()
        immutable.setflags(write=False)
        object.__setattr__(self, "w_hat", immutable)
        settings = dict(self.radius_settings)
        if settings != make_radius_settings(settings.get("gap_floor_fraction", -1.0)):
            raise ValueError(f"unsupported radius settings: {settings}")
        object.__setattr__(self, "radius_settings", settings)

    @property
    def class_count(self) -> int:
        return len(self.class_ids)

    @property
    def maximum_shell_count(self) -> int:
        return smax(self.class_count)

    @property
    def semantic_payload(self) -> dict[str, Any]:
        return {
            "class_ids": list(self.class_ids),
            "shell_count": self.shell_count,
            "capacities": list(self.capacities),
            "assignment": list(self.assignment),
            "w_hat": {
                "dtype": self.w_hat.dtype.str,
                "shape": list(self.w_hat.shape),
                "hash": hash_array(self.w_hat),
            },
            "radius_settings": dict(self.radius_settings),
        }

    @property
    def plan_semantic_hash(self) -> str:
        return stable_hash(self.semantic_payload)

    @property
    def provenance_payload(self) -> dict[str, Any]:
        return {
            **self.semantic_payload,
            "family": self.family,
            "confusion_hash": self.confusion_hash,
            "selection": dict(self.selection),
            "sorted_class_ids": list(self.sorted_class_ids),
        }

    @property
    def plan_provenance_hash(self) -> str:
        return stable_hash(self.provenance_payload)

    def to_metadata(self) -> dict[str, Any]:
        return {
            **self.provenance_payload,
            "class_count": self.class_count,
            "smax": self.maximum_shell_count,
            "plan_semantic_hash": self.plan_semantic_hash,
            "plan_provenance_hash": self.plan_provenance_hash,
        }


def build_shell_plan(
    difficulty: Sequence[float] | np.ndarray,
    confusion: Any,
    shell_count: int,
    *,
    class_ids: Sequence[int] | np.ndarray | None = None,
    family: str = "fixed_s",
    confusion_hash: str | None = None,
    selection: Mapping[str, Any] | None = None,
    radius_settings: Mapping[str, Any] | None = None,
) -> ShellPlan:
    values, ids = _validated_classes(difficulty, class_ids)
    matrix = np.asarray(confusion, dtype=np.float64)
    if matrix.shape != (values.size, values.size):
        raise ValueError("confusion shape must match difficulty")
    capacities = shell_capacities(values.size, shell_count)
    assignment, sorted_ids = assign_classes_to_shells(
        values, shell_count, class_ids=ids, capacities=capacities
    )
    return ShellPlan(
        class_ids=tuple(int(value) for value in ids),
        shell_count=int(shell_count),
        capacities=capacities,
        assignment=tuple(int(value) for value in assignment),
        sorted_class_ids=tuple(int(value) for value in sorted_ids),
        w_hat=normalize_confusion(matrix),
        family=str(family),
        confusion_hash=confusion_hash,
        selection=dict(selection or {}),
        radius_settings=dict(RADIUS_SETTINGS if radius_settings is None else radius_settings),
    )


def save_plan(plan: ShellPlan, directory: str | Path, *, overwrite: bool = False) -> None:
    save_array_artifact(
        directory,
        {"w_hat": plan.w_hat},
        plan.to_metadata(),
        basename="plan",
        overwrite=overwrite,
    )


def load_plan(directory: str | Path) -> ShellPlan:
    arrays, metadata = load_array_artifact(directory, basename="plan")
    plan = ShellPlan(
        class_ids=tuple(metadata["class_ids"]),
        shell_count=int(metadata["shell_count"]),
        capacities=tuple(metadata["capacities"]),
        assignment=tuple(metadata["assignment"]),
        sorted_class_ids=tuple(metadata["sorted_class_ids"]),
        w_hat=arrays["w_hat"],
        family=metadata["family"],
        confusion_hash=metadata.get("confusion_hash"),
        selection=metadata.get("selection", {}),
        radius_settings=metadata["radius_settings"],
    )
    if plan.plan_semantic_hash != metadata["plan_semantic_hash"]:
        raise ValueError("plan semantic hash mismatch")
    if plan.plan_provenance_hash != metadata["plan_provenance_hash"]:
        raise ValueError("plan provenance hash mismatch")
    return plan


__all__ = [
    "RADIUS_SETTINGS",
    "ShellPlan",
    "assign_classes_to_shells",
    "build_shell_plan",
    "expand_fixed_shell_counts",
    "load_plan",
    "make_radius_settings",
    "maximum_shell_count",
    "save_plan",
    "shell_capacities",
    "smax",
]
