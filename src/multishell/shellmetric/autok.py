"""Deterministic paired-bootstrap shell-count selection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import load_array_artifact, save_array_artifact
from ..confusion import ConfusionArtifact, soft_directed_confusion
from .plan import ShellPlan, build_shell_plan, load_plan, save_plan, shell_capacities, smax

DEFAULT_BOOTSTRAP_REPEATS = 200
FLAT_DIFFICULTY_TOLERANCE = 1.0e-12


def _readonly_float64(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).copy()
    array.setflags(write=False)
    return array


def _seed_from_confusion_hash(confusion_hash: str) -> int:
    """Map the first 64 hash bits to the recorded NumPy bootstrap seed."""

    try:
        return int(confusion_hash[:16], 16)
    except (TypeError, ValueError) as error:
        raise ValueError("confusion artifact hash must begin with hexadecimal digits") from error


def _bootstrap_difficulties(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    class_count = probabilities.shape[1]
    class_probabilities = [probabilities[labels == index] for index in range(class_count)]
    rng = np.random.default_rng(seed)
    left = np.empty((repeats, class_count), dtype=np.float64)
    right = np.empty_like(left)

    for repetition in range(repeats):
        Q_left = np.empty((class_count, class_count), dtype=np.float64)
        Q_right = np.empty_like(Q_left)
        for class_index, values in enumerate(class_probabilities):
            sample_count = values.shape[0]
            left_indices = rng.integers(sample_count, size=sample_count)
            right_indices = rng.integers(sample_count, size=sample_count)
            Q_left[class_index] = values[left_indices].mean(axis=0, dtype=np.float64)
            Q_right[class_index] = values[right_indices].mean(axis=0, dtype=np.float64)
        left[repetition] = _difficulty_from_Q(Q_left)
        right[repetition] = _difficulty_from_Q(Q_right)
    return left, right


def _difficulty_from_Q(Q: np.ndarray) -> np.ndarray:
    """Compute raw difficulty directly from a bootstrap directed confusion."""

    W = 0.5 * (Q + Q.T)
    np.fill_diagonal(W, 0.0)
    return W.sum(axis=1, dtype=np.float64)


def _assignment(
    difficulty: np.ndarray,
    class_ids: np.ndarray,
    capacities: tuple[int, ...],
) -> np.ndarray:
    order = np.lexsort((class_ids, difficulty))
    assignment = np.empty(difficulty.size, dtype=np.int64)
    offset = 0
    for shell_index, capacity in enumerate(capacities):
        assignment[order[offset : offset + capacity]] = shell_index
        offset += capacity
    return assignment


def _directed_risk(
    planning_difficulty: np.ndarray,
    evaluation_difficulty: np.ndarray,
    class_ids: np.ndarray,
    capacities: tuple[int, ...],
) -> float:
    assignment = _assignment(planning_difficulty, class_ids, capacities)
    shell_count = len(capacities)
    shell_means = np.bincount(
        assignment,
        weights=planning_difficulty,
        minlength=shell_count,
    ) / np.asarray(capacities, dtype=np.float64)
    residual = evaluation_difficulty - shell_means[assignment]
    return float(np.mean(residual * residual, dtype=np.float64))


@dataclass(frozen=True)
class AutoKResult:
    """Auto-K decision, its complete paired risk trace, and final shell plan."""

    plan: ShellPlan
    candidate_shell_counts: tuple[int, ...]
    bootstrap_risks: np.ndarray
    mean_risks: np.ndarray
    standard_errors: np.ndarray
    normalized_full_difficulty: np.ndarray
    best_shell_count: int
    chosen_shell_count: int
    one_se_threshold: float
    bootstrap_seed: int

    def __post_init__(self) -> None:
        candidates = tuple(int(value) for value in self.candidate_shell_counts)
        expected = tuple(range(1, smax(self.plan.class_count) + 1))
        if candidates != expected:
            raise ValueError(f"candidate_shell_counts must equal {expected}")
        risks = _readonly_float64(self.bootstrap_risks)
        means = _readonly_float64(self.mean_risks)
        standard_errors = _readonly_float64(self.standard_errors)
        normalized = _readonly_float64(self.normalized_full_difficulty)
        if risks.ndim != 2 or risks.shape[1] != len(candidates):
            raise ValueError("bootstrap_risks must have shape [repetitions, candidates]")
        if risks.shape[0] < 2:
            raise ValueError("at least two bootstrap repetitions are required")
        if means.shape != (len(candidates),) or standard_errors.shape != means.shape:
            raise ValueError("risk summaries must have one value per candidate")
        if normalized.shape != (self.plan.class_count,):
            raise ValueError("normalized_full_difficulty has the wrong shape")
        values = (risks, means, standard_errors, normalized)
        if not all(np.isfinite(value).all() for value in values):
            raise ValueError("Auto-K risk values must be finite")
        if np.any(risks < 0.0) or np.any(standard_errors < 0.0):
            raise ValueError("Auto-K risks and standard errors must be non-negative")
        if int(self.best_shell_count) not in candidates:
            raise ValueError("best_shell_count is not a candidate")
        if int(self.chosen_shell_count) not in candidates:
            raise ValueError("chosen_shell_count is not a candidate")
        if self.plan.shell_count != int(self.chosen_shell_count) or self.plan.family != "autok":
            raise ValueError("plan must be the chosen Auto-K plan")

        object.__setattr__(self, "candidate_shell_counts", candidates)
        object.__setattr__(self, "bootstrap_risks", risks)
        object.__setattr__(self, "mean_risks", means)
        object.__setattr__(self, "standard_errors", standard_errors)
        object.__setattr__(self, "normalized_full_difficulty", normalized)
        object.__setattr__(self, "best_shell_count", int(self.best_shell_count))
        object.__setattr__(self, "chosen_shell_count", int(self.chosen_shell_count))
        object.__setattr__(self, "one_se_threshold", float(self.one_se_threshold))
        object.__setattr__(self, "bootstrap_seed", int(self.bootstrap_seed))

    @property
    def bootstrap_repeats(self) -> int:
        return int(self.bootstrap_risks.shape[0])

    @property
    def shell_count(self) -> int:
        return self.chosen_shell_count

    @property
    def plan_semantic_hash(self) -> str:
        return self.plan.plan_semantic_hash

    @property
    def plan_provenance_hash(self) -> str:
        return self.plan.plan_provenance_hash

    @property
    def selection_frequency(self) -> np.ndarray:
        """Stability: share of repetitions whose risk is minimized by each count."""

        winners = np.argmin(self.bootstrap_risks, axis=1)  # ties favor smaller S
        counts = np.bincount(winners, minlength=len(self.candidate_shell_counts))
        return counts / float(self.bootstrap_repeats)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "candidate_shell_counts": list(self.candidate_shell_counts),
            "mean_risks": self.mean_risks.tolist(),
            "standard_errors": self.standard_errors.tolist(),
            "selection_frequency": self.selection_frequency.tolist(),
            "best_shell_count": self.best_shell_count,
            "chosen_shell_count": self.chosen_shell_count,
            "one_se_threshold": self.one_se_threshold,
            "bootstrap_seed": self.bootstrap_seed,
            "bootstrap_repeats": self.bootstrap_repeats,
            "plan_semantic_hash": self.plan_semantic_hash,
            "plan_provenance_hash": self.plan_provenance_hash,
        }


def select_shell_count(
    labels: Any,
    probabilities: Any,
    confusion: ConfusionArtifact,
    *,
    bootstrap_repeats: int = DEFAULT_BOOTSTRAP_REPEATS,
    radius_settings: Mapping[str, Any] | None = None,
) -> AutoKResult:
    """Select ``S`` by paired class-stratified bootstrap predictive risk."""

    labels_array = np.asarray(labels, dtype=np.int64)
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    if labels_array.shape != confusion.sample_ids.shape:
        raise ValueError("OOF rows do not match the confusion artifact sample IDs")
    Q = soft_directed_confusion(
        labels_array,
        probabilities_array,
        class_count=confusion.class_ids.size,
    )
    if not np.allclose(Q, confusion.Q, atol=1.0e-12, rtol=1.0e-12):
        raise ValueError("OOF labels/probabilities do not match the confusion artifact")
    repeats = int(bootstrap_repeats)
    if repeats < 2:
        raise ValueError("bootstrap_repeats must be at least two")

    class_count = confusion.class_ids.size
    candidates = tuple(range(1, smax(class_count) + 1))
    full_difficulty = np.asarray(confusion.h, dtype=np.float64)
    minimum = float(full_difficulty.min())
    span = float(full_difficulty.max() - minimum)
    seed = _seed_from_confusion_hash(confusion.artifact_hash)

    if span < FLAT_DIFFICULTY_TOLERANCE:
        normalized_full = np.zeros_like(full_difficulty)
        risks = np.zeros((repeats, len(candidates)), dtype=np.float64)
        means = np.zeros(len(candidates), dtype=np.float64)
        standard_errors = np.zeros_like(means)
        best = chosen = 1
        threshold = 0.0
    else:
        normalized_full = (full_difficulty - minimum) / span
        left, right = _bootstrap_difficulties(
            labels_array,
            probabilities_array,
            repeats=repeats,
            seed=seed,
        )
        left = (left - minimum) / span
        right = (right - minimum) / span
        risks = np.empty((repeats, len(candidates)), dtype=np.float64)

        for candidate_index, shell_count in enumerate(candidates):
            capacities = shell_capacities(class_count, shell_count)
            for repetition in range(repeats):
                left_to_right = _directed_risk(
                    left[repetition],
                    right[repetition],
                    confusion.class_ids,
                    capacities,
                )
                right_to_left = _directed_risk(
                    right[repetition],
                    left[repetition],
                    confusion.class_ids,
                    capacities,
                )
                risks[repetition, candidate_index] = 0.5 * (left_to_right + right_to_left)

        means = risks.mean(axis=0, dtype=np.float64)
        standard_errors = risks.std(axis=0, ddof=1) / np.sqrt(float(repeats))
        best_index = int(np.argmin(means))
        best = candidates[best_index]
        threshold = float(means[best_index] + standard_errors[best_index])
        chosen_index = int(np.flatnonzero(means <= threshold)[0])
        chosen = candidates[chosen_index]

    selection = {
        "method": "paired_class_stratified_bootstrap_predictive_risk",
        "candidate_shell_counts": list(candidates),
        "bootstrap_risks": risks.tolist(),
        "mean_risks": means.tolist(),
        "standard_errors": standard_errors.tolist(),
        "best_shell_count": best,
        "chosen_shell_count": chosen,
        "one_se_threshold": threshold,
        "bootstrap_seed": seed,
        "bootstrap_repeats": repeats,
    }
    plan = build_shell_plan(
        full_difficulty,
        confusion.W,
        chosen,
        class_ids=confusion.class_ids,
        family="autok",
        confusion_hash=confusion.artifact_hash,
        selection=selection,
        radius_settings=radius_settings,
    )
    return AutoKResult(
        plan=plan,
        candidate_shell_counts=candidates,
        bootstrap_risks=risks,
        mean_risks=means,
        standard_errors=standard_errors,
        normalized_full_difficulty=normalized_full,
        best_shell_count=best,
        chosen_shell_count=chosen,
        one_se_threshold=threshold,
        bootstrap_seed=seed,
    )


def save_autok(result: AutoKResult, directory: str | Path) -> None:
    """Persist the chosen plan plus the complete bootstrap risk trace."""

    save_plan(result.plan, directory)
    save_array_artifact(
        directory,
        {
            "bootstrap_risks": result.bootstrap_risks,
            "mean_risks": result.mean_risks,
            "standard_errors": result.standard_errors,
            "normalized_full_difficulty": result.normalized_full_difficulty,
        },
        {**result.to_metadata(), "confusion_hash": result.plan.confusion_hash},
        basename="autok",
    )


def load_autok(directory: str | Path) -> AutoKResult:
    plan = load_plan(directory)
    arrays, metadata = load_array_artifact(directory, basename="autok")
    result = AutoKResult(
        plan=plan,
        candidate_shell_counts=tuple(metadata["candidate_shell_counts"]),
        bootstrap_risks=arrays["bootstrap_risks"],
        mean_risks=arrays["mean_risks"],
        standard_errors=arrays["standard_errors"],
        normalized_full_difficulty=arrays["normalized_full_difficulty"],
        best_shell_count=metadata["best_shell_count"],
        chosen_shell_count=metadata["chosen_shell_count"],
        one_se_threshold=metadata["one_se_threshold"],
        bootstrap_seed=metadata["bootstrap_seed"],
    )
    if result.plan_provenance_hash != metadata["plan_provenance_hash"]:
        raise ValueError("cached AutoK metadata references another plan")
    return result


__all__ = [
    "AutoKResult",
    "DEFAULT_BOOTSTRAP_REPEATS",
    "FLAT_DIFFICULTY_TOLERANCE",
    "load_autok",
    "save_autok",
    "select_shell_count",
]
