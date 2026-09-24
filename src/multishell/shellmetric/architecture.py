"""Section 5.4 auxiliary architecture study: stages, validation, and decisions.

Stage A varies only the inner activation, Stage B varies only the output map,
and the auxiliary stage materializes the frozen winner.  Every stage reuses the
official ReLU-derived pilot/plan and the locked primary loss configuration.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import inf
from typing import Any

import numpy as np

from ..artifacts import stable_hash
from .activations import ACTIVATION_NAMES

ACTIVATION_STAGE = "activation"
OUTPUT_MAP_STAGE = "output_map"
AUXILIARY_STAGE = "auxiliary"
STAGES = (ACTIVATION_STAGE, OUTPUT_MAP_STAGE, AUXILIARY_STAGE)
STAGE_A_CANDIDATES = ("relu", "silu", "gelu_exact", "channel_radial_silu")
STAGE_B_CANDIDATES = ("linear_no_bias", "radial_power_gate")
PRIMARY_ACTIVATION = "relu"
PRIMARY_HEAD = "linear_no_bias"
SELECTION_SEEDS = frozenset({0, 1, 2})
TOLERANCE_PERCENTAGE_POINTS = 0.2

_HASH = re.compile(r"^[0-9a-f]{64}$")
_REFERENCE_KEYS = (
    "locked_loss_hash",
    "reference_plan_semantic_hash",
    "reference_plan_provenance_hash",
)
_STAGE_KEYS = {
    ACTIVATION_STAGE: {"stage", "candidates", *_REFERENCE_KEYS},
    OUTPUT_MAP_STAGE: {
        "stage",
        "candidates",
        "activation",
        "activation_decision_hash",
        *_REFERENCE_KEYS,
    },
    AUXILIARY_STAGE: {
        "stage",
        "activation",
        "embedding_head",
        "activation_decision_hash",
        "output_map_decision_hash",
        *_REFERENCE_KEYS,
    },
}
_STAGE_LABELS = {ACTIVATION_STAGE: "StageA", OUTPUT_MAP_STAGE: "StageB"}


@dataclass(frozen=True)
class ArchitectureVariant:
    """One representation-encoder architecture; ``candidate`` names a decision entry."""

    activation: str
    embedding_head: str
    candidate: str | None = None

    @property
    def is_primary(self) -> bool:
        return (self.activation, self.embedding_head) == (PRIMARY_ACTIVATION, PRIMARY_HEAD)


def architecture_section(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    return config.get("study", {}).get("architecture") or None


def stage_of(config: Mapping[str, Any]) -> str | None:
    section = architecture_section(config)
    return None if section is None else str(section["stage"])


def stage_row_name(stage: str, candidate: str) -> str:
    return f"ShellMetric-{_STAGE_LABELS[stage]}({candidate})"


def architecture_variants(config: Mapping[str, Any]) -> tuple[ArchitectureVariant, ...]:
    """Expand the (single-factor) architecture axis of a study."""

    section = architecture_section(config)
    model = config["model"]
    if section is None:
        return (
            ArchitectureVariant(str(model["backbone_activation"]), str(model["embedding_head"])),
        )
    stage = section["stage"]
    if stage == ACTIVATION_STAGE:
        return tuple(ArchitectureVariant(name, PRIMARY_HEAD, name) for name in STAGE_A_CANDIDATES)
    if stage == OUTPUT_MAP_STAGE:
        activation = str(section["activation"])
        return tuple(ArchitectureVariant(activation, name, name) for name in STAGE_B_CANDIDATES)
    return (ArchitectureVariant(str(section["activation"]), str(section["embedding_head"])),)


def validate_architecture_study(config: Mapping[str, Any]) -> None:
    """Reject anything but one locked single-factor stage in the canonical cell."""

    section = architecture_section(config)
    if section is None:
        return
    stage = section.get("stage")
    if stage not in STAGES:
        raise ValueError(f"study.architecture.stage must be one of {list(STAGES)}")
    unknown = sorted(set(section) - _STAGE_KEYS[stage])
    if unknown:
        raise ValueError(
            f"study.architecture keys {unknown} are not allowed in the {stage} stage; "
            "the suite rejects activation/output-map Cartesian products"
        )
    declared = {ACTIVATION_STAGE: STAGE_A_CANDIDATES, OUTPUT_MAP_STAGE: STAGE_B_CANDIDATES}
    if stage in declared and "candidates" in section:
        requested = tuple(section["candidates"])
        other = STAGE_B_CANDIDATES if stage == ACTIVATION_STAGE else STAGE_A_CANDIDATES
        if set(requested) & set(other):
            raise ValueError("the suite rejects an activation/output-map Cartesian product")
        if requested != declared[stage]:
            raise ValueError(f"{stage} candidates must equal {list(declared[stage])} in order")
    if (
        stage in {OUTPUT_MAP_STAGE, AUXILIARY_STAGE}
        and section.get("activation") not in ACTIVATION_NAMES
    ):
        raise ValueError("the stage must name the literal Stage-A winning activation")
    if stage == AUXILIARY_STAGE and section.get("embedding_head") not in STAGE_B_CANDIDATES:
        raise ValueError("the auxiliary stage must name the literal Stage-B winning output map")
    required = list(_REFERENCE_KEYS)
    if stage in {OUTPUT_MAP_STAGE, AUXILIARY_STAGE}:
        required.append("activation_decision_hash")
    if stage == AUXILIARY_STAGE:
        required.append("output_map_decision_hash")
    invalid = [key for key in required if not _HASH.match(str(section.get(key, "")))]
    if invalid:
        raise ValueError(
            f"study.architecture must reference 64-hex hashes for {invalid}; copy them from the "
            "primary study's provenance.json and the saved architecture decisions"
        )

    model, study, data = config["model"], config["study"], config["data"]
    primary = (model.get("backbone_activation"), model.get("embedding_head"))
    if primary != (PRIMARY_ACTIVATION, PRIMARY_HEAD):
        raise ValueError(
            "architecture stages keep model.backbone_activation=relu and "
            "model.embedding_head=linear_no_bias so the official ReLU pilot/plan is reused"
        )
    cell = (
        str(data.get("dataset")),
        str(model.get("backbone")),
        int(model.get("embedding_dim", 0)),
    )
    if cell != ("cifar100", "resnet18", 3) and not study.get("rehearsal", False):
        raise ValueError(
            "the architecture study runs only in the CIFAR-100/ResNet-18/3D cell "
            "(set study.rehearsal: true to rehearse the workflow elsewhere)"
        )
    if cell[1:] != ("resnet18", 3):
        raise ValueError("the architecture study needs the scratch ResNet-18 at d=3")
    if study.get("fixed_shell_counts") not in (None, "none", []) or study.get("controls"):
        raise ValueError("architecture stages run AutoK only: no FixedS sweep and no controls")
    crossed = study.get("baselines") or study.get("external_baselines")
    if not study.get("run_auto_k", True) or crossed:
        raise ValueError("architecture stages run AutoK only and are not crossed with baselines")
    if study.get("embedding_dims") not in (None, [3], (3,)):
        raise ValueError("architecture stages are not crossed with embedding dimensions")
    if stage != AUXILIARY_STAGE and not {int(seed) for seed in study["seeds"]} <= SELECTION_SEEDS:
        raise ValueError("Stage A/B selection uses only seeds [0, 1, 2]")


def summarize_candidate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Seed means of the four validation-only decision quantities."""

    def mean(name: str) -> float | None:
        values = [float(record[name]) for record in records if record.get(name) is not None]
        return float(np.mean(values)) if values else None

    return {
        "euclidean_1nn": mean("euclidean_1nn"),
        "map_at_r": mean("map_at_r"),
        "undefined_spoke_classes": mean("undefined_spoke_classes"),
        "mean_spoke_fraction": mean("mean_spoke_fraction"),
        "seed_count": len(records),
    }


def select_architecture_candidate(
    candidates: Mapping[str, Mapping[str, Any]],
    *,
    declared_order: Sequence[str],
    preferred_exact_tie: str,
    tolerance_percentage_points: float = TOLERANCE_PERCENTAGE_POINTS,
) -> dict[str, Any]:
    """Apply the fixed validation-only decision rule and record its trace."""

    if not candidates or set(candidates) != set(declared_order):
        raise ValueError("candidate results must match the declared candidate order exactly")
    tolerance = float(tolerance_percentage_points) / 100.0

    def value(name: str, key: str, missing: float) -> float:
        raw = candidates[name].get(key)
        return missing if raw is None else float(raw)

    best = max(value(name, "euclidean_1nn", -inf) for name in declared_order)
    retained = [
        name
        for name in declared_order
        if best - value(name, "euclidean_1nn", -inf) <= tolerance + 1.0e-15
    ]
    trace: list[dict[str, Any]] = [
        {"rule": "euclidean_1nn_within_tolerance", "best": best, "retained": list(retained)}
    ]
    for key, direction in (
        ("map_at_r", "max"),
        ("undefined_spoke_classes", "min"),
        ("mean_spoke_fraction", "min"),
    ):
        if len(retained) == 1:
            break
        missing = -inf if direction == "max" else inf
        scores = {name: value(name, key, missing) for name in retained}
        target = max(scores.values()) if direction == "max" else min(scores.values())
        retained = [name for name in retained if scores[name] == target]
        trace.append({"rule": f"{direction}_{key}", "target": target, "retained": list(retained)})
    winner = retained[0]
    if len(retained) > 1:
        winner = preferred_exact_tie if preferred_exact_tie in retained else retained[0]
        trace.append({"rule": "exact_tie_preference", "preferred": preferred_exact_tie})
    result = {
        "candidates": {name: dict(candidates[name]) for name in declared_order},
        "declared_order": list(declared_order),
        "accuracy_tolerance_percentage_points": float(tolerance_percentage_points),
        "trace": trace,
        "selected": winner,
        "selection_uses_test": False,
    }
    result["decision_hash"] = stable_hash(result)
    return result


def decide_stage(
    config: Mapping[str, Any],
    per_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Build the immutable, content-hashed decision artifact for Stage A or B."""

    section = architecture_section(config)
    stage = None if section is None else section["stage"]
    if stage not in _STAGE_LABELS:
        raise ValueError("only Stage A and Stage B produce architecture decisions")
    order = STAGE_A_CANDIDATES if stage == ACTIVATION_STAGE else STAGE_B_CANDIDATES
    preferred = PRIMARY_ACTIVATION if stage == ACTIVATION_STAGE else PRIMARY_HEAD
    decision = select_architecture_candidate(
        {name: summarize_candidate(per_candidate[name]) for name in order},
        declared_order=order,
        preferred_exact_tie=preferred,
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "architecture_decision",
        "stage": stage,
        "fixed_activation": section.get("activation") if stage == OUTPUT_MAP_STAGE else None,
        "parent_decision_hash": section.get("activation_decision_hash"),
        **{key: section[key] for key in _REFERENCE_KEYS},
        "inputs": {name: [dict(record) for record in per_candidate[name]] for name in order},
        **{key: value for key, value in decision.items() if key != "decision_hash"},
    }
    payload["decision_hash"] = stable_hash(payload)
    return payload


def verify_decision(decision: Mapping[str, Any], *, stage: str, selected: str) -> None:
    body = {key: value for key, value in decision.items() if key != "decision_hash"}
    if stable_hash(body) != decision.get("decision_hash"):
        raise ValueError("architecture decision content does not match its hash")
    if decision.get("stage") != stage or decision.get("selected") != selected:
        raise ValueError(
            f"the {stage} decision selected {decision.get('selected')!r}, not {selected!r}"
        )


__all__ = [
    "ACTIVATION_STAGE",
    "AUXILIARY_STAGE",
    "OUTPUT_MAP_STAGE",
    "STAGES",
    "STAGE_A_CANDIDATES",
    "STAGE_B_CANDIDATES",
    "ArchitectureVariant",
    "architecture_section",
    "architecture_variants",
    "decide_stage",
    "select_architecture_candidate",
    "stage_of",
    "stage_row_name",
    "summarize_candidate",
    "validate_architecture_study",
    "verify_decision",
]
