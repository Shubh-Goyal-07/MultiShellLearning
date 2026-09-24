"""Study expansion: reporting rows, semantic training jobs, aliases, and gates.

A reporting row is what a table prints; a training job is what a GPU runs.
Rows are expanded over seeds, embedding dimensions, the single-factor
architecture axis, controls, and baselines, then deduplicated into jobs by the
hash of everything that can change representation training (Section 7).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import hash_array, save_json, stable_hash
from ..baselines import BASELINES, DISPLAY_NAMES, EXTERNAL_BASELINES
from ..confusion import ConfusionArtifact
from ..models import EXTERNAL_BACKBONES
from ..reproducibility import derive_stage_seed
from ..train_baseline import BaselineTrainingConfig, normalize_baseline_method
from .architecture import (
    AUXILIARY_STAGE,
    ArchitectureVariant,
    architecture_variants,
    stage_of,
    stage_row_name,
)
from .autok import AutoKResult
from .plan import ShellPlan, build_shell_plan, expand_fixed_shell_counts

SHELLMETRIC = "shellmetric"
BASELINE = "baseline"
READY = "ready"
NOT_APPLICABLE = "not_applicable"
BLOCKED_EXTERNAL_PIN = "blocked_external_pin"
REQUIRES_EXTERNAL_ADAPTER = "requires_external_adapter"
AUTOK = "ShellMetric-AutoK"
SHUFFLED = "ShellMetric-AutoK-ShuffledConfusion"
NO_SHELL_LOSS = "ShellMetric-AutoK-NoShellLoss"
ACTIVATION_VARIANT = "ShellMetric-ActivationVariant"
DIMENSION_PRESETS = {
    "dimension_curve": (2, 3, 8, 16, 32, 128, 512, 1024),
    "core": (3, 32, 128),
}
CONTROLS = ("shuffled_confusion", "no_shell_loss")
NON_SEMANTIC_TRAINING_KEYS = frozenset({"keep_epoch_checkpoints"})


def fixed_s_name(shell_count: int) -> str:
    return f"ShellMetric-FixedS({shell_count})"


def semantic_training(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config["training"].items()
        if key not in NON_SEMANTIC_TRAINING_KEYS
    }


def locked_recipe_hash(config: Mapping[str, Any]) -> str:
    """Hash of the locked representation recipe: loss, sampler, optimizer, budget."""

    return stable_hash(
        {
            "loss": config["loss"],
            "sampling": config["sampling"],
            "training": semantic_training(config),
        }
    )


@dataclass(frozen=True)
class EncoderSpec:
    embedding_dim: int
    backbone_activation: str
    embedding_head: str


@dataclass(frozen=True)
class ReportingRow:
    row_id: str
    method: str
    seed: int | None
    embedding_dim: int | None
    training_job_id: str | None
    status: str = READY
    table: str = "controlled"
    control: bool = False
    evaluation_scope: str = "test_eligible"
    plan_provenance_hash: str | None = None
    plan_semantic_hash: str | None = None
    parents: Mapping[str, str] | None = None
    architecture: Mapping[str, str] | None = None
    note: str | None = None
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainingJob:
    job_id: str
    kind: str
    method: str
    seed: int
    encoder: EncoderSpec
    spec: Mapping[str, Any] = field(repr=False)
    plan_key: str | None = None
    plan_semantic_hash: str | None = None
    no_shell_loss: bool = False
    control: bool = False
    aliases: tuple[str, ...] = ()
    plan_provenance_hashes: tuple[str, ...] = ()

    @property
    def uses_pk_schedule(self) -> bool:
        return self.spec["sampler"]["kind"] == "pk"


@dataclass(frozen=True)
class StudyManifest:
    config_hash: str
    confusion_hash: str | None
    split_hash: str
    reporting_rows: tuple[ReportingRow, ...]
    training_jobs: tuple[TrainingJob, ...]
    fixed_shell_counts: tuple[int, ...]
    embedding_dims: tuple[int, ...]
    autok_shell_count: int | None
    architecture_stage: str | None = None
    comparisons: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        rows = [asdict(row) for row in self.reporting_rows]
        return {
            "schema_version": 2,
            "config_hash": self.config_hash,
            "confusion_hash": self.confusion_hash,
            "split_hash": self.split_hash,
            "fixed_shell_counts": list(self.fixed_shell_counts),
            "embedding_dims": list(self.embedding_dims),
            "autok_shell_count": self.autok_shell_count,
            "architecture_stage": self.architecture_stage,
            "reporting_row_count": len(rows),
            "unique_training_job_count": len(self.training_jobs),
            "status_counts": {
                status: sum(row["status"] == status for row in rows)
                for status in sorted({row["status"] for row in rows})
            },
            "comparisons": [dict(item) for item in self.comparisons],
            "reporting_rows": rows,
            "training_jobs": [asdict(job) for job in self.training_jobs],
        }


def expand_embedding_dims(requested: Any, default: int) -> tuple[int, ...]:
    """Resolve ``None`` (the model dimension), a named preset, or an explicit list."""

    if requested is None:
        return (int(default),)
    if isinstance(requested, str):
        if requested not in DIMENSION_PRESETS:
            raise ValueError(f"embedding_dims preset must be one of {sorted(DIMENSION_PRESETS)}")
        return DIMENSION_PRESETS[requested]
    values = sorted({int(value) for value in requested})
    if not values or values[0] < 1:
        raise ValueError("embedding_dims must contain positive integers")
    return tuple(values)


def expand_baselines(requested: Any) -> tuple[str, ...]:
    if requested in (None, "none", []):
        return ()
    if requested == "all":
        return BASELINES
    names = [normalize_baseline_method(name) for name in requested]
    return tuple(name for name in BASELINES if name in names)


def external_status(config: Mapping[str, Any], name: str) -> str:
    """Gate status of an externally pinned protocol; never picks a 'latest' pin."""

    pins = config.get("external_pins") or {}
    if name == "vit":
        return REQUIRES_EXTERNAL_ADAPTER if pins.get("vit") else BLOCKED_EXTERNAL_PIN
    pin = pins.get("hyperspacex")
    if not pin:
        return BLOCKED_EXTERNAL_PIN
    if name == "hyperspacex_matched" and not (
        pin.get("official_reproduction_hash") and pin.get("matched_deviations")
    ):
        return BLOCKED_EXTERNAL_PIN  # Official reproduced first; every deviation recorded
    return REQUIRES_EXTERNAL_ADAPTER


def canonical_control_cell(config: Mapping[str, Any], dims: Sequence[int]) -> bool:
    data, model = config["data"], config["model"]
    return (
        str(data.get("dataset")) == "cifar100"
        and str(model.get("backbone")) == "resnet18"
        and tuple(dims) == (3,)
        and stage_of(config) is None
    )


def job_spec(
    config: Mapping[str, Any],
    *,
    kind: str,
    method: str,
    seed: int,
    encoder: EncoderSpec,
    split_hash: str,
    schedule_hash: str | None,
    plan: ShellPlan | None = None,
    no_shell_loss_w_hat: np.ndarray | None = None,
) -> dict[str, Any]:
    """Everything that can change representation training, and nothing else."""

    data, model = config["data"], config["model"]
    spec: dict[str, Any] = {
        "kind": kind,
        "method": method,
        "seed": int(seed),
        "data": {
            "dataset": data["dataset"],
            "version": data.get("version"),
            "split_hash": split_hash,
            "input_shape": data.get("input_shape"),
        },
        "model": {
            "backbone": model["backbone"],
            "pretrained": bool(model.get("pretrained", False)),
            "pretrained_weights": model.get("pretrained_weights"),
            "backbone_activation": encoder.backbone_activation,
            "embedding_dim": encoder.embedding_dim,
            "embedding_head": encoder.embedding_head,
        },
        "training": semantic_training(config),
    }
    if kind == SHELLMETRIC:
        no_shell_loss = no_shell_loss_w_hat is not None
        spec["loss"] = {**config["loss"], **({"shell_weight": 0.0} if no_shell_loss else {})}
        spec["plan"] = (
            {"w_hat_hash": hash_array(no_shell_loss_w_hat), "radii": None}
            if no_shell_loss
            else plan.plan_semantic_hash  # type: ignore[union-attr]
        )
        spec["sampler"] = {"kind": "pk", "schedule_hash": schedule_hash}
        return spec
    baseline = BaselineTrainingConfig.from_config(config, method)
    spec["objective"] = baseline.objective
    spec["sampler"] = (
        {"kind": "pk", "schedule_hash": schedule_hash}
        if baseline.uses_pk_sampler
        else {"kind": "shuffled", "batch_size": baseline.batch_size}
    )
    return spec


def _derangement(rng: np.random.Generator, class_count: int) -> np.ndarray:
    while True:  # expected ~e draws; a derangement exists for every C >= 2
        permutation = rng.permutation(class_count)
        if np.all(permutation != np.arange(class_count)):
            return permutation


def shuffled_confusion_plan(
    confusion: ConfusionArtifact,
    autok_plan: ShellPlan,
    *,
    seed: int,
    attempts: int = 100,
) -> tuple[ShellPlan, tuple[int, ...]] | None:
    """Pair one seed with a deterministic derangement, or ``None`` if invariant.

    The real AutoK count and capacities are frozen; assignment and margins are
    recomputed from ``W[pi, pi]``.
    """

    class_count = confusion.class_ids.size
    rng = np.random.default_rng(
        derive_stage_seed(int(seed), "shuffled_confusion", confusion.artifact_hash)
    )
    for _ in range(attempts):
        permutation = _derangement(rng, class_count)
        shuffled = confusion.W[np.ix_(permutation, permutation)]
        if np.array_equal(shuffled, confusion.W):
            continue
        plan = build_shell_plan(
            shuffled.sum(axis=1),
            shuffled,
            autok_plan.shell_count,
            class_ids=confusion.class_ids,
            family="autok_shuffled_confusion",
            confusion_hash=confusion.artifact_hash,
            selection={
                "parent_plan_provenance_hash": autok_plan.plan_provenance_hash,
                "parent_plan_semantic_hash": autok_plan.plan_semantic_hash,
                "permutation": permutation.tolist(),
                "seed": int(seed),
            },
            radius_settings=autok_plan.radius_settings,
        )
        if plan.assignment != autok_plan.assignment or not np.allclose(
            plan.w_hat, autok_plan.w_hat
        ):
            return plan, tuple(int(value) for value in permutation)
    return None


class _Builder:
    """Accumulate rows and deduplicate their jobs by semantic spec hash."""

    def __init__(self) -> None:
        self.rows: list[ReportingRow] = []
        self.jobs: dict[str, TrainingJob] = {}
        self.provenance: dict[str, set[str]] = {}

    def add(self, row: ReportingRow, job: TrainingJob | None = None) -> None:
        if job is not None:
            existing = self.jobs.get(job.job_id, job)
            self.jobs[job.job_id] = replace(existing, aliases=(*existing.aliases, row.row_id))
            if row.plan_provenance_hash is not None:
                self.provenance.setdefault(job.job_id, set()).add(row.plan_provenance_hash)
        self.rows.append(row)

    def finish(self) -> tuple[tuple[ReportingRow, ...], tuple[TrainingJob, ...]]:
        jobs = {
            job_id: replace(
                job, plan_provenance_hashes=tuple(sorted(self.provenance.get(job_id, ())))
            )
            for job_id, job in self.jobs.items()
        }
        rows = tuple(
            row
            if row.training_job_id is None
            else replace(
                row,
                aliases=tuple(
                    alias for alias in jobs[row.training_job_id].aliases if alias != row.row_id
                ),
            )
            for row in self.rows
        )
        if len({row.row_id for row in rows}) != len(rows):
            raise RuntimeError("study expansion produced a duplicate reporting row")
        return rows, tuple(jobs[job_id] for job_id in sorted(jobs))


def row_id(method: str, embedding_dim: int | None, seed: int | None) -> str:
    if seed is None:
        return f"{method}/reference"
    return f"{method}/d={embedding_dim}/seed={seed}"


def build_study_manifest(
    config: Mapping[str, Any],
    confusion: ConfusionArtifact,
    autok: AutoKResult,
    *,
    split_hash: str,
    schedule_hashes: Mapping[int, str] | None = None,
) -> tuple[StudyManifest, dict[str, ShellPlan]]:
    """Expand every reporting row and its deduplicated encoder-training job."""

    study, model = config["study"], config["model"]
    class_count = confusion.class_ids.size
    fixed_counts = expand_fixed_shell_counts(
        study.get("fixed_shell_counts"),
        class_count,
        ensure_one_shell_control=bool(study.get("ensure_one_shell_control", False)),
    )
    dims = expand_embedding_dims(study.get("embedding_dims"), model["embedding_dim"])
    seeds = tuple(int(seed) for seed in study["seeds"])
    if len(set(seeds)) != len(seeds):
        raise ValueError("study seeds must be unique")
    controls = tuple(study.get("controls") or ())
    if controls and not (canonical_control_cell(config, dims) or study.get("rehearsal", False)):
        raise ValueError(
            "controls are allowed only in the CIFAR-100/ResNet-18/3D canonical cell "
            "(or with study.rehearsal: true)"
        )
    if controls and (1 not in fixed_counts or not study.get("run_auto_k", True)):
        raise ValueError(
            "the canonical control suite needs AutoK and FixedS(1); "
            "set run_auto_k and ensure_one_shell_control"
        )
    stage = stage_of(config)
    variants = architecture_variants(config)
    baselines = expand_baselines(study.get("baselines"))
    external = tuple(study.get("external_baselines") or ())
    schedule_hashes = dict(schedule_hashes or {})

    plans: dict[str, ShellPlan] = {
        fixed_s_name(count): build_shell_plan(
            confusion.h,
            confusion.W,
            count,
            class_ids=confusion.class_ids,
            family="fixed_s",
            confusion_hash=confusion.artifact_hash,
            selection={"requested_shell_count": count},
            radius_settings=autok.plan.radius_settings,
        )
        for count in fixed_counts
    }
    if study.get("run_auto_k", True):
        plans[AUTOK] = autok.plan
    control_plans: dict[str, ShellPlan] = {}  # per-seed; never expanded into study rows
    parents = {
        "confusion_hash": confusion.artifact_hash,
        "autok_plan_provenance_hash": autok.plan_provenance_hash,
        "autok_plan_semantic_hash": autok.plan_semantic_hash,
    }
    builder = _Builder()

    def spec(kind: str, method: str, seed: int, encoder: EncoderSpec, **extra: Any) -> dict:
        return job_spec(
            config,
            kind=kind,
            method=method,
            seed=seed,
            encoder=encoder,
            split_hash=split_hash,
            schedule_hash=schedule_hashes.get(seed),
            **extra,
        )

    def shellmetric_row(
        name: str,
        seed: int,
        encoder: EncoderSpec,
        plan_key: str,
        plan: ShellPlan,
        variant: ArchitectureVariant,
        **row_fields: Any,
    ) -> None:
        payload = spec(SHELLMETRIC, SHELLMETRIC, seed, encoder, plan=plan)
        job = TrainingJob(
            job_id=stable_hash(payload),
            kind=SHELLMETRIC,
            method=SHELLMETRIC,
            seed=seed,
            encoder=encoder,
            spec=payload,
            plan_key=plan_key,
            plan_semantic_hash=plan.plan_semantic_hash,
            control=bool(row_fields.get("control", False)),
        )
        row = ReportingRow(
            row_id=row_id(name, encoder.embedding_dim, seed),
            method=name,
            seed=seed,
            embedding_dim=encoder.embedding_dim,
            training_job_id=job.job_id,
            plan_provenance_hash=plan.plan_provenance_hash,
            plan_semantic_hash=plan.plan_semantic_hash,
            architecture={
                "activation": variant.activation,
                "embedding_head": variant.embedding_head,
            },
            **row_fields,
        )
        builder.add(row, job)

    for seed in seeds:
        for dim in dims:
            for variant in variants:
                encoder = EncoderSpec(dim, variant.activation, variant.embedding_head)
                scope = (
                    "validation_only" if stage not in (None, AUXILIARY_STAGE) else "test_eligible"
                )
                for plan_key, plan in plans.items():
                    name, note = plan_key, None
                    if variant.candidate is not None:
                        name = stage_row_name(stage, variant.candidate)  # type: ignore[arg-type]
                    elif stage == AUXILIARY_STAGE and not variant.is_primary:
                        name = ACTIVATION_VARIANT
                    elif stage == AUXILIARY_STAGE:
                        note = (
                            "auxiliary winner equals primary; no distinct architecture variant won"
                        )
                    shellmetric_row(
                        name,
                        seed,
                        encoder,
                        plan_key,
                        plan,
                        variant,
                        evaluation_scope=scope,
                        note=note,
                    )
            if not controls:
                continue
            encoder = EncoderSpec(
                dim, str(model["backbone_activation"]), str(model["embedding_head"])
            )
            primary = ArchitectureVariant(encoder.backbone_activation, encoder.embedding_head)
            if "shuffled_confusion" in controls:
                shuffled = shuffled_confusion_plan(confusion, autok.plan, seed=seed)
                if shuffled is None:
                    builder.add(
                        ReportingRow(
                            row_id(SHUFFLED, dim, seed),
                            SHUFFLED,
                            seed,
                            dim,
                            None,
                            status=NOT_APPLICABLE,
                            control=True,
                            parents=parents,
                            note="confusion graph is permutation-invariant",
                        )
                    )
                else:
                    key = f"{SHUFFLED}/seed={seed}"
                    control_plans[key] = shuffled[0]
                    shellmetric_row(
                        SHUFFLED,
                        seed,
                        encoder,
                        key,
                        shuffled[0],
                        primary,
                        control=True,
                        parents=parents,
                    )
            if "no_shell_loss" in controls:
                payload = spec(
                    SHELLMETRIC, SHELLMETRIC, seed, encoder, no_shell_loss_w_hat=confusion.W_hat
                )
                job = TrainingJob(
                    job_id=stable_hash(payload),
                    kind=SHELLMETRIC,
                    method=SHELLMETRIC,
                    seed=seed,
                    encoder=encoder,
                    spec=payload,
                    plan_key=AUTOK,
                    no_shell_loss=True,
                    control=True,
                )
                builder.add(
                    ReportingRow(
                        row_id(NO_SHELL_LOSS, dim, seed),
                        NO_SHELL_LOSS,
                        seed,
                        dim,
                        job.job_id,
                        control=True,
                        plan_provenance_hash=autok.plan_provenance_hash,
                        parents=parents,
                    ),
                    job,
                )
        for dim in dims:
            encoder = EncoderSpec(
                dim, str(model["backbone_activation"]), str(model["embedding_head"])
            )
            for method in baselines:
                payload = spec(BASELINE, method, seed, encoder)
                job = TrainingJob(
                    job_id=stable_hash(payload),
                    kind=BASELINE,
                    method=method,
                    seed=seed,
                    encoder=encoder,
                    spec=payload,
                )
                builder.add(
                    ReportingRow(
                        row_id(DISPLAY_NAMES[method], dim, seed),
                        DISPLAY_NAMES[method],
                        seed,
                        dim,
                        job.job_id,
                    ),
                    job,
                )
            if "hyperspacex_matched" in external:
                name = EXTERNAL_BASELINES["hyperspacex_matched"]
                builder.add(
                    ReportingRow(
                        row_id(name, dim, seed),
                        name,
                        seed,
                        dim,
                        None,
                        status=external_status(config, "hyperspacex_matched"),
                    )
                )
    if "hyperspacex_official" in external:
        name = EXTERNAL_BASELINES["hyperspacex_official"]
        builder.add(
            ReportingRow(
                row_id(name, None, None),
                name,
                None,
                None,
                None,
                status=external_status(config, "hyperspacex_official"),
                table="reference",
                note="unchanged pinned upstream protocol; never enters paired statistics",
            )
        )

    rows, jobs = builder.finish()
    manifest = StudyManifest(
        config_hash=stable_hash(config),
        confusion_hash=confusion.artifact_hash,
        split_hash=split_hash,
        reporting_rows=rows,
        training_jobs=jobs,
        fixed_shell_counts=fixed_counts,
        embedding_dims=dims,
        autok_shell_count=autok.chosen_shell_count,
        architecture_stage=stage,
        comparisons=_control_comparisons(controls, rows, autok.chosen_shell_count),
    )
    return manifest, {**plans, **control_plans}


def blocked_study_manifest(
    config: Mapping[str, Any], *, class_count: int, split_hash: str
) -> StudyManifest:
    """Rows of a cell whose backbone awaits an approved pin; nothing is planned or trained."""

    study, model = config["study"], config["model"]
    fixed_counts = expand_fixed_shell_counts(
        study.get("fixed_shell_counts"),
        class_count,
        ensure_one_shell_control=bool(study.get("ensure_one_shell_control", False)),
    )
    dims = expand_embedding_dims(study.get("embedding_dims"), model["embedding_dim"])
    methods = [
        *(fixed_s_name(count) for count in fixed_counts),
        *([AUTOK] if study.get("run_auto_k", True) else []),
        *(DISPLAY_NAMES[name] for name in expand_baselines(study.get("baselines"))),
    ]
    note = f"backbone {model['backbone']} awaits an approved immutable pin manifest"
    rows = tuple(
        ReportingRow(
            row_id(method, dim, int(seed)),
            method,
            int(seed),
            dim,
            None,
            status=external_status(config, "vit"),
            note=note,
        )
        for seed in study["seeds"]
        for dim in dims
        for method in methods
    )
    return StudyManifest(
        config_hash=stable_hash(config),
        confusion_hash=None,
        split_hash=split_hash,
        reporting_rows=rows,
        training_jobs=(),
        fixed_shell_counts=fixed_counts,
        embedding_dims=dims,
        autok_shell_count=None,
    )


def is_gated_backbone(config: Mapping[str, Any]) -> bool:
    return str(config["model"]["backbone"]) in EXTERNAL_BACKBONES


def _control_comparisons(
    controls: Iterable[str], rows: Sequence[ReportingRow], autok_shell_count: int
) -> tuple[dict[str, Any], ...]:
    controls = tuple(controls)
    if not controls:
        return ()
    one_shell = {
        "name": "multi_shell_vs_one_shell",
        "flagship": AUTOK,
        "control": fixed_s_name(1),
        "status": READY,
    }
    if autok_shell_count == 1:
        one_shell.update(
            status=NOT_APPLICABLE,
            reason="AutoK selected one shell; AutoK, FixedS(1), and the control share one encoder",
        )
    result = [one_shell]
    for control, method in (("shuffled_confusion", SHUFFLED), ("no_shell_loss", NO_SHELL_LOSS)):
        if control in controls:
            ready = any(row.method == method and row.status == READY for row in rows)
            result.append(
                {
                    "name": control,
                    "flagship": AUTOK,
                    "control": method,
                    "status": READY if ready else NOT_APPLICABLE,
                }
            )
    return tuple(result)


def save_study_manifest(
    manifest: StudyManifest, output_dir: str | Path, *, overwrite: bool = False
) -> Path:
    return save_json(
        Path(output_dir) / "study_manifest.json", manifest.to_dict(), overwrite=overwrite
    )


__all__ = [
    "ACTIVATION_VARIANT",
    "AUTOK",
    "BLOCKED_EXTERNAL_PIN",
    "DIMENSION_PRESETS",
    "NO_SHELL_LOSS",
    "REQUIRES_EXTERNAL_ADAPTER",
    "SHUFFLED",
    "EncoderSpec",
    "ReportingRow",
    "StudyManifest",
    "TrainingJob",
    "blocked_study_manifest",
    "build_study_manifest",
    "expand_baselines",
    "expand_embedding_dims",
    "external_status",
    "fixed_s_name",
    "is_gated_backbone",
    "job_spec",
    "locked_recipe_hash",
    "row_id",
    "save_study_manifest",
    "shuffled_confusion_plan",
]
