"""Restartable orchestration for the final ShellMetric study protocol.

One command expands a study (FixedS sweep, AutoK, controls, baselines,
dimension curves, or one architecture stage), prints and saves every reporting
row and deduplicated encoder job, and then runs or resumes the jobs through a
job store shared by semantic hash.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from ..artifacts import load_json, save_json, to_jsonable, utc_timestamp
from ..config import load_config, resolve_config, validate_config
from ..data import build_transforms, dataset_sample_ids, load_dataset
from ..models import build_encoder, build_pilot_classifier
from ..reproducibility import environment_metadata
from ..training import resolve_device
from .architecture import (
    ACTIVATION_STAGE,
    AUXILIARY_STAGE,
    OUTPUT_MAP_STAGE,
    architecture_section,
    decide_stage,
    verify_decision,
)
from .jobs import EncoderFactory, JobContext, JobStore, evaluation_hash, run_job
from .plan import save_plan
from .sampler import PKSchedule
from .stages import (
    DatasetFactory,
    PilotFactory,
    Planning,
    partition,
    prepare_planning,
    prepare_split,
    save_immutable_json,
)
from .study import (
    EncoderSpec,
    ReportingRow,
    StudyManifest,
    blocked_study_manifest,
    build_study_manifest,
    is_gated_backbone,
    locked_recipe_hash,
)

LOGGER = logging.getLogger(__name__)
ManifestCallback = Callable[[Mapping[str, Any]], None]
VALIDATION_ONLY_STAGES = (ACTIVATION_STAGE, OUTPUT_MAP_STAGE)


@dataclass(frozen=True)
class JobRun:
    job_id: str
    status: str
    output_dir: Path
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class StudyRun:
    output_dir: Path
    manifest: StudyManifest
    stages: Mapping[str, str]
    jobs: tuple[JobRun, ...]
    dry_run: bool
    decision: Mapping[str, Any] | None = None

    @property
    def failed(self) -> tuple[JobRun, ...]:
        return tuple(job for job in self.jobs if job.status == "failed")


def _default_output_dir(config: Mapping[str, Any]) -> Path:
    if config.get("output_dir") is not None:
        return Path(config["output_dir"])
    experiment = str(config.get("experiment", {}).get("name", "shellmetric"))
    return Path("artifacts") / str(config["data"]["dataset"]) / experiment


def _default_dataset_factory(config: Mapping[str, Any]) -> DatasetFactory:
    def factory(split: str, training: bool) -> Dataset[Any]:
        requested = copy.deepcopy(dict(config))
        requested["data"] = {**requested["data"], "split": split}
        transform = build_transforms(str(config["data"]["dataset"]), training=training)
        return load_dataset(requested, split=split, transform=transform)

    return factory


def _default_encoder_factory(config: Mapping[str, Any]) -> EncoderFactory:
    def factory(spec: EncoderSpec) -> nn.Module:
        return build_encoder(
            config,
            embedding_dimension=spec.embedding_dim,
            activation=spec.backbone_activation,
            head=spec.embedding_head,
        )

    return factory


def _input_shape(dataset: Dataset[Any]) -> list[int]:
    sample = dataset[0]
    inputs = sample["inputs"] if isinstance(sample, Mapping) else sample[0]
    return [int(value) for value in torch.as_tensor(inputs).shape]


def _load_decision(cache_root: Path, decision_hash: str) -> dict[str, Any]:
    path = cache_root / "decisions" / f"{decision_hash}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"architecture decision {decision_hash} is not in {path.parent}; "
            "run the preceding stage with the same cache_dir first"
        )
    return load_json(path)


def _verify_architecture_references(
    config: Mapping[str, Any], planning: Planning, cache_root: Path
) -> None:
    """Every stage must reuse the locked recipe and the official ReLU-derived plan."""

    section = architecture_section(config)
    if section is None:
        return
    expected = {
        "locked_loss_hash": locked_recipe_hash(config),
        "reference_plan_semantic_hash": planning.autok.plan_semantic_hash,
        "reference_plan_provenance_hash": planning.autok.plan_provenance_hash,
    }
    mismatched = {key: value for key, value in expected.items() if section[key] != value}
    if mismatched:
        raise ValueError(
            f"architecture stage does not match the locked primary study: {mismatched}"
        )
    if section["stage"] in {OUTPUT_MAP_STAGE, AUXILIARY_STAGE}:
        verify_decision(
            _load_decision(cache_root, section["activation_decision_hash"]),
            stage=ACTIVATION_STAGE,
            selected=section["activation"],
        )
    if section["stage"] == AUXILIARY_STAGE:
        output_map = _load_decision(cache_root, section["output_map_decision_hash"])
        verify_decision(output_map, stage=OUTPUT_MAP_STAGE, selected=section["embedding_head"])
        if (output_map["fixed_activation"], output_map["parent_decision_hash"]) != (
            section["activation"],
            section["activation_decision_hash"],
        ):
            raise ValueError("the output-map decision was made for another Stage-A winner")


def _row_directory(root: Path, row: ReportingRow) -> Path:
    if row.seed is None:
        return root / "results" / row.method / "reference"
    return root / "results" / row.method / f"d{row.embedding_dim}" / f"seed{row.seed}"


def _row_metadata(config: Mapping[str, Any], manifest: StudyManifest, row: ReportingRow) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "study_row_evaluation",
        "dataset": str(config["data"]["dataset"]),
        "method": row.method,
        "seed": row.seed,
        "embedding_dim": row.embedding_dim,
        "row_id": row.row_id,
        "training_job_id": row.training_job_id,
        "aliases": list(row.aliases),
        "control": row.control,
        "table": row.table,
        "evaluation_scope": row.evaluation_scope,
        "plan_provenance_hash": row.plan_provenance_hash,
        "plan_semantic_hash": row.plan_semantic_hash,
        "parents": row.parents,
        "architecture": row.architecture,
        "note": row.note,
        "config_hash": manifest.config_hash,
        "split_hash": manifest.split_hash,
        "confusion_hash": manifest.confusion_hash,
    }


def _publish_rows(
    root: Path, config: Mapping[str, Any], manifest: StudyManifest, store: JobStore
) -> None:
    """Materialize each reporting row's metrics so a study summarizes on its own."""

    for row in manifest.reporting_rows:
        if row.training_job_id is None:
            continue
        job_dir = store.directory(row.training_job_id)
        target = _row_directory(root, row)
        metadata = {**_row_metadata(config, manifest, row), "job_dir": str(job_dir)}
        if (job_dir / "failure.json").is_file():
            save_json(
                target / "failure.json",
                {**metadata, **load_json(job_dir / "failure.json")},
                overwrite=True,
            )
            continue
        (target / "failure.json").unlink(missing_ok=True)
        for name in ("validation", "test"):
            source = job_dir / name / "metrics.json"
            if not source.is_file() or (name == "test" and row.evaluation_scope != "test_eligible"):
                continue
            save_json(target / name / "metrics.json", load_json(source), overwrite=True)
            save_json(
                target / name / "evaluation_metadata.json",
                {**metadata, "partition": name},
                overwrite=True,
            )


def _stage_decision(
    config: Mapping[str, Any],
    manifest: StudyManifest,
    store: JobStore,
    schedules: Mapping[int, PKSchedule],
    root: Path,
    cache_root: Path,
) -> dict[str, Any] | None:
    """Decide Stage A/B once every candidate's validation metrics exist."""

    stage = manifest.architecture_stage
    if stage not in VALIDATION_ONLY_STAGES:
        return None
    factor = "activation" if stage == ACTIVATION_STAGE else "embedding_head"
    per_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in manifest.reporting_rows:
        assert row.seed is not None and row.architecture is not None, row.row_id
        job_id = str(row.training_job_id)
        metrics_path = store.directory(job_id) / "validation" / "metrics.json"
        if not metrics_path.is_file():
            return None
        metrics, training = load_json(metrics_path), store.training_result(job_id) or {}
        raw, spoke = metrics["raw_euclidean"], metrics["geometry"]["spoke"]
        per_candidate.setdefault(str(row.architecture[factor]), []).append(
            {
                "seed": row.seed,
                "job_id": job_id,
                "batch_schedule_hash": schedules[int(row.seed)].schedule_hash,
                "euclidean_1nn": raw["accuracy"],
                "map_at_r": raw["retrieval"]["map_at_r"],
                "undefined_spoke_classes": spoke["undefined_class_count"],
                "mean_spoke_fraction": spoke["macro_mean_spoke_fraction"],
                "parameter_counts": training.get("parameter_counts"),
                "training_samples_per_second": training.get("training_samples_per_second"),
            }
        )
    decision = decide_stage(config, per_candidate)
    save_immutable_json(root / "architecture_decision.json", decision)
    save_immutable_json(cache_root / "decisions" / f"{decision['decision_hash']}.json", decision)
    return decision


def run_shellmetric_study(
    config: Mapping[str, Any],
    output_dir: str | Path | None = None,
    *,
    resume: bool = False,
    dry_run: bool = False,
    allow_test: bool = False,
    device: str | torch.device = "auto",
    dataset_factory: DatasetFactory | None = None,
    pilot_factory: PilotFactory | None = None,
    encoder_factory: EncoderFactory | None = None,
    manifest_callback: ManifestCallback | None = None,
) -> StudyRun:
    """Run or resume one complete ShellMetric suite.

    Test embeddings are accessed only when both ``allow_test`` and
    ``evaluation.locked`` are true, and never for Stage-A/B candidates. Creating
    the immutable split manifest reads official test IDs and labels only.
    """

    resolved = resolve_config(config)
    section = architecture_section(resolved)
    if allow_test and not bool(resolved["evaluation"].get("locked", False)):
        raise ValueError("--allow-test requires evaluation.locked: true")
    if allow_test and section is not None and section["stage"] in VALIDATION_ONLY_STAGES:
        raise ValueError("Stage-A/B candidates are validation-only; test only the frozen winner")
    device = str(resolve_device(device))
    root = Path(output_dir) if output_dir is not None else _default_output_dir(resolved)
    root.mkdir(parents=True, exist_ok=True)
    factory = dataset_factory or _default_dataset_factory(resolved)

    official_train = factory("train", True)
    official_train_eval = factory("train", False)
    if not np.array_equal(
        dataset_sample_ids(official_train), dataset_sample_ids(official_train_eval)
    ):
        raise ValueError("training and deterministic evaluation datasets are not aligned")
    resolved["data"] = dict(resolved["data"])
    resolved["data"].setdefault("input_shape", _input_shape(official_train_eval))
    split, split_status = prepare_split(resolved, factory, official_train_eval)
    validate_config(resolved, class_count=split.class_ids.size)
    stages: dict[str, str] = {
        "split": split_status,
        "config": save_immutable_json(root / "resolved_config.json", resolved),
    }

    if is_gated_backbone(resolved):  # no pilot can be built before the pin is approved
        blocked = blocked_study_manifest(
            resolved, class_count=int(split.class_ids.size), split_hash=split.manifest_hash
        )
        stages["manifest"] = save_immutable_json(root / "study_manifest.json", blocked.to_dict())
        if manifest_callback is not None:
            manifest_callback({**blocked.to_dict(), "job_store": None, "job_status": {}})
        return StudyRun(root, blocked, stages, (), dry_run)

    train = partition(official_train, split, split.train_mask)
    train_eval = partition(official_train_eval, split, split.train_mask)
    validation = partition(official_train_eval, split, split.validation_mask)
    cache_root = Path(resolved.get("cache_dir") or root / "shared")
    planning = prepare_planning(
        resolved,
        split,
        train,
        train_eval,
        pilot_factory or (lambda count: build_pilot_classifier(resolved, num_classes=count)),
        cache_root,
        class_names=tuple(str(name) for name in official_train_eval.class_names),
    )
    stages.update(planning.statuses)
    _verify_architecture_references(resolved, planning, cache_root)

    sampling = resolved["sampling"]
    schedules = {
        int(seed): PKSchedule.create(
            train.targets,
            seed=int(seed),
            classes_per_batch=sampling.get("classes_per_batch"),
            samples_per_class=sampling.get("samples_per_class"),
            target_batch_size=int(sampling["target_batch_size"]),
        )
        for seed in resolved["study"]["seeds"]
    }
    manifest, plans = build_study_manifest(
        resolved,
        planning.confusion,
        planning.autok,
        split_hash=split.manifest_hash,
        schedule_hashes={seed: schedule.schedule_hash for seed, schedule in schedules.items()},
    )
    for plan in plans.values():
        plan_dir = root / "plans" / plan.plan_provenance_hash
        if not (plan_dir / "plan_metadata.json").is_file():
            save_plan(plan, plan_dir)
    stages["manifest"] = save_immutable_json(root / "study_manifest.json", manifest.to_dict())
    store = JobStore(cache_root / "jobs")
    eval_hash = evaluation_hash(resolved)
    save_immutable_json(
        root / "provenance.json",
        {
            "split_manifest": str(Path(resolved["data"]["split_manifest"])),
            "split_hash": split.manifest_hash,
            "planning_hash": split.planning_hash,
            "pilot_dir": str(planning.directories["pilot"]),
            "oof_hash": planning.oof_hash,
            "confusion_dir": str(planning.directories["confusion"]),
            "confusion_hash": planning.confusion.artifact_hash,
            "autok_dir": str(planning.directories["autok"]),
            "autok_shell_count": planning.autok.chosen_shell_count,
            "autok_plan_semantic_hash": planning.autok.plan_semantic_hash,
            "autok_plan_provenance_hash": planning.autok.plan_provenance_hash,
            "locked_loss_hash": locked_recipe_hash(resolved),
            "config_hash": manifest.config_hash,
            "evaluation_hash": eval_hash,
            "batch_schedule_hashes": {str(seed): s.schedule_hash for seed, s in schedules.items()},
            "job_store": str(store.root),
        },
    )
    with (root / "runs.jsonl").open("a", encoding="utf-8") as handle:
        record = {
            "started_at": utc_timestamp(),
            "resume": resume,
            "dry_run": dry_run,
            "allow_test": allow_test,
            "device": device,
            "environment": environment_metadata(),
        }
        handle.write(json.dumps(to_jsonable(record), sort_keys=True) + "\n")

    test_jobs = (
        {
            row.training_job_id
            for row in manifest.reporting_rows
            if row.training_job_id is not None and row.evaluation_scope == "test_eligible"
        }
        if allow_test
        else set()
    )
    statuses = {
        job.job_id: store.status(job, eval_hash, need_test=job.job_id in test_jobs)
        for job in manifest.training_jobs
    }
    if manifest_callback is not None:
        manifest_callback(
            {**manifest.to_dict(), "job_store": str(store.root), "job_status": statuses}
        )
    if dry_run:
        runs = tuple(
            JobRun(job.job_id, statuses[job.job_id], store.directory(job.job_id), job.aliases)
            for job in manifest.training_jobs
        )
        return StudyRun(root, manifest, stages, runs, True)

    context = JobContext(
        config=resolved,
        train=train,
        train_eval=train_eval,
        validation=validation,
        test=partition(factory("test", False), split, split.test_mask) if allow_test else None,
        class_count=int(split.class_ids.size),
        schedules=schedules,
        encoder_factory=encoder_factory or _default_encoder_factory(resolved),
        device=device,
    )
    job_runs: list[JobRun] = []
    for index, job in enumerate(manifest.training_jobs, start=1):
        LOGGER.info(
            "job %d/%d %s (%s seed=%d d=%d): %s",
            index,
            len(manifest.training_jobs),
            job.job_id[:12],
            job.method,
            job.seed,
            job.encoder.embedding_dim,
            statuses[job.job_id],
        )
        if statuses[job.job_id] == "complete":
            status = "reused"
        else:
            plan = plans[job.plan_key] if job.plan_key is not None else None
            status = run_job(
                job, plan, context, store, resume=resume, evaluate_test=job.job_id in test_jobs
            )
        job_runs.append(JobRun(job.job_id, status, store.directory(job.job_id), job.aliases))
    _publish_rows(root, resolved, manifest, store)
    decision = _stage_decision(resolved, manifest, store, schedules, root, cache_root)
    save_json(
        root / "study_result.json",
        {
            "config_hash": manifest.config_hash,
            "finished_at": utc_timestamp(),
            "test_evaluated": allow_test,
            "stages": stages,
            "jobs": [
                {"job_id": run.job_id, "status": run.status, "aliases": list(run.aliases)}
                for run in job_runs
            ],
            "architecture_decision_hash": None if decision is None else decision["decision_hash"],
        },
        overwrite=True,
    )
    return StudyRun(root, manifest, stages, tuple(job_runs), False, decision)


def _fixed_shell_override(value: str) -> str | list[int]:
    normalized = value.strip().lower()
    if normalized in {"compact", "all", "none"}:
        return normalized
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "fixed shell counts must be compact, all, none, or comma-separated integers"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the restartable ShellMetric study")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fixed-shell-counts", type=_fixed_shell_override)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--device", default="auto", help="auto (CUDA when available), cpu, or cuda")
    return parser


def _print_manifest(payload: Mapping[str, Any]) -> None:
    """Print every reporting row and deduplicated job before anything trains."""

    status = payload["job_status"]
    summary = {
        "reporting_rows": payload["reporting_row_count"],
        "row_status": payload["status_counts"],
        "unique_training_jobs": payload["unique_training_job_count"],
        "jobs_by_store_status": {
            name: sum(value == name for value in status.values())
            for name in sorted(set(status.values()))
        },
        "autok_shell_count": payload["autok_shell_count"],
        "fixed_shell_counts": payload["fixed_shell_counts"],
        "embedding_dims": payload["embedding_dims"],
        "comparisons": payload["comparisons"],
        "job_store": payload["job_store"],
    }
    rows = [
        {
            key: row[key]
            for key in (
                "row_id",
                "status",
                "training_job_id",
                "control",
                "evaluation_scope",
                "aliases",
            )
        }
        for row in payload["reporting_rows"]
    ]
    jobs = [
        {
            "job_id": job["job_id"],
            "store_status": status[job["job_id"]],
            "method": job["method"],
            "seed": job["seed"],
            "encoder": job["encoder"],
            "plan_semantic_hash": job["plan_semantic_hash"],
            "control": job["control"],
            "aliases": job["aliases"],
        }
        for job in payload["training_jobs"]
    ]
    print(json.dumps({"summary": summary, "rows": rows, "jobs": jobs}, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stderr)
    config = load_config(args.config, validate=False)
    if args.fixed_shell_counts is not None:
        config["study"]["fixed_shell_counts"] = args.fixed_shell_counts
    result = run_shellmetric_study(
        config,
        args.output_dir,
        resume=args.resume,
        dry_run=args.dry_run,
        allow_test=args.allow_test,
        device=args.device,
        manifest_callback=_print_manifest,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "dry_run": result.dry_run,
                "job_statuses": {job.job_id: job.status for job in result.jobs},
                "architecture_decision": None
                if result.decision is None
                else {key: result.decision[key] for key in ("stage", "selected", "decision_hash")},
            },
            indent=2,
        )
    )
    if result.failed:
        print(
            f"{len(result.failed)} job(s) failed; see failure.json in each job directory",
            file=sys.stderr,
        )
        return 1
    return 0


__all__ = ["JobRun", "StudyRun", "build_parser", "main", "run_shellmetric_study"]
