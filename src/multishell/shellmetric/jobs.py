"""Semantic job store: train one encoder job and write its evaluation artifacts.

A job directory is named by the hash of its training semantics, so any study
that needs the same encoder (AutoK/FixedS aliases, Stage-A ReLU, dimension
curves) reuses it.  Evaluation follows the fixed order of Section 6: raw
Euclidean metrics first, then ShellMetric's native shell-then-cosine kNN, then
the post-hoc affine probe on the frozen encoder, and finally native heads.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from ..artifacts import (
    hash_array,
    load_json,
    load_npz,
    save_json,
    save_npz,
    stable_hash,
    utc_timestamp,
)
from ..reproducibility import derive_stage_seed, environment_metadata, seed_everything
from ..train_baseline import (
    BaselineModel,
    BaselineTrainingConfig,
    native_predictions,
    restore_baseline_model,
    train_baseline,
)
from ..training import count_parameters, extract_embeddings
from .evaluate import (
    ShellKNNResult,
    classification_metrics,
    evaluate_raw_embeddings,
    evaluate_shell_then_cosine_knn,
    geometry_diagnostics,
    select_k,
)
from .plan import ShellPlan
from .probe import train_linear_probe
from .sampler import PKSchedule
from .stages import save_immutable_json
from .study import SHELLMETRIC, EncoderSpec, TrainingJob
from .train import TrainingConfig, load_representation, train_shellmetric

EncoderFactory = Callable[[EncoderSpec], nn.Module]
EVALUATION_PROTOCOL = "shellmetric_evaluation_v2"
NOT_APPLICABLE = "not_applicable"
_SAVED_NEIGHBORS = 11


@dataclass(frozen=True)
class JobContext:
    """Datasets and settings shared by every job of one study invocation."""

    config: Mapping[str, Any]
    train: Dataset[Any]
    train_eval: Dataset[Any]
    validation: Dataset[Any]
    test: Dataset[Any] | None
    class_count: int
    schedules: Mapping[int, PKSchedule]
    encoder_factory: EncoderFactory
    device: str = "cpu"

    @property
    def evaluation_hash(self) -> str:
        return evaluation_hash(self.config)


def evaluation_hash(config: Mapping[str, Any]) -> str:
    settings = {key: value for key, value in config["evaluation"].items() if key != "locked"}
    return stable_hash({"protocol": EVALUATION_PROTOCOL, "evaluation": settings})


class JobStore:
    """Job directories keyed by semantic hash, shared across studies."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def directory(self, job_id: str) -> Path:
        return self.root / job_id

    def register(self, job: TrainingJob) -> Path:
        directory = self.directory(job.job_id)
        save_immutable_json(directory / "job_spec.json", {"job_id": job.job_id, "spec": job.spec})
        return directory

    def training_result(self, job_id: str) -> dict[str, Any] | None:
        path = self.directory(job_id) / "training_result.json"
        return load_json(path) if path.is_file() else None

    def evaluated(self, job_id: str, partition: str, expected_hash: str) -> bool:
        path = self.directory(job_id) / partition / "evaluation_metadata.json"
        if not path.is_file():
            return False
        if load_json(path).get("evaluation_hash") != expected_hash:
            raise ValueError(
                f"job {job_id} was evaluated under another evaluation protocol; "
                "use a separate cache_dir for a changed protocol"
            )
        return True

    def status(self, job: TrainingJob, expected_hash: str, *, need_test: bool) -> str:
        directory = self.directory(job.job_id)
        if (directory / "failure.json").is_file():
            return "failed"
        if self.training_result(job.job_id) is None:
            return "partial" if (directory / "checkpoints" / "last.pt").is_file() else "new"
        partitions = ("validation", "test") if need_test else ("validation",)
        if all(self.evaluated(job.job_id, name, expected_hash) for name in partitions):
            return "complete"
        return "needs_evaluation"


def run_job(
    job: TrainingJob,
    plan: ShellPlan | None,
    context: JobContext,
    store: JobStore,
    *,
    resume: bool,
    evaluate_test: bool,
) -> str:
    """Train (or reuse) and evaluate one job; failures are recorded, not raised."""

    directory = store.register(job)
    interrupted = (directory / "checkpoints" / "last.pt").is_file()
    if store.training_result(job.job_id) is None and interrupted and not resume:
        raise FileExistsError(f"job {job.job_id} was interrupted; pass --resume to continue it")
    failure = directory / "failure.json"
    try:
        if store.training_result(job.job_id) is None:
            _train(job, plan, context, directory, resume=resume)
        for partition in ("validation", "test") if evaluate_test else ("validation",):
            if not store.evaluated(job.job_id, partition, context.evaluation_hash):
                evaluate_partition(job, plan, context, directory, partition)
    except Exception as error:  # record and continue with the remaining jobs
        save_json(
            failure,
            {
                "job_id": job.job_id,
                "method": job.method,
                "seed": job.seed,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "failed_at": utc_timestamp(),
            },
            overwrite=True,
        )
        return "failed"
    failure.unlink(missing_ok=True)
    return "completed"


def _train(
    job: TrainingJob, plan: ShellPlan | None, context: JobContext, directory: Path, *, resume: bool
) -> dict[str, Any]:
    seed_everything(job.seed)  # identical initial encoder state for every paired method
    encoder = context.encoder_factory(job.encoder)
    schedule = context.schedules[job.seed] if job.uses_pk_schedule else None
    if job.kind == SHELLMETRIC:
        result = train_shellmetric(
            encoder,
            context.train,
            context.validation,
            plan,
            config=TrainingConfig.from_config(context.config),
            seed=job.seed,
            output_dir=directory,
            device=context.device,
            schedule=schedule,
            gallery_dataset=context.train_eval,
            no_shell_loss=job.no_shell_loss,
            job_hash=job.job_id,
            resume=resume,
        )
        final = result.history[-1]
        selection = {
            "best_epoch": result.best_epoch,
            "best_validation_euclidean_1nn": result.best_validation_accuracy,
            "checkpoints": {"representation": "checkpoints/best.pt"},
            "final_state": {
                key: final.get(key) for key in ("radii", "radial_gate_beta", "radial_gate_power")
            },
        }
        total = count_parameters(encoder, trainable_only=False) + result.parameter_counts["radii"]
    else:
        config = BaselineTrainingConfig.from_config(context.config, job.method)
        model = BaselineModel.from_config(
            encoder, context.class_count, config, embedding_dimension=job.encoder.embedding_dim
        )
        result = train_baseline(
            model,
            context.train,
            context.validation,
            config=config,
            seed=job.seed,
            output_dir=directory,
            device=context.device,
            schedule=schedule,
            gallery_dataset=context.train_eval,
            job_hash=job.job_id,
            resume=resume,
        )
        selection = {
            "best_epoch": result.representation_best_epoch,
            "best_validation_euclidean_1nn": result.representation_best_accuracy,
            "native_best_epoch": result.native_best_epoch,
            "native_best_accuracy": result.native_best_accuracy,
            "checkpoints": {
                "representation": "checkpoints/representation_best.pt",
                "native": None
                if result.native_checkpoint is None
                else "checkpoints/native_best.pt",
            },
        }
        total = count_parameters(model, trainable_only=False)
    payload = {
        "schema_version": 1,
        "artifact_type": "training_result",
        "job_id": job.job_id,
        "kind": job.kind,
        "method": job.method,
        "seed": job.seed,
        "encoder": asdict(job.encoder),
        **selection,
        "training_runtime_seconds": result.training_seconds,
        "training_samples_per_second": result.samples_per_second,
        "parameter_counts": dict(result.parameter_counts),
        "trainable_parameter_count": result.parameter_counts["trainable"],
        "total_parameter_count": total,
        "environment": environment_metadata(),
        "completed_at": utc_timestamp(),
    }
    save_json(directory / "training_result.json", payload, overwrite=True)
    return payload


def _classification_block(truth: np.ndarray, predictions: np.ndarray, **extra: Any) -> dict:
    return {**classification_metrics(truth, predictions), **extra}


def _native_arrays(result: ShellKNNResult) -> dict[str, np.ndarray]:
    return {
        "predictions": result.predictions,
        "predicted_shells": result.predicted_shells,
        "true_shells": result.true_shells,
        "candidate_gallery_sizes": result.candidate_gallery_sizes,
        "effective_k": result.effective_k,
        "neighbor_indices": result.neighbor_indices,
        "neighbor_sample_ids": np.asarray(
            [
                ["" if value is None else str(value) for value in row]
                for row in result.neighbor_sample_ids
            ],
            dtype=str,
        ),
        "neighbor_labels": result.neighbor_labels,
        "neighbor_distances": result.neighbor_distances,
        "oracle_predictions": result.oracle_predictions,
    }


def _load_encoder(
    job: TrainingJob, plan: ShellPlan | None, context: JobContext, directory: Path, checkpoint: str
) -> tuple[nn.Module, np.ndarray | None, BaselineModel | None]:
    """Rebuild the frozen encoder (and radii, or the baseline model) from a checkpoint."""

    seed_everything(job.seed)
    encoder = context.encoder_factory(job.encoder)
    path = directory / checkpoint
    if job.kind == SHELLMETRIC:
        encoder, radii = load_representation(encoder, path, None if job.no_shell_loss else plan)
        return encoder, radii, None
    config = BaselineTrainingConfig.from_config(context.config, job.method)
    model = BaselineModel.from_config(
        encoder, context.class_count, config, embedding_dimension=job.encoder.embedding_dim
    )
    restore_baseline_model(model, path)
    return model.encoder, None, model


@dataclass(frozen=True)
class _Embeddings:
    """Frozen-encoder embeddings of one query partition and the training gallery."""

    queries: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray
    gallery: np.ndarray
    gallery_labels: np.ndarray
    gallery_ids: np.ndarray
    seconds_per_sample: float


def _embed(encoder: nn.Module, context: JobContext, directory: Path, partition: str) -> _Embeddings:
    dataset = context.validation if partition == "validation" else context.test
    if dataset is None:
        raise RuntimeError("test evaluation requires authorized test access")
    started = time.perf_counter()
    queries, labels, ids = extract_embeddings(encoder, dataset, device=context.device)
    seconds = (time.perf_counter() - started) / max(len(labels), 1)
    if partition == "validation":
        gallery = extract_embeddings(encoder, context.train_eval, device=context.device)
    else:  # the frozen gallery cached during validation
        cached = load_npz(directory / "validation" / "embeddings.npz")
        gallery = tuple(cached[f"train_{name}"] for name in ("embeddings", "labels", "sample_ids"))
    arrays = {
        f"{partition}_embeddings": queries,
        f"{partition}_labels": labels,
        f"{partition}_sample_ids": ids,
    }
    if partition == "validation":
        arrays.update(
            train_embeddings=gallery[0], train_labels=gallery[1], train_sample_ids=gallery[2]
        )
    save_npz(directory / partition / "embeddings.npz", arrays, overwrite=True)
    return _Embeddings(queries, labels, ids, *gallery, seconds)


def _raw_section(
    data: _Embeddings, evaluation: Mapping[str, Any], selected: int | None, output: Path
) -> tuple[dict[str, Any], Any, int, float]:
    """Decoder-free raw Euclidean kNN and retrieval (the common representation table)."""

    recall_ks = tuple(
        int(name.rsplit("_", 1)[1])
        for name in evaluation["retrieval"]
        if name.startswith("recall_at_")
    )
    ks = tuple(evaluation["euclidean_knn_k"]) if selected is None else tuple(sorted({1, selected}))
    started = time.perf_counter()
    raw = evaluate_raw_embeddings(
        data.queries,
        data.labels,
        data.gallery,
        data.gallery_labels,
        data.gallery_ids,
        k_values=ks,
        recall_ks=recall_ks,
    )
    seconds = (time.perf_counter() - started) / len(data.labels)
    k = selected or select_k({key: float(m["accuracy"]) for key, m in raw.metrics_by_k.items()})
    save_json(
        output / "raw_metrics.json",
        {
            "metrics_by_k": {str(key): value for key, value in raw.metrics_by_k.items()},
            "selected_k": k,
            "retrieval": dict(raw.retrieval),
            "cosine_1nn_accuracy": raw.cosine_1nn_accuracy,
        },
        overwrite=True,
    )
    save_npz(
        output / "raw_predictions.npz",
        {
            **{f"predictions_k_{key}": value for key, value in raw.predictions_by_k.items()},
            "euclidean_neighbor_indices": raw.euclidean_neighbors.indices[:, :_SAVED_NEIGHBORS],
            "euclidean_neighbor_distances": raw.euclidean_neighbors.distances[:, :_SAVED_NEIGHBORS],
            "cosine_1nn_predictions": raw.cosine_1nn_predictions,
        },
        overwrite=True,
    )
    metrics = {
        "raw_euclidean": {
            **raw.metrics_by_k[1],
            "retrieval": dict(raw.retrieval),
            "cosine_1nn_accuracy": raw.cosine_1nn_accuracy,
        },
        "raw_euclidean_selected_k": {**raw.metrics_by_k[k], "k": k},
    }
    return metrics, raw, k, seconds


def _native_section(
    data: _Embeddings,
    radii: np.ndarray,
    plan: ShellPlan,
    evaluation: Mapping[str, Any],
    selected: int | None,
    output: Path,
) -> tuple[dict[str, Any], int, float]:
    """ShellMetric-native shell-then-cosine kNN; reports 1-NN and the selected k."""

    ks = (
        tuple(evaluation["shellmetric_native_k"])
        if selected is None
        else tuple(sorted({1, selected}))
    )
    started = time.perf_counter()
    native = evaluate_shell_then_cosine_knn(
        data.queries,
        data.labels,
        data.gallery,
        data.gallery_labels,
        data.gallery_ids,
        radii,
        plan.assignment,
        class_ids=plan.class_ids,
        k_values=ks,
    )
    seconds = (time.perf_counter() - started) / len(data.labels)
    k = selected or select_k(
        {key: float(result.metrics["accuracy"]) for key, result in native.items()}
    )
    hashes = {
        "plan_semantic_hash": plan.plan_semantic_hash,
        "plan_provenance_hash": plan.plan_provenance_hash,
        "radius_hash": hash_array(np.asarray(radii, dtype=np.float64)),
    }
    save_json(
        output / "native_metrics.json",
        {
            "metrics_by_k": {str(key): dict(r.metrics) for key, r in native.items()},
            "selected_k": k,
            **hashes,
        },
        overwrite=True,
    )
    for key in sorted({1, k}):
        save_npz(
            output / f"native_predictions_k{key}.npz", _native_arrays(native[key]), overwrite=True
        )
    metrics = {
        "shellmetric_native": {**native[1].metrics, **hashes},
        "shellmetric_native_selected_k": {**native[k].metrics, **hashes},
    }
    return metrics, k, seconds


def _probe_section(
    data: _Embeddings,
    encoder: nn.Module,
    context: JobContext,
    job: TrainingJob,
    directory: Path,
    selections: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Common affine probe, trained only after the encoder is frozen."""

    evaluation = context.config["evaluation"]
    probe_path = directory / "probe.pt"
    if selections is None:
        probe = train_linear_probe(
            data.gallery,
            data.gallery_labels,
            data.queries,
            data.labels,
            class_count=context.class_count,
            encoder=encoder,
            seed=derive_stage_seed(0, "linear_probe", job.job_id),
            learning_rate=float(evaluation["probe_learning_rate"]),
            weight_decays=tuple(evaluation["probe_weight_decays"]),
            max_epochs=int(evaluation["probe_max_epochs"]),
            patience=int(evaluation["probe_patience"]),
            batch_size=evaluation.get("probe_batch_size"),
        )
        torch.save(probe.model.state_dict(), probe_path)
        model = probe.model
        selection = {
            "selected_weight_decay": probe.selected_weight_decay,
            "selected_epoch": probe.selected_epoch,
            "encoder_checksum": probe.encoder_checksum,
        }
        extra = {**selection, "trials": list(probe.trials)}
    else:
        model = nn.Linear(data.queries.shape[1], context.class_count)
        model.load_state_dict(torch.load(probe_path, map_location="cpu", weights_only=True))
        selection = extra = dict(selections)
    with torch.no_grad():
        predictions = model(torch.as_tensor(data.queries, dtype=torch.float32)).argmax(1).numpy()
    return {**classification_metrics(data.labels, predictions), **extra}, selection


def _native_head_section(
    job: TrainingJob, context: JobContext, directory: Path, partition: str, training: Mapping
) -> dict[str, Any]:
    """Native classification heads use their own validation-native checkpoint."""

    checkpoint = training["checkpoints"].get("native")
    if checkpoint is None:
        return {"status": NOT_APPLICABLE}
    _, _, model = _load_encoder(job, None, context, directory, checkpoint)
    assert model is not None
    model.to(context.device)
    dataset = context.validation if partition == "validation" else context.test
    embeddings, labels, _ = extract_embeddings(model, dataset, device=context.device)
    predictions = native_predictions(model, embeddings)
    save_npz(
        directory / partition / "native_head_predictions.npz",
        {"predictions": predictions},
        overwrite=True,
    )
    return {
        **classification_metrics(labels, predictions),
        "selection_rule": "validation_native_top1",
        "epoch": training["native_best_epoch"],
    }


def evaluate_partition(
    job: TrainingJob, plan: ShellPlan | None, context: JobContext, directory: Path, partition: str
) -> dict[str, Any]:
    """Write every evaluation artifact for ``validation`` or (authorized) ``test``.

    Test evaluation reuses the k values and probe selected on validation.
    """

    evaluation = context.config["evaluation"]
    training = load_json(directory / "training_result.json")
    output = directory / partition
    selections = (
        None
        if partition == "validation"
        else load_json(directory / "validation" / "evaluation_metadata.json")["selections"]
    )
    encoder, radii, _ = _load_encoder(
        job, plan, context, directory, training["checkpoints"]["representation"]
    )
    encoder.to(context.device)
    data = _embed(encoder, context, directory, partition)

    metrics, raw, raw_k, raw_seconds = _raw_section(
        data, evaluation, None if selections is None else int(selections["raw_k"]), output
    )
    native_k = native_seconds = None
    if radii is None:
        reason = {
            "status": NOT_APPLICABLE,
            "reason": "no_shell_loss" if job.kind == SHELLMETRIC else "baseline",
        }
        metrics.update(shellmetric_native=reason, shellmetric_native_selected_k=reason)
    else:
        assert plan is not None
        native, native_k, native_seconds = _native_section(
            data,
            radii,
            plan,
            evaluation,
            None if selections is None else selections["native_k"],
            output,
        )
        metrics.update(native)
    metrics["geometry"] = geometry_diagnostics(
        data.queries,
        data.labels,
        radii=radii,
        class_assignment=None if radii is None else plan.assignment,  # type: ignore[union-attr]
        class_ids=None if radii is None else plan.class_ids,  # type: ignore[union-attr]
        euclidean_1nn_predictions=raw.predictions_by_k[1],
        cosine_1nn_predictions=raw.cosine_1nn_predictions,
    )
    save_json(output / "geometry.json", metrics["geometry"], overwrite=True)
    metrics["linear_probe"], probe_selection = _probe_section(
        data, encoder, context, job, directory, None if selections is None else selections["probe"]
    )
    save_json(output / "probe_metrics.json", metrics["linear_probe"], overwrite=True)
    metrics["native_head"] = _native_head_section(job, context, directory, partition, training)
    metrics.update(
        efficiency={
            "embedding_seconds_per_sample": data.seconds_per_sample,
            "gallery_memory_bytes": int(np.asarray(data.gallery, dtype=np.float32).nbytes),
            "raw_knn_seconds_per_query": raw_seconds,
            "native_knn_seconds_per_query": native_seconds,
        },
        training_runtime_seconds=training["training_runtime_seconds"],
        trainable_parameter_count=training["trainable_parameter_count"],
        total_parameter_count=training["total_parameter_count"],
    )
    save_json(output / "metrics.json", metrics, overwrite=True)
    save_json(
        output / "evaluation_metadata.json",
        {
            "schema_version": 2,
            "artifact_type": "job_evaluation",
            "evaluation_hash": evaluation_hash(context.config),
            "partition": partition,
            "job_id": job.job_id,
            "kind": job.kind,
            "method": job.method,
            "seed": job.seed,
            "encoder": asdict(job.encoder),
            "representation": "raw_encoder_output",
            "checkpoint_selection": "validation_raw_euclidean_1nn",
            "plan_semantic_hash": None if job.no_shell_loss else job.plan_semantic_hash,
            "radii": None if radii is None else np.asarray(radii).tolist(),
            "radius_hash": None if radii is None else metrics["shellmetric_native"]["radius_hash"],
            "selections": {"raw_k": raw_k, "native_k": native_k, "probe": probe_selection},
            "evaluated_at": utc_timestamp(),
        },
        overwrite=True,
    )
    return metrics


__all__ = [
    "EVALUATION_PROTOCOL",
    "EncoderFactory",
    "JobContext",
    "JobStore",
    "evaluate_partition",
    "evaluation_hash",
    "run_job",
]
