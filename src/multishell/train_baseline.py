"""Training for the eight matched in-repository baselines.

Every method trains the same raw ``d``-dimensional Cartesian encoder.  A native
classification head exists only for CE, label smoothing, ArcFace, and CosFace;
it is saved in a separate checkpoint and never enters the common representation.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from .artifacts import save_json
from .baselines import (
    BASELINES,
    CLASSIFICATION_BASELINES,
    METRIC_BASELINES,
    batch_hard_triplet_loss,
    build_native_head,
    multi_similarity_loss,
    pair_contrastive_loss,
    supervised_contrastive_loss,
)
from .data import dataset_labels, unpack_batch
from .reproducibility import (
    capture_rng_state,
    derive_stage_seed,
    restore_rng_state,
    seed_everything,
    seed_worker,
)
from .shellmetric.evaluate import euclidean_1nn_accuracy
from .shellmetric.sampler import ManifestBatchSampler, PKSchedule
from .training import (
    atomic_torch_save,
    autocast,
    build_optimizer,
    build_scheduler,
    count_parameters,
    cpu_state,
    extract_embeddings,
    grad_scaler,
    load_checkpoint,
    resolve_device,
)

BASELINE_METHODS = BASELINES
LOGGER = logging.getLogger(__name__)

_ALIASES = {
    "ce": "cross_entropy",
    "crossentropy": "cross_entropy",
    "label_smoothing_ce": "label_smoothing",
    "ce_label_smoothing": "label_smoothing",
    "supervised_contrastive": "supcon",
    "contrastive": "pair_contrastive",
    "pairwise_contrastive": "pair_contrastive",
    "triplet": "batch_hard_triplet",
    "batch_hard": "batch_hard_triplet",
    "multisimilarity": "multi_similarity",
}

# Objective constants that can change each method's training (Section 8.3).
OBJECTIVE_FIELDS: dict[str, tuple[str, ...]] = {
    "cross_entropy": (),
    "label_smoothing": ("label_smoothing",),
    "arcface": ("scale", "arcface_margin"),
    "cosface": ("scale", "cosface_margin"),
    "pair_contrastive": ("pair_margin",),
    "supcon": ("supcon_temperature",),
    "batch_hard_triplet": ("triplet_margin",),
    "multi_similarity": (
        "multi_similarity_alpha",
        "multi_similarity_beta",
        "multi_similarity_base",
        "multi_similarity_miner_epsilon",
    ),
}


def normalize_baseline_method(method: str) -> str:
    """Return one canonical final-spec baseline name."""

    name = str(method).strip().lower().replace("-", "_").replace(" ", "_")
    name = _ALIASES.get(name, name)
    if name not in BASELINE_METHODS:
        raise ValueError(
            f"unsupported baseline method {method!r}; choose one of {list(BASELINE_METHODS)}"
        )
    return name


@dataclass(frozen=True)
class BaselineTrainingConfig:
    """Shared optimizer budget and locked objective constants."""

    method: str = "cross_entropy"
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    optimizer: str = "adamw"
    momentum: float = 0.9
    warmup_epochs: int = 0
    scheduler: str = "cosine"
    workers: int = 0
    mixed_precision: bool = False
    grad_clip_norm: float | None = None
    keep_epoch_checkpoints: bool = True
    classes_per_batch: int | None = None
    samples_per_class: int | None = None
    label_smoothing: float = 0.1
    scale: float = 30.0
    arcface_margin: float = 0.50
    cosface_margin: float = 0.35
    pair_margin: float = 1.0
    supcon_temperature: float = 0.07
    triplet_margin: float = 0.20
    multi_similarity_alpha: float = 2.0
    multi_similarity_beta: float = 50.0
    multi_similarity_base: float = 0.5
    multi_similarity_miner_epsilon: float = 0.1

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", normalize_baseline_method(self.method))
        object.__setattr__(self, "optimizer", self.optimizer.lower())
        object.__setattr__(self, "scheduler", self.scheduler.lower())
        if self.epochs < 1 or self.batch_size < 2:
            raise ValueError("epochs >= 1 and batch_size >= 2 are required")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.optimizer not in {"adamw", "sgd"} or self.scheduler not in {"cosine", "none"}:
            raise ValueError("unsupported optimizer or scheduler")
        if not 0 <= self.warmup_epochs < self.epochs or self.workers < 0:
            raise ValueError("warmup_epochs must lie in [0, epochs) and workers be >= 0")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must lie in [0, 1)")
        positive = (
            self.scale,
            self.pair_margin,
            self.supcon_temperature,
            self.triplet_margin,
            self.multi_similarity_alpha,
            self.multi_similarity_beta,
        )
        if any(not np.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("scales, temperatures, margins, alpha, and beta must be positive")
        if not 0 <= self.arcface_margin < np.pi or self.cosface_margin < 0:
            raise ValueError("invalid ArcFace or CosFace margin")
        if self.multi_similarity_miner_epsilon < 0:
            raise ValueError("multi_similarity_miner_epsilon must be non-negative")

    @classmethod
    def from_config(cls, config: Mapping[str, Any], method: str) -> BaselineTrainingConfig:
        """Build one method's config from resolved ``training``/``sampling``/``baseline``."""

        allowed = {item.name for item in fields(cls)}
        sampling = dict(config.get("sampling", {}))
        values: dict[str, Any] = {
            key: value for key, value in config.get("training", {}).items() if key in allowed
        }
        values.update({key: value for key, value in sampling.items() if key in allowed})
        if sampling.get("target_batch_size") is not None:
            values["batch_size"] = int(sampling["target_batch_size"])
        values.update(
            {key: value for key, value in config.get("baseline", {}).items() if key in allowed}
        )
        return cls(**{**values, "method": method})

    @property
    def uses_pk_sampler(self) -> bool:
        return self.method in METRIC_BASELINES

    @property
    def objective(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in OBJECTIVE_FIELDS[self.method]}


class BaselineModel(nn.Module):
    """A raw Cartesian encoder and an optional, separately saved native head."""

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        *,
        method: str,
        embedding_dimension: int | None = None,
        scale: float = 30.0,
        arcface_margin: float = 0.50,
        cosface_margin: float = 0.35,
    ) -> None:
        super().__init__()
        self.method = normalize_baseline_method(method)
        self.num_classes = int(num_classes)
        dimension = embedding_dimension or getattr(encoder, "embedding_dimension", None)
        if self.num_classes < 2 or dimension is None or int(dimension) < 1:
            raise ValueError("num_classes >= 2 and a positive embedding dimension are required")
        self.embedding_dimension = int(dimension)
        self.encoder = encoder
        self.native_head = build_native_head(
            self.method,
            self.embedding_dimension,
            self.num_classes,
            scale=scale,
            arcface_margin=arcface_margin,
            cosface_margin=cosface_margin,
        )

    @classmethod
    def from_config(
        cls,
        encoder: nn.Module,
        num_classes: int,
        config: BaselineTrainingConfig,
        *,
        embedding_dimension: int | None = None,
    ) -> BaselineModel:
        return cls(
            encoder,
            num_classes,
            method=config.method,
            embedding_dimension=embedding_dimension,
            scale=config.scale,
            arcface_margin=config.arcface_margin,
            cosface_margin=config.cosface_margin,
        )

    @property
    def has_native_head(self) -> bool:
        return self.native_head is not None

    def forward(self, inputs: Tensor) -> Tensor:
        """Return the raw pre-normalization representation for every method."""

        embeddings = self.encoder(inputs)
        expected = (inputs.shape[0], self.embedding_dimension)
        if embeddings.shape != expected:
            raise ValueError(f"encoder must return raw embeddings with shape {expected}")
        return embeddings

    def native_logits(self, embeddings: Tensor, targets: Tensor | None = None) -> Tensor:
        """Native-head scores; margins are applied only when targets are given."""

        if self.native_head is None:
            raise RuntimeError(f"{self.method} has no native classifier")
        if self.method in {"arcface", "cosface"}:
            return self.native_head(embeddings, targets)
        return self.native_head(embeddings)

    def training_loss(
        self,
        embeddings: Tensor,
        targets: Tensor,
        config: BaselineTrainingConfig,
        *,
        second_view: Tensor | None = None,
    ) -> Tensor:
        if config.method != self.method:
            raise ValueError("model and training config methods differ")
        if self.method in CLASSIFICATION_BASELINES:
            logits = self.native_logits(
                embeddings, targets if self.method in {"arcface", "cosface"} else None
            )
            smoothing = config.label_smoothing if self.method == "label_smoothing" else 0.0
            return F.cross_entropy(logits, targets, label_smoothing=smoothing)
        if self.method == "pair_contrastive":
            return pair_contrastive_loss(embeddings, targets, margin=config.pair_margin)
        if self.method == "supcon":
            if second_view is None:
                raise ValueError("SupCon requires two independently augmented views")
            return supervised_contrastive_loss(
                (embeddings, second_view), targets, temperature=config.supcon_temperature
            )
        if self.method == "batch_hard_triplet":
            return batch_hard_triplet_loss(embeddings, targets, margin=config.triplet_margin)
        return multi_similarity_loss(
            embeddings,
            targets,
            alpha=config.multi_similarity_alpha,
            beta=config.multi_similarity_beta,
            base=config.multi_similarity_base,
            miner_epsilon=config.multi_similarity_miner_epsilon,
        )


@dataclass(frozen=True)
class BaselineTrainingResult:
    method: str
    job_hash: str | None
    representation_best_epoch: int
    representation_best_accuracy: float
    representation_checkpoint: Path
    native_best_epoch: int | None
    native_best_accuracy: float | None
    native_checkpoint: Path | None
    last_checkpoint: Path
    history: tuple[dict[str, Any], ...]
    training_seconds: float
    samples_per_second: float
    parameter_counts: Mapping[str, int]


class _TwoViewDataset(Dataset[tuple[Any, Any, int]]):
    """Two independently augmented views of each item, for SupCon."""

    def __init__(self, dataset: Dataset[Any]) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Any, Any, int]:
        first, label, _ = unpack_batch(self.dataset[index])
        second, second_label, _ = unpack_batch(self.dataset[index])
        if int(label) != int(second_label):
            raise RuntimeError("a dataset changed its label between SupCon views")
        return first, second, int(label)


def _epoch_loader(
    dataset: Dataset[Any],
    config: BaselineTrainingConfig,
    schedule: PKSchedule | None,
    *,
    seed: int,
    epoch: int,
) -> DataLoader[Any]:
    if schedule is None:  # the one shared ordinary shuffled sampler
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.workers,
            generator=torch.Generator().manual_seed(derive_stage_seed(seed, "shuffle", epoch)),
            worker_init_fn=seed_worker,
        )
    source: Dataset[Any] = _TwoViewDataset(dataset) if config.method == "supcon" else dataset
    return DataLoader(
        source,
        batch_sampler=ManifestBatchSampler(schedule.epoch(epoch), dataset_size=len(dataset)),
        num_workers=config.workers,
        worker_init_fn=seed_worker,
    )


def baseline_parameter_counts(model: BaselineModel) -> dict[str, int]:
    encoder = model.encoder
    counts = {
        "encoder": count_parameters(encoder),
        "backbone": count_parameters(getattr(encoder, "backbone", None)),
        "output_map": count_parameters(getattr(encoder, "head", None)),
        "native_head": count_parameters(model.native_head),
    }
    counts["trainable"] = counts["encoder"] + counts["native_head"]
    return counts


def load_baseline_checkpoint(path: str | Path, *, map_location: Any = "cpu") -> dict[str, Any]:
    payload = load_checkpoint(path, map_location=map_location)
    kinds = {"baseline_representation", "baseline_native", "baseline_training_state"}
    if not isinstance(payload, dict) or payload.get("artifact_type") not in kinds:
        raise ValueError(f"{path} is not a ShellMetric baseline checkpoint")
    return payload


def _checkpoint(
    model: BaselineModel, kind: str, *, epoch: int, accuracy: float, job_hash: str | None
) -> dict[str, Any]:
    payload = {
        "schema_version": 3,
        "artifact_type": kind,
        "method": model.method,
        "job_hash": job_hash,
        "epoch": epoch,
        "embedding_dimension": model.embedding_dimension,
        "encoder_state": cpu_state(model.encoder),
    }
    if kind == "baseline_representation":
        payload.update(
            selection_rule="validation_raw_euclidean_1nn", validation_raw_euclidean_1nn=accuracy
        )
    else:
        assert model.native_head is not None
        payload.update(
            selection_rule="validation_native_top1",
            validation_native_accuracy=accuracy,
            native_head_state=cpu_state(model.native_head),
        )
    return payload


@torch.no_grad()
def native_predictions(model: BaselineModel, embeddings: np.ndarray) -> np.ndarray:
    """Inference-time native-head predictions from cached raw embeddings."""

    device = next(model.native_head.parameters()).device  # type: ignore[union-attr]
    logits = model.native_logits(torch.as_tensor(embeddings, dtype=torch.float32, device=device))
    return logits.argmax(dim=1).cpu().numpy()


def train_baseline(
    model: BaselineModel,
    train_dataset: Dataset[Any],
    validation_dataset: Dataset[Any],
    *,
    config: BaselineTrainingConfig,
    seed: int = 0,
    output_dir: str | Path,
    device: str | torch.device = "cpu",
    schedule: PKSchedule | None = None,
    gallery_dataset: Dataset[Any] | None = None,
    job_hash: str | None = None,
    resume: bool = False,
) -> BaselineTrainingResult:
    """Train one matched baseline and save both validation checkpoint selections."""

    if config.method != model.method:
        raise ValueError("model and training config methods differ")
    seed_everything(seed)
    target = resolve_device(device)
    model.to(target)
    gallery_dataset = train_dataset if gallery_dataset is None else gallery_dataset
    labels = dataset_labels(train_dataset)
    if not np.array_equal(np.unique(labels), np.arange(model.num_classes)):
        raise ValueError("training labels must contain every contiguous internal class index")
    if not np.array_equal(dataset_labels(gallery_dataset), labels):
        raise ValueError("gallery_dataset must align index-by-index with train_dataset")
    if config.uses_pk_sampler:
        schedule = schedule or PKSchedule.create(
            labels,
            seed=seed,
            classes_per_batch=config.classes_per_batch,
            samples_per_class=config.samples_per_class,
            target_batch_size=config.batch_size,
        )
        if not np.array_equal(schedule.labels, labels):
            raise ValueError("the paired batch schedule was built for different training labels")
    elif schedule is not None:
        raise ValueError("classification baselines use the shared shuffled sampler")
    optimizer = build_optimizer(
        [{"params": model.parameters(), "weight_decay": config.weight_decay}],
        name=config.optimizer,
        learning_rate=config.learning_rate,
        momentum=config.momentum,
    )
    scheduler = build_scheduler(
        optimizer, name=config.scheduler, epochs=config.epochs, warmup_epochs=config.warmup_epochs
    )
    scaler = grad_scaler(target, config.mixed_precision)
    checkpoints = Path(output_dir) / "checkpoints"
    last_path = checkpoints / "last.pt"

    start_epoch, elapsed = 1, 0.0
    best = {"representation": (0, -1.0), "native": (0, -1.0)}
    history: list[dict[str, Any]] = []
    if resume and last_path.is_file():
        payload = load_baseline_checkpoint(last_path, map_location=target)
        if payload["job_hash"] != job_hash:
            raise ValueError("resume checkpoint does not match this baseline job")
        model.encoder.load_state_dict(payload["encoder_state"])
        if model.native_head is not None:
            model.native_head.load_state_dict(payload["native_head_state"])
        optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
        restore_rng_state(payload["rng_state"])
        start_epoch = int(payload["epoch"]) + 1
        best = {name: tuple(value) for name, value in payload["best"].items()}
        elapsed = float(payload["training_seconds"])
        history = list(payload["history"])

    for epoch in range(start_epoch, config.epochs + 1):
        started = time.perf_counter()
        model.train()
        loss_sum, batches, samples = 0.0, 0, 0
        for batch in _epoch_loader(train_dataset, config, schedule, seed=seed, epoch=epoch):
            if config.method == "supcon":
                first, second, batch_labels = batch
                inputs, second_inputs = first.to(target), second.to(target)
            else:
                inputs, batch_labels, _ = unpack_batch(batch)
                inputs, second_inputs = torch.as_tensor(inputs).to(target), None
            targets = torch.as_tensor(batch_labels, dtype=torch.long, device=target)
            optimizer.zero_grad(set_to_none=True)
            with autocast(target, config.mixed_precision):
                embeddings = model(inputs)
                second_embeddings = None if second_inputs is None else model(second_inputs)
            loss = model.training_loss(
                embeddings.float(),
                targets,
                config,
                second_view=None if second_embeddings is None else second_embeddings.float(),
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {model.method} loss at epoch {epoch}")
            scaler.scale(loss).backward()
            if config.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach())
            batches += 1
            samples += int(targets.numel())
        if scheduler is not None:
            scheduler.step()
        elapsed += time.perf_counter() - started

        gallery_z, gallery_y, gallery_ids = extract_embeddings(
            model, gallery_dataset, device=target, workers=config.workers
        )
        validation_z, validation_y, _ = extract_embeddings(
            model, validation_dataset, device=target, workers=config.workers
        )
        accuracy = {
            "representation": euclidean_1nn_accuracy(
                validation_z, validation_y, gallery_z, gallery_y, gallery_ids
            )
        }
        if model.native_head is not None:
            accuracy["native"] = float(
                np.mean(native_predictions(model, validation_z) == validation_y)
            )
        history.append(
            {
                "epoch": epoch,
                "loss": loss_sum / max(batches, 1),
                "samples": samples,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "validation_raw_euclidean_1nn": accuracy["representation"],
                "validation_native_accuracy": accuracy.get("native"),
            }
        )
        LOGGER.info(
            "%s epoch %d/%d loss=%.4f validation_1nn=%.4f",
            model.method,
            epoch,
            config.epochs,
            history[-1]["loss"],
            accuracy["representation"],
        )
        for name, value in accuracy.items():
            if value > best[name][1]:
                best[name] = (epoch, value)
                kind = f"baseline_{name}"
                atomic_torch_save(
                    checkpoints / f"{name}_best.pt",
                    _checkpoint(model, kind, epoch=epoch, accuracy=value, job_hash=job_hash),
                )
        if config.keep_epoch_checkpoints:
            atomic_torch_save(
                checkpoints / f"epoch_{epoch:04d}.pt",
                {
                    "epoch": epoch,
                    "encoder_state": cpu_state(model.encoder),
                    "native_head_state": (
                        None if model.native_head is None else cpu_state(model.native_head)
                    ),
                },
            )
        atomic_torch_save(
            last_path,
            {
                **_checkpoint(
                    model,
                    "baseline_representation",
                    epoch=epoch,
                    accuracy=accuracy["representation"],
                    job_hash=job_hash,
                ),
                "artifact_type": "baseline_training_state",
                "native_head_state": (
                    None if model.native_head is None else cpu_state(model.native_head)
                ),
                "optimizer": optimizer.state_dict(),
                "scheduler": None if scheduler is None else scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "rng_state": capture_rng_state(),
                "best": {name: list(value) for name, value in best.items()},
                "history": history,
                "training_seconds": elapsed,
            },
        )
        save_json(Path(output_dir) / "history.json", history, overwrite=True)

    has_native = model.native_head is not None
    total_samples = sum(int(record["samples"]) for record in history)
    return BaselineTrainingResult(
        method=model.method,
        job_hash=job_hash,
        representation_best_epoch=best["representation"][0],
        representation_best_accuracy=best["representation"][1],
        representation_checkpoint=checkpoints / "representation_best.pt",
        native_best_epoch=best["native"][0] if has_native else None,
        native_best_accuracy=best["native"][1] if has_native else None,
        native_checkpoint=checkpoints / "native_best.pt" if has_native else None,
        last_checkpoint=last_path,
        history=tuple(history),
        training_seconds=elapsed,
        samples_per_second=total_samples / max(elapsed, 1e-12),
        parameter_counts=baseline_parameter_counts(model),
    )


def restore_baseline_model(model: BaselineModel, checkpoint: str | Path) -> dict[str, Any]:
    """Load a representation or native checkpoint into ``model``."""

    payload = load_baseline_checkpoint(checkpoint)
    if payload.get("method") != model.method:
        raise ValueError("checkpoint and model methods differ")
    model.encoder.load_state_dict(payload["encoder_state"])
    if payload.get("native_head_state") is not None:
        if model.native_head is None:
            raise ValueError("checkpoint contains a native head but the model does not")
        model.native_head.load_state_dict(payload["native_head_state"])
    return payload


__all__ = [
    "BASELINE_METHODS",
    "OBJECTIVE_FIELDS",
    "BaselineModel",
    "BaselineTrainingConfig",
    "BaselineTrainingResult",
    "baseline_parameter_counts",
    "load_baseline_checkpoint",
    "native_predictions",
    "normalize_baseline_method",
    "restore_baseline_model",
    "train_baseline",
]
