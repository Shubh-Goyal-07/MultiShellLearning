"""Decoder-free ShellMetric representation training."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, default_collate

from ..artifacts import save_json, stable_hash
from ..data import dataset_labels, unpack_batch
from ..reproducibility import capture_rng_state, restore_rng_state, seed_everything, seed_worker
from ..training import (
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
from .evaluate import euclidean_1nn_accuracy
from .heads import RadialPowerGate
from .loss import ShellMetricLoss
from .plan import ShellPlan
from .radii import OrderedRadii
from .sampler import ManifestBatchSampler, PKSchedule

LOGGER = logging.getLogger(__name__)
LOSS_FIELDS = (
    "positive_margin",
    "negative_margin_base",
    "confusion_margin_delta",
    "negative_weight",
    "shell_weight",
    "shell_huber_beta",
    "epsilon",
)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    optimizer: str = "adamw"
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    momentum: float = 0.9
    warmup_epochs: int = 0
    scheduler: str = "cosine"
    mixed_precision: bool = False
    workers: int = 0
    keep_epoch_checkpoints: bool = True
    target_batch_size: int = 128
    classes_per_batch: int | None = None
    samples_per_class: int | None = None
    positive_margin: float = 0.50
    negative_margin_base: float = 1.00
    confusion_margin_delta: float = 0.50
    negative_weight: float = 1.00
    shell_weight: float = 1.00
    shell_huber_beta: float = 0.10
    epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer budget")
        if self.optimizer not in {"adamw", "sgd"} or self.scheduler not in {"cosine", "none"}:
            raise ValueError("unsupported optimizer or scheduler")
        if not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError("warmup_epochs must lie in [0, epochs)")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> TrainingConfig:
        """Merge the resolved ``training``, ``loss``, and ``sampling`` sections."""

        allowed = {item.name for item in fields(cls)}
        values: dict[str, Any] = {}
        for section in ("training", "loss", "sampling"):
            values.update(
                {key: value for key, value in config.get(section, {}).items() if key in allowed}
            )
        return cls(**values)

    @property
    def loss_options(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in LOSS_FIELDS}


@dataclass(frozen=True)
class TrainingResult:
    job_hash: str
    best_epoch: int
    best_validation_accuracy: float
    checkpoint: Path
    last_checkpoint: Path
    history: tuple[dict[str, Any], ...]
    training_seconds: float
    samples_per_second: float
    parameter_counts: Mapping[str, int]


def _label_positions(labels: np.ndarray, class_ids: tuple[int, ...]) -> np.ndarray:
    lookup = {int(class_id): index for index, class_id in enumerate(class_ids)}
    try:
        return np.asarray([lookup[int(label)] for label in labels], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"dataset class {exc.args[0]} is absent from the shell plan") from exc


def parameter_counts(encoder: nn.Module, radii: nn.Module | None = None) -> dict[str, int]:
    """Trainable counts; the output map excludes the backbone, per the protocol."""

    backbone = getattr(encoder, "backbone", None)
    head = getattr(encoder, "head", None)
    counts = {
        "encoder": count_parameters(encoder),
        "backbone": count_parameters(backbone),
        "output_map": count_parameters(head),
        "radii": count_parameters(radii),
    }
    counts["trainable"] = counts["encoder"] + counts["radii"]
    return counts


def _gate_state(encoder: nn.Module) -> dict[str, float]:
    head = getattr(encoder, "head", None)
    if not isinstance(head, RadialPowerGate):
        return {}
    return {"radial_gate_beta": float(head.beta.detach()), "radial_gate_power": float(head.power)}


def _check_initial_variance(
    encoder: nn.Module, dataset: Dataset[Any], indices: tuple[int, ...], device: torch.device
) -> float:
    """Abort before training on a collapsed or non-finite initial embedding batch."""

    state = capture_rng_state()
    try:
        inputs, _, _ = unpack_batch(default_collate([dataset[index] for index in indices]))
        encoder.eval()
        with torch.no_grad():
            variance = encoder(torch.as_tensor(inputs).to(device)).float().var(unbiased=False)
    finally:
        restore_rng_state(state)
    if not torch.isfinite(variance) or float(variance) <= 0.0:
        raise RuntimeError("initial embedding variance must be finite and nonzero")
    return float(variance)


def representation_payload(
    encoder: nn.Module,
    radii: OrderedRadii | None,
    *,
    job_hash: str,
    plan: ShellPlan,
    epoch: int,
    validation_accuracy: float,
    no_shell_loss: bool,
) -> dict[str, Any]:
    """The saved representation: encoder plus shared radii, never a decoder."""

    return {
        "schema_version": 2,
        "kind": "shellmetric_representation",
        "selection_rule": "validation_raw_euclidean_1nn",
        "job_hash": job_hash,
        "plan_semantic_hash": None if no_shell_loss else plan.plan_semantic_hash,
        "plan_provenance_hash": plan.plan_provenance_hash,
        "epoch": epoch,
        "validation_euclidean_1nn": validation_accuracy,
        "no_shell_loss": no_shell_loss,
        "encoder": cpu_state(encoder),
        "radii": None if radii is None else cpu_state(radii),
        "radii_values": None if radii is None else radii().detach().cpu().tolist(),
    }


def train_shellmetric(
    encoder: nn.Module,
    train_dataset: Dataset[Any],
    validation_dataset: Dataset[Any],
    plan: ShellPlan,
    *,
    config: TrainingConfig | None = None,
    seed: int = 0,
    output_dir: str | Path,
    device: str | torch.device = "cpu",
    schedule: PKSchedule | None = None,
    gallery_dataset: Dataset[Any] | None = None,
    no_shell_loss: bool = False,
    job_hash: str | None = None,
    resume: bool = False,
) -> TrainingResult:
    """Train only the embedding encoder and, unless ablated, the shared radii.

    ``gallery_dataset`` must be the training split under deterministic
    evaluation transforms; it supplies the gallery for per-epoch checkpoint
    selection by validation raw-Euclidean 1-NN accuracy.
    """

    config = TrainingConfig() if config is None else config
    seed_everything(seed)
    target = resolve_device(device)
    encoder.to(target)
    gallery_dataset = train_dataset if gallery_dataset is None else gallery_dataset
    train_labels = dataset_labels(train_dataset)
    if not np.array_equal(dataset_labels(gallery_dataset), train_labels):
        raise ValueError("gallery_dataset must align index-by-index with train_dataset")
    positions = _label_positions(train_labels, plan.class_ids)
    schedule = schedule or PKSchedule.create(
        positions,
        seed=seed,
        classes_per_batch=config.classes_per_batch,
        samples_per_class=config.samples_per_class,
        target_batch_size=config.target_batch_size,
    )
    if not np.array_equal(schedule.labels, positions):
        raise ValueError("the batch schedule was built for different training labels")
    radii = None if no_shell_loss else OrderedRadii.from_plan(plan).to(target)
    criterion = ShellMetricLoss(
        plan.w_hat,
        None if no_shell_loss else plan.assignment,
        **{**config.loss_options, "shell_weight": 0.0 if no_shell_loss else config.shell_weight},
    ).to(target)
    groups = [{"params": encoder.parameters(), "weight_decay": config.weight_decay}]
    if radii is not None:
        groups.append({"params": radii.parameters(), "weight_decay": 0.0})
    optimizer = build_optimizer(
        groups, name=config.optimizer, learning_rate=config.learning_rate, momentum=config.momentum
    )
    scheduler = build_scheduler(
        optimizer, name=config.scheduler, epochs=config.epochs, warmup_epochs=config.warmup_epochs
    )
    scaler = grad_scaler(target, config.mixed_precision)
    run_hash = job_hash or stable_hash(
        {
            "plan": None if no_shell_loss else plan.plan_semantic_hash,
            "w_hat": plan.w_hat.tolist() if no_shell_loss else None,
            "training": asdict(config),
            "seed": int(seed),
            "schedule": schedule.schedule_hash,
            "encoder_shapes": {
                key: list(value.shape) for key, value in encoder.state_dict().items()
            },
        }
    )
    root = Path(output_dir)
    checkpoints = root / "checkpoints"
    last_path = checkpoints / "last.pt"
    label_map = {int(class_id): index for index, class_id in enumerate(plan.class_ids)}

    start_epoch, best_epoch, best_accuracy, elapsed = 1, 0, -1.0, 0.0
    history: list[dict[str, Any]] = []
    if resume and last_path.is_file():
        payload = load_checkpoint(last_path, map_location=target)
        if payload["job_hash"] != run_hash:
            raise ValueError("resume checkpoint does not match the resolved training semantics")
        encoder.load_state_dict(payload["encoder"])
        if radii is not None:
            radii.load_state_dict(payload["radii"])
        optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
        restore_rng_state(payload["rng_state"])
        start_epoch = int(payload["epoch"]) + 1
        best_epoch = int(payload["best_epoch"])
        best_accuracy = float(payload["best_validation_accuracy"])
        elapsed = float(payload["training_seconds"])
        history = list(payload["history"])
    else:
        _check_initial_variance(encoder, gallery_dataset, schedule.epoch(1)[0], target)

    steps_per_epoch = schedule.steps_per_epoch
    samples_per_step = schedule.classes_per_batch * schedule.samples_per_class
    for epoch in range(start_epoch, config.epochs + 1):
        started = time.perf_counter()
        encoder.train()
        loader = DataLoader(
            train_dataset,
            batch_sampler=ManifestBatchSampler(
                schedule.epoch(epoch), dataset_size=len(train_dataset)
            ),
            num_workers=config.workers,
            worker_init_fn=seed_worker,
        )
        totals = {"total": 0.0, "positive": 0.0, "negative": 0.0, "shell": 0.0}
        for batch in loader:
            inputs, labels, _ = unpack_batch(batch)
            targets = torch.as_tensor(
                [label_map[int(label)] for label in torch.as_tensor(labels).tolist()],
                dtype=torch.long,
                device=target,
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast(target, config.mixed_precision):
                embeddings = encoder(torch.as_tensor(inputs).to(target))
            output = criterion(embeddings.float(), targets, None if radii is None else radii())
            if not torch.isfinite(output.total):
                raise FloatingPointError(f"non-finite ShellMetric loss at epoch {epoch}")
            scaler.scale(output.total).backward()
            scaler.step(optimizer)
            scaler.update()
            for name in totals:
                totals[name] += float(getattr(output, name).detach())
        if scheduler is not None:
            scheduler.step()
        elapsed += time.perf_counter() - started

        gallery_z, gallery_y, gallery_ids = extract_embeddings(
            encoder, gallery_dataset, device=target, workers=config.workers
        )
        validation_z, validation_y, _ = extract_embeddings(
            encoder, validation_dataset, device=target, workers=config.workers
        )
        accuracy = euclidean_1nn_accuracy(
            validation_z, validation_y, gallery_z, gallery_y, gallery_ids
        )
        record: dict[str, Any] = {
            "epoch": epoch,
            **{name: value / steps_per_epoch for name, value in totals.items()},
            "validation_euclidean_1nn": accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "radii": None if radii is None else radii().detach().cpu().tolist(),
            **_gate_state(encoder),
        }
        history.append(record)
        LOGGER.info(
            "shellmetric epoch %d/%d loss=%.4f validation_1nn=%.4f radii=%s",
            epoch,
            config.epochs,
            record["total"],
            accuracy,
            record["radii"],
        )
        representation = representation_payload(
            encoder,
            radii,
            job_hash=run_hash,
            plan=plan,
            epoch=epoch,
            validation_accuracy=accuracy,
            no_shell_loss=no_shell_loss,
        )
        if accuracy > best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch
            atomic_torch_save(checkpoints / "best.pt", representation)
        if config.keep_epoch_checkpoints:
            atomic_torch_save(checkpoints / f"epoch_{epoch:04d}.pt", representation)
        atomic_torch_save(
            last_path,
            {
                **representation,
                "kind": "shellmetric_training_state",
                "best_epoch": best_epoch,
                "best_validation_accuracy": best_accuracy,
                "optimizer": optimizer.state_dict(),
                "scheduler": None if scheduler is None else scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "rng_state": capture_rng_state(),
                "history": history,
                "training_seconds": elapsed,
            },
        )
        save_json(root / "history.json", history, overwrite=True)

    trained_epochs = config.epochs
    return TrainingResult(
        job_hash=run_hash,
        best_epoch=best_epoch,
        best_validation_accuracy=best_accuracy,
        checkpoint=checkpoints / "best.pt",
        last_checkpoint=last_path,
        history=tuple(history),
        training_seconds=elapsed,
        samples_per_second=trained_epochs
        * steps_per_epoch
        * samples_per_step
        / max(elapsed, 1e-12),
        parameter_counts=parameter_counts(encoder, radii),
    )


def load_representation(
    encoder: nn.Module, checkpoint: str | Path | Mapping[str, Any], plan: ShellPlan | None = None
) -> tuple[nn.Module, np.ndarray | None]:
    """Load a saved representation; return the encoder and learned radii (if any)."""

    payload = checkpoint if isinstance(checkpoint, Mapping) else load_checkpoint(checkpoint)
    encoder.load_state_dict(payload["encoder"])
    if payload.get("radii") is None:
        return encoder, None
    if plan is None:
        raise ValueError("a plan is required to rebuild the learned radii")
    radii = OrderedRadii.from_plan(plan)
    radii.load_state_dict(payload["radii"])
    with torch.no_grad():
        values: Tensor = radii()
    return encoder, values.double().numpy()


__all__ = [
    "TrainingConfig",
    "TrainingResult",
    "load_representation",
    "parameter_counts",
    "representation_payload",
    "train_shellmetric",
]
