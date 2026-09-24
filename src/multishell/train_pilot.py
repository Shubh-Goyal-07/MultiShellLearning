"""Leakage-safe cross-fitted pilot training.

The functions in this module deliberately operate on an already selected
development dataset.  They never construct, import, or inspect an official test
dataset.  This makes the boundary enforced by the split/pipeline layer easy to
audit and easy to test with a dataset that raises when accessed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

from .artifacts import hash_file, save_json, save_npz, stable_hash, to_jsonable
from .data import dataset_labels, dataset_sample_ids, unpack_batch
from .reproducibility import derive_stage_seed, seed_everything
from .training import (
    autocast,
    build_optimizer,
    build_scheduler,
    grad_scaler,
    resolve_device,
)

ProgressCallback = Callable[[Mapping[str, Any]], None]

# Public names kept for callers that hash or verify pilot artifacts.
canonical_json_hash = stable_hash
file_sha256 = hash_file


def _progress_enabled(requested: bool | None) -> bool:
    """Resolve terminal progress without making it part of the science config.

    ``MULTISHELL_PROGRESS=1`` forces display in a captured terminal and
    ``MULTISHELL_PROGRESS=0`` suppresses it; CI output stays quiet.
    """

    if requested is not None:
        return bool(requested)
    override = os.environ.get("MULTISHELL_PROGRESS", "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return True
    if override in {"0", "false", "no", "off"}:
        return False
    if os.environ.get("CI", "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    return bool(getattr(sys.stderr, "isatty", lambda: False)())


def _epoch_iterator(total: int, *, enabled: bool, description: str) -> Iterable[int]:
    if not enabled:
        return range(total)
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - exercised only in minimal installs
        print(f"[pilot] {description}", file=sys.stderr, flush=True)
        return range(total)
    return tqdm(range(total), desc=description, unit="epoch", dynamic_ncols=True)


class _JsonlProgressLog:
    """Append small, crash-resilient progress records outside scientific artifacts."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")  # a new invocation is a new stream

    def emit(self, event: Mapping[str, Any]) -> None:
        payload = {
            "schema_version": 1,
            "component": "pilot_crossfit",
            "time_utc": datetime.now(UTC).isoformat(),
            **dict(event),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(to_jsonable(payload), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _emit(callback: ProgressCallback | None, event: str, **fields: Any) -> None:
    if callback is not None:
        callback({"event": event, **fields})


@dataclass(frozen=True)
class PilotTrainingConfig:
    """Configuration for every model in a cross-fitting run."""

    epochs: int = 5
    batch_size: int = 128
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"
    momentum: float = 0.9
    scheduler: str = "cosine"
    warmup_epochs: int = 0
    min_learning_rate: float = 0.0
    label_smoothing: float = 0.0
    num_workers: int = 0
    pin_memory: bool = False
    mixed_precision: bool = False
    grad_clip_norm: float | None = None
    deterministic: bool = True
    device: str = "auto"
    seeds: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("pilot epochs, batch_size, and learning_rate must be positive")
        if not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError("pilot warmup_epochs must lie in [0, epochs)")
        if not self.seeds:
            raise ValueError("at least one pilot seed is required")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | PilotTrainingConfig | None
    ) -> PilotTrainingConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        raw: Mapping[str, Any] = value.get("pilot", value)  # type: ignore[assignment]
        allowed = {item.name for item in dataclasses.fields(cls)}
        kwargs = {key: item for key, item in raw.items() if key in allowed}
        if "seeds" in kwargs:
            kwargs["seeds"] = tuple(int(seed) for seed in kwargs["seeds"])
        return cls(**kwargs)


@dataclass
class OOFPredictions:
    """Complete out-of-fold predictions, with one prediction per seed/sample."""

    sample_ids: np.ndarray
    labels: np.ndarray
    fold_ids: np.ndarray
    seeds: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray
    class_ids: np.ndarray
    histories: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.sample_ids = np.asarray(self.sample_ids).astype(str)
        self.labels = np.asarray(self.labels, dtype=np.int64)
        self.fold_ids = np.asarray(self.fold_ids, dtype=np.int64)
        self.seeds = np.asarray(self.seeds, dtype=np.int64)
        self.logits = np.asarray(self.logits)
        self.probabilities = np.asarray(self.probabilities)
        self.predictions = np.asarray(self.predictions, dtype=np.int64)
        self.class_ids = np.asarray(self.class_ids, dtype=np.int64)

        n, k, c = len(self.labels), len(self.seeds), len(self.class_ids)
        if len(self.sample_ids) != n or self.fold_ids.shape != (n,):
            raise ValueError("OOF sample_ids, labels, and fold_ids must have the same length")
        if len(set(self.sample_ids.tolist())) != n:
            raise ValueError("OOF sample IDs must be unique")
        if self.logits.shape != (k, n, c):
            raise ValueError(f"expected logits shape {(k, n, c)}, got {self.logits.shape}")
        if self.probabilities.shape != self.logits.shape:
            raise ValueError("OOF probabilities and logits must have identical shapes")
        if self.predictions.shape != (k, n):
            raise ValueError(f"expected predictions shape {(k, n)}, got {self.predictions.shape}")
        if not np.isfinite(self.logits).all() or not np.isfinite(self.probabilities).all():
            raise ValueError("OOF predictions contain NaN or infinity")
        if np.any(self.fold_ids < 0):
            raise ValueError("every OOF sample must have a non-negative fold ID")
        if not np.allclose(self.probabilities.sum(axis=-1), 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError("OOF probabilities do not sum to one")

    @property
    def mean_probabilities(self) -> np.ndarray:
        return self.probabilities.mean(axis=0)

    @property
    def mean_logits(self) -> np.ndarray:
        return self.logits.mean(axis=0)

    def save(self, output_dir: str | os.PathLike[str]) -> tuple[Path, Path]:
        self.validate()
        root = Path(output_dir)
        arrays_path = save_npz(
            root / "oof_predictions.npz",
            {
                "sample_ids": self.sample_ids,
                "labels": self.labels,
                "fold_ids": self.fold_ids,
                "seeds": self.seeds,
                "logits": self.logits,
                "probabilities": self.probabilities,
                "predictions": self.predictions,
                "class_ids": self.class_ids,
            },
            overwrite=True,
        )
        metadata = {
            **self.metadata,
            "schema_version": 1,
            "artifact_type": "oof_predictions",
            "arrays_sha256": hash_file(arrays_path),
            "histories": self.histories,
        }
        metadata_path = save_json(root / "oof_metadata.json", metadata, overwrite=True)
        return arrays_path, metadata_path

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> OOFPredictions:
        source = Path(path)
        root = source if source.is_dir() else source.parent
        arrays_path = source if source.suffix == ".npz" else root / "oof_predictions.npz"
        metadata_path = root / "oof_metadata.json"
        with np.load(arrays_path, allow_pickle=False) as arrays:
            values = {name: arrays[name] for name in arrays.files}
        metadata: dict[str, Any] = {}
        histories: list[dict[str, Any]] = []
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected = metadata.get("arrays_sha256")
            if expected and expected != hash_file(arrays_path):
                raise ValueError(f"OOF array hash mismatch for {arrays_path}")
            histories = list(metadata.pop("histories", []))
        result = cls(**values, histories=histories, metadata=metadata)
        result.validate()
        return result


def derive_seed(root_seed: int, *parts: Any) -> int:
    return derive_stage_seed(root_seed, "pilot", *parts)


def model_tensor_output(output: Any, *, purpose: str) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, Mapping):
        preferred = ("logits", "output") if purpose == "logits" else ("embedding", "z", "output")
        for key in preferred:
            if isinstance(output.get(key), Tensor):
                return output[key]
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        return output[0]
    raise TypeError(f"model did not return a tensor suitable for {purpose}")


def _seed_worker(worker_id: int, *, base_seed: int) -> None:
    worker_seed = (base_seed + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _loader(
    dataset: Dataset[Any],
    indices: Sequence[int],
    config: PilotTrainingConfig,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader[Any]:
    return DataLoader(
        Subset(dataset, [int(index) for index in indices]),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        generator=torch.Generator().manual_seed(seed),
        worker_init_fn=partial(_seed_worker, base_seed=seed),
        persistent_workers=config.num_workers > 0,
    )


def _make_model(factory: Callable[..., nn.Module], num_classes: int) -> nn.Module:
    errors: list[Exception] = []
    for attempt in (
        lambda: factory(num_classes=num_classes),
        lambda: factory(num_classes),
        lambda: factory(),
    ):
        try:
            model = attempt()
        except TypeError as exc:
            errors.append(exc)
            continue
        if not isinstance(model, nn.Module):
            raise TypeError("model_factory must return torch.nn.Module")
        return model
    raise TypeError(f"could not call model_factory: {errors[-1]}")


def train_pilot_fold(
    dataset: Dataset[Any],
    train_indices: Sequence[int],
    holdout_indices: Sequence[int],
    model_factory: Callable[..., nn.Module],
    num_classes: int,
    config: PilotTrainingConfig | Mapping[str, Any] | None = None,
    *,
    evaluation_dataset: Dataset[Any] | None = None,
    seed: int = 0,
    show_progress: bool | None = None,
    progress_callback: ProgressCallback | None = None,
    progress_context: Mapping[str, Any] | None = None,
) -> tuple[nn.Module, np.ndarray, np.ndarray, dict[str, Any]]:
    """Train one fold for its full schedule and predict its untouched holdout fold."""

    cfg = PilotTrainingConfig.from_mapping(config)
    device = resolve_device(cfg.device)
    context = dict(progress_context or {})
    seed_label = str(context.get("root_seed", seed))
    if context.get("seed_number") is not None:
        seed_label += f" ({context['seed_number']}/{context['seed_count']})"
    fold_label = (
        f"{context['fold_number']}/{context['fold_count']}"
        if context.get("fold_number") is not None
        else str(context.get("fold_id", "?"))
    )
    description = f"Pilot seed {seed_label}, fold {fold_label}"
    fold_started = time.perf_counter()
    _emit(
        progress_callback,
        "fold_start",
        **context,
        fold_seed=int(seed),
        train_count=len(train_indices),
        holdout_count=len(holdout_indices),
        epochs=cfg.epochs,
        device=str(device),
    )
    seed_everything(seed, deterministic=cfg.deterministic)
    model = _make_model(model_factory, num_classes).to(device)
    optimizer = build_optimizer(
        [{"params": model.parameters(), "weight_decay": cfg.weight_decay}],
        name=cfg.optimizer,
        learning_rate=cfg.learning_rate,
        momentum=cfg.momentum,
    )
    scheduler = build_scheduler(
        optimizer,
        name=cfg.scheduler,
        epochs=cfg.epochs,
        warmup_epochs=cfg.warmup_epochs,
        floor=cfg.min_learning_rate / cfg.learning_rate,
    )
    scaler = grad_scaler(device, cfg.mixed_precision)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    history: list[dict[str, float]] = []

    progress = _epoch_iterator(
        cfg.epochs, enabled=_progress_enabled(show_progress), description=description
    )
    for epoch in progress:
        epoch_started = time.perf_counter()
        model.train()
        loader = _loader(
            dataset, train_indices, cfg, shuffle=True, seed=derive_seed(seed, "train", epoch)
        )
        loss_sum, correct, sample_count = 0.0, 0, 0
        for batch in loader:
            inputs, targets, _ = unpack_batch(batch)
            inputs = torch.as_tensor(inputs).to(device, non_blocking=True)
            targets = torch.as_tensor(targets).to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device, cfg.mixed_precision):
                logits = model_tensor_output(model(inputs), purpose="logits")
                if logits.ndim != 2 or logits.shape[1] != num_classes:
                    raise ValueError(
                        f"pilot logits must have shape [B,{num_classes}], got {tuple(logits.shape)}"
                    )
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            if cfg.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            batch_size = int(targets.numel())
            loss_sum += float(loss.detach()) * batch_size
            correct += int((logits.detach().argmax(dim=1) == targets).sum())
            sample_count += batch_size
        if scheduler is not None:
            scheduler.step()
        metrics = {
            "epoch": float(epoch),
            "loss": loss_sum / max(sample_count, 1),
            "accuracy": correct / max(sample_count, 1),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(metrics)
        _emit(
            progress_callback,
            "epoch_end",
            **context,
            fold_seed=int(seed),
            epoch=epoch + 1,
            epoch_index=epoch,
            epoch_count=cfg.epochs,
            loss=metrics["loss"],
            accuracy=metrics["accuracy"],
            learning_rate=metrics["learning_rate"],
            samples=sample_count,
            duration_seconds=time.perf_counter() - epoch_started,
        )
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(  # type: ignore[attr-defined]
                loss=f"{metrics['loss']:.4f}", accuracy=f"{metrics['accuracy']:.4f}"
            )

    model.eval()
    holdout_loader = _loader(
        dataset if evaluation_dataset is None else evaluation_dataset,
        holdout_indices,
        cfg,
        shuffle=False,
        seed=derive_seed(seed, "holdout"),
    )
    logits_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in holdout_loader:
            inputs, targets, _ = unpack_batch(batch)
            logits = model_tensor_output(
                model(torch.as_tensor(inputs).to(device, non_blocking=True)), purpose="logits"
            )
            logits_parts.append(logits.float().cpu().numpy())
            labels_parts.append(torch.as_tensor(targets, dtype=torch.long).cpu().numpy())
    holdout_logits = np.concatenate(logits_parts, axis=0)
    holdout_labels = np.concatenate(labels_parts, axis=0)
    if len(holdout_logits) != len(holdout_indices):
        raise RuntimeError("holdout prediction count does not match holdout fold size")
    _emit(
        progress_callback,
        "fold_end",
        **context,
        fold_seed=int(seed),
        train_count=len(train_indices),
        holdout_count=len(holdout_indices),
        holdout_accuracy=float((holdout_logits.argmax(axis=1) == holdout_labels).mean()),
        duration_seconds=time.perf_counter() - fold_started,
    )
    details = {
        "seed": int(seed),
        "train_count": len(train_indices),
        "holdout_count": len(holdout_indices),
        "history": history,
    }
    return model.cpu(), holdout_logits, holdout_labels, details


def train_pilot_crossfit(
    dataset: Dataset[Any],
    fold_ids: Sequence[int] | np.ndarray,
    model_factory: Callable[..., nn.Module],
    *,
    evaluation_dataset: Dataset[Any] | None = None,
    num_classes: int | None = None,
    labels: Sequence[int] | np.ndarray | None = None,
    sample_ids: Sequence[Any] | np.ndarray | None = None,
    class_ids: Sequence[int] | np.ndarray | None = None,
    config: PilotTrainingConfig | Mapping[str, Any] | None = None,
    split_hash: str | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    show_progress: bool | None = None,
    progress_callback: ProgressCallback | None = None,
) -> OOFPredictions:
    """Produce verified out-of-fold predictions for every sample and seed.

    ``dataset`` must contain only development samples.  ``fold_ids`` is aligned
    with that dataset and every model trains on all folds except the one it
    predicts.  Fixed-schedule training avoids selecting a checkpoint on the same
    examples whose OOF predictions are being recorded.
    """

    cfg = PilotTrainingConfig.from_mapping(config)
    if evaluation_dataset is not None:
        if len(evaluation_dataset) != len(dataset):
            raise ValueError("evaluation_dataset must align one-to-one with the training dataset")
        if not np.array_equal(dataset_labels(evaluation_dataset), dataset_labels(dataset)):
            raise ValueError("evaluation_dataset labels do not match the training dataset")
        if not np.array_equal(dataset_sample_ids(evaluation_dataset), dataset_sample_ids(dataset)):
            raise ValueError("evaluation_dataset sample IDs do not match the training dataset")
    fold_array = np.asarray(fold_ids, dtype=np.int64)
    if fold_array.shape != (len(dataset),):
        raise ValueError("fold_ids must have one entry per development sample")
    unique_folds = np.unique(fold_array)
    if len(unique_folds) < 2 or not np.array_equal(unique_folds, np.arange(len(unique_folds))):
        raise ValueError("fold IDs must be contiguous integers starting at zero with >= 2 folds")

    labels_array = dataset_labels(dataset) if labels is None else np.asarray(labels, dtype=np.int64)
    ids_array = (
        dataset_sample_ids(dataset) if sample_ids is None else np.asarray(sample_ids).astype(str)
    )
    if labels_array.shape != (len(dataset),) or ids_array.shape != (len(dataset),):
        raise ValueError("labels and sample_ids must align with the development dataset")
    if len(set(ids_array.tolist())) != len(ids_array):
        raise ValueError("development sample IDs must be unique")
    if class_ids is None:
        num_classes = int(labels_array.max()) + 1 if num_classes is None else int(num_classes)
        class_array = np.arange(num_classes, dtype=np.int64)
    else:
        class_array = np.asarray(class_ids, dtype=np.int64)
        num_classes = len(class_array) if num_classes is None else int(num_classes)
    if num_classes != len(class_array):
        raise ValueError("num_classes and class_ids disagree")
    if np.any(labels_array < 0) or np.any(labels_array >= num_classes):
        raise ValueError("pilot labels must be contiguous internal class indices")

    display_progress = _progress_enabled(show_progress)
    log = _JsonlProgressLog(Path(output_dir) / "progress.jsonl") if output_dir is not None else None

    def forward(event: Mapping[str, Any]) -> None:
        if log is not None:
            log.emit(event)
        if progress_callback is not None:
            progress_callback(event)

    callback = forward if log is not None or progress_callback is not None else None
    k, n, c = len(cfg.seeds), len(dataset), int(num_classes)
    run_started = time.perf_counter()
    _emit(
        callback,
        "run_start",
        seeds=list(cfg.seeds),
        seed_count=k,
        fold_count=len(unique_folds),
        epochs_per_fold=cfg.epochs,
        sample_count=n,
        class_count=c,
        split_hash=split_hash,
    )
    all_logits = np.full((k, n, c), np.nan, dtype=np.float32)
    histories: list[dict[str, Any]] = []
    all_indices = np.arange(n)
    for seed_index, root_seed in enumerate(cfg.seeds):
        seed_started = time.perf_counter()
        seed_context = {
            "root_seed": int(root_seed),
            "seed_index": seed_index,
            "seed_number": seed_index + 1,
            "seed_count": k,
        }
        _emit(callback, "seed_start", **seed_context)
        predicted = np.zeros(n, dtype=bool)
        for fold_index, fold_id in enumerate(unique_folds):
            holdout = all_indices[fold_array == fold_id]
            train = all_indices[fold_array != fold_id]
            _, fold_logits, observed_labels, details = train_pilot_fold(
                dataset,
                train,
                holdout,
                model_factory,
                c,
                cfg,
                seed=derive_seed(root_seed, "fold", int(fold_id)),
                evaluation_dataset=evaluation_dataset,
                show_progress=display_progress,
                progress_callback=callback,
                progress_context={
                    **seed_context,
                    "fold_id": int(fold_id),
                    "fold_index": fold_index,
                    "fold_number": fold_index + 1,
                    "fold_count": len(unique_folds),
                },
            )
            if not np.array_equal(observed_labels, labels_array[holdout]):
                raise RuntimeError("dataset order/labels changed while creating OOF predictions")
            if predicted[holdout].any():
                raise RuntimeError("a development sample was predicted by more than one fold model")
            all_logits[seed_index, holdout] = fold_logits
            predicted[holdout] = True
            histories.append({**details, "root_seed": int(root_seed), "fold_id": int(fold_id)})
        if not predicted.all():
            raise RuntimeError(
                f"missing OOF predictions for indices {np.flatnonzero(~predicted)[:10]}"
            )
        _emit(
            callback,
            "seed_end",
            **seed_context,
            oof_accuracy=float((all_logits[seed_index].argmax(axis=1) == labels_array).mean()),
            duration_seconds=time.perf_counter() - seed_started,
        )

    probabilities = (
        torch.softmax(torch.from_numpy(all_logits.astype(np.float64)), dim=-1)
        .numpy()
        .astype(np.float32)
    )
    predictions = probabilities.argmax(axis=-1).astype(np.int64)
    result = OOFPredictions(
        sample_ids=ids_array,
        labels=labels_array,
        fold_ids=fold_array,
        seeds=np.asarray(cfg.seeds, dtype=np.int64),
        logits=all_logits,
        probabilities=probabilities,
        predictions=predictions,
        class_ids=class_array,
        histories=histories,
        metadata={
            "config": asdict(cfg),
            "config_hash": canonical_json_hash(cfg),
            "split_hash": split_hash,
            "fold_count": len(unique_folds),
            "fixed_epoch_training": True,
            "test_data_accessed": False,
        },
    )
    result.validate()
    if output_dir is not None:
        result.save(output_dir)
    seed_accuracies = [float((predictions[index] == labels_array).mean()) for index in range(k)]
    _emit(
        callback,
        "run_end",
        seed_accuracies=seed_accuracies,
        mean_seed_accuracy=float(np.mean(seed_accuracies)),
        duration_seconds=time.perf_counter() - run_started,
        artifact_dir=None if output_dir is None else str(Path(output_dir)),
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Validate an existing OOF artifact without constructing any dataset."""

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--validate", type=Path, required=True, help="OOF artifact directory")
    args = parser.parse_args(argv)
    result = OOFPredictions.load(args.validate)
    print(
        json.dumps(
            {"samples": len(result.labels), "seeds": result.seeds.tolist(), "status": "valid"}
        )
    )
    return 0


__all__ = [
    "OOFPredictions",
    "PilotTrainingConfig",
    "canonical_json_hash",
    "derive_seed",
    "file_sha256",
    "model_tensor_output",
    "seed_everything",
    "train_pilot_crossfit",
    "train_pilot_fold",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
