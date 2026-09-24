"""Method-agnostic helpers shared by the pilot, ShellMetric, and baseline trainers."""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager, nullcontext
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from .data import unpack_batch


def resolve_device(requested: str | torch.device = "auto") -> torch.device:
    if isinstance(requested, torch.device):
        return requested
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def build_optimizer(
    groups: Iterable[Mapping[str, Any]],
    *,
    name: str,
    learning_rate: float,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    """Build SGD, AdamW, or Adam over groups that each declare their weight decay."""

    param_groups = [{**group, "params": list(group["params"])} for group in groups]
    param_groups = [group for group in param_groups if group["params"]]
    normalized = name.strip().lower()
    if normalized == "sgd":
        return torch.optim.SGD(param_groups, lr=learning_rate, momentum=momentum)
    if normalized == "adamw":
        return torch.optim.AdamW(param_groups, lr=learning_rate)
    if normalized == "adam":
        return torch.optim.Adam(param_groups, lr=learning_rate)
    raise ValueError(f"unsupported optimizer {name!r}; choose sgd, adamw, or adam")


def warmup_cosine_factor(
    epoch: int, *, epochs: int, warmup_epochs: int = 0, floor: float = 0.0
) -> float:
    """Learning-rate multiplier after ``epoch`` completed epochs.

    Linear warmup over ``warmup_epochs`` is followed by cosine decay to ``floor``.
    """

    if warmup_epochs and epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return floor + (1.0 - floor) * cosine


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    name: str,
    epochs: int,
    warmup_epochs: int = 0,
    floor: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    normalized = name.strip().lower()
    if normalized in {"", "none", "constant"}:
        return None
    if normalized != "cosine":
        raise ValueError(f"unsupported scheduler {name!r}; choose cosine or none")
    if not 0 <= warmup_epochs < epochs:
        raise ValueError("warmup_epochs must lie in [0, epochs)")
    factor = partial(warmup_cosine_factor, epochs=epochs, warmup_epochs=warmup_epochs, floor=floor)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def autocast(device: torch.device, enabled: bool) -> AbstractContextManager[Any]:
    """Mixed precision is honored only on CUDA; CPU runs stay in full precision."""

    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def grad_scaler(device: torch.device, enabled: bool) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=bool(enabled and device.type == "cuda"))


def atomic_torch_save(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
        torch.save(dict(payload), temporary)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def load_checkpoint(path: str | Path, *, map_location: Any = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def cpu_state(module: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def count_parameters(module: nn.Module | None, *, trainable_only: bool = True) -> int:
    if module is None:
        return 0
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad or not trainable_only
    )


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    dataset: Dataset[Any],
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 256,
    workers: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(embeddings, labels, sample_ids)`` in dataset order, in eval mode."""

    target = resolve_device(device)
    model.eval()
    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    sample_ids: list[str] = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers):
        inputs, targets, ids = unpack_batch(batch)
        output = model(torch.as_tensor(inputs).to(target))
        embeddings.append(output.detach().float().cpu().numpy())
        labels.append(torch.as_tensor(targets).cpu().numpy().astype(np.int64, copy=False))
        start = len(sample_ids)
        count = output.shape[0]
        sample_ids.extend(
            (str(index) for index in range(start, start + count))
            if ids is None
            else (str(value) for value in ids)
        )
    return np.concatenate(embeddings), np.concatenate(labels), np.asarray(sample_ids, dtype=str)


__all__ = [
    "atomic_torch_save",
    "autocast",
    "build_optimizer",
    "build_scheduler",
    "count_parameters",
    "cpu_state",
    "extract_embeddings",
    "grad_scaler",
    "load_checkpoint",
    "resolve_device",
    "warmup_cosine_factor",
]
