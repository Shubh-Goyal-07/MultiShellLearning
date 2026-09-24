"""Common affine probe trained only on cached, frozen embeddings."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


def module_checksum(module: nn.Module) -> str:
    """Hash parameters and buffers without depending on mapping iteration quirks."""

    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ProbeTrial:
    weight_decay: float
    best_epoch: int
    validation_accuracy: float
    state_dict: dict[str, Tensor]


@dataclass(frozen=True)
class ProbeResult:
    model: nn.Linear
    selected_weight_decay: float
    selected_epoch: int
    validation_accuracy: float
    trials: tuple[dict[str, Any], ...]
    encoder_checksum: str | None


def _arrays(embeddings: Any, labels: Any) -> tuple[Tensor, Tensor]:
    x = torch.as_tensor(embeddings, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    if x.ndim != 2 or y.shape != (x.shape[0],) or x.shape[0] == 0:
        raise ValueError("embeddings/labels must have shapes [N,d] and [N]")
    if not torch.isfinite(x).all():
        raise ValueError("embeddings must be finite")
    return x, y


def _accuracy(model: nn.Module, x: Tensor, y: Tensor) -> float:
    with torch.no_grad():
        return float((model(x).argmax(dim=1) == y).float().mean().item())


def train_linear_probe(
    train_embeddings: Any,
    train_labels: Any,
    validation_embeddings: Any,
    validation_labels: Any,
    *,
    class_count: int | None = None,
    encoder: nn.Module | None = None,
    seed: int = 0,
    learning_rate: float = 1.0e-2,
    weight_decays: Sequence[float] = (0.0, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3),
    max_epochs: int = 200,
    patience: int = 20,
    batch_size: int | None = None,
    device: str | torch.device = "cpu",
) -> ProbeResult:
    """Fit the protocol's common decoder while proving the encoder stayed frozen."""

    train_x, train_y = _arrays(train_embeddings, train_labels)
    validation_x, validation_y = _arrays(validation_embeddings, validation_labels)
    if train_x.shape[1] != validation_x.shape[1]:
        raise ValueError("train and validation embedding dimensions differ")
    inferred_classes = int(max(train_y.max(), validation_y.max()).item()) + 1
    classes = inferred_classes if class_count is None else int(class_count)
    if classes < inferred_classes:
        raise ValueError("class_count excludes an observed label")
    if max_epochs < 1 or patience < 1 or learning_rate <= 0:
        raise ValueError("invalid probe optimization settings")
    decays = tuple(float(value) for value in weight_decays)
    if not decays or any(value < 0 for value in decays):
        raise ValueError("weight_decays must be non-empty and non-negative")

    original_modes: dict[nn.Module, bool] = {}
    before: str | None = None
    if encoder is not None:
        before = module_checksum(encoder)
        for module in encoder.modules():
            original_modes[module] = module.training
        encoder.requires_grad_(False)
        encoder.eval()

    target_device = torch.device(device)
    train_x, train_y = train_x.to(target_device), train_y.to(target_device)
    validation_x, validation_y = validation_x.to(target_device), validation_y.to(target_device)
    actual_batch = min(int(batch_size or 1024), train_x.shape[0])
    trials: list[ProbeTrial] = []

    try:
        for decay in decays:
            torch.manual_seed(int(seed))
            model = nn.Linear(train_x.shape[1], classes, bias=True).to(target_device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=decay)
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            best_accuracy = -1.0
            best_epoch = 0
            best_state: dict[str, Tensor] | None = None
            stale = 0
            for epoch in range(1, int(max_epochs) + 1):
                model.train()
                order = torch.randperm(train_x.shape[0], generator=generator)
                for start in range(0, train_x.shape[0], actual_batch):
                    indices = order[start : start + actual_batch].to(target_device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.cross_entropy(model(train_x[indices]), train_y[indices])
                    loss.backward()
                    optimizer.step()
                model.eval()
                accuracy = _accuracy(model, validation_x, validation_y)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_epoch = epoch
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                if stale >= patience:
                    break
            assert best_state is not None
            trials.append(ProbeTrial(decay, best_epoch, best_accuracy, best_state))
    finally:
        if encoder is not None:
            after = module_checksum(encoder)
            if after != before:
                raise RuntimeError("encoder parameters or buffers changed during probe training")
            for module, was_training in original_modes.items():
                module.train(was_training)

    # Accuracy first, then stronger regularization, then earlier epoch.
    selected = min(trials, key=lambda trial: (-trial.validation_accuracy, -trial.weight_decay, trial.best_epoch))
    final_model = nn.Linear(train_x.shape[1], classes, bias=True)
    final_model.load_state_dict(selected.state_dict)
    final_model.eval()
    return ProbeResult(
        model=final_model,
        selected_weight_decay=selected.weight_decay,
        selected_epoch=selected.best_epoch,
        validation_accuracy=selected.validation_accuracy,
        trials=tuple(
            {
                "weight_decay": trial.weight_decay,
                "best_epoch": trial.best_epoch,
                "validation_accuracy": trial.validation_accuracy,
            }
            for trial in trials
        ),
        encoder_checksum=before,
    )


__all__ = ["ProbeResult", "ProbeTrial", "module_checksum", "train_linear_probe"]
