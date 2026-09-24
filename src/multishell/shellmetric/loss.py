"""Prototype-free supervised ShellMetric objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class ShellMetricLossOutput:
    total: Tensor
    positive: Tensor
    negative: Tensor
    shell: Tensor
    positive_pairs: int
    negative_pairs: int


def _zero(reference: Tensor) -> Tensor:
    return reference.sum() * 0.0


def shellmetric_loss(
    embeddings: Tensor,
    targets: Tensor,
    w_hat: Tensor,
    *,
    class_to_shell: Tensor | None,
    radii: Tensor | None,
    positive_margin: float = 0.50,
    negative_margin_base: float = 1.00,
    confusion_margin_delta: float = 0.50,
    negative_weight: float = 1.00,
    shell_weight: float = 1.00,
    shell_huber_beta: float = 0.10,
    epsilon: float = 1.0e-8,
) -> ShellMetricLossOutput:
    """Compute positive, negative, and radial terms over unordered pairs."""

    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError("embeddings must be a [B,d] tensor with B >= 2")
    labels = targets.to(device=embeddings.device, dtype=torch.long)
    if labels.shape != (embeddings.shape[0],):
        raise ValueError("targets must have shape [B]")
    if w_hat.ndim != 2 or w_hat.shape[0] != w_hat.shape[1]:
        raise ValueError("w_hat must be square")
    if labels.min() < 0 or labels.max() >= w_hat.shape[0]:
        raise ValueError("targets contain an invalid class index")
    if positive_margin < 0 or negative_margin_base < 0 or confusion_margin_delta < 0:
        raise ValueError("margins must be non-negative")
    if negative_weight < 0 or shell_weight < 0 or shell_huber_beta <= 0 or epsilon <= 0:
        raise ValueError("weights must be non-negative and beta/epsilon positive")

    pair_indices = torch.triu_indices(
        embeddings.shape[0], embeddings.shape[0], offset=1, device=embeddings.device
    )
    left, right = pair_indices[0], pair_indices[1]
    distances = torch.linalg.vector_norm(embeddings[left] - embeddings[right], dim=1)
    same = labels[left] == labels[right]
    different = ~same
    positive = (
        torch.relu(distances[same] - float(positive_margin)).square().mean()
        if bool(same.any())
        else _zero(embeddings)
    )
    if bool(different.any()):
        pair_weights = w_hat.to(device=embeddings.device, dtype=embeddings.dtype)[
            labels[left[different]], labels[right[different]]
        ]
        negative_margins = float(negative_margin_base) + float(confusion_margin_delta) * pair_weights
        negative = torch.relu(negative_margins - distances[different]).square().mean()
    else:
        negative = _zero(embeddings)

    if shell_weight == 0.0:
        shell = _zero(embeddings)
    else:
        if class_to_shell is None or radii is None:
            raise ValueError("class_to_shell and radii are required when shell_weight is nonzero")
        assignment = class_to_shell.to(device=embeddings.device, dtype=torch.long)
        radius_values = radii.to(device=embeddings.device, dtype=embeddings.dtype)
        if assignment.ndim != 1 or assignment.numel() != w_hat.shape[0]:
            raise ValueError("class_to_shell must have one entry per class")
        # Public plans use 1-based shell indices.  Accepting zero here would hide
        # a malformed plan, so reject it explicitly.
        if assignment.min() < 1 or assignment.max() > radius_values.numel():
            raise ValueError("class_to_shell contains an invalid 1-based shell index")
        target_radii = radius_values[assignment[labels] - 1]
        observed = torch.sqrt(embeddings.square().sum(dim=1) + float(epsilon))
        shell = F.smooth_l1_loss(
            torch.log(observed),
            torch.log(target_radii),
            beta=float(shell_huber_beta),
            reduction="mean",
        )

    total = positive + float(negative_weight) * negative + float(shell_weight) * shell
    return ShellMetricLossOutput(
        total=total,
        positive=positive,
        negative=negative,
        shell=shell,
        positive_pairs=int(same.sum().item()),
        negative_pairs=int(different.sum().item()),
    )


class ShellMetricLoss(nn.Module):
    """Stateful wrapper holding the fixed confusion and class-shell plan."""

    def __init__(
        self,
        w_hat: Any,
        class_to_shell: Any | None,
        **kwargs: float,
    ) -> None:
        super().__init__()
        matrix = torch.tensor(np.asarray(w_hat).copy(), dtype=torch.float32)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("w_hat must be square")
        self.register_buffer("w_hat", matrix)
        if class_to_shell is None:
            self.register_buffer("class_to_shell", None)
        else:
            self.register_buffer("class_to_shell", torch.as_tensor(class_to_shell, dtype=torch.long))
        self.options = dict(kwargs)

    def forward(self, embeddings: Tensor, targets: Tensor, radii: Tensor | None = None) -> ShellMetricLossOutput:
        return shellmetric_loss(
            embeddings,
            targets,
            self.w_hat,
            class_to_shell=self.class_to_shell,
            radii=radii,
            **self.options,
        )


__all__ = ["ShellMetricLoss", "ShellMetricLossOutput", "shellmetric_loss"]
