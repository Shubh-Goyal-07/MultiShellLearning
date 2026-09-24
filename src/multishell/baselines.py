"""Canonical comparison objectives from the final ShellMetric protocol."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

CLASSIFICATION_BASELINES = ("cross_entropy", "label_smoothing", "arcface", "cosface")
METRIC_BASELINES = ("pair_contrastive", "supcon", "batch_hard_triplet", "multi_similarity")
BASELINES = (*CLASSIFICATION_BASELINES, *METRIC_BASELINES)

# Reporting names used in every table row.
DISPLAY_NAMES = {
    "cross_entropy": "CE",
    "label_smoothing": "CE+LS",
    "arcface": "ArcFace",
    "cosface": "CosFace",
    "pair_contrastive": "PairContrastive",
    "supcon": "SupCon",
    "batch_hard_triplet": "BatchHardTriplet",
    "multi_similarity": "MultiSimilarity",
}

# Externally pinned protocols: never executed by this repository's runner.
EXTERNAL_BASELINES = {
    "hyperspacex_matched": "HyperSpaceX-Matched",
    "hyperspacex_official": "HyperSpaceX-Official",
}


def _validate(embeddings: Tensor, targets: Tensor) -> Tensor:
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError("embeddings must have shape [B,d] with B >= 2")
    labels = targets.to(device=embeddings.device, dtype=torch.long)
    if labels.shape != (embeddings.shape[0],):
        raise ValueError("targets must have shape [B]")
    return labels


class LinearClassificationHead(nn.Linear):
    def __init__(self, embedding_dim: int, class_count: int) -> None:
        super().__init__(embedding_dim, class_count, bias=True)

    def loss(self, embeddings: Tensor, targets: Tensor, *, label_smoothing: float = 0.0) -> Tensor:
        return F.cross_entropy(self(embeddings), targets, label_smoothing=label_smoothing)


class _MarginHead(nn.Module):
    def __init__(self, embedding_dim: int, class_count: int, *, scale: float) -> None:
        super().__init__()
        if embedding_dim < 1 or class_count < 2 or scale <= 0:
            raise ValueError("invalid margin-head dimensions or scale")
        self.weight = nn.Parameter(torch.empty(class_count, embedding_dim))
        self.scale = float(scale)
        nn.init.xavier_uniform_(self.weight)

    def cosine(self, embeddings: Tensor) -> Tensor:
        return F.linear(F.normalize(embeddings, dim=1), F.normalize(self.weight, dim=1))


class ArcFaceHead(_MarginHead):
    def __init__(
        self, embedding_dim: int, class_count: int, *, scale: float = 30.0, margin: float = 0.50
    ) -> None:
        super().__init__(embedding_dim, class_count, scale=scale)
        self.margin = float(margin)

    def forward(self, embeddings: Tensor, targets: Tensor | None = None) -> Tensor:
        cosine = self.cosine(embeddings)
        if targets is None:
            return self.scale * cosine
        labels = _validate(embeddings, targets)
        rows = torch.arange(labels.numel(), device=labels.device)
        selected = cosine[rows, labels].clamp(-1.0 + 1.0e-7, 1.0 - 1.0e-7)
        logits = cosine.clone()
        logits[rows, labels] = torch.cos(torch.acos(selected) + self.margin)
        return self.scale * logits


class CosFaceHead(_MarginHead):
    def __init__(
        self, embedding_dim: int, class_count: int, *, scale: float = 30.0, margin: float = 0.35
    ) -> None:
        super().__init__(embedding_dim, class_count, scale=scale)
        self.margin = float(margin)

    def forward(self, embeddings: Tensor, targets: Tensor | None = None) -> Tensor:
        logits = self.cosine(embeddings)
        if targets is not None:
            labels = _validate(embeddings, targets)
            rows = torch.arange(labels.numel(), device=labels.device)
            logits = logits.clone()
            logits[rows, labels] -= self.margin
        return self.scale * logits


def pair_contrastive_loss(embeddings: Tensor, targets: Tensor, *, margin: float = 1.0) -> Tensor:
    labels = _validate(embeddings, targets)
    pairs = torch.triu_indices(labels.numel(), labels.numel(), offset=1, device=embeddings.device)
    distances = torch.linalg.vector_norm(embeddings[pairs[0]] - embeddings[pairs[1]], dim=1)
    same = labels[pairs[0]] == labels[pairs[1]]
    losses = torch.where(same, distances.square(), torch.relu(margin - distances).square())
    return losses.mean()


def supervised_contrastive_loss(
    views: Tensor | Sequence[Tensor], targets: Tensor, *, temperature: float = 0.07
) -> Tensor:
    """Standard SupCon loss for two or more augmented views per sample."""

    if isinstance(views, Sequence):
        values = torch.stack(tuple(views), dim=1)
    else:
        values = views
    if values.ndim != 3 or values.shape[1] < 2 or temperature <= 0:
        raise ValueError("views must have shape [B,V,d] with V >= 2")
    labels = targets.to(values.device, torch.long)
    if labels.shape != (values.shape[0],):
        raise ValueError("targets must have shape [B]")
    batch, view_count, dimension = values.shape
    features = F.normalize(values, dim=-1).reshape(batch * view_count, dimension)
    repeated = labels[:, None].expand(batch, view_count).reshape(-1)
    logits = features @ features.T / temperature
    identity = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    positive = repeated[:, None].eq(repeated[None, :]) & ~identity
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits).masked_fill(identity, 0.0)
    log_probability = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1.0e-12))
    counts = positive.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("every SupCon anchor requires a positive")
    return -(log_probability.masked_fill(~positive, 0.0).sum(dim=1) / counts).mean()


def batch_hard_triplet_loss(embeddings: Tensor, targets: Tensor, *, margin: float = 0.20) -> Tensor:
    labels = _validate(embeddings, targets)
    distances = torch.cdist(embeddings, embeddings)
    same = labels[:, None].eq(labels[None, :])
    same.fill_diagonal_(False)
    different = ~labels[:, None].eq(labels[None, :])
    valid = same.any(dim=1) & different.any(dim=1)
    if not bool(valid.any()):
        raise ValueError("batch-hard triplet requires positive and negative examples")
    hardest_positive = distances.masked_fill(~same, -torch.inf).max(dim=1).values
    hardest_negative = distances.masked_fill(~different, torch.inf).min(dim=1).values
    return torch.relu(hardest_positive[valid] - hardest_negative[valid] + margin).mean()


def multi_similarity_loss(
    embeddings: Tensor,
    targets: Tensor,
    *,
    alpha: float = 2.0,
    beta: float = 50.0,
    base: float = 0.5,
    miner_epsilon: float = 0.1,
) -> Tensor:
    labels = _validate(embeddings, targets)
    similarity = F.normalize(embeddings, dim=1) @ F.normalize(embeddings, dim=1).T
    losses: list[Tensor] = []
    for index in range(labels.numel()):
        positive = similarity[index][labels == labels[index]]
        positive = positive[
            torch.arange(labels.numel(), device=labels.device)[labels == labels[index]] != index
        ]
        negative = similarity[index][labels != labels[index]]
        if positive.numel() == 0 or negative.numel() == 0:
            continue
        mined_positive = positive[positive < negative.max() + miner_epsilon]
        mined_negative = negative[negative > positive.min() - miner_epsilon]
        if mined_positive.numel() == 0 or mined_negative.numel() == 0:
            continue
        positive_term = torch.log1p(torch.exp(-alpha * (mined_positive - base)).sum()) / alpha
        negative_term = torch.log1p(torch.exp(beta * (mined_negative - base)).sum()) / beta
        losses.append(positive_term + negative_term)
    if not losses:
        return embeddings.sum() * 0.0
    return torch.stack(losses).mean()


def build_native_head(
    method: str,
    embedding_dim: int,
    class_count: int,
    *,
    scale: float = 30.0,
    arcface_margin: float = 0.50,
    cosface_margin: float = 0.35,
) -> nn.Module | None:
    """Return the declared native head, or ``None`` for metric-only baselines."""

    normalized = method.lower().replace("-", "_")
    if normalized in {"cross_entropy", "label_smoothing"}:
        return LinearClassificationHead(embedding_dim, class_count)
    if normalized == "arcface":
        return ArcFaceHead(embedding_dim, class_count, scale=scale, margin=arcface_margin)
    if normalized == "cosface":
        return CosFaceHead(embedding_dim, class_count, scale=scale, margin=cosface_margin)
    if normalized in METRIC_BASELINES:
        return None
    raise ValueError(f"unsupported baseline: {method!r}")


__all__ = [
    "BASELINES",
    "CLASSIFICATION_BASELINES",
    "DISPLAY_NAMES",
    "EXTERNAL_BASELINES",
    "METRIC_BASELINES",
    "ArcFaceHead",
    "CosFaceHead",
    "LinearClassificationHead",
    "batch_hard_triplet_loss",
    "build_native_head",
    "multi_similarity_loss",
    "pair_contrastive_loss",
    "supervised_contrastive_loss",
]
