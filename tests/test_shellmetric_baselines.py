from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from multishell.baselines import BASELINES
from multishell.data import TensorSampleDataset
from multishell.train_baseline import (
    BaselineModel,
    BaselineTrainingConfig,
    load_baseline_checkpoint,
    train_baseline,
)


class TinyEncoder(nn.Module):
    embedding_dimension = 3

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(4, 3, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(inputs)


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(3)
    return torch.randn(12, 4), torch.arange(3).repeat_interleave(4)


def _datasets() -> tuple[TensorSampleDataset, TensorSampleDataset]:
    rng = np.random.default_rng(7)
    centers = np.eye(3, 4, dtype=np.float32) * 2.0

    def make(split: str, count: int) -> TensorSampleDataset:
        labels = np.repeat(np.arange(3), count)
        values = centers[labels] + rng.normal(0.0, 0.2, (labels.size, 4))
        return TensorSampleDataset(
            values.astype(np.float32),
            labels,
            dataset_name="baseline_fixture",
            split=split,
        )

    return make("train", 8), make("validation", 4)


def test_all_and_only_declared_baselines_have_finite_objectives() -> None:
    assert BASELINES == (
        "cross_entropy",
        "label_smoothing",
        "arcface",
        "cosface",
        "pair_contrastive",
        "supcon",
        "batch_hard_triplet",
        "multi_similarity",
    )
    inputs, labels = _batch()
    for method in BASELINES:
        model = BaselineModel(TinyEncoder(), 3, method=method, embedding_dimension=3)
        embeddings = model(inputs)
        config = BaselineTrainingConfig(
            method=method,
            epochs=1,
            batch_size=12,
            classes_per_batch=3,
            samples_per_class=4,
        )
        second = model(inputs + 0.01) if method == "supcon" else None
        loss = model.training_loss(embeddings, labels, config, second_view=second)
        assert loss.ndim == 0 and torch.isfinite(loss)
        loss.backward()
        if method in {"cross_entropy", "label_smoothing", "arcface", "cosface"}:
            assert model.native_head is not None
        else:
            assert model.native_head is None


def test_baseline_training_saves_separate_representation_and_native_selections(
    tmp_path: Path,
) -> None:
    train, validation = _datasets()
    model = BaselineModel(TinyEncoder(), 3, method="cross_entropy", embedding_dimension=3)
    config = BaselineTrainingConfig(
        method="cross_entropy",
        epochs=1,
        batch_size=12,
        learning_rate=1.0e-2,
    )
    result = train_baseline(
        model,
        train,
        validation,
        config=config,
        seed=2,
        output_dir=tmp_path,
    )
    assert result.representation_checkpoint is not None
    assert result.native_checkpoint is not None
    representation = load_baseline_checkpoint(result.representation_checkpoint)
    native = load_baseline_checkpoint(result.native_checkpoint)
    assert "native_head_state" not in representation
    assert "native_head_state" in native
    assert representation["selection_rule"] == "validation_raw_euclidean_1nn"
    assert native["selection_rule"] == "validation_native_top1"
