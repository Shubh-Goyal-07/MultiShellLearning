from __future__ import annotations

import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from multishell.train_pilot import (
    PilotTrainingConfig,
    canonical_json_hash,
    file_sha256,
    train_pilot_crossfit,
)


def test_pilot_writes_epoch_progress_without_changing_science_config(
    tmp_path, capsys
) -> None:
    inputs = torch.tensor(
        [
            [-2.0, -1.0],
            [-1.5, -0.5],
            [-1.0, -2.0],
            [-0.5, -1.5],
            [2.0, 1.0],
            [1.5, 0.5],
            [1.0, 2.0],
            [0.5, 1.5],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    folds = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    config = PilotTrainingConfig(
        epochs=2,
        batch_size=4,
        learning_rate=0.05,
        scheduler="none",
        seeds=(17,),
        device="cpu",
    )
    callback_events: list[dict[str, object]] = []

    result = train_pilot_crossfit(
        TensorDataset(inputs, labels),
        folds,
        lambda num_classes: nn.Linear(2, num_classes),
        num_classes=2,
        labels=labels.numpy(),
        sample_ids=np.asarray([f"sample-{index}" for index in range(len(labels))]),
        class_ids=np.asarray([0, 1]),
        config=config,
        split_hash="test-split",
        output_dir=tmp_path,
        show_progress=False,
        progress_callback=lambda event: callback_events.append(dict(event)),
    )
    reference = train_pilot_crossfit(
        TensorDataset(inputs, labels),
        folds,
        lambda num_classes: nn.Linear(2, num_classes),
        num_classes=2,
        labels=labels.numpy(),
        sample_ids=np.asarray([f"sample-{index}" for index in range(len(labels))]),
        class_ids=np.asarray([0, 1]),
        config=config,
        split_hash="test-split",
        show_progress=False,
    )
    reference_dir = tmp_path / "reference"
    reference.save(reference_dir)

    assert capsys.readouterr().err == ""
    np.testing.assert_array_equal(result.logits, reference.logits)
    np.testing.assert_array_equal(result.probabilities, reference.probabilities)
    assert file_sha256(tmp_path / "oof_predictions.npz") == file_sha256(
        reference_dir / "oof_predictions.npz"
    )
    assert file_sha256(tmp_path / "oof_metadata.json") == file_sha256(
        reference_dir / "oof_metadata.json"
    )
    assert result.metadata["config_hash"] == canonical_json_hash(config)
    assert "show_progress" not in result.metadata["config"]

    progress_path = tmp_path / "progress.jsonl"
    records = [json.loads(line) for line in progress_path.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "run_start",
        "seed_start",
        "fold_start",
        "epoch_end",
        "epoch_end",
        "fold_end",
        "fold_start",
        "epoch_end",
        "epoch_end",
        "fold_end",
        "seed_end",
        "run_end",
    ]
    epoch_records = [record for record in records if record["event"] == "epoch_end"]
    assert len(epoch_records) == 4
    assert all(record["root_seed"] == 17 for record in epoch_records)
    assert all(record["fold_number"] in (1, 2) for record in epoch_records)
    assert all(record["epoch"] in (1, 2) for record in epoch_records)
    assert all(0.0 <= record["accuracy"] <= 1.0 for record in epoch_records)
    assert all(record["loss"] >= 0.0 for record in epoch_records)
    assert [event["event"] for event in callback_events] == [
        record["event"] for record in records
    ]
