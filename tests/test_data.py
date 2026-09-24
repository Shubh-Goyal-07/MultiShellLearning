from __future__ import annotations

import gzip
import struct

import numpy as np
import pytest
import torch

from multishell.data import (
    MNISTIDXDataset,
    SealedDataset,
    TensorSampleDataset,
    load_dataset,
)


def _write_idx_fixture(root) -> None:
    raw = root / "MNIST" / "raw"
    raw.mkdir(parents=True)
    train_images = np.arange(4 * 2 * 3, dtype=np.uint8).reshape(4, 2, 3)
    train_labels = np.array([4, 7, 4, 7], dtype=np.uint8)
    test_images = train_images[:2]
    test_labels = train_labels[:2]
    fixtures = [
        ("train-images-idx3-ubyte.gz", struct.pack(">IIII", 2051, 4, 2, 3) + train_images.tobytes()),
        ("train-labels-idx1-ubyte.gz", struct.pack(">II", 2049, 4) + train_labels.tobytes()),
        ("t10k-images-idx3-ubyte.gz", struct.pack(">IIII", 2051, 2, 2, 3) + test_images.tobytes()),
        ("t10k-labels-idx1-ubyte.gz", struct.pack(">II", 2049, 2) + test_labels.tobytes()),
    ]
    for name, payload in fixtures:
        with gzip.open(raw / name, "wb") as handle:
            handle.write(payload)


def test_local_idx_dataset_maps_external_labels_and_stable_ids(tmp_path) -> None:
    _write_idx_fixture(tmp_path)
    dataset = MNISTIDXDataset(tmp_path, train=True)
    assert len(dataset) == 4
    image, label, sample_id = dataset[1]
    assert image.shape == (1, 2, 3)
    assert image.dtype == torch.float32
    assert label == 1
    assert sample_id == "mnist:train:00000001"
    np.testing.assert_array_equal(dataset.class_ids, [4, 7])


def test_tensor_dataset_has_immutable_identity_contract() -> None:
    dataset = TensorSampleDataset(torch.eye(3), [10, 20, 10], dataset_name="toy")
    assert dataset[0][1:] == (0, "toy:train:00000000")
    assert dataset[1][1] == 1
    assert len(set(dataset.sample_ids.tolist())) == len(dataset)


def test_sealed_dataset_raises_on_all_access() -> None:
    sealed = SealedDataset()
    with pytest.raises(RuntimeError):
        len(sealed)
    with pytest.raises(RuntimeError):
        sealed[0]


def test_configured_synthetic_splits_are_deterministic_and_disjoint() -> None:
    config = {
        "experiment": {"seed": 31},
        "data": {
            "dataset": "synthetic",
            "num_classes": 4,
            "samples_per_class": 20,
            "input_dimension": 5,
            "test_fraction": 0.25,
            "confusing_pairs": [[0, 1]],
        },
    }
    train_a = load_dataset(config, split="train")
    train_b = load_dataset(config, split="train")
    test = load_dataset(config, split="test")
    assert len(train_a) == 60
    assert len(test) == 20
    torch.testing.assert_close(train_a.inputs, train_b.inputs)
    assert set(train_a.sample_ids).isdisjoint(set(test.sample_ids))
