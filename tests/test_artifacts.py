from __future__ import annotations

import numpy as np
import pytest

from multishell.artifacts import (
    ArtifactMismatchError,
    hash_array,
    load_array_artifact,
    save_array_artifact,
    stable_hash,
    validate_hash,
)


def test_stable_hash_ignores_mapping_insertion_order() -> None:
    assert stable_hash({"a": 1, "b": [2, 3]}) == stable_hash({"b": [2, 3], "a": 1})


def test_array_hash_includes_shape_and_dtype() -> None:
    base = np.arange(6, dtype=np.int64)
    assert hash_array(base) != hash_array(base.reshape(2, 3))
    assert hash_array(base) != hash_array(base.astype(np.float64))


def test_array_artifact_round_trip_and_hash_validation(tmp_path) -> None:
    arrays = {"x": np.arange(5), "matrix": np.eye(3)}
    paths = save_array_artifact(
        tmp_path / "artifact",
        arrays,
        {"artifact_type": "test", "schema_version": 1},
    )
    loaded_arrays, metadata = load_array_artifact(paths.arrays.parent)
    np.testing.assert_array_equal(loaded_arrays["x"], arrays["x"])
    assert metadata["artifact_type"] == "test"


def test_tampered_array_artifact_is_rejected(tmp_path) -> None:
    root = tmp_path / "artifact"
    paths = save_array_artifact(root, {"x": np.arange(4)}, {"artifact_type": "test"})
    np.savez_compressed(paths.arrays, x=np.arange(5))
    with pytest.raises(ArtifactMismatchError):
        load_array_artifact(root)


def test_explicit_hash_mismatch_raises() -> None:
    with pytest.raises(ArtifactMismatchError):
        validate_hash("codebook", "expected", "actual")
