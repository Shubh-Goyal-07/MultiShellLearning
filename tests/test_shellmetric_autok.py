from __future__ import annotations

import numpy as np

import multishell.confusion as confusion_module
from multishell.confusion import (
    build_confusion_artifact,
    load_confusion_artifact,
    save_confusion_artifact,
)
from multishell.shellmetric.autok import select_shell_count
from multishell.shellmetric.plan import smax


def _structured_oof(class_count: int = 9) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(17)
    labels = np.repeat(np.arange(class_count), 12)
    probabilities = np.empty((labels.size, class_count), dtype=np.float64)
    for row, label in enumerate(labels):
        correct = 0.96 - 0.075 * label
        probabilities[row] = (1.0 - correct) / (class_count - 1)
        probabilities[row, label] = correct
        probabilities[row] += rng.normal(0.0, 0.004, class_count)
        probabilities[row] = np.maximum(probabilities[row], 0.0)
        probabilities[row] /= probabilities[row].sum()
    return labels, probabilities


def test_confusion_artifact_is_exact_raw_q_w_w_hat_h(tmp_path) -> None:
    labels = np.repeat(np.arange(3), 2)
    probabilities = np.asarray(
        [
            [0.8, 0.2, 0.0],
            [0.6, 0.4, 0.0],
            [0.1, 0.9, 0.0],
            [0.3, 0.7, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    artifact = build_confusion_artifact(
        labels,
        probabilities,
        sample_ids=[f"sample-{index}" for index in range(labels.size)],
        fold_ids=[0, 1, 0, 1, 0, 1],
        pilot_seeds=[0, 1, 2],
        split_hash="split",
    )

    expected_Q = np.asarray(
        [[0.7, 0.3, 0.0], [0.2, 0.8, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    expected_W = np.asarray(
        [[0.0, 0.25, 0.0], [0.25, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(artifact.Q, expected_Q)
    np.testing.assert_allclose(artifact.W, expected_W)
    np.testing.assert_allclose(artifact.h, [0.25, 0.25, 0.0])
    np.testing.assert_allclose(artifact.W_hat, expected_W / (0.25 + 1.0e-12))
    assert artifact.W[0, 2] == 0.0
    assert artifact.W_hat[0, 2] == 0.0

    save_confusion_artifact(artifact, tmp_path)
    with np.load(tmp_path / "confusion.npz", allow_pickle=False) as payload:
        assert set(payload.files) == {
            "class_ids",
            "Q",
            "W",
            "W_hat",
            "h",
            "sample_ids",
            "fold_ids",
            "pilot_seeds",
        }
    restored = load_confusion_artifact(tmp_path)
    assert restored.artifact_hash == artifact.artifact_hash
    np.testing.assert_array_equal(restored.W, artifact.W)
    np.testing.assert_array_equal(restored.sample_ids, artifact.sample_ids)


def test_removed_confusion_path_has_no_floor_or_calibration_api() -> None:
    assert not hasattr(confusion_module, "build_confusion_weights")
    assert not hasattr(confusion_module, "fit_temperature")
    assert not hasattr(confusion_module, "hard_directed_confusion")


def test_autok_is_deterministic_paired_and_evaluates_every_count() -> None:
    labels, probabilities = _structured_oof()
    artifact = build_confusion_artifact(
        labels,
        probabilities,
        pilot_seeds=[0, 1, 2],
        split_hash="training-only-split",
    )

    first = select_shell_count(labels, probabilities, artifact, bootstrap_repeats=32)
    second = select_shell_count(labels, probabilities, artifact, bootstrap_repeats=32)

    expected_candidates = tuple(range(1, smax(9) + 1))
    assert first.candidate_shell_counts == expected_candidates
    assert first.bootstrap_risks.shape == (32, len(expected_candidates))
    assert first.bootstrap_risks.dtype == np.float64
    np.testing.assert_array_equal(first.bootstrap_risks, second.bootstrap_risks)
    np.testing.assert_array_equal(first.mean_risks, second.mean_risks)
    assert first.bootstrap_seed == int(artifact.artifact_hash[:16], 16)
    assert first.chosen_shell_count == second.chosen_shell_count

    best_index = int(np.argmin(first.mean_risks))
    threshold = first.mean_risks[best_index] + first.standard_errors[best_index]
    expected = expected_candidates[int(np.flatnonzero(first.mean_risks <= threshold)[0])]
    assert first.best_shell_count == expected_candidates[best_index]
    assert first.chosen_shell_count == expected
    assert first.plan.shell_count == expected
    assert first.plan.confusion_hash == artifact.artifact_hash


def test_autok_returns_one_shell_for_flat_full_difficulty() -> None:
    class_count = 4
    labels = np.repeat(np.arange(class_count), 5)
    probabilities = np.full((labels.size, class_count), 0.1, dtype=np.float64)
    probabilities[np.arange(labels.size), labels] = 0.7
    artifact = build_confusion_artifact(labels, probabilities, split_hash="flat")

    result = select_shell_count(labels, probabilities, artifact, bootstrap_repeats=8)

    assert result.chosen_shell_count == 1
    assert result.best_shell_count == 1
    np.testing.assert_array_equal(result.bootstrap_risks, 0.0)
    np.testing.assert_array_equal(result.normalized_full_difficulty, 0.0)
