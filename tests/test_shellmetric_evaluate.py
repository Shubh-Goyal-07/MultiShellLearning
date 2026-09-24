from __future__ import annotations

import numpy as np
import pytest

from multishell.shellmetric.evaluate import (
    dense_exact_knn,
    effective_rank,
    evaluate_raw_embeddings,
    exact_knn,
    geometry_diagnostics,
    majority_vote,
    nearest_log_radius_shells,
    safe_directions,
    select_k,
    shell_then_cosine_knn,
    spoke_fractions,
)


def test_blockwise_exact_knn_matches_dense_and_orders_distance_ties_by_id() -> None:
    queries = np.array([[0.0], [1.5]], dtype=np.float64)
    gallery = np.array([[-1.0], [1.0], [-2.0], [2.0]], dtype=np.float64)
    labels = np.array([1, 0, 1, 0])
    sample_ids = np.array([20, 10, 40, 30])

    blockwise = exact_knn(
        queries,
        gallery,
        labels,
        sample_ids,
        k=4,
        query_block_size=1,
        gallery_block_size=1,
    )
    dense = dense_exact_knn(queries, gallery, labels, sample_ids, k=4)

    np.testing.assert_array_equal(blockwise.indices, dense.indices)
    np.testing.assert_array_equal(blockwise.sample_ids, dense.sample_ids)
    np.testing.assert_allclose(blockwise.distances, dense.distances, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(blockwise.sample_ids[0], [10, 20, 30, 40])


def test_majority_vote_uses_summed_distance_then_class_id() -> None:
    labels = np.array([[2, 1, 2, 1], [2, 1, 2, 1]])
    distances = np.array([[1.0, 1.0, 4.0, 2.0], [1.0, 1.0, 2.0, 2.0]])
    np.testing.assert_array_equal(majority_vote(labels, distances), [1, 1])


def test_k_selection_uses_validation_accuracy_and_prefers_smaller_k() -> None:
    assert select_k({11: 0.8, 5: 0.9, 3: 0.9, 1: 0.7}) == 3


def test_raw_evaluation_reports_knn_retrieval_and_cosine_diagnostic() -> None:
    gallery = np.array([[1.0, 0.0], [1.1, 0.0], [0.0, 1.0], [0.0, 1.1]])
    gallery_labels = np.array([0, 0, 1, 1])
    queries = np.array([[1.05, 0.0], [0.0, 1.05]])
    targets = np.array([0, 1])

    result = evaluate_raw_embeddings(
        queries,
        targets,
        gallery,
        gallery_labels,
        np.array([4, 3, 2, 1]),
        k_values=(1, 3),
        query_block_size=1,
        gallery_block_size=2,
    )

    assert result.metrics_by_k[1]["accuracy"] == 1.0
    assert result.retrieval == {
        "recall_at_1": 1.0,
        "recall_at_5": 1.0,
        "map_at_r": 1.0,
    }
    assert result.cosine_1nn_accuracy == 1.0


def test_safe_directions_and_log_radius_boundary_follow_spec() -> None:
    epsilon = 1e-8
    directions = safe_directions(
        np.array([[0.0, 0.0], [epsilon / 2, 0.0], [epsilon, 0.0], [3.0, 4.0]])
    )
    np.testing.assert_array_equal(directions[:2], np.zeros((2, 2)))
    np.testing.assert_allclose(directions[2], [1.0, 0.0])
    np.testing.assert_allclose(directions[3], [0.6, 0.8])

    boundary = np.sqrt(1.0 * 4.0 - epsilon)
    shells = nearest_log_radius_shells(
        np.array([[boundary, 0.0], [boundary + 1e-6, 0.0]]), [1.0, 4.0]
    )
    np.testing.assert_array_equal(shells, [1, 2])


def test_shell_classifier_uses_assignment_gallery_k_eff_and_decomposition() -> None:
    gallery = np.array(
        [
            [4.0, 0.0],  # assigned shell 1 despite its observed outer radius
            [2.0, 0.0],
            [0.0, 1.0],  # assigned shell 2 despite its observed inner radius
        ]
    )
    gallery_labels = np.array([0, 1, 2])
    queries = np.array([[1.0, 0.0], [0.0, 4.0], [1.0, 0.0]])
    targets = np.array([0, 2, 2])

    result = shell_then_cosine_knn(
        queries,
        targets,
        gallery,
        gallery_labels,
        np.array([10, 20, 30]),
        [1.0, 4.0],
        {0: 1, 1: 1, 2: 2},
        k=3,
        query_block_size=1,
        gallery_block_size=1,
    )

    np.testing.assert_array_equal(result.predicted_shells, [1, 2, 1])
    np.testing.assert_array_equal(result.true_shells, [1, 2, 2])
    np.testing.assert_array_equal(result.candidate_gallery_sizes, [2, 1, 2])
    np.testing.assert_array_equal(result.effective_k, [2, 1, 2])
    assert result.neighbor_labels[1, 0] == 2
    assert result.metrics["shell_selection_numerator"] == 2
    assert result.metrics["shell_selection_denominator"] == 3
    assert result.metrics["conditional_cosine_knn_denominator"] == 2
    assert result.metrics["oracle_shell_cosine_knn_accuracy"] == 1.0


def test_shell_neighbor_and_vote_ties_follow_ids_then_classes() -> None:
    result = shell_then_cosine_knn(
        np.array([[1.0, 0.0]]),
        np.array([0]),
        np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]),
        np.array([1, 0, 1, 0]),
        np.array([40, 10, 30, 20]),
        [1.0],
        {0: 1, 1: 1},
        k=4,
    )
    np.testing.assert_array_equal(result.neighbor_sample_ids[0], [10, 20, 30, 40])
    assert result.predictions[0] == 0


def test_one_shell_classifier_equals_global_cosine_knn() -> None:
    gallery = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    labels = np.array([0, 1, 2])
    ids = np.array([8, 4, 6])
    queries = np.array([[0.9, 0.1], [-0.8, 0.2]])
    targets = np.array([0, 2])
    native = shell_then_cosine_knn(
        queries, targets, gallery, labels, ids, [1.0], {0: 1, 1: 1, 2: 1}, k=3
    )
    global_cosine = exact_knn(queries, gallery, labels, ids, k=3, metric="cosine")

    np.testing.assert_array_equal(native.neighbor_indices, global_cosine.indices)
    np.testing.assert_allclose(native.neighbor_distances, global_cosine.distances, atol=1e-12)
    np.testing.assert_array_equal(native.predictions, global_cosine.predict(3))


def test_effective_rank_rank_one_isotropic_and_zero_are_finite() -> None:
    assert effective_rank([2.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert effective_rank(np.ones(5)) == pytest.approx(5.0)
    assert effective_rank(np.zeros(5)) == 0.0


def test_spoke_fraction_marks_zero_mean_class_undefined() -> None:
    embeddings = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    labels = np.array([0, 0, 1, 1])
    result = spoke_fractions(embeddings, labels)

    assert result["by_class"]["0"]["spoke_fraction"] == pytest.approx(1.0)
    assert result["by_class"]["1"]["spoke_fraction"] is None
    assert result["undefined_class_count"] == 1


def test_geometry_diagnostics_reports_ranks_shells_and_distance_distributions() -> None:
    embeddings = np.array([[1.0, 0.0], [1.2, 0.0], [0.0, 4.0], [0.0, 4.2]])
    labels = np.array([0, 0, 1, 1])
    result = geometry_diagnostics(
        embeddings,
        labels,
        radii=[1.0, 4.0],
        class_assignment={0: 1, 1: 2},
        max_distance_pairs=20,
        euclidean_1nn_predictions=labels,
        cosine_1nn_predictions=np.array([0, 0, 0, 0]),
    )

    assert np.isfinite(result["global_covariance"]["effective_rank"])
    assert np.isfinite(result["within_class_covariance"]["effective_rank"])
    assert result["shells"]["class_occupancy"] == [1, 1]
    assert result["shells"]["sample_occupancy"] == [2, 2]
    assert result["shells"]["nearest_shell_adherence_rate"] == 1.0
    assert result["pair_distances"]["same_class"]["population_count"] == 2
    assert result["euclidean_vs_cosine_1nn"]["accuracy_difference"] == 0.5


def test_streamed_raw_evaluation_matches_one_chunk_and_keeps_only_top_k() -> None:
    rng = np.random.default_rng(2)
    gallery = rng.normal(size=(60, 3))
    labels = np.repeat(np.arange(3), 20)
    ids = np.arange(60)[::-1]
    queries = rng.normal(size=(17, 3))
    targets = rng.integers(0, 3, 17)
    whole = evaluate_raw_embeddings(queries, targets, gallery, labels, ids)
    streamed = evaluate_raw_embeddings(
        queries,
        targets,
        gallery,
        labels,
        ids,
        query_chunk_size=4,
        query_block_size=3,
        gallery_block_size=7,
    )
    assert whole.retrieval == streamed.retrieval
    for k, predictions in whole.predictions_by_k.items():
        np.testing.assert_array_equal(predictions, streamed.predictions_by_k[k])
    np.testing.assert_array_equal(
        whole.euclidean_neighbors.indices, streamed.euclidean_neighbors.indices
    )
    assert whole.euclidean_neighbors.k == 11
    np.testing.assert_array_equal(
        whole.euclidean_neighbors.sample_ids, ids[whole.euclidean_neighbors.indices]
    )
