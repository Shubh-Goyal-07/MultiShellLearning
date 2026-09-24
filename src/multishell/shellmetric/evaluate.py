"""Decoder-free evaluation for prototype-free ShellMetric embeddings.

All searches are exact.  Query and gallery blocking bounds working memory while
preserving the specified distance, sample-ID, and class-ID tie rules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

DEFAULT_K_VALUES = (1, 3, 5, 11)
RADIUS_EPSILON = 1.0e-8
SAFE_DIRECTION_EPSILON = 1.0e-8
_PAD_LABEL = np.iinfo(np.int64).min

DistanceMetric = Literal["squared_euclidean", "euclidean", "cosine"]


def _matrix(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty [N, d] matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _labels(value: Any, count: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.int64)
    if array.shape != (count,):
        raise ValueError(f"{name} must have shape [{count}]")
    return array


def _ks(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(sorted({int(value) for value in values}))
    if not result or result[0] < 1:
        raise ValueError("k values must be positive and non-empty")
    return result


def _sample_ids(value: Any, count: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(value)
    if ids.shape != (count,):
        raise ValueError(f"gallery_sample_ids must have shape [{count}]")
    python_ids = ids.tolist()
    try:
        if len(set(python_ids)) != count:
            raise ValueError("gallery_sample_ids must be unique")
        order = sorted(range(count), key=python_ids.__getitem__)
    except TypeError as error:
        raise ValueError("gallery_sample_ids must be hashable and mutually ordered") from error
    ranks = np.empty(count, dtype=np.int64)
    ranks[np.asarray(order, dtype=np.int64)] = np.arange(count, dtype=np.int64)
    return ids, ranks


def safe_directions(embeddings: Any, *, epsilon: float = SAFE_DIRECTION_EPSILON) -> np.ndarray:
    """Return unit directions, using the exact zero-vector rule below ``epsilon``."""

    values = _matrix(embeddings, "embeddings")
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    norms = np.linalg.norm(values, axis=1)
    directions = np.zeros_like(values)
    valid = norms >= epsilon
    directions[valid] = values[valid] / norms[valid, None]
    return directions


def _distance_block(queries: np.ndarray, gallery: np.ndarray, metric: str) -> np.ndarray:
    products = queries @ gallery.T
    if metric == "cosine":
        return np.clip(1.0 - products, 0.0, 2.0)
    query_energy = np.einsum("ij,ij->i", queries, queries)[:, None]
    gallery_energy = np.einsum("ij,ij->i", gallery, gallery)[None, :]
    return np.maximum(query_energy + gallery_energy - 2.0 * products, 0.0)


def _stable_topk(values: np.ndarray, k: int) -> np.ndarray:
    """Per-row positions of the ``k`` smallest values, ordered by (value, position)."""

    rows, width = values.shape
    if k == 1:  # argmin returns the first (lowest-position) minimum
        return np.argmin(values, axis=1)[:, None]
    if k >= width:
        return np.argsort(values, axis=1, kind="stable")
    threshold = np.partition(values, k - 1, axis=1)[:, k - 1 : k]
    below = values < threshold
    tied = values == threshold
    quota = k - below.sum(axis=1, keepdims=True)
    chosen = below | (tied & (np.cumsum(tied, axis=1) <= quota))
    positions = np.nonzero(chosen)[1].reshape(rows, k)
    order = np.argsort(np.take_along_axis(values, positions, axis=1), axis=1, kind="stable")
    return np.take_along_axis(positions, order, axis=1)


@dataclass(frozen=True)
class KNNResult:
    """Ranked exact neighbors. Distances are squared Euclidean or cosine."""

    indices: np.ndarray
    labels: np.ndarray
    distances: np.ndarray
    metric: str
    gallery_sample_ids: np.ndarray = field(repr=False)

    @property
    def k(self) -> int:
        return int(self.indices.shape[1])

    @property
    def sample_ids(self) -> np.ndarray:
        """Neighbor sample IDs, materialized only on request (they can be large)."""

        return self.gallery_sample_ids[self.indices]

    def predict(self, k: int | None = None) -> np.ndarray:
        chosen = self.k if k is None else int(k)
        if not 1 <= chosen <= self.k:
            raise ValueError(f"k must lie in [1, {self.k}]")
        return majority_vote(self.labels[:, :chosen], self.distances[:, :chosen])


def exact_knn(
    query_embeddings: Any,
    gallery_embeddings: Any,
    gallery_labels: Any,
    gallery_sample_ids: Any,
    *,
    k: int,
    metric: DistanceMetric = "squared_euclidean",
    query_block_size: int = 256,
    gallery_block_size: int = 4096,
    direction_epsilon: float = SAFE_DIRECTION_EPSILON,
) -> KNNResult:
    """Find exact neighbors without constructing the full query-gallery matrix."""

    queries = _matrix(query_embeddings, "query_embeddings")
    gallery = _matrix(gallery_embeddings, "gallery_embeddings")
    if queries.shape[1] != gallery.shape[1]:
        raise ValueError("query and gallery dimensions must match")
    labels = _labels(gallery_labels, gallery.shape[0], "gallery_labels")
    ids, id_ranks = _sample_ids(gallery_sample_ids, gallery.shape[0])
    k = int(k)
    if not 1 <= k <= gallery.shape[0]:
        raise ValueError(f"k must lie in [1, {gallery.shape[0]}]")
    if int(query_block_size) < 1 or int(gallery_block_size) < 1:
        raise ValueError("block sizes must be positive")
    canonical_metric = "squared_euclidean" if metric == "euclidean" else str(metric)
    if canonical_metric not in {"squared_euclidean", "cosine"}:
        raise ValueError("metric must be squared_euclidean, euclidean, or cosine")
    if canonical_metric == "cosine":
        queries = safe_directions(queries, epsilon=direction_epsilon)
        gallery = safe_directions(gallery, epsilon=direction_epsilon)

    # Visit the gallery in sample-ID order: a stable positional tie-break is
    # then exactly the required sample-ID tie-break, also across blocks.
    id_order = np.argsort(id_ranks)
    ordered_gallery = gallery[id_order]
    neighbor_indices = np.empty((queries.shape[0], k), dtype=np.int64)
    neighbor_distances = np.empty((queries.shape[0], k), dtype=np.float64)
    query_step = int(query_block_size)
    gallery_step = int(gallery_block_size)
    for query_start in range(0, queries.shape[0], query_step):
        query_block = queries[query_start : query_start + query_step]
        best_positions = np.empty((query_block.shape[0], 0), dtype=np.int64)
        best_distances = np.empty((query_block.shape[0], 0), dtype=np.float64)
        for gallery_start in range(0, gallery.shape[0], gallery_step):
            distances = _distance_block(
                query_block,
                ordered_gallery[gallery_start : gallery_start + gallery_step],
                canonical_metric,
            )
            positions = np.broadcast_to(
                np.arange(gallery_start, gallery_start + distances.shape[1]), distances.shape
            )
            candidate_distances = np.concatenate((best_distances, distances), axis=1)
            candidate_positions = np.concatenate((best_positions, positions), axis=1)
            top = _stable_topk(candidate_distances, min(k, candidate_distances.shape[1]))
            best_distances = np.take_along_axis(candidate_distances, top, axis=1)
            best_positions = np.take_along_axis(candidate_positions, top, axis=1)
        stop = query_start + query_block.shape[0]
        neighbor_indices[query_start:stop] = id_order[best_positions]
        neighbor_distances[query_start:stop] = best_distances

    return KNNResult(
        indices=neighbor_indices,
        labels=labels[neighbor_indices],
        distances=neighbor_distances,
        metric=canonical_metric,
        gallery_sample_ids=ids,
    )


blockwise_exact_knn = exact_knn


def dense_exact_knn(
    query_embeddings: Any,
    gallery_embeddings: Any,
    gallery_labels: Any,
    gallery_sample_ids: Any,
    *,
    k: int,
    metric: DistanceMetric = "squared_euclidean",
    direction_epsilon: float = SAFE_DIRECTION_EPSILON,
) -> KNNResult:
    """Small-fixture reference using one query block and one gallery block."""

    queries = _matrix(query_embeddings, "query_embeddings")
    gallery = _matrix(gallery_embeddings, "gallery_embeddings")
    return exact_knn(
        queries,
        gallery,
        gallery_labels,
        gallery_sample_ids,
        k=k,
        metric=metric,
        query_block_size=queries.shape[0],
        gallery_block_size=gallery.shape[0],
        direction_epsilon=direction_epsilon,
    )


def euclidean_1nn_accuracy(
    query_embeddings: Any,
    query_labels: Any,
    gallery_embeddings: Any,
    gallery_labels: Any,
    gallery_sample_ids: Any,
) -> float:
    """Primary representation endpoint used for every checkpoint selection."""

    neighbors = exact_knn(
        query_embeddings, gallery_embeddings, gallery_labels, gallery_sample_ids, k=1
    )
    truth = _labels(query_labels, neighbors.indices.shape[0], "query_labels")
    return float(np.mean(neighbors.labels[:, 0] == truth))


def majority_vote(neighbor_labels: Any, neighbor_distances: Any) -> np.ndarray:
    """Vote by count, then summed distance, then the smallest class ID."""

    labels = np.asarray(neighbor_labels, dtype=np.int64)
    distances = np.asarray(neighbor_distances, dtype=np.float64)
    if labels.ndim != 2 or labels.shape != distances.shape or labels.shape[1] == 0:
        raise ValueError("neighbor labels and distances must be equal non-empty matrices")
    if not np.isfinite(distances).all():
        raise ValueError("neighbor distances must be finite")
    predictions = np.empty(labels.shape[0], dtype=np.int64)
    for row in range(labels.shape[0]):
        classes, inverse, counts = np.unique(labels[row], return_inverse=True, return_counts=True)
        tied = np.flatnonzero(counts == counts.max())
        sums = np.zeros(classes.size, dtype=np.float64)
        np.add.at(sums, inverse, distances[row])
        best_sum = sums[tied].min()
        predictions[row] = classes[tied[sums[tied] == best_sum]].min()
    return predictions


def classification_metrics(targets: Any, predictions: Any) -> dict[str, Any]:
    """Return the required top-1, macro, balanced, and worst-class metrics."""

    truth = np.asarray(targets, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    if truth.ndim != 1 or truth.size == 0 or predicted.shape != truth.shape:
        raise ValueError("targets and predictions must be equal non-empty vectors")
    classes = np.unique(np.concatenate((truth, predicted)))
    recalls: list[float] = []
    f1_scores: list[float] = []
    per_class: dict[str, dict[str, float | int | None]] = {}
    for class_id in classes:
        true_positive = int(np.count_nonzero((truth == class_id) & (predicted == class_id)))
        support = int(np.count_nonzero(truth == class_id))
        predicted_count = int(np.count_nonzero(predicted == class_id))
        recall = None if support == 0 else true_positive / support
        precision = None if predicted_count == 0 else true_positive / predicted_count
        if recall is None:
            f1 = 0.0
        elif precision is None or precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        if recall is not None:
            recalls.append(recall)
        f1_scores.append(f1)
        per_class[str(int(class_id))] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    accuracy = float(np.mean(truth == predicted))
    return {
        "accuracy": accuracy,
        "top1_accuracy": accuracy,
        "macro_f1": float(np.mean(f1_scores)),
        "balanced_accuracy": float(np.mean(recalls)),
        "worst_class_accuracy": float(np.min(recalls)),
        "correct_count": int(np.count_nonzero(truth == predicted)),
        "sample_count": int(truth.size),
        "per_class": per_class,
    }


def select_k(validation_accuracy: Mapping[int, float]) -> int:
    """Select the best validation k, resolving an exact tie toward smaller k."""

    if not validation_accuracy:
        raise ValueError("validation_accuracy cannot be empty")
    scores = {int(k): float(value) for k, value in validation_accuracy.items()}
    if any(k < 1 for k in scores) or not np.isfinite(list(scores.values())).all():
        raise ValueError("k values must be positive and accuracies finite")
    best = max(scores.values())
    return min(k for k, value in scores.items() if value == best)


def _retrieval_terms(
    truth: np.ndarray, ranked: np.ndarray, counts: np.ndarray, recall_ks: Sequence[int]
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Per-query Recall@K hits and average precision at R."""

    hits = {
        k: np.any(ranked[:, : min(k, ranked.shape[1])] == truth[:, None], axis=1) for k in recall_ks
    }
    average_precision = np.empty(truth.size, dtype=np.float64)
    for row, relevant_count in enumerate(counts):
        relevant = ranked[row, :relevant_count] == truth[row]
        precision = np.cumsum(relevant) / np.arange(1, relevant_count + 1)
        average_precision[row] = float(np.sum(precision * relevant) / relevant_count)
    return hits, average_precision


def retrieval_metrics(
    targets: Any,
    ranked_gallery_labels: Any,
    *,
    relevant_counts: Any,
    recall_ks: Sequence[int] = (1, 5),
) -> dict[str, float]:
    """Compute retrieval Recall@K and mAP@R from exact ranked neighbors."""

    truth = np.asarray(targets, dtype=np.int64)
    ranked = np.asarray(ranked_gallery_labels, dtype=np.int64)
    counts = np.asarray(relevant_counts, dtype=np.int64)
    if truth.ndim != 1 or truth.size == 0 or ranked.ndim != 2:
        raise ValueError("targets must be a vector and ranked labels a matrix")
    if ranked.shape[0] != truth.size or counts.shape != truth.shape:
        raise ValueError("retrieval arrays must have the same query count")
    if np.any(counts < 1) or np.any(counts > ranked.shape[1]):
        raise ValueError("ranked neighbors must include R relevant-rank positions")
    hits, average_precision = _retrieval_terms(truth, ranked, counts, _ks(recall_ks))
    return {
        **{f"recall_at_{k}": float(value.mean()) for k, value in hits.items()},
        "map_at_r": float(average_precision.mean()),
    }


@dataclass(frozen=True)
class RawEvaluationResult:
    euclidean_neighbors: KNNResult
    predictions_by_k: Mapping[int, np.ndarray]
    metrics_by_k: Mapping[int, Mapping[str, Any]]
    retrieval: Mapping[str, float]
    cosine_1nn_neighbors: KNNResult
    cosine_1nn_predictions: np.ndarray
    cosine_1nn_accuracy: float


def evaluate_raw_embeddings(
    query_embeddings: Any,
    query_labels: Any,
    gallery_embeddings: Any,
    gallery_labels: Any,
    gallery_sample_ids: Any,
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    recall_ks: Sequence[int] = (1, 5),
    query_block_size: int = 256,
    gallery_block_size: int = 4096,
    query_chunk_size: int = 1024,
) -> RawEvaluationResult:
    """Evaluate raw Euclidean kNN/retrieval plus diagnostic cosine 1-NN.

    mAP@R needs each query's ranking to depth R (its class's gallery size), so
    queries are streamed in chunks; only the top ``max(k)`` neighbors are kept.
    """

    queries = _matrix(query_embeddings, "query_embeddings")
    gallery = _matrix(gallery_embeddings, "gallery_embeddings")
    truth = _labels(query_labels, queries.shape[0], "query_labels")
    gallery_truth = _labels(gallery_labels, gallery.shape[0], "gallery_labels")
    candidates, recall = _ks(k_values), _ks(recall_ks)
    if candidates[-1] > gallery.shape[0]:
        raise ValueError("a requested k exceeds the gallery size")
    gallery_classes, gallery_counts = np.unique(gallery_truth, return_counts=True)
    count_by_class = dict(zip(gallery_classes.tolist(), gallery_counts.tolist(), strict=True))
    try:
        relevant_counts = np.asarray([count_by_class[int(label)] for label in truth])
    except KeyError as error:
        raise ValueError(f"query class {error.args[0]} is absent from the gallery") from error
    depth = min(gallery.shape[0], max(candidates[-1], recall[-1], int(relevant_counts.max())))
    keep = candidates[-1]
    count = queries.shape[0]
    indices = np.empty((count, keep), dtype=np.int64)
    distances = np.empty((count, keep), dtype=np.float64)
    predictions = {k: np.empty(count, dtype=np.int64) for k in candidates}
    hits = {k: np.empty(count, dtype=bool) for k in recall}
    average_precision = np.empty(count, dtype=np.float64)
    options = {"query_block_size": query_block_size, "gallery_block_size": gallery_block_size}
    for start in range(0, count, max(1, int(query_chunk_size))):
        chunk = slice(start, min(start + int(query_chunk_size), count))
        ranked = exact_knn(
            queries[chunk], gallery, gallery_truth, gallery_sample_ids, k=depth, **options
        )
        for k in candidates:
            predictions[k][chunk] = ranked.predict(k)
        chunk_hits, chunk_ap = _retrieval_terms(
            truth[chunk], ranked.labels, relevant_counts[chunk], recall
        )
        for k in recall:
            hits[k][chunk] = chunk_hits[k]
        average_precision[chunk] = chunk_ap
        indices[chunk], distances[chunk] = ranked.indices[:, :keep], ranked.distances[:, :keep]
    euclidean = KNNResult(
        indices=indices,
        labels=gallery_truth[indices],
        distances=distances,
        metric="squared_euclidean",
        gallery_sample_ids=np.asarray(gallery_sample_ids),
    )
    cosine = exact_knn(
        queries, gallery, gallery_truth, gallery_sample_ids, k=1, metric="cosine", **options
    )
    cosine_predictions = cosine.predict(1)
    return RawEvaluationResult(
        euclidean_neighbors=euclidean,
        predictions_by_k=predictions,
        metrics_by_k={k: classification_metrics(truth, predictions[k]) for k in candidates},
        retrieval={
            **{f"recall_at_{k}": float(value.mean()) for k, value in hits.items()},
            "map_at_r": float(average_precision.mean()),
        },
        cosine_1nn_neighbors=cosine,
        cosine_1nn_predictions=cosine_predictions,
        cosine_1nn_accuracy=float(np.mean(cosine_predictions == truth)),
    )


def nearest_log_radius_shells(
    embeddings: Any, radii: Any, *, epsilon: float = RADIUS_EPSILON
) -> np.ndarray:
    """Return 1-based nearest shells; exact boundaries belong to the inner shell."""

    values = _matrix(embeddings, "embeddings")
    radii_array = np.asarray(radii, dtype=np.float64)
    if (
        radii_array.ndim != 1
        or radii_array.size == 0
        or not np.isfinite(radii_array).all()
        or np.any(radii_array <= 0)
        or np.any(np.diff(radii_array) <= 0)
    ):
        raise ValueError("radii must be a non-empty, positive, strictly increasing vector")
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    smoothed_squared_radius = np.einsum("ij,ij->i", values, values) + epsilon
    boundaries_squared = radii_array[:-1] * radii_array[1:]
    return (
        np.searchsorted(boundaries_squared, smoothed_squared_radius, side="left").astype(np.int64)
        + 1
    )


def _assignment_map(
    class_assignment: Mapping[int, int] | Any,
    class_ids: Any | None,
    shell_count: int,
) -> dict[int, int]:
    if isinstance(class_assignment, Mapping):
        assignment = {int(key): int(value) for key, value in class_assignment.items()}
    else:
        values = np.asarray(class_assignment, dtype=np.int64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("class_assignment must be a non-empty vector or mapping")
        ids = (
            np.arange(values.size, dtype=np.int64)
            if class_ids is None
            else np.asarray(class_ids, dtype=np.int64)
        )
        if ids.shape != values.shape or np.unique(ids).size != ids.size:
            raise ValueError("class_ids must be unique and match class_assignment")
        assignment = dict(zip(ids.tolist(), values.tolist(), strict=True))
    if not assignment or any(not 1 <= shell <= shell_count for shell in assignment.values()):
        raise ValueError(f"class assignments must lie in [1, {shell_count}]")
    return assignment


def _assigned_shells(labels: np.ndarray, assignment: Mapping[int, int]) -> np.ndarray:
    try:
        return np.asarray([assignment[int(label)] for label in labels], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"class {error.args[0]} is absent from class_assignment") from error


@dataclass(frozen=True)
class _ShellNeighbors:
    indices: np.ndarray
    distances: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray
    candidate_counts: np.ndarray


def _search_shells(
    query_directions: np.ndarray,
    query_shells: np.ndarray,
    gallery_directions: np.ndarray,
    gallery_shells: np.ndarray,
    gallery_labels: np.ndarray,
    gallery_ids: np.ndarray,
    *,
    shell_count: int,
    max_k: int,
    query_block_size: int,
    gallery_block_size: int,
) -> _ShellNeighbors:
    count = query_directions.shape[0]
    indices = np.full((count, max_k), -1, dtype=np.int64)
    distances = np.full((count, max_k), np.inf, dtype=np.float64)
    labels = np.full((count, max_k), _PAD_LABEL, dtype=np.int64)
    sample_ids = np.empty((count, max_k), dtype=object)
    sample_ids.fill(None)
    candidate_counts = np.empty(count, dtype=np.int64)
    for shell in range(1, shell_count + 1):
        gallery_indices = np.flatnonzero(gallery_shells == shell)
        if gallery_indices.size == 0:
            raise ValueError(f"shell {shell} has an empty gallery")
        query_indices = np.flatnonzero(query_shells == shell)
        if query_indices.size == 0:
            continue
        candidate_counts[query_indices] = gallery_indices.size
        effective_max = min(max_k, int(gallery_indices.size))
        local = exact_knn(
            query_directions[query_indices],
            gallery_directions[gallery_indices],
            gallery_labels[gallery_indices],
            gallery_ids[gallery_indices],
            k=effective_max,
            metric="cosine",
            query_block_size=query_block_size,
            gallery_block_size=gallery_block_size,
        )
        global_indices = gallery_indices[local.indices]
        indices[np.ix_(query_indices, np.arange(effective_max))] = global_indices
        distances[np.ix_(query_indices, np.arange(effective_max))] = local.distances
        labels[np.ix_(query_indices, np.arange(effective_max))] = local.labels
        sample_ids[np.ix_(query_indices, np.arange(effective_max))] = local.sample_ids
    return _ShellNeighbors(indices, distances, labels, sample_ids, candidate_counts)


def _variable_vote(neighbors: _ShellNeighbors, requested_k: int) -> tuple[np.ndarray, np.ndarray]:
    effective = np.minimum(int(requested_k), neighbors.candidate_counts)
    predictions = np.empty(effective.size, dtype=np.int64)
    for row, row_k in enumerate(effective):
        predictions[row] = majority_vote(
            neighbors.labels[row : row + 1, :row_k],
            neighbors.distances[row : row + 1, :row_k],
        )[0]
    return predictions, effective


@dataclass(frozen=True)
class ShellKNNResult:
    k: int
    predictions: np.ndarray
    predicted_shells: np.ndarray
    true_shells: np.ndarray
    candidate_gallery_sizes: np.ndarray
    effective_k: np.ndarray
    neighbor_indices: np.ndarray
    neighbor_sample_ids: np.ndarray
    neighbor_labels: np.ndarray
    neighbor_distances: np.ndarray
    oracle_predictions: np.ndarray
    oracle_effective_k: np.ndarray
    metrics: Mapping[str, Any]


def _conditional_metric(correct: np.ndarray, condition: np.ndarray) -> dict[str, Any]:
    denominator = int(np.count_nonzero(condition))
    numerator = int(np.count_nonzero(correct & condition))
    return {
        "value": None if denominator == 0 else numerator / denominator,
        "numerator": numerator,
        "denominator": denominator,
    }


def evaluate_shell_then_cosine_knn(
    query_embeddings: Any,
    query_labels: Any,
    gallery_embeddings: Any,
    gallery_labels: Any,
    gallery_sample_ids: Any,
    radii: Any,
    class_assignment: Mapping[int, int] | Any,
    *,
    class_ids: Any | None = None,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    radius_epsilon: float = RADIUS_EPSILON,
    direction_epsilon: float = SAFE_DIRECTION_EPSILON,
    query_block_size: int = 256,
    gallery_block_size: int = 4096,
) -> dict[int, ShellKNNResult]:
    """Evaluate all native k values with one deployable and one oracle search."""

    queries = _matrix(query_embeddings, "query_embeddings")
    gallery = _matrix(gallery_embeddings, "gallery_embeddings")
    if queries.shape[1] != gallery.shape[1]:
        raise ValueError("query and gallery dimensions must match")
    truth = _labels(query_labels, queries.shape[0], "query_labels")
    gallery_truth = _labels(gallery_labels, gallery.shape[0], "gallery_labels")
    gallery_ids, _ = _sample_ids(gallery_sample_ids, gallery.shape[0])
    radii_array = np.asarray(radii, dtype=np.float64)
    # Validation, including strict ordering, is centralized here.
    predicted_shells = nearest_log_radius_shells(queries, radii_array, epsilon=radius_epsilon)
    shell_count = int(radii_array.size)
    if shell_count == 1 and radii_array[0] != 1.0:
        raise ValueError("the one-shell classifier requires rho_1 = 1")
    assignment = _assignment_map(class_assignment, class_ids, shell_count)
    true_shells = _assigned_shells(truth, assignment)
    gallery_shells = _assigned_shells(gallery_truth, assignment)
    query_directions = safe_directions(queries, epsilon=direction_epsilon)
    gallery_directions = safe_directions(gallery, epsilon=direction_epsilon)
    candidates = _ks(k_values)
    deployable_neighbors = _search_shells(
        query_directions,
        predicted_shells,
        gallery_directions,
        gallery_shells,
        gallery_truth,
        gallery_ids,
        shell_count=shell_count,
        max_k=candidates[-1],
        query_block_size=query_block_size,
        gallery_block_size=gallery_block_size,
    )
    oracle_neighbors = _search_shells(
        query_directions,
        true_shells,
        gallery_directions,
        gallery_shells,
        gallery_truth,
        gallery_ids,
        shell_count=shell_count,
        max_k=candidates[-1],
        query_block_size=query_block_size,
        gallery_block_size=gallery_block_size,
    )
    shell_correct = predicted_shells == true_shells
    query_zero_count = int(np.count_nonzero(np.linalg.norm(queries, axis=1) < direction_epsilon))
    gallery_zero_count = int(np.count_nonzero(np.linalg.norm(gallery, axis=1) < direction_epsilon))
    results: dict[int, ShellKNNResult] = {}
    for k in candidates:
        predictions, effective_k = _variable_vote(deployable_neighbors, k)
        oracle_predictions, oracle_effective_k = _variable_vote(oracle_neighbors, k)
        metrics = classification_metrics(truth, predictions)
        shell_metric = _conditional_metric(shell_correct, np.ones_like(shell_correct))
        conditional = _conditional_metric(predictions == truth, shell_correct)
        oracle = _conditional_metric(oracle_predictions == truth, np.ones_like(shell_correct))
        metrics.update(
            {
                "k": k,
                "shell_selection_accuracy": shell_metric["value"],
                "shell_selection_numerator": shell_metric["numerator"],
                "shell_selection_denominator": shell_metric["denominator"],
                "conditional_cosine_knn_accuracy": conditional["value"],
                "conditional_cosine_knn_numerator": conditional["numerator"],
                "conditional_cosine_knn_denominator": conditional["denominator"],
                "oracle_shell_cosine_knn_accuracy": oracle["value"],
                "oracle_shell_cosine_knn_numerator": oracle["numerator"],
                "oracle_shell_cosine_knn_denominator": oracle["denominator"],
                "effective_k_min": int(effective_k.min()),
                "effective_k_max": int(effective_k.max()),
                "zero_norm_query_count": query_zero_count,
                "zero_norm_gallery_count": gallery_zero_count,
                "geometry_warning": bool(query_zero_count or gallery_zero_count),
            }
        )
        width = min(k, deployable_neighbors.indices.shape[1])
        results[k] = ShellKNNResult(
            k=k,
            predictions=predictions,
            predicted_shells=predicted_shells,
            true_shells=true_shells,
            candidate_gallery_sizes=deployable_neighbors.candidate_counts,
            effective_k=effective_k,
            neighbor_indices=deployable_neighbors.indices[:, :width],
            neighbor_sample_ids=deployable_neighbors.sample_ids[:, :width],
            neighbor_labels=deployable_neighbors.labels[:, :width],
            neighbor_distances=deployable_neighbors.distances[:, :width],
            oracle_predictions=oracle_predictions,
            oracle_effective_k=oracle_effective_k,
            metrics=metrics,
        )
    return results


def shell_then_cosine_knn(
    query_embeddings: Any,
    query_labels: Any,
    gallery_embeddings: Any,
    gallery_labels: Any,
    gallery_sample_ids: Any,
    radii: Any,
    class_assignment: Mapping[int, int] | Any,
    *,
    k: int = 1,
    class_ids: Any | None = None,
    radius_epsilon: float = RADIUS_EPSILON,
    direction_epsilon: float = SAFE_DIRECTION_EPSILON,
    query_block_size: int = 256,
    gallery_block_size: int = 4096,
) -> ShellKNNResult:
    """Run the parameter-free ShellMetric-native classifier for one k."""

    return evaluate_shell_then_cosine_knn(
        query_embeddings,
        query_labels,
        gallery_embeddings,
        gallery_labels,
        gallery_sample_ids,
        radii,
        class_assignment,
        class_ids=class_ids,
        k_values=(k,),
        radius_epsilon=radius_epsilon,
        direction_epsilon=direction_epsilon,
        query_block_size=query_block_size,
        gallery_block_size=gallery_block_size,
    )[int(k)]


def effective_rank(eigenvalues: Any, *, epsilon: float = np.finfo(np.float64).eps) -> float:
    """Return covariance participation ratio, finite and zero at zero covariance."""

    values = np.asarray(eigenvalues, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("eigenvalues must be a finite vector")
    tolerance = max(float(epsilon), np.finfo(np.float64).eps)
    if np.any(values < -tolerance):
        raise ValueError("eigenvalues cannot be negative")
    values = np.maximum(values, 0.0)
    denominator = float(values @ values)
    if denominator == 0.0:
        return 0.0
    return float(values.sum() ** 2 / (denominator + float(epsilon)))


def covariance_diagnostics(embeddings: Any) -> dict[str, Any]:
    """Return the centered covariance spectrum and rotationally invariant rank."""

    values = _matrix(embeddings, "embeddings")
    centered = values - values.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / values.shape[0]
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)[::-1]
    rank = effective_rank(eigenvalues)
    return {
        "eigenvalues": eigenvalues.tolist(),
        "effective_rank": rank,
        "effective_rank_fraction": rank / values.shape[1],
    }


def spoke_fractions(
    embeddings: Any,
    labels: Any,
    *,
    class_ids: Any | None = None,
    epsilon: float = RADIUS_EPSILON,
    mean_norm_threshold: float = SAFE_DIRECTION_EPSILON,
) -> dict[str, Any]:
    """Compute per-class radial/tangential residual variance and spoke fraction."""

    values = _matrix(embeddings, "embeddings")
    targets = _labels(labels, values.shape[0], "labels")
    classes = np.unique(targets) if class_ids is None else np.asarray(class_ids, dtype=np.int64)
    if classes.ndim != 1 or classes.size == 0 or np.unique(classes).size != classes.size:
        raise ValueError("class_ids must be a unique non-empty vector")
    if not np.isin(targets, classes).all():
        raise ValueError("labels contain a class absent from class_ids")
    by_class: dict[str, dict[str, float | int | None]] = {}
    defined: list[float] = []
    undefined = 0
    for class_id in classes:
        members = values[targets == class_id]
        if members.shape[0] == 0:
            raise ValueError(f"class {int(class_id)} has no samples")
        mean = members.mean(axis=0)
        mean_norm = float(np.linalg.norm(mean))
        if mean_norm < mean_norm_threshold:
            radial = tangential = fraction = None
            undefined += 1
        else:
            direction = mean / mean_norm
            residuals = members - mean
            radial_components = residuals @ direction
            radial = float(np.mean(np.square(radial_components)))
            tangential_residuals = residuals - radial_components[:, None] * direction
            tangential = float(
                np.mean(np.einsum("ij,ij->i", tangential_residuals, tangential_residuals))
            )
            fraction = radial / (radial + tangential + float(epsilon))
            defined.append(fraction)
        by_class[str(int(class_id))] = {
            "sample_count": int(members.shape[0]),
            "mean_norm": mean_norm,
            "radial_variance": radial,
            "tangential_variance": tangential,
            "spoke_fraction": fraction,
        }
    return {
        "by_class": by_class,
        "undefined_class_count": undefined,
        "defined_class_count": len(defined),
        "macro_mean_spoke_fraction": None if not defined else float(np.mean(defined)),
    }


def _distribution(values: np.ndarray, *, population_count: int | None = None) -> dict[str, Any]:
    if values.size == 0:
        return {
            "population_count": int(population_count or 0),
            "sample_count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    quantiles = np.quantile(values, (0.05, 0.25, 0.5, 0.75, 0.95))
    return {
        "population_count": int(values.size if population_count is None else population_count),
        "sample_count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "max": float(values.max()),
    }


def pair_distance_diagnostics(
    embeddings: Any,
    labels: Any,
    *,
    max_pairs: int = 100_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Summarize same/different-class Euclidean distances with bounded storage."""

    values = _matrix(embeddings, "embeddings")
    targets = _labels(labels, values.shape[0], "labels")
    max_pairs = int(max_pairs)
    if max_pairs < 1:
        raise ValueError("max_pairs must be positive")
    total_pairs = values.shape[0] * (values.shape[0] - 1) // 2
    if total_pairs == 0:
        empty = np.empty(0, dtype=np.float64)
        return {"same_class": _distribution(empty), "different_class": _distribution(empty)}
    if total_pairs <= max_pairs:
        left, right = np.triu_indices(values.shape[0], 1)
    else:
        ordinals = np.sort(
            np.random.default_rng(int(seed)).choice(total_pairs, size=max_pairs, replace=False)
        )
        rows = np.arange(values.shape[0] - 1, dtype=np.int64)
        starts = rows * (2 * values.shape[0] - rows - 1) // 2
        left = np.searchsorted(starts, ordinals, side="right") - 1
        right = left + 1 + (ordinals - starts[left])
    distances = np.linalg.norm(values[left] - values[right], axis=1)
    same = targets[left] == targets[right]
    _, class_counts = np.unique(targets, return_counts=True)
    same_population = int(np.sum(class_counts * (class_counts - 1) // 2))
    return {
        "same_class": _distribution(distances[same], population_count=same_population),
        "different_class": _distribution(
            distances[~same], population_count=total_pairs - same_population
        ),
        "sampling_seed": int(seed),
        "sampled": total_pairs > max_pairs,
    }


def geometry_diagnostics(
    embeddings: Any,
    labels: Any,
    *,
    radii: Any | None = None,
    class_assignment: Mapping[int, int] | Any | None = None,
    class_ids: Any | None = None,
    radius_epsilon: float = RADIUS_EPSILON,
    max_distance_pairs: int = 100_000,
    euclidean_1nn_predictions: Any | None = None,
    cosine_1nn_predictions: Any | None = None,
) -> dict[str, Any]:
    """Compute the mandatory raw-space geometry diagnostics available per split."""

    values = _matrix(embeddings, "embeddings")
    targets = _labels(labels, values.shape[0], "labels")
    norms = np.linalg.norm(values, axis=1)
    global_covariance = covariance_diagnostics(values)
    residuals = np.empty_like(values)
    for class_id in np.unique(targets):
        mask = targets == class_id
        residuals[mask] = values[mask] - values[mask].mean(axis=0, keepdims=True)
    within_covariance = covariance_diagnostics(residuals)
    result: dict[str, Any] = {
        "embedding_norm": _distribution(norms),
        "global_covariance": global_covariance,
        "within_class_covariance": within_covariance,
        "spoke": spoke_fractions(values, targets, class_ids=class_ids),
        "pair_distances": pair_distance_diagnostics(values, targets, max_pairs=max_distance_pairs),
    }
    if (radii is None) != (class_assignment is None):
        raise ValueError("radii and class_assignment must be provided together")
    if radii is not None:
        radii_array = np.asarray(radii, dtype=np.float64)
        nearest = nearest_log_radius_shells(values, radii_array, epsilon=radius_epsilon)
        assignment = _assignment_map(class_assignment, class_ids, radii_array.size)
        assigned = _assigned_shells(targets, assignment)
        smoothed_radii = np.sqrt(np.square(norms) + float(radius_epsilon))
        radial_error = np.abs(np.log(smoothed_radii) - np.log(radii_array[assigned - 1]))
        class_occupancy = [
            int(sum(shell == index for shell in assignment.values()))
            for index in range(1, radii_array.size + 1)
        ]
        sample_occupancy = [
            int(np.count_nonzero(assigned == index)) for index in range(1, radii_array.size + 1)
        ]
        result["shells"] = {
            "radii": radii_array.tolist(),
            "adjacent_radius_gaps": np.diff(radii_array).tolist(),
            "class_occupancy": class_occupancy,
            "sample_occupancy": sample_occupancy,
            "absolute_log_radius_error": _distribution(radial_error),
            "nearest_shell_adherence_rate": float(np.mean(nearest == assigned)),
        }
    if (euclidean_1nn_predictions is None) != (cosine_1nn_predictions is None):
        raise ValueError("both Euclidean and cosine predictions are required")
    if euclidean_1nn_predictions is not None:
        euclidean = _labels(euclidean_1nn_predictions, values.shape[0], "euclidean_1nn_predictions")
        cosine = _labels(cosine_1nn_predictions, values.shape[0], "cosine_1nn_predictions")
        result["euclidean_vs_cosine_1nn"] = {
            "prediction_disagreement_rate": float(np.mean(euclidean != cosine)),
            "euclidean_accuracy": float(np.mean(euclidean == targets)),
            "cosine_accuracy": float(np.mean(cosine == targets)),
            "accuracy_difference": float(
                np.mean(euclidean == targets) - np.mean(cosine == targets)
            ),
        }
    return result


__all__ = [
    "DEFAULT_K_VALUES",
    "KNNResult",
    "RADIUS_EPSILON",
    "RawEvaluationResult",
    "SAFE_DIRECTION_EPSILON",
    "ShellKNNResult",
    "blockwise_exact_knn",
    "classification_metrics",
    "covariance_diagnostics",
    "dense_exact_knn",
    "effective_rank",
    "euclidean_1nn_accuracy",
    "evaluate_raw_embeddings",
    "evaluate_shell_then_cosine_knn",
    "exact_knn",
    "geometry_diagnostics",
    "majority_vote",
    "nearest_log_radius_shells",
    "pair_distance_diagnostics",
    "retrieval_metrics",
    "safe_directions",
    "select_k",
    "shell_then_cosine_knn",
    "spoke_fractions",
]
