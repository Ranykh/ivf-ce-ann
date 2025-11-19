"""Common evaluation metrics for ANN experiments."""

from __future__ import annotations

import numpy as np


def recall_at_k(ground_truth: np.ndarray, predictions: np.ndarray, k: int) -> float:
    """Compute recall@k."""
    if ground_truth.ndim != 2 or predictions.ndim != 2:
        raise ValueError("ground_truth and predictions must be 2D arrays")
    if ground_truth.shape[0] != predictions.shape[0]:
        raise ValueError("ground_truth and predictions must have same rows")
    if k <= 0:
        raise ValueError("k must be positive")

    num_queries = ground_truth.shape[0]
    limit = min(k, predictions.shape[1], ground_truth.shape[1])
    hits = 0

    for gt, pred in zip(ground_truth, predictions):
        gt_set = set(gt[:limit])
        hits += sum(1 for candidate in pred[:limit] if candidate in gt_set)

    return hits / (num_queries * limit)


def queries_per_second(num_queries: int, elapsed_seconds: float) -> float:
    """Return throughput in queries per second given elapsed time."""
    if elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be positive")
    return num_queries / elapsed_seconds


def mean_distance(distances: np.ndarray) -> float:
    """Compute the arithmetic mean of a set of distances."""
    if distances.size == 0:
        return float("nan")
    return float(np.mean(distances))
