"""Shared helpers for experiment scripts."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

from evaluation import EvaluationResult


def select_queries(
    queries: np.ndarray,
    ground_truth: np.ndarray,
    num_queries: int,
    *,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a subset of queries and aligned ground truth entries."""
    limit = min(num_queries, queries.shape[0], ground_truth.shape[0])
    if seed is None or limit == queries.shape[0]:
        return queries[:limit], ground_truth[:limit]

    rng = np.random.default_rng(seed)
    indices = rng.choice(queries.shape[0], size=limit, replace=False)
    return queries[indices], ground_truth[indices]


def format_results(name: str, result: EvaluationResult) -> str:
    """Format evaluation metrics into a compact human-readable string."""
    recall_parts = ", ".join(
        f"R@{k}={result.recalls[k]:.4f}" for k in sorted(result.recalls.keys())
    )
    latency_detail = ""
    if result.latency_p50_ms > 0 or result.latency_p95_ms > 0:
        latency_detail = (
            f", p50_ms={result.latency_p50_ms:.3f}, p95_ms={result.latency_p95_ms:.3f}"
        )
    return (
        f"{name}: {recall_parts}, "
        f"avg_query_time_ms={result.query_time_ms:.3f}, "
        f"qps={result.qps:.2f}"
        f"{latency_detail}"
    )
