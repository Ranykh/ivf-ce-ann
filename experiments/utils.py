"""Shared helpers for experiment scripts."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

from evaluation import EvaluationResult


def select_queries(
    queries: np.ndarray,
    ground_truth: np.ndarray,
    num_queries: int,
) -> Tuple[np.ndarray, np.ndarray]:
    limit = min(num_queries, queries.shape[0], ground_truth.shape[0])
    return queries[:limit], ground_truth[:limit]


def format_results(name: str, result: EvaluationResult) -> str:
    recall_parts = ", ".join(
        f"R@{k}={result.recalls[k]:.4f}" for k in sorted(result.recalls.keys())
    )
    return (
        f"{name}: {recall_parts}, "
        f"avg_query_time_ms={result.query_time_ms:.3f}, "
        f"qps={result.qps:.2f}"
    )
