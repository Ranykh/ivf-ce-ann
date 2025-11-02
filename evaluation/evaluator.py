"""Evaluation orchestrator for ANN methods."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, Mapping, Sequence

import numpy as np

from evaluation.metrics import queries_per_second, recall_at_k


@dataclass
class EvaluationResult:
    recalls: Mapping[int, float]
    query_time_ms: float
    qps: float


class Evaluator:
    def __init__(self, ground_truth: np.ndarray) -> None:
        if ground_truth.ndim != 2:
            raise ValueError("ground_truth must be 2D")
        self.ground_truth = ground_truth.astype(np.int32, copy=False)

    def evaluate(
        self,
        searcher,
        queries: np.ndarray,
        k_values: Sequence[int],
    ) -> EvaluationResult:
        if queries.ndim != 2:
            raise ValueError("queries must be 2D")
        if not k_values:
            raise ValueError("k_values must be non-empty")

        max_k = max(k_values)
        predictions = np.full((queries.shape[0], max_k), -1, dtype=np.int32)

        start = perf_counter()
        for idx, query in enumerate(queries):
            ids, _ = searcher.search(query, max_k)
            fill = min(ids.shape[0], max_k)
            predictions[idx, :fill] = ids[:fill]
        elapsed = perf_counter() - start

        recalls = {k: recall_at_k(self.ground_truth, predictions, k) for k in k_values}
        query_time_ms = (elapsed / queries.shape[0]) * 1e3 if queries.size else 0.0
        qps = queries_per_second(queries.shape[0], elapsed) if elapsed > 0 else 0.0
        return EvaluationResult(recalls=recalls, query_time_ms=query_time_ms, qps=qps)
