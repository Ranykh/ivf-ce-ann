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
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0


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
        per_query_latencies: list[float] = []
        for idx, query in enumerate(queries):
            q_start = perf_counter()
            ids, _ = searcher.search(query, max_k)
            per_query_latencies.append((perf_counter() - q_start) * 1e3)
            fill = min(ids.shape[0], max_k)
            predictions[idx, :fill] = ids[:fill]
        elapsed = perf_counter() - start

        recalls = {k: recall_at_k(self.ground_truth, predictions, k) for k in k_values}
        query_time_ms = (elapsed / queries.shape[0]) * 1e3 if queries.size else 0.0
        qps = queries_per_second(queries.shape[0], elapsed) if elapsed > 0 else 0.0
        latencies = np.asarray(per_query_latencies, dtype=np.float32)
        if latencies.size == 0:
            latency_p50 = 0.0
            latency_p95 = 0.0
        else:
            latency_p50 = float(np.percentile(latencies, 50))
            latency_p95 = float(np.percentile(latencies, 95))

        return EvaluationResult(
            recalls=recalls,
            query_time_ms=query_time_ms,
            qps=qps,
            latency_p50_ms=latency_p50,
            latency_p95_ms=latency_p95,
        )
