"""Exact brute-force search used for ground-truth comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class BruteForceIndex:
    vectors: np.ndarray


class BruteForceSearch:
    """Reference implementation computing exact L2 distances."""

    def __init__(self) -> None:
        self.index: BruteForceIndex | None = None

    def build(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        self.index = BruteForceIndex(vectors.astype(np.float32, copy=True))

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise RuntimeError("Index has not been built yet.")
        if query.ndim != 1:
            raise ValueError("query must be 1D")
        if query.shape[0] != self.index.vectors.shape[1]:
            raise ValueError("query dimension mismatch.")
        if k <= 0:
            raise ValueError("k must be positive")

        diffs = self.index.vectors - query
        distances = np.einsum("ij,ij->i", diffs, diffs, dtype=np.float32)
        k = min(k, distances.shape[0])
        idx = np.argpartition(distances, kth=k - 1)[:k]
        ordering = np.argsort(distances[idx])
        return idx[ordering].astype(np.int32), distances[idx][ordering]
