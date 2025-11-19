"""Wrappers around FAISS indices used for external baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


@dataclass
class FAISSIndexWrapper:
    """Thin wrapper to hold the constructed FAISS index object."""
    index: "faiss.Index"  # type: ignore[name-defined]


class FAISSIVFBaseline:
    """Minimal wrapper around faiss.IndexIVFFlat."""

    def __init__(
        self,
        dimension: int,
        *,
        nlist: int,
        nprobe: int,
    ) -> None:
        """Configure the FAISS IVF Flat baseline."""
        if faiss is None:
            raise ImportError("faiss is required for the FAISS baselines.")

        self.dimension = dimension
        self.nlist = nlist
        self.nprobe = nprobe
        self.wrapper: FAISSIndexWrapper | None = None

    def build(self, vectors: np.ndarray) -> None:
        """Train and add vectors to a FAISS IVF Flat index."""
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        if vectors.shape[1] != self.dimension:
            raise ValueError("Input vectors dimension mismatch.")

        quantizer = faiss.IndexFlatL2(self.dimension)
        index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, faiss.METRIC_L2)

        # Train and add vectors.
        if not index.is_trained:
            index.train(vectors)
        index.add(vectors)
        index.nprobe = self.nprobe
        self.wrapper = FAISSIndexWrapper(index=index)

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search the FAISS index for the nearest neighbors of a query."""
        if self.wrapper is None:
            raise RuntimeError("Index has not been built yet.")
        if query.ndim != 1:
            raise ValueError("query must be 1D")
        if query.shape[0] != self.dimension:
            raise ValueError("query dimension mismatch.")
        if k <= 0:
            raise ValueError("k must be positive")

        query_batch = np.expand_dims(query.astype(np.float32, copy=False), axis=0)
        distances, indices = self.wrapper.index.search(query_batch, k)
        return indices[0].astype(np.int32), distances[0]
