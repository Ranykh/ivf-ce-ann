"""Standard IVF searcher."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from src.index import IVFIndex
from src.search.base_searcher import BaseSearcher
from src.search.utils import deduplicate_keep_best


class IVFSearcher(BaseSearcher):
    """Searcher that probes a fixed number of IVF clusters per query."""

    def __init__(self, index: IVFIndex, *, nprobe: int) -> None:
        """Store the IVF index reference and number of centroids to probe."""
        if not index.is_built:
            raise ValueError("Index must be built before constructing a searcher.")
        super().__init__(index_dimension=index.dimension)
        self.index = index
        self.nprobe = nprobe

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve the approximate top-k neighbors via IVF scanning."""
        self._validate_query(query)
        if k <= 0:
            raise ValueError("k must be positive")

        cluster_ids, _ = self.index.nearest_centroids(query, self.nprobe)

        all_ids: list[int] = []
        all_distances: list[float] = []
        for cluster_id in cluster_ids:
            ids, distances = self.index.search_cluster(cluster_id, query)
            all_ids.extend(ids.tolist())
            all_distances.extend(distances.tolist())

        if not all_ids:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)

        all_ids_arr = np.asarray(all_ids, dtype=np.int32)
        all_dists_arr = np.asarray(all_distances, dtype=np.float32)

        unique_ids, unique_dists = deduplicate_keep_best(all_ids_arr, all_dists_arr)

        k = min(k, unique_ids.shape[0])
        idx = np.argpartition(unique_dists, kth=k - 1)[:k]
        ordering = np.argsort(unique_dists[idx])

        return unique_ids[idx][ordering], unique_dists[idx][ordering]
