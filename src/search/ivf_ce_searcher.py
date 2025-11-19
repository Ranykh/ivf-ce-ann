"""IVF-CE two-stage search implementation."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Set, Tuple

import numpy as np

from src.index import IVFCEIndex
from src.index.cross_links import CrossLink
from src.search.base_searcher import BaseSearcher
from src.search.utils import deduplicate_keep_best


class IVFCEExplorer(BaseSearcher):
    """Two-stage IVF-CE searcher that explores cross-cluster links."""

    def __init__(
        self,
        index: IVFCEIndex,
        *,
        n1: int,
        n2: int,
        k2: int,
    ) -> None:
        """Configure the explorer with stage budgets and candidate counts."""
        if not index.is_built:
            raise ValueError("Index must be built before constructing a searcher.")
        super().__init__(index_dimension=index.dimension)
        if n1 <= 0 or n2 < 0 or k2 <= 0:
            raise ValueError("n1 and k2 must be positive, n2 must be non-negative.")
        self.index = index
        self.n1 = n1
        self.n2 = n2
        self.k2 = k2

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search using centroid routing followed by link-guided exploration."""
        self._validate_query(query)
        if k <= 0:
            raise ValueError("k must be positive")

        stage1_clusters = self._stage0_routing(query)
        stage1_cluster_list = stage1_clusters.tolist()
        stage1_ids, stage1_dists = self._search_clusters(query, stage1_cluster_list)

        if stage1_ids.size == 0:
            return stage1_ids, stage1_dists

        # Stage 2: cross-cluster exploration guidance.
        candidate_pairs = self._select_candidates(stage1_ids, stage1_dists)
        stage2_clusters = self._stage2_propose_clusters(
            candidate_pairs, searched_clusters=set(stage1_cluster_list)
        )

        if len(stage2_clusters) < self.n2:
            needed = self.n2 - len(stage2_clusters)
            excluded = set(stage1_cluster_list)
            excluded.update(stage2_clusters)
            fallback = self._fallback_clusters(query, excluded=excluded, limit=needed)
            stage2_clusters.extend(fallback)

        stage2_ids, stage2_dists = self._search_clusters(query, stage2_clusters)

        combined_ids = np.concatenate([stage1_ids, stage2_ids])
        combined_dists = np.concatenate([stage1_dists, stage2_dists])
        final_ids, final_dists = deduplicate_keep_best(combined_ids, combined_dists)

        if final_ids.size == 0:
            return final_ids, final_dists

        top_k = min(k, final_ids.size)
        idx = np.argpartition(final_dists, kth=top_k - 1)[:top_k]
        ordering = np.argsort(final_dists[idx])
        return final_ids[idx][ordering], final_dists[idx][ordering]

    def _stage0_routing(self, query: np.ndarray) -> np.ndarray:
        """Route the query to the n1 nearest coarse centroids."""
        cluster_ids, _ = self.index.nearest_centroids(query, self.n1)
        return cluster_ids

    def _search_clusters(
        self, query: np.ndarray, clusters: Sequence[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search all vectors within the provided cluster ids."""
        if not isinstance(clusters, np.ndarray) and not isinstance(clusters, list):
            clusters = list(clusters)
        if len(clusters) == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)

        all_ids: List[int] = []
        all_dists: List[float] = []
        for cluster_id in clusters:
            ids, dists = self.index.search_cluster(cluster_id, query)
            all_ids.extend(ids.tolist())
            all_dists.extend(dists.tolist())

        if not all_ids:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)

        ids_arr = np.asarray(all_ids, dtype=np.int32)
        dists_arr = np.asarray(all_dists, dtype=np.float32)
        return deduplicate_keep_best(ids_arr, dists_arr)

    def _select_candidates(
        self, ids: np.ndarray, distances: np.ndarray
    ) -> Sequence[Tuple[int, float]]:
        """Choose the current best vector candidates to seed link expansion."""
        if ids.size == 0:
            return []
        limit = min(self.k2, ids.size)
        return [
            (int(ids[i]), float(distances[i]))
            for i in np.argsort(distances)[:limit]
        ]

    def _stage2_propose_clusters(
        self,
        candidates: Sequence[Tuple[int, float]],
        *,
        searched_clusters: Set[int],
    ) -> List[int]:
        """Use cross-links from promising candidates to propose new clusters."""
        if self.n2 == 0 or not candidates:
            return []

        accumulator: dict[int, float] = {}
        for vector_id, _score in candidates:
            links = self.index.cross_links.get(int(vector_id), [])
            for link in links:
                if link.cluster_id in searched_clusters:
                    continue
                current = accumulator.get(link.cluster_id, float("inf"))
                if link.distance < current:
                    accumulator[link.cluster_id] = link.distance

        if not accumulator:
            return []

        sorted_pairs = sorted(accumulator.items(), key=lambda x: x[1])
        return [cluster for cluster, _ in sorted_pairs[: self.n2]]

    def _fallback_clusters(
        self, query: np.ndarray, excluded: Set[int], limit: int
    ) -> List[int]:
        """Select additional nearest centroids if links supply too few clusters."""
        if limit <= 0:
            return []

        nprobe = min(
            self.index.n_clusters, self.n1 + self.n2 + limit
        )
        ordered, _ = self.index.nearest_centroids(query, nprobe)
        fallback: List[int] = []
        for cluster_id in ordered:
            cid = int(cluster_id)
            if cid in excluded:
                continue
            excluded.add(cid)
            fallback.append(cid)
            if len(fallback) == limit:
                break
        return fallback
