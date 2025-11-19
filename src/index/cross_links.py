"""Cross-link computation for IVF-CE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from src.index.ivf_index import IVFIndex


@dataclass(frozen=True)
class CrossLink:
    """Represents a candidate cluster linked to a given database vector."""
    cluster_id: int
    distance: float


class CrossLinkBuilder:
    """Compute cross-cluster links for vectors stored in an IVF index."""

    def __init__(
        self,
        index: IVFIndex,
        *,
        k1: int,
        m_max: int,
        p_index: int,
    ) -> None:
        """Initialize the builder with link hyperparameters and source index."""
        if not index.is_built:
            raise ValueError("Index must be built before computing cross-links.")
        if k1 <= 0 or m_max <= 0 or p_index <= 0:
            raise ValueError("k1, m_max, and p_index must be positive")

        self.index = index
        self.k1 = k1
        self.m_max = m_max
        self.p_index = p_index

    def build_for_vector(self, vector_id: int) -> List[CrossLink]:
        """Compute cross-cluster links for a single database vector."""
        vector = self.index.database[vector_id]
        own_cluster = int(self.index.assignments[vector_id])

        centroid_ids, _ = self.index.nearest_centroids(vector, self.p_index)
        candidate_clusters = [c for c in centroid_ids if c != own_cluster]
        neighbors: List[Tuple[int, int, float]] = []

        for cluster_id in candidate_clusters:
            member_ids, distances = self.index.search_cluster(cluster_id, vector)
            for vid, dist in zip(member_ids, distances):
                if vid == vector_id:
                    continue
                neighbors.append((cluster_id, int(vid), float(dist)))

        if not neighbors:
            return []

        neighbors.sort(key=lambda x: x[2])
        neighbors = neighbors[: self.k1]

        # Aggregate by cluster, keeping the minimum distance per cluster.
        best_per_cluster: Dict[int, float] = {}
        for cluster_id, _, dist in neighbors:
            current = best_per_cluster.get(cluster_id, float("inf"))
            if dist < current:
                best_per_cluster[cluster_id] = dist

        sorted_clusters = sorted(best_per_cluster.items(), key=lambda x: x[1])
        limited = sorted_clusters[: self.m_max]
        return [CrossLink(cluster_id=c, distance=d) for c, d in limited]
