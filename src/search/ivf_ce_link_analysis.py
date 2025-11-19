"""IVF-CE explorer variant that surfaces detailed link diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from src.search.ivf_ce_searcher import IVFCEExplorer


@dataclass
class LinkDiagnostics:
    """Container describing how cross-cluster links were used for a query."""

    query_id: int
    B: int
    n1: int
    n2: int
    link_clusters_used: int
    fallback_clusters_used: int
    n2_total: int
    link_recommended_clusters: List[int]
    fallback_clusters: List[int]
    nprobe_next_clusters: List[int]
    overlap_rate: float


class IVFCEExplorerDiagnostics(IVFCEExplorer):
    """Extends the IVF-CE explorer with link-usage instrumentation."""

    def search_with_diagnostics(
        self,
        query: np.ndarray,
        k: int,
        *,
        query_id: int,
        B: int,
    ) -> Tuple[np.ndarray, np.ndarray, LinkDiagnostics]:
        """
        Execute a search and return IVF-CE results plus per-query diagnostics.

        Parameters
        ----------
        query:
            Query vector.
        k:
            Top-k to retrieve.
        query_id:
            Index of the query within the evaluation batch (for logging).
        B:
            Total cluster budget (n1 + n2). Needed to simulate IVF overlap.
        """

        self._validate_query(query)
        if k <= 0:
            raise ValueError("k must be positive.")

        stage1_clusters = self._stage0_routing(query)
        stage1_list = stage1_clusters.tolist()
        stage1_ids, stage1_dists = self._search_clusters(query, stage1_list)

        if self.n2 == 0:
            final_ids, final_dists = self._deduplicate_and_select(
                stage1_ids, stage1_dists, k
            )
            diagnostics = LinkDiagnostics(
                query_id=query_id,
                B=B,
                n1=self.n1,
                n2=0,
                link_clusters_used=0,
                fallback_clusters_used=0,
                n2_total=0,
                link_recommended_clusters=[],
                fallback_clusters=[],
                nprobe_next_clusters=[],
                overlap_rate=0.0,
            )
            return final_ids, final_dists, diagnostics

        if stage1_ids.size == 0:
            diagnostics = LinkDiagnostics(
                query_id=query_id,
                B=B,
                n1=self.n1,
                n2=self.n2,
                link_clusters_used=0,
                fallback_clusters_used=0,
                n2_total=0,
                link_recommended_clusters=[],
                fallback_clusters=[],
                nprobe_next_clusters=[],
                overlap_rate=0.0,
            )
            return stage1_ids, stage1_dists, diagnostics

        candidate_pairs = self._select_candidates(stage1_ids, stage1_dists)
        link_clusters = self._stage2_propose_clusters(
            candidate_pairs, searched_clusters=set(stage1_list)
        )
        link_clusters = link_clusters[: self.n2]

        stage2_clusters = list(link_clusters)
        fallback_clusters: List[int] = []
        if len(stage2_clusters) < self.n2:
            needed = self.n2 - len(stage2_clusters)
            excluded = set(stage1_list)
            excluded.update(stage2_clusters)
            fallback_clusters = self._fallback_clusters(
                query, excluded=excluded, limit=needed
            )
            stage2_clusters.extend(fallback_clusters)

        stage2_ids, stage2_dists = self._search_clusters(query, stage2_clusters)
        combined_ids = np.concatenate([stage1_ids, stage2_ids])
        combined_dists = np.concatenate([stage1_dists, stage2_dists])
        final_ids, final_dists = self._deduplicate_and_select(combined_ids, combined_dists, k)

        nprobe_order, _ = self.index.nearest_centroids(
            query, min(B, self.index.n_clusters)
        )
        nprobe_list = [int(cid) for cid in nprobe_order.tolist()[self.n1 : self.n1 + self.n2]]

        overlap = 0.0
        if self.n2 > 0 and nprobe_list:
            overlap_count = len(set(link_clusters) & set(nprobe_list))
            overlap = overlap_count / float(self.n2)

        diagnostics = LinkDiagnostics(
            query_id=query_id,
            B=B,
            n1=self.n1,
            n2=self.n2,
            link_clusters_used=len(link_clusters),
            fallback_clusters_used=len(fallback_clusters),
            n2_total=len(stage2_clusters),
            link_recommended_clusters=[int(cid) for cid in link_clusters],
            fallback_clusters=[int(cid) for cid in fallback_clusters],
            nprobe_next_clusters=nprobe_list,
            overlap_rate=overlap,
        )
        return final_ids, final_dists, diagnostics

    def _deduplicate_and_select(
        self, ids: np.ndarray, distances: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Deduplicate candidate ids and return the best k results."""
        final_ids, final_dists = (
            ids,
            distances,
        )
        if final_ids.size > 0:
            from src.search.utils import deduplicate_keep_best

            final_ids, final_dists = deduplicate_keep_best(final_ids, final_dists)

        if final_ids.size == 0:
            return final_ids, final_dists

        top_k = min(k, final_ids.size)
        idx = np.argpartition(final_dists, kth=top_k - 1)[:top_k]
        ordering = np.argsort(final_dists[idx])
        return final_ids[idx][ordering], final_dists[idx][ordering]
