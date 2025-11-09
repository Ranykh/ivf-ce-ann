"""IVF-CE index that augments IVF with cross-cluster links."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Dict, List

import numpy as np

from src.index.cross_links import CrossLink, CrossLinkBuilder
from src.index.ivf_index import IVFIndex
from src.index.metadata import BuildStats


class IVFCEIndex(IVFIndex):
    def __init__(
        self,
        dimension: int,
        *,
        n_clusters: int,
        k1: int,
        m_max: int,
        p_index: int,
        n_init: int = 10,
        max_iter: int = 100,
        seed: int | None = None,
        progress_log_interval: int = 10_000,
        progress_logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            dimension=dimension,
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            seed=seed,
        )
        self.k1 = k1
        self.m_max = m_max
        self.p_index = p_index
        self.cross_links: Dict[int, List[CrossLink]] = {}
        if progress_log_interval <= 0:
            raise ValueError("progress_log_interval must be positive.")
        self._progress_log_interval = progress_log_interval
        self._progress_logger = progress_logger or logging.getLogger(
            "ivfce.cross_links"
        )

    def build(self, vectors: np.ndarray) -> None:
        super().build(vectors)
        cross_start = perf_counter()
        self.cross_links = self._build_cross_links()
        cross_time = perf_counter() - cross_start
        if self.build_stats is None:
            self.build_stats = BuildStats(components={})
        self.build_stats.add_component("cross_links", cross_time)

    def _build_cross_links(self) -> Dict[int, List[CrossLink]]:
        builder = CrossLinkBuilder(
            self,
            k1=self.k1,
            m_max=self.m_max,
            p_index=self.p_index,
        )
        cross_links: Dict[int, List[CrossLink]] = {}
        total_vectors = self.database.shape[0]
        next_log = self._progress_log_interval
        for vector_id in range(total_vectors):
            cross_links[vector_id] = builder.build_for_vector(vector_id)
            processed = vector_id + 1
            if processed >= next_log or processed == total_vectors:
                percent = (processed / total_vectors) * 100 if total_vectors else 100.0
                self._progress_logger.info(
                    "Cross-links: processed %d/%d vectors (%.1f%%)",
                    processed,
                    total_vectors,
                    percent,
                )
                next_log += self._progress_log_interval
        return cross_links

    def _serialize_state(self) -> Dict[str, Any]:
        state = super()._serialize_state()
        state.update(
            {
                "k1": self.k1,
                "m_max": self.m_max,
                "p_index": self.p_index,
                "cross_links": self.cross_links,
            }
        )
        return state

    @classmethod
    def _hydrate_from_state(cls, state: Dict[str, Any]) -> "IVFCEIndex":
        class_name = state.get("class_name")
        if class_name != cls.__name__:
            raise ValueError(
                f"Serialized index was built as {class_name}; "
                f"use IVFCEIndex.load to restore it."
            )

        index = cls(
            dimension=int(state["dimension"]),
            n_clusters=int(state["n_clusters"]),
            k1=int(state["k1"]),
            m_max=int(state["m_max"]),
            p_index=int(state["p_index"]),
            n_init=int(state.get("n_init", 10)),
            max_iter=int(state.get("max_iter", 100)),
            seed=state.get("seed"),
        )
        cls._apply_serialized_state(index, state)
        cross_links_serialized: Dict[int, List[CrossLink]] = state["cross_links"]
        index.cross_links = {
            int(vec_id): list(links) for vec_id, links in cross_links_serialized.items()
        }
        return index
