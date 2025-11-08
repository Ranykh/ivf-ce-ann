"""Standard IVF index implemented with NumPy."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from src.clustering import train_kmeans
from src.index.base_index import BaseIndex
from src.index.metadata import BuildStats, IndexMetadata
from src.utils import pairwise_squared_l2


@dataclass
class InvertedList:
    ids: np.ndarray  # shape: (n_members,)


class IVFIndex(BaseIndex):
    """Inverted File Index without cross-cluster links."""

    _SERIALIZATION_VERSION = 1

    def __init__(
        self,
        dimension: int,
        *,
        n_clusters: int,
        n_init: int = 10,
        max_iter: int = 100,
        seed: int | None = None,
    ) -> None:
        super().__init__(dimension=dimension)
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.seed = seed

        self.centroids: np.ndarray | None = None
        self.database: np.ndarray | None = None
        self.inverted_lists: Dict[int, InvertedList] = {}
        self.assignments: np.ndarray | None = None

    def build(self, vectors: np.ndarray) -> None:
        self._validate_input(vectors)
        total_start = perf_counter()
        clustering_start = perf_counter()

        result = train_kmeans(
            vectors,
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            max_iter=self.max_iter,
            seed=self.seed,
        )
        clustering_time = perf_counter() - clustering_start

        assign_start = perf_counter()
        self.centroids = result.centroids
        self.database = vectors.astype(np.float32, copy=True)
        self.assignments = result.assignments.astype(np.int32)
        assign_time = perf_counter() - assign_start

        lists_start = perf_counter()
        self._build_inverted_lists(result.assignments)
        inverted_time = perf_counter() - lists_start

        total_time = perf_counter() - total_start
        overhead = max(0.0, total_time - clustering_time - assign_time - inverted_time)

        stats = BuildStats(
            components={
                "clustering": clustering_time,
                "copy_assign": assign_time,
                "inverted_lists": inverted_time,
            }
        )
        if overhead > 0.0:
            stats.add_component("overhead", overhead)
        self.build_stats = stats
        self.is_built = True

    def assign(self, vectors: np.ndarray) -> np.ndarray:
        self._ensure_ready()
        self._validate_input(vectors)

        distances = pairwise_squared_l2(vectors, self.centroids)
        return np.argmin(distances, axis=1).astype(np.int32)

    def search_cluster(
        self, cluster_id: int, query: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return vector ids and squared distances for a query within a cluster."""
        self._ensure_ready()

        member_ids = self.get_cluster_member_ids(cluster_id)
        if member_ids.size == 0:
            return member_ids, np.empty(0, dtype=np.float32)

        vectors = self.database[member_ids]
        diffs = vectors - query
        distances = np.einsum("ij,ij->i", diffs, diffs, dtype=np.float32)
        return member_ids, distances

    def get_cluster_member_ids(self, cluster_id: int) -> np.ndarray:
        self._ensure_ready()
        inverted_list = self.inverted_lists.get(cluster_id)
        if inverted_list is None:
            return np.empty(0, dtype=np.int32)
        return inverted_list.ids

    def nearest_centroids(
        self, query: np.ndarray, nprobe: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the closest ``nprobe`` centroids to the query."""
        self._ensure_ready()
        if query.shape[0] != self.dimension:
            raise ValueError("query dimension does not match index dimension")
        if nprobe <= 0:
            raise ValueError("nprobe must be positive")

        diffs = self.centroids - query
        distances = np.einsum("ij,ij->i", diffs, diffs, dtype=np.float32)
        nprobe = min(nprobe, self.centroids.shape[0])
        idx = np.argpartition(distances, kth=nprobe - 1)[:nprobe]
        ordering = np.argsort(distances[idx])
        return idx[ordering], distances[idx][ordering]

    def _build_inverted_lists(self, assignments: np.ndarray) -> None:
        self.inverted_lists.clear()
        for cluster_id in range(self.n_clusters):
            member_ids = np.where(assignments == cluster_id)[0].astype(np.int32)
            self.inverted_lists[cluster_id] = InvertedList(ids=member_ids)

    def _ensure_ready(self) -> None:
        if (
            not self.is_built
            or self.centroids is None
            or self.database is None
            or getattr(self, "assignments", None) is None
        ):
            raise RuntimeError("Index has not been built yet.")

    def save(self, path: str | Path) -> None:
        state = self._serialize_state()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "IVFIndex":
        path = Path(path)
        with path.open("rb") as handle:
            state = pickle.load(handle)
        version = state.get("version", 0)
        if version != cls._SERIALIZATION_VERSION:
            raise ValueError(
                f"Unsupported IVF index serialization version {version} "
                f"(expected {cls._SERIALIZATION_VERSION})."
            )
        return cls._hydrate_from_state(state)

    def _serialize_state(self) -> Dict[str, Any]:
        self._ensure_ready()
        return {
            "version": self._SERIALIZATION_VERSION,
            "class_name": self.__class__.__name__,
            "dimension": self.dimension,
            "n_clusters": self.n_clusters,
            "n_init": self.n_init,
            "max_iter": self.max_iter,
            "seed": self.seed,
            "centroids": self.centroids,
            "database": self.database,
            "assignments": self.assignments,
            "inverted_lists": {
                cid: inv.ids for cid, inv in self.inverted_lists.items()
            },
            "build_stats": self.build_stats.to_serializable()
            if self.build_stats
            else None,
            "metadata": self.metadata.to_serializable() if self.metadata else None,
        }

    @classmethod
    def _hydrate_from_state(cls, state: Dict[str, Any]) -> "IVFIndex":
        class_name = state.get("class_name")
        if class_name != cls.__name__:
            raise ValueError(
                f"Serialized index was built as {class_name}; "
                f"use the matching class to load."
            )

        index = cls(
            dimension=int(state["dimension"]),
            n_clusters=int(state["n_clusters"]),
            n_init=int(state.get("n_init", 10)),
            max_iter=int(state.get("max_iter", 100)),
            seed=state.get("seed"),
        )
        cls._apply_serialized_state(index, state)
        return index

    @staticmethod
    def _apply_serialized_state(index: "IVFIndex", state: Dict[str, Any]) -> None:
        index.centroids = np.array(state["centroids"], copy=True)
        index.database = np.array(state["database"], copy=True)
        index.assignments = np.array(state["assignments"], dtype=np.int32, copy=True)

        inverted_lists_serialized: Dict[int, np.ndarray] = state["inverted_lists"]
        reconstructed: Dict[int, InvertedList] = {}
        for cid, ids in inverted_lists_serialized.items():
            reconstructed[int(cid)] = InvertedList(
                ids=np.array(ids, dtype=np.int32, copy=True)
            )
        index.inverted_lists = reconstructed
        index.is_built = True

        build_stats_payload = state.get("build_stats")
        if build_stats_payload:
            index.build_stats = BuildStats.from_serializable(build_stats_payload)
        metadata_payload = state.get("metadata")
        if metadata_payload:
            index.metadata = IndexMetadata.from_serializable(metadata_payload)
