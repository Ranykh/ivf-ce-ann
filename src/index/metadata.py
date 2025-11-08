"""Index metadata and build statistics helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class BuildStats:
    """Stores per-component build timings for an index."""

    components: Dict[str, float] = field(default_factory=dict)

    def add_component(self, name: str, duration_seconds: float) -> None:
        self.components[name] = float(duration_seconds)

    @property
    def total(self) -> float:
        return float(sum(self.components.values()))

    def component(self, name: str) -> Optional[float]:
        return self.components.get(name)

    def as_dict(self) -> Dict[str, float]:
        """Return a copy of all components."""
        return dict(self.components)

    def to_serializable(self) -> Dict[str, Any]:
        return {"components": dict(self.components)}

    @classmethod
    def from_serializable(cls, payload: Dict[str, Any]) -> "BuildStats":
        components = payload.get("components", {})
        return cls(components={str(k): float(v) for k, v in components.items()})


@dataclass
class IndexMetadata:
    """Describes how an IVF-CE index instance was produced."""

    config_name: str
    dataset_name: str
    dataset_path: str | None
    dataset_size: int
    n_clusters: int
    dimension: int
    m_max: Optional[int]
    k1: Optional[int]
    p_index: Optional[int]
    seed: Optional[int]
    built_at: str
    build_stats: BuildStats
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        config_name: str,
        dataset_name: str,
        dataset_path: str | None,
        dataset_size: int,
        n_clusters: int,
        dimension: int,
        m_max: Optional[int],
        k1: Optional[int],
        p_index: Optional[int],
        seed: Optional[int],
        build_stats: BuildStats,
        extra: Optional[Dict[str, Any]] = None,
    ) -> "IndexMetadata":
        return cls(
            config_name=config_name,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            dataset_size=int(dataset_size),
            n_clusters=int(n_clusters),
            dimension=int(dimension),
            m_max=int(m_max) if m_max is not None else None,
            k1=int(k1) if k1 is not None else None,
            p_index=int(p_index) if p_index is not None else None,
            seed=int(seed) if seed is not None else None,
            built_at=datetime.utcnow().isoformat(),
            build_stats=build_stats,
            extra=extra or {},
        )

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "config_name": self.config_name,
            "dataset_name": self.dataset_name,
            "dataset_path": self.dataset_path,
            "dataset_size": self.dataset_size,
            "n_clusters": self.n_clusters,
            "dimension": self.dimension,
            "m_max": self.m_max,
            "k1": self.k1,
            "p_index": self.p_index,
            "seed": self.seed,
            "built_at": self.built_at,
            "build_stats": self.build_stats.to_serializable(),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_serializable(cls, payload: Dict[str, Any]) -> "IndexMetadata":
        build_stats_payload = payload.get("build_stats", {})
        build_stats = BuildStats.from_serializable(build_stats_payload)
        return cls(
            config_name=payload.get("config_name", "unknown"),
            dataset_name=payload.get("dataset_name", "unknown"),
            dataset_path=payload.get("dataset_path"),
            dataset_size=int(payload.get("dataset_size", 0)),
            n_clusters=int(payload.get("n_clusters", 0)),
            dimension=int(payload.get("dimension", 0)),
            m_max=payload.get("m_max"),
            k1=payload.get("k1"),
            p_index=payload.get("p_index"),
            seed=payload.get("seed"),
            built_at=payload.get("built_at", datetime.utcnow().isoformat()),
            build_stats=build_stats,
            extra=dict(payload.get("extra", {})),
        )
