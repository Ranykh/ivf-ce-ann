"""Named IVF-CE configuration presets loaded from YAML files."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset location/name referenced by a preset."""
    name: str
    path: str


@dataclass(frozen=True)
class ClusteringConfig:
    """Parameters controlling coarse quantizer training."""
    n_clusters: int
    n_init: int
    max_iter: int
    seed: Optional[int]


@dataclass(frozen=True)
class IndexingConfig:
    """Cross-link construction hyperparameters."""
    m_max: int
    k1: int
    p_index: int


@dataclass(frozen=True)
class SearchConfig:
    """Query-time sweep specification for a preset."""
    B_values: List[int]
    k2_values: List[int]
    query_seed: int
    query_count: Optional[int]


@dataclass(frozen=True)
class IVFCEPreset:
    """Container bundling dataset, build, and search settings."""
    name: str
    dataset: DatasetConfig
    clustering: ClusteringConfig
    indexing: IndexingConfig
    search: SearchConfig


_PRESET_DIR = Path(__file__).resolve().parents[2] / "config" / "presets"


def available_preset_names() -> List[str]:
    """List preset identifiers detected in the presets directory."""
    if not _PRESET_DIR.is_dir():
        return []
    return sorted(path.stem.upper() for path in _PRESET_DIR.glob("*.yaml"))


def _ensure_mapping(section: str, data, path: Path) -> dict:
    """Ensure a preset section decodes to a mapping structure."""
    if not isinstance(data, dict):
        raise ValueError(f"Section '{section}' in preset {path} must be a mapping.")
    return data


def _load_dataset(section: dict, path: Path) -> DatasetConfig:
    """Parse the dataset portion of a preset file."""
    section = _ensure_mapping("dataset", section, path)
    try:
        name = str(section["name"])
        dataset_path = str(section["path"])
    except KeyError as exc:
        raise ValueError(f"Dataset section in {path} missing key {exc}.") from exc
    return DatasetConfig(name=name, path=dataset_path)


def _load_clustering(section: dict, path: Path) -> ClusteringConfig:
    """Parse the clustering parameters from a preset."""
    section = _ensure_mapping("clustering", section, path)
    try:
        n_clusters = int(section["n_clusters"])
        n_init = int(section.get("n_init", 10))
        max_iter = int(section.get("max_iter", 100))
        seed_val = section.get("seed")
        seed = int(seed_val) if seed_val is not None else None
    except KeyError as exc:
        raise ValueError(f"Clustering section in {path} missing key {exc}.") from exc
    return ClusteringConfig(
        n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, seed=seed
    )


def _load_indexing(section: dict, path: Path) -> IndexingConfig:
    """Parse indexing hyperparameters from a preset."""
    section = _ensure_mapping("indexing", section, path)
    try:
        m_max = int(section["m_max"])
        k1 = int(section["k1"])
        p_index = int(section["p_index"])
    except KeyError as exc:
        raise ValueError(f"Indexing section in {path} missing key {exc}.") from exc
    return IndexingConfig(m_max=m_max, k1=k1, p_index=p_index)


def _load_search(section: dict, path: Path) -> SearchConfig:
    """Parse query-time sweep parameters from a preset."""
    section = _ensure_mapping("search", section, path)
    try:
        b_values = [int(v) for v in section.get("B_values", [])]
        k2_values = [int(v) for v in section.get("k2_values", [])]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Search section arrays in {path} must be integers.") from exc
    if not b_values:
        raise ValueError(f"Search section in {path} must define B_values.")
    if not k2_values:
        raise ValueError(f"Search section in {path} must define k2_values.")
    if min(b_values) < 2:
        raise ValueError(f"All B_values must be >= 2 in {path}.")
    if min(k2_values) <= 0:
        raise ValueError(f"All k2_values must be positive in {path}.")
    query_seed = int(section.get("query_seed", 0))
    query_count_val = section.get("query_count")
    query_count = int(query_count_val) if query_count_val is not None else None
    return SearchConfig(
        B_values=sorted(set(b_values)),
        k2_values=sorted(set(k2_values)),
        query_seed=query_seed,
        query_count=query_count,
    )


@lru_cache(maxsize=None)
def get_preset(name: str) -> IVFCEPreset:
    """Load and cache a preset by name."""
    normalized = name.upper()
    path = _PRESET_DIR / f"{normalized}.yaml"
    if not path.is_file():
        available = ", ".join(available_preset_names())
        raise ValueError(
            f"Unknown IVF-CE preset '{name}'. Available presets: {available or 'none found'}."
        )

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Preset file {path} must contain a mapping.")

    preset_name = str(data.get("name", normalized)).upper()
    dataset = _load_dataset(data.get("dataset", {}), path)
    clustering = _load_clustering(data.get("clustering", {}), path)
    indexing = _load_indexing(data.get("indexing", {}), path)
    search = _load_search(data.get("search", {}), path)
    return IVFCEPreset(
        name=preset_name,
        dataset=dataset,
        clustering=clustering,
        indexing=indexing,
        search=search,
    )
