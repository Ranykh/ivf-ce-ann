"""Minimal dataset loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from data.utils import read_fvecs, read_ivecs


DatasetTuple = Tuple[np.ndarray, np.ndarray, np.ndarray]


def load_dataset(name: str, root: str | Path = "data") -> DatasetTuple:
    """Load a supported dataset by name."""
    name = name.lower()
    if name != "sift1m":
        raise ValueError(f"Unsupported dataset '{name}'. Only 'sift1m' is available.")
    return load_sift1m(root)


def load_sift1m(root: str | Path = "data") -> DatasetTuple:
    """Load the SIFT1M dataset from disk."""
    root_path = Path(root)
    base_path, query_path, gt_path = _find_sift_files(root_path)

    base = read_fvecs(base_path)
    queries = read_fvecs(query_path)
    ground_truth = read_ivecs(gt_path)
    return base, queries, ground_truth


def _find_sift_files(root: Path) -> tuple[Path, Path, Path]:
    """Locate SIFT1M base/query/ground-truth files under known directories."""
    candidates = [
        root / "raw" / "sift1m" / "sift",
        root / "sift1m",
    ]

    for directory in candidates:
        base = directory / "sift_base.fvecs"
        query = directory / "sift_query.fvecs"
        gt = directory / "sift_groundtruth.ivecs"
        if base.is_file() and query.is_file() and gt.is_file():
            return base, query, gt

    raise FileNotFoundError(
        "SIFT1M files not found. "
        "Expected to see sift_base.fvecs, sift_query.fvecs, and sift_groundtruth.ivecs "
        f"under one of: {', '.join(str(p) for p in candidates)}."
    )
