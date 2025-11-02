"""Low-level dataset file readers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import numpy as np


PathLike = Union[str, os.PathLike]


def _ensure_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return path


def read_fvecs(path: PathLike) -> np.ndarray:
    """Load a .fvecs file into a float32 NumPy array."""
    file_path = _ensure_file(Path(path))

    data = np.fromfile(file_path, dtype=np.int32)
    if data.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    dim = data[0]
    if dim <= 0:
        raise ValueError(f"Invalid vector dimension ({dim}) in file {file_path}")

    data = data.reshape(-1, dim + 1)
    if not np.all(data[:, 0] == dim):
        raise ValueError(f"Inconsistent vector dimensions found in {file_path}")

    vectors = data[:, 1:].view(np.float32).copy()
    return vectors


def read_bvecs(path: PathLike) -> np.ndarray:
    """Load a .bvecs file into an unsigned byte NumPy array."""
    file_path = _ensure_file(Path(path))

    raw = np.fromfile(file_path, dtype=np.uint8)
    if raw.size == 0:
        return np.empty((0, 0), dtype=np.uint8)

    if raw.size < 4:
        raise ValueError(f"Corrupted bvecs file: {file_path}")
    dim = int(np.frombuffer(raw[:4], dtype=np.int32, count=1)[0])
    if dim <= 0:
        raise ValueError(f"Invalid vector dimension ({dim}) in file {file_path}")

    record_size = dim + 4
    if raw.size % record_size != 0:
        raise ValueError(f"File size is not a multiple of record size in {file_path}")

    raw = raw.reshape(-1, record_size)
    dims = np.frombuffer(raw[:, :4].tobytes(), dtype=np.int32)
    if not np.all(dims == dim):
        raise ValueError(f"Inconsistent vector dimensions found in {file_path}")

    vectors = raw[:, 4:].copy()
    return vectors


def read_ivecs(path: PathLike) -> np.ndarray:
    """Load a .ivecs file into an int32 NumPy array."""
    file_path = _ensure_file(Path(path))

    data = np.fromfile(file_path, dtype=np.int32)
    if data.size == 0:
        return np.empty((0, 0), dtype=np.int32)

    dim = data[0]
    if dim <= 0:
        raise ValueError(f"Invalid vector dimension ({dim}) in file {file_path}")

    data = data.reshape(-1, dim + 1)
    if not np.all(data[:, 0] == dim):
        raise ValueError(f"Inconsistent vector dimensions found in {file_path}")

    return data[:, 1:].copy()
