"""Abstract base class for inverted file indices."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.index.metadata import BuildStats, IndexMetadata


class BaseIndex(ABC):
    """Shared interface for all ANN indices in the project."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.is_built = False
        self.build_stats: Optional[BuildStats] = None
        self.metadata: Optional[IndexMetadata] = None

    @abstractmethod
    def build(self, vectors: np.ndarray) -> None:
        """Build the index from the provided database vectors."""

    @abstractmethod
    def assign(self, vectors: np.ndarray) -> np.ndarray:
        """Assign each vector to the closest coarse centroid."""

    def save(self, path: str | Path) -> None:
        """Persist index state to disk."""
        raise NotImplementedError("save is not implemented for this index.")

    @classmethod
    def load(cls, path: str | Path) -> "BaseIndex":
        """Restore index state from disk."""
        raise NotImplementedError("load is not implemented for this index.")

    def _validate_input(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected vectors with dimension {self.dimension}, "
                f"got {vectors.shape[1]}"
            )

    def _common_state(self) -> Dict[str, Any]:
        """Return common metadata shared across subclasses."""
        return {"dimension": self.dimension, "is_built": self.is_built}
