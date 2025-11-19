"""Abstract searcher interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class BaseSearcher(ABC):
    """Shared interface for ANN searchers."""

    def __init__(self, index_dimension: int) -> None:
        """Record the dimensionality all queries must adhere to."""
        self.dimension = index_dimension

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``k`` nearest vectors for the given query."""

    def _validate_query(self, query: np.ndarray) -> None:
        """Ensure a query is 1D and matches the index dimensionality."""
        if query.ndim != 1:
            raise ValueError("query must be a 1D array")
        if query.shape[0] != self.dimension:
            raise ValueError(
                f"Query dimension {query.shape[0]} does not match index dimension {self.dimension}"
            )
