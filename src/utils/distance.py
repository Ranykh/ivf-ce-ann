"""Distance helpers shared across the project."""

from __future__ import annotations

import numpy as np


def pairwise_squared_l2(queries: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Compute pairwise squared L2 distances between two batches."""
    if queries.ndim != 2 or vectors.ndim != 2:
        raise ValueError("queries and vectors must be 2D arrays")
    if queries.shape[1] != vectors.shape[1]:
        raise ValueError("Input matrices must have the same feature dimension")

    # Use ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a dot b for efficiency.
    q_norms = np.sum(queries**2, axis=1)[:, None]
    v_norms = np.sum(vectors**2, axis=1)[None, :]
    cross = queries @ vectors.T
    return q_norms + v_norms - 2.0 * cross
