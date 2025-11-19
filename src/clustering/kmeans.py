"""K-means clustering utilities for coarse quantizer training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "faiss is required for clustering. Install faiss-cpu from requirements.txt."
    ) from exc


@dataclass
class KMeansResult:
    """K-means centroids and assignment matrix returned by training."""
    centroids: np.ndarray
    assignments: np.ndarray


def train_kmeans(
    vectors: np.ndarray,
    *,
    n_clusters: int,
    n_init: int = 10,
    max_iter: int = 100,
    seed: Optional[int] = None,
) -> KMeansResult:
    """Train k-means using Faiss if available, otherwise scikit-learn."""
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array")
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    if vectors.shape[0] < n_clusters:
        raise ValueError("Number of vectors must be at least n_clusters")

    return _train_faiss_kmeans(
        vectors,
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        seed=seed,
    )


def _train_faiss_kmeans(
    vectors: np.ndarray,
    *,
    n_clusters: int,
    n_init: int,
    max_iter: int,
    seed: Optional[int],
) -> KMeansResult:
    """Run Faiss KMeans and return centroids plus cluster assignments."""
    dim = vectors.shape[1]
    kmeans = faiss.Kmeans(
        d=dim,
        k=n_clusters,
        niter=max_iter,
        nredo=n_init,
        verbose=False,
        seed=seed or 0,
    )
    kmeans.train(vectors)
    distances, assignments = kmeans.index.search(vectors, 1)
    # Faiss returns assignments as int64 column vector.
    assignments = assignments[:, 0].astype(np.int32)
    centroids = np.array(kmeans.centroids, dtype=np.float32)
    return KMeansResult(centroids=centroids, assignments=assignments)
