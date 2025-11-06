import numpy as np

from src.clustering import train_kmeans


def test_train_kmeans_returns_centroids_and_assignments():
    # Create two obvious clusters in 2D space.
    cluster_a = np.array([[0.0, 0.0], [0.1, 0.0], [-0.1, 0.0]], dtype=np.float32)
    cluster_b = np.array([[5.0, 5.0], [5.1, 5.0], [4.9, 5.0]], dtype=np.float32)
    data = np.vstack([cluster_a, cluster_b])

    result = train_kmeans(data, n_clusters=2, n_init=5, max_iter=50, seed=123)

    assert result.centroids.shape == (2, 2)
    # Each point should be assigned to exactly one cluster.
    assert result.assignments.shape == (data.shape[0],)

    # Check that points from the same group share the same assignment.
    assignments_a = result.assignments[: len(cluster_a)]
    assignments_b = result.assignments[len(cluster_a) :]
    assert len(set(assignments_a.tolist())) == 1
    assert len(set(assignments_b.tolist())) == 1

