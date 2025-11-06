import numpy as np

from src.index import IVFCEIndex, IVFIndex
from src.search import IVFCEExplorer, IVFSearcher


def _toy_vectors():
    return np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [1.0, 0.0],
            [1.1, 0.0],
            [1.0, 0.1],
        ],
        dtype=np.float32,
    )


def test_ivf_index_assignment_and_search():
    data = _toy_vectors()
    index = IVFIndex(dimension=2, n_clusters=2, seed=42, n_init=5, max_iter=50)
    index.build(data)

    assert index.centroids.shape == (2, 2)
    assert index.database.shape == data.shape

    # Assign a query near the first cluster.
    assignments = index.assign(np.array([[0.05, 0.02]], dtype=np.float32))
    assert assignments.shape == (1,)

    searcher = IVFSearcher(index, nprobe=2)
    ids, distances = searcher.search(np.array([0.05, 0.02], dtype=np.float32), k=3)
    assert ids.shape == (3,)
    assert np.isfinite(distances).all()


def test_ivfce_cross_links_and_search():
    data = _toy_vectors()
    index = IVFCEIndex(
        dimension=2,
        n_clusters=2,
        k1=3,
        m_max=2,
        p_index=2,
        seed=0,
        n_init=5,
        max_iter=50,
    )
    index.build(data)

    assert index.cross_links, "Expected non-empty cross-link mapping."

    explorer = IVFCEExplorer(index, n1=1, n2=1, k2=2)
    ids, distances = explorer.search(np.array([0.05, 0.02], dtype=np.float32), k=3)

    assert ids.shape[0] <= 3
    assert np.isfinite(distances).all()
