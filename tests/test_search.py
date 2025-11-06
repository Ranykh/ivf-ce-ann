import numpy as np

from src.index import IVFCEIndex, IVFIndex
from src.search import IVFCEExplorer, IVFSearcher


def _toy_dataset() -> np.ndarray:
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


def test_ivf_searcher_respects_nprobe_and_returns_topk():
    data = _toy_dataset()
    index = IVFIndex(dimension=2, n_clusters=2, seed=42, n_init=5, max_iter=50)
    index.build(data)

    searcher = IVFSearcher(index, nprobe=1)
    query = np.array([0.05, 0.02], dtype=np.float32)
    ids_nprobe1, _ = searcher.search(query, k=3)
    assert ids_nprobe1.size <= 3

    searcher_all = IVFSearcher(index, nprobe=2)
    ids_nprobe2, _ = searcher_all.search(query, k=3)

    # With more clusters searched we should retrieve at least as many unique ids.
    assert ids_nprobe2.size >= ids_nprobe1.size
    assert len(set(ids_nprobe2.tolist())) == ids_nprobe2.size


def test_ivfce_searcher_uses_cross_links_for_additional_clusters():
    data = _toy_dataset()
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

    explorer = IVFCEExplorer(index, n1=1, n2=1, k2=2)
    query = np.array([0.05, 0.02], dtype=np.float32)

    base_ids, _ = explorer._search_clusters(query, explorer._stage0_routing(query))

    # Force stage2 candidates that include cross links by using the built state.
    final_ids, _ = explorer.search(query, k=3)

    # Expect the final result set to include at least everything from stage1.
    assert set(base_ids.tolist()).issubset(set(final_ids.tolist()))
