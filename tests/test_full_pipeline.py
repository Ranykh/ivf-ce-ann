import numpy as np
import pytest

from data.dataset_loader import load_dataset
from src.index import IVFCEIndex, IVFIndex
from src.search import IVFCEExplorer, IVFSearcher


def _recall_at_k(ground_truth: np.ndarray, predictions: np.ndarray, k: int) -> float:
    hits = 0
    limit = min(k, predictions.shape[0], ground_truth.shape[0])
    gt_set = set(int(idx) for idx in ground_truth[:limit])
    for cand in predictions[:limit]:
        if int(cand) in gt_set:
            hits += 1
    return hits / limit if limit else 0.0


@pytest.mark.integration
def test_full_pipeline_ivf_and_ivfce():
    try:
        base, queries, ground_truth = load_dataset("sift1m", "data")
    except FileNotFoundError:
        pytest.skip("SIFT1M dataset not available; skipping integration test.")

    # Use a manageable subset for quick testing.
    base_subset = base[:2000]
    queries_subset = queries[:5]
    ground_truth_subset = ground_truth[:5]

    dim = base_subset.shape[1]
    n_clusters = 16

    ivf_index = IVFIndex(
        dimension=dim,
        n_clusters=n_clusters,
        n_init=5,
        max_iter=50,
        seed=123,
    )
    ivf_index.build(base_subset)

    ivf_searcher = IVFSearcher(ivf_index, nprobe=4)

    ivfce_index = IVFCEIndex(
        dimension=dim,
        n_clusters=n_clusters,
        k1=15,
        m_max=8,
        p_index=8,
        n_init=5,
        max_iter=50,
        seed=123,
    )
    ivfce_index.build(base_subset)

    ivfce_searcher = IVFCEExplorer(ivfce_index, n1=2, n2=3, k2=10)

    for query, gt_neighbors in zip(queries_subset, ground_truth_subset):
        ivf_ids, _ = ivf_searcher.search(query, k=10)
        ivf_recall = _recall_at_k(gt_neighbors, ivf_ids, k=10)
        assert 0.0 <= ivf_recall <= 1.0

        ivfce_ids, _ = ivfce_searcher.search(query, k=10)
        ivfce_recall = _recall_at_k(gt_neighbors, ivfce_ids, k=10)
        assert 0.0 <= ivfce_recall <= 1.0

        # IVF-CE should not perform worse than IVF on this tiny subset.
        assert ivfce_recall >= ivf_recall
