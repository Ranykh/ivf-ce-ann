import numpy as np

from experiments.utils import select_queries


def test_select_queries_returns_prefix_when_no_seed():
    queries = np.arange(20, dtype=np.float32).reshape(10, 2)
    ground_truth = np.arange(20, dtype=np.int32).reshape(10, 2)

    subset_q, subset_gt = select_queries(queries, ground_truth, 4)

    assert subset_q.shape == (4, 2)
    assert np.array_equal(subset_q, queries[:4])
    assert np.array_equal(subset_gt, ground_truth[:4])


def test_select_queries_seeded_sampling_matches_rng():
    queries = np.arange(40, dtype=np.float32).reshape(20, 2)
    ground_truth = np.arange(40, dtype=np.int32).reshape(20, 2)
    limit = 5
    seed = 13

    rng = np.random.default_rng(seed)
    expected_indices = rng.choice(queries.shape[0], size=limit, replace=False)

    sampled_q, sampled_gt = select_queries(
        queries, ground_truth, limit, seed=seed
    )

    assert np.array_equal(sampled_q, queries[expected_indices])
    assert np.array_equal(sampled_gt, ground_truth[expected_indices])
