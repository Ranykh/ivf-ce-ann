import numpy as np

from src.index import IVFIndex, IVFCEIndex, CrossLink, IndexMetadata


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


def test_ivf_index_builds_inverted_lists():
    data = _toy_dataset()
    index = IVFIndex(dimension=2, n_clusters=2, seed=42, n_init=5, max_iter=50)
    index.build(data)

    assert index.centroids.shape == (2, 2)
    total_in_list = sum(len(invl.ids) for invl in index.inverted_lists.values())
    assert total_in_list == data.shape[0]

    assignments = index.assign(np.array([[0.05, 0.05]], dtype=np.float32))
    assert assignments.shape == (1,)


def test_ivfce_cross_links_are_built():
    data = _toy_dataset()
    index = IVFCEIndex(
        dimension=2,
        n_clusters=2,
        k1=3,
        m_max=2,
        p_index=2,
        seed=123,
        n_init=5,
        max_iter=50,
    )
    index.build(data)

    # Ensure at least one vector has cross-links computed.
    assert any(index.cross_links.values())


def test_ivfce_index_persistence_roundtrip(tmp_path):
    data = _toy_dataset()
    index = IVFCEIndex(
        dimension=2,
        n_clusters=2,
        k1=3,
        m_max=2,
        p_index=2,
        seed=123,
        n_init=5,
        max_iter=50,
    )
    index.build(data)
    index.metadata = IndexMetadata.create(
        config_name="TEST",
        dataset_name="toy",
        dataset_path="data/toy",
        dataset_size=data.shape[0],
        n_clusters=2,
        dimension=2,
        m_max=2,
        k1=3,
        p_index=2,
        seed=123,
        build_stats=index.build_stats,
    )

    path = tmp_path / "ivfce.idx"
    index.save(path)

    restored = IVFCEIndex.load(path)
    assert restored.is_built
    assert restored.metadata is not None
    assert restored.metadata.config_name == "TEST"
    assert np.allclose(restored.centroids, index.centroids)
    assert restored.cross_links.keys() == index.cross_links.keys()
