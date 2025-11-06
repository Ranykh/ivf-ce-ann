import numpy as np
import pytest

from data.dataset_loader import load_dataset, load_sift1m
from data.utils import read_bvecs, read_fvecs, read_ivecs


def _write_fvecs(path, vectors):
    dim = vectors.shape[1]
    with open(path, "wb") as handle:
        for vec in vectors:
            np.array([dim], dtype=np.int32).tofile(handle)
            vec.astype(np.float32).tofile(handle)


def _write_bvecs(path, vectors):
    dim = vectors.shape[1]
    with open(path, "wb") as handle:
        for vec in vectors:
            np.array([dim], dtype=np.int32).tofile(handle)
            vec.astype(np.uint8).tofile(handle)


def _write_ivecs(path, vectors):
    dim = vectors.shape[1]
    with open(path, "wb") as handle:
        for vec in vectors:
            np.array([dim], dtype=np.int32).tofile(handle)
            vec.astype(np.int32).tofile(handle)


def test_read_vector_formats_roundtrip(tmp_path):
    base = np.array([[0.0, 1.0], [2.5, -3.5]], dtype=np.float32)
    bvec = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    ivec = np.array([[7, 8], [9, 10]], dtype=np.int32)

    f_path = tmp_path / "sample.fvecs"
    b_path = tmp_path / "sample.bvecs"
    i_path = tmp_path / "sample.ivecs"

    _write_fvecs(f_path, base)
    _write_bvecs(b_path, bvec)
    _write_ivecs(i_path, ivec)

    loaded_f = read_fvecs(f_path)
    loaded_b = read_bvecs(b_path)
    loaded_i = read_ivecs(i_path)

    assert np.allclose(loaded_f, base)
    assert np.array_equal(loaded_b, bvec)
    assert np.array_equal(loaded_i, ivec)


def test_load_sift1m_reads_expected_files(tmp_path):
    root = tmp_path / "data_root"
    dataset_dir = root / "sift1m"
    dataset_dir.mkdir(parents=True)

    base = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    ground_truth = np.array([[5, 6]], dtype=np.int32)

    _write_fvecs(dataset_dir / "sift_base.fvecs", base)
    _write_fvecs(dataset_dir / "sift_query.fvecs", queries)
    _write_ivecs(dataset_dir / "sift_groundtruth.ivecs", ground_truth)

    loaded_base, loaded_queries, loaded_gt = load_sift1m(root)

    assert np.allclose(loaded_base, base)
    assert np.allclose(loaded_queries, queries)
    assert np.array_equal(loaded_gt, ground_truth)


def test_load_dataset_dispatches_to_known_dataset(tmp_path):
    root = tmp_path / "data_root"
    dataset_dir = root / "sift1m"
    dataset_dir.mkdir(parents=True)

    vectors = np.array([[0.0]], dtype=np.float32)
    gt = np.array([[0]], dtype=np.int32)

    _write_fvecs(dataset_dir / "sift_base.fvecs", vectors)
    _write_fvecs(dataset_dir / "sift_query.fvecs", vectors)
    _write_ivecs(dataset_dir / "sift_groundtruth.ivecs", gt)

    base, queries, ground_truth = load_dataset("sift1m", root)
    assert base.shape == (1, 1)
    assert queries.shape == (1, 1)
    assert ground_truth.shape == (1, 1)


def test_load_dataset_unknown_name_raises(tmp_path):
    with pytest.raises(ValueError):
        load_dataset("unknown", tmp_path)
