import pytest

from src.index import available_preset_names, get_preset


def test_preset_files_are_discoverable():
    names = set(available_preset_names())
    assert {"A", "B", "C"}.issubset(names)

    preset_a = get_preset("a")
    assert preset_a.name == "A"
    assert preset_a.indexing.m_max == 8
    assert preset_a.indexing.k1 == 15
    assert preset_a.indexing.p_index == 5
    assert preset_a.dataset.name == "sift1m"
    assert preset_a.dataset.path == "data/"
    assert preset_a.clustering.n_clusters == 80
    assert preset_a.search.B_values == [4, 6, 8]
    assert preset_a.search.k2_values == [50, 100, 200]


def test_unknown_preset_raises_value_error():
    with pytest.raises(ValueError):
        get_preset("Z")
