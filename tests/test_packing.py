import pytest

from steptronoss.data.packing.non_truncation import PackingResult, pack

pytestmark = pytest.mark.cpu


def test_pack_drop_policy():
    result = pack([2, 3, 5, 7, 1], max_len=6, oversize_policy="drop")
    assert isinstance(result, PackingResult)
    assert result.num_packed_samples == 3
    assert result.num_droped == 1
    assert result.packed_sample_ranges == [(0, 2), (2, 1), (4, 1)]


def test_pack_extend_policy():
    result = pack([2, 3, 5, 7, 1], max_len=6, oversize_policy="extend")
    assert isinstance(result, PackingResult)
    assert result.num_packed_samples == 4
    assert result.num_droped == 0
    assert result.packed_sample_ranges == [(0, 2), (2, 1), (3, 1), (4, 1)]


def test_pack_empty():
    result = pack([], max_len=6, oversize_policy="drop")
    assert isinstance(result, PackingResult)
    assert result.num_packed_samples == 0
    assert result.num_droped == 0
    assert result.packed_sample_ranges == []
