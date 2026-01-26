import pytest
from torch.utils.data import Dataset

from steptronoss.data.dataloader.packed_dataloader import MixedPackedDataloader

pytestmark = pytest.mark.cpu


class SizedDataset(Dataset):
    def __init__(self, sizes: list[int], tag: int):
        self._sizes = sizes
        self._tag = tag

    def __len__(self) -> int:
        return len(self._sizes)

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += len(self._sizes)
        if idx < 0 or idx >= len(self._sizes):
            raise IndexError("index out of range")
        return [self._tag] * self._sizes[idx]


def _build_loader():
    ds_a = SizedDataset([2, 3], tag=0)
    ds_b = SizedDataset([1, 4], tag=1)
    return MixedPackedDataloader(
        datasets=[ds_a, ds_b],
        epochs=[1.0, 1.0],
        max_length=5,
        oversize_policy="extend",
    )


def test_packed_dataloader_get_and_len():
    loader = _build_loader()

    expected = []
    for start, count in loader.packing_result.packed_sample_ranges:
        items = []
        total_len = 0
        for domain_idx, in_domain_idx in loader.piece_order[start : start + count]:
            item = loader.datasets[domain_idx][in_domain_idx]
            items.append(item)
            total_len += len(item)
        expected.append(items)
        assert total_len <= 5

    assert len(loader) == len(expected)

    for items in expected:
        assert loader.get() == items
        loader.update()

    assert loader.get() == expected[0]


def test_packed_dataloader_state_roundtrip():
    loader = _build_loader()
    for _ in range(3):
        next(loader)

    state = loader.state_dict()
    loader2 = _build_loader()
    loader2.load_state_dict(state)

    seq1 = [next(loader) for _ in range(6)]
    seq2 = [next(loader2) for _ in range(6)]
    assert seq1 == seq2
