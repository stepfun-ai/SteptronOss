import pytest
import torch.multiprocessing as mp
from torch.utils.data import Dataset

from steptronoss.data.datasets.compile_dataset import CompiledDataset, compile_dataset

pytestmark = pytest.mark.cpu


class ListDataset(Dataset):
    def __init__(self, items: list[dict]) -> None:
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        return self._items[idx]


class TemplateDataset(Dataset):
    def __init__(self, items: list[dict], template) -> None:
        self._items = items
        self._template = template

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        return self._template(self._items[idx])


def test_compiled_dataset_roundtrip(tmp_path):
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("compile_dataset requires fork")

    items = [
        {"id": 0, "text": "item-0", "payload": [0, 1]},
        {"id": 1, "text": "item-1", "payload": [1, 2]},
        {"id": 2, "text": "item-2", "payload": [2, 3]},
        {"id": 3, "text": "item-3", "payload": [3, 4]},
    ]
    dataset = ListDataset(items)
    save_path = tmp_path / "compiled"

    output = compile_dataset(
        dataset,
        str(save_path),
        num_workers=1,
        sample_meta_extractor=lambda x: x["id"],
    )
    assert output == str(save_path)

    compiled = CompiledDataset(str(save_path))
    assert len(compiled) == len(items)
    assert compiled[0] == items[0]
    assert compiled[2] == items[2]
    assert compiled[-1] == items[-1]
    assert compiled.sample_meta == [item["id"] for item in items]

    with pytest.raises(IndexError):
        _ = compiled[len(items)]

    compiled.close()


def test_compiled_dataset_template_error(tmp_path):
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("compile_dataset requires fork")

    def bad_template(_item: dict) -> dict:
        raise ValueError("bad template")

    dataset = TemplateDataset([{"id": 0}], template=bad_template)
    save_path = tmp_path / "compiled"

    with pytest.raises(ValueError, match="bad template"):
        compile_dataset(dataset, str(save_path), num_workers=1)
