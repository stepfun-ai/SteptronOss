"""Compile Dataset instances into indexed binary shards."""

from __future__ import annotations

import os
import pickle
from collections.abc import Callable
from typing import Any

import megfile
import torch.multiprocessing as mp
from torch.utils.data import Dataset
from tqdm import tqdm

from steptronoss.data.utils.indexed_bin_file import IndexedBinReader, IndexedBinWriter


def _compile_shard(
    dataset: Dataset,
    total: int,
    shard_id: int,
    num_shards: int,
    shard_path: str,
    sample_meta_extractor: Callable | None,
    progress: Any | None = None,
):
    megfile.smart_makedirs(shard_path, exist_ok=True)
    sample_meta_extractor = sample_meta_extractor or (lambda x: None)
    sample_meta = []
    with IndexedBinWriter(shard_path) as writer:
        for idx in range(shard_id, total, num_shards):
            data = dataset[idx]
            sample_meta.append(sample_meta_extractor(data))
            writer.write(pickle.dumps(data))
            if progress is not None:
                progress.update(1)
    return shard_id, sample_meta


def _shard_size(total: int, shard_id: int, num_shards: int) -> int:
    if total == 0:
        return 0
    if shard_id >= total:
        return 0
    return (total - shard_id + num_shards - 1) // num_shards


def _assign_sample_meta(
    sample_meta: list[Any],
    shard_meta: list[Any],
    total: int,
    shard_id: int,
    num_shards: int,
) -> None:
    for offset, idx in enumerate(range(shard_id, total, num_shards)):
        sample_meta[idx] = shard_meta[offset]


def _compile_shard_worker(
    dataset: Dataset,
    total: int,
    shard_id: int,
    num_shards: int,
    shard_path: str,
    sample_meta_extractor: Callable | None,
    output_queue: mp.Queue,
) -> None:
    result = _compile_shard(dataset, total, shard_id, num_shards, shard_path, sample_meta_extractor)
    output_queue.put(result)


def compile_dataset(
    dataset: Dataset,
    save_path: str,
    num_workers: int | None = None,
    sample_meta_extractor: Callable[[Any], Any] | None = None,
) -> str:
    """Compile a dataset into one or more indexed binary shards.
    sample_meta_extractor: extract sample-wise info from each sample,
    store it in CompiledDataset.sample_meta: list
    """
    assert "fork" in mp.get_all_start_methods(), "compile_dataset requires fork start method"

    total = len(dataset)

    megfile.smart_makedirs(save_path, exist_ok=True)

    num_workers = num_workers or os.cpu_count() or 1
    num_workers = max(1, min(num_workers, total)) if total > 0 else 1
    num_shards = num_workers

    shards: list[dict[str, Any]] = []
    tasks: list[tuple[int, str]] = []

    for shard_id in range(num_shards):
        shard_name = f"shard-{shard_id:05d}"
        shard_path = megfile.smart_path_join(save_path, shard_name)
        shards.append({"path": shard_name, "size": _shard_size(total, shard_id, num_shards)})
        tasks.append((shard_id, shard_path))

    progress = tqdm(total=total, desc="Compiling")
    sample_meta = [None] * total
    try:
        if num_workers == 1:
            for shard_id, shard_path in tasks:
                shard_id, shard_meta = _compile_shard(
                    dataset,
                    total,
                    shard_id,
                    num_shards,
                    shard_path,
                    sample_meta_extractor,
                    progress=progress,
                )
                _assign_sample_meta(sample_meta, shard_meta, total, shard_id, num_shards)
        else:
            ctx = mp.get_context("fork")
            error_queue: mp.Queue = ctx.Queue()
            active: list[tuple[mp.Process, int]] = []
            for shard_id, shard_path in tasks:
                proc = ctx.Process(
                    target=_compile_shard_worker,
                    args=(
                        dataset,
                        total,
                        shard_id,
                        num_shards,
                        shard_path,
                        sample_meta_extractor,
                        error_queue,
                    ),
                )
                proc.start()
                active.append(proc)

            errors = []
            for _i in range(len(active)):
                output = error_queue.get()
                if isinstance(output, Exception):
                    errors.append(output)
                else:
                    shard_id, shard_info = output
                    _assign_sample_meta(sample_meta, shard_info, total, shard_id, num_shards)
                    progress.update(len(shard_info))
            if errors:
                raise RuntimeError(f"Shard compilation errors: {errors}")
            progress.set_description("Success")
            for proc in active:
                proc.join()

    finally:
        progress.close()
    manifest = {
        "length": total,
        "shards": shards,
        "sample_meta": sample_meta,
    }
    manifest_path = megfile.smart_path_join(save_path, "compiled_dataset.pkl")
    with megfile.smart_open(manifest_path, "wb") as f:
        pickle.dump(manifest, f)

    return save_path


class CompiledDataset(Dataset):
    path: str
    sample_meta: list

    def __init__(self, path: str) -> None:
        self.path = path
        self._readers: list[IndexedBinReader] = []
        self._shard_sizes: list[int] = []

        manifest_path = megfile.smart_path_join(path, "compiled_dataset.pkl")
        assert megfile.smart_exists(manifest_path), f"manifest file not found, is '{path}' a compiled dataset?"
        with megfile.smart_open(manifest_path, "rb") as f:
            manifest = pickle.load(f)

        shards = manifest.get("shards", [])
        for shard in shards:
            shard_path = megfile.smart_path_join(path, shard["path"])
            reader = IndexedBinReader(shard_path)
            self._readers.append(reader)
            self._shard_sizes.append(int(shard["size"]))
        self.sample_meta = manifest["sample_meta"]
        self._num_shards = len(self._shard_sizes)

        self._length = sum(self._shard_sizes)

    def __len__(self) -> int:
        return self._length

    def _locate(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += self._length
        if idx < 0 or idx >= self._length:
            raise IndexError("index out of range")
        shard_idx = idx % self._num_shards
        local_idx = idx // self._num_shards
        if local_idx >= self._shard_sizes[shard_idx]:
            raise IndexError("index out of range")
        return shard_idx, local_idx

    def __getitem__(self, idx: int) -> Any:
        shard_idx, local_idx = self._locate(idx)
        data = self._readers[shard_idx][local_idx]
        return pickle.loads(data)

    def close(self) -> None:
        for reader in self._readers:
            reader.close()

    def __del__(self) -> None:
        self.close()
