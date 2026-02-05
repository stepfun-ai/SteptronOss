from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from loguru import logger
from tqdm import tqdm

from steptronoss.data.datasets.base_language_dataset import Dataset
from steptronoss.data.datasets.compile_dataset import CompiledDataset
from steptronoss.data.nextable import LazyUpdateNextable
from steptronoss.data.packing.non_truncation import PackingResult, pack
from steptronoss.data.samplers.base_sampler import (
    LoopedShuffleSampler,
    WeightedRandomSampler,
)


class MixedPackedDataloader(LazyUpdateNextable):
    def __init__(
        self,
        datasets: list[Dataset | CompiledDataset],
        epochs: list[float],
        max_length: int,
        oversize_policy: Literal["drop", "extend"] = "extend",
        transform: Callable | None = None,
    ):
        if len(datasets) == 0:
            raise ValueError("datasets cannot be empty.")
        if len(datasets) != len(epochs):
            raise ValueError("datasets and epochs must have the same length.")

        self.datasets = datasets
        self.ds_epochs = epochs
        self.transform = transform

        self.piece_order, self.packing_result = self._schedule_all(
            max_length=max_length, oversize_policy=oversize_policy
        )
        self._pack_ranges = self.packing_result.packed_sample_ranges
        if len(self._pack_ranges) == 0:
            raise ValueError("packed dataset is empty.")
        self.data_idx = 0
        self._pending_pack_idx: int | None = None

    def _schedule_all(
        self,
        max_length: int,
        oversize_policy: str = "drop",
    ) -> tuple[list[tuple[int, int]], PackingResult]:
        inter_domain_sampler: WeightedRandomSampler
        in_domain_samplers: list[LoopedShuffleSampler] = []
        _weights: list[float] = []

        for idx, (dataset, epoch) in enumerate(zip(self.datasets, self.ds_epochs)):
            size = len(dataset)
            if size <= 0:
                raise ValueError("Dataset is empty.")
            if float(epoch) <= 0:
                raise ValueError("Epoch must be positive.")
            if (
                isinstance(dataset, CompiledDataset)
                and dataset.sample_meta
                and isinstance(dataset.sample_meta[0], int)
                and dataset.sample_meta[0] == len(dataset[0]["tokens"])
            ):
                dataset.sizes = dataset.sample_meta
            else:
                logger.warning(
                    "Getting length of each sample, this might be slow...Use CompiledDataset "
                    "with length as meta for optimization. "
                    "```"
                    "compile_dataset(dataset, save_path, sample_meta_extractor=lambda x: len(x['tokens']))"
                    "```"
                )
                dataset.sizes = [len(x) for x in tqdm(dataset)]

            in_domain_samplers.append(LoopedShuffleSampler(size=size, base_seed=1234 + idx))

            weight = float(epoch) * float(size)  # use weight = N_sample x Epoch
            if weight <= 0:
                raise ValueError("Sampling weight must be positive.")
            _weights.append(weight)

        inter_domain_sampler = WeightedRandomSampler(
            size=len(self.datasets),
            base_seed=1234,
            weights=_weights,
        )

        scheduled_piece_order: list[tuple[int, int]] = []
        scheduled_piece_sizes: list[int] = []
        for _i in tqdm(range(int(sum(_weights))), desc="Sampling"):
            domain_idx = next(inter_domain_sampler)
            in_domain_idx = next(in_domain_samplers[domain_idx])
            scheduled_piece_order.append((domain_idx, in_domain_idx))
            scheduled_piece_sizes.append(self.datasets[domain_idx].sizes[in_domain_idx])

        packing_result = pack(scheduled_piece_sizes, max_len=max_length, oversize_policy=oversize_policy)
        if packing_result.num_droped:
            logger.warning(f"{packing_result.num_droped} sample droped for oversize!")
        return scheduled_piece_order, packing_result

        # dataset_sizes: list[list[int]] = [[len(x) for x in ds] for ds in self.datasets]

    def __len__(self):
        return len(self._pack_ranges)

    def get(self) -> list[Any]:
        if self._pending_pack_idx is None:
            self._pending_pack_idx = self.data_idx % len(self._pack_ranges)
        start, count = self._pack_ranges[self._pending_pack_idx]
        packed_items: list[Any] = []
        for domain_idx, in_domain_idx in self.piece_order[start : start + count]:
            packed_items.append(self.datasets[domain_idx][in_domain_idx])
        if self.transform:
            packed_items = self.transform(packed_items)
        return packed_items

    def update(self) -> None:
        if self._pending_pack_idx is None:
            self._pending_pack_idx = self.data_idx % len(self._pack_ranges)
        self._pending_pack_idx = None
        self.data_idx += 1

    def state_dict(self) -> dict[str, Any]:
        return {"data_idx": self.data_idx}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "data_idx" in state_dict:
            self.data_idx = state_dict["data_idx"]
        self._pending_pack_idx = None
