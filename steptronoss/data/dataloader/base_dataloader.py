from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset

from steptronoss.data.nextable import LazyUpdateNextable
from steptronoss.data.samplers.base_sampler import BaseSampler


class BaseDataloader(LazyUpdateNextable):
    """Simple Nextable dataloader that indexes a dataset with a sampler."""

    def __init__(self, dataset: Dataset, sampler: BaseSampler) -> None:
        self.dataset = dataset
        self.sampler = sampler
        if not hasattr(self.sampler, "get") or not hasattr(self.sampler, "update"):
            raise TypeError("sampler must implement get() and update()")

    def __len__(self) -> int:
        return len(self.dataset)

    def get(self) -> Any:
        idx = self.sampler.get()
        return self.dataset[idx]

    def update(self) -> None:
        self.sampler.update()

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        if hasattr(self.sampler, "state_dict"):
            state["sampler"] = self.sampler.state_dict()
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        sampler_state = state_dict.get("sampler")
        if sampler_state is not None and hasattr(self.sampler, "load_state_dict"):
            self.sampler.load_state_dict(sampler_state)
