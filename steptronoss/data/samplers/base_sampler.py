import heapq
from abc import abstractmethod

import torch
from loguru import logger

from steptronoss.data.nextable import LazyUpdateNextable


class BaseSampler(LazyUpdateNextable):
    size: int

    @abstractmethod
    def __next__(self) -> int:
        pass


class LoopedSequentialSampler(LazyUpdateNextable):
    """size=3: [0, 1, 2, 0, 1, 2, ...]"""

    def __init__(self, size: int):
        self.size = size
        self.data_idx = 0

    def update(self):
        self.data_idx += 1

    def get(self):
        return self.data_idx % self.size

    def state_dict(self):
        return {"data_idx": self.data_idx}

    def load_state_dict(self, state_dict):
        self.data_idx = state_dict["data_idx"]


class LoopedShuffleSampler(LazyUpdateNextable):
    """Define a sampler that can be used to sample from multiple epochs.
    size: the total number of samples to sample from.
    size = 3,  e.g. [0, 2, 1] + [1, 0, 2] + [2, 0, 1]
    """

    def __init__(
        self,
        size: int = 0,
        base_seed: int = 1234,
        same_order_for_each_epoch=False,
    ):
        self.size = size
        self.base_seed = base_seed
        self.same_order_for_each_epoch = same_order_for_each_epoch

        self.data_idx = 0
        self._idx_cur_epoch: list[int] = []

        self._reset_idx_cur_epoch()

    def get(self):
        return self._idx_cur_epoch[self.data_idx % self.size]

    def update(self):
        self.data_idx += 1
        if self.data_idx % self.size == 0:  # start of new epoch
            self._reset_idx_cur_epoch()

    def state_dict(self):
        return dict(size=self.size, data_idx=self.data_idx)

    def load_state_dict(self, state_dict):
        if "size" in state_dict:
            assert self.size == state_dict["size"]
        self.data_idx = state_dict["data_idx"]
        # NOTE: must ret idx cur epoch to make this correct when epoch > 0
        self._reset_idx_cur_epoch()

    def _reset_idx_cur_epoch(self):
        epoch = self.data_idx // self.size
        # logger.info(f"> reset idx cur epoch, Current Epoch: {epoch}, offset: {self.data_idx}")
        seed = self.base_seed
        if not self.same_order_for_each_epoch:
            seed += epoch
        rng = torch.Generator().manual_seed(seed)
        self._idx_cur_epoch = torch.randperm(self.size, generator=rng).tolist()


class WeightedRandomSampler(LazyUpdateNextable):
    """Sample with balanced history via argmin(yields) and seed-specific randomness."""

    def __init__(
        self,
        size: int = 0,
        base_seed: int = 1234,
        weights=None,
    ):
        self.size = size
        self.base_seed = base_seed

        self.data_idx = 0
        self._pending_idx = None
        self._rng = torch.Generator().manual_seed(int(self.base_seed))

        self._weights = (
            torch.ones(self.size, dtype=torch.float32)
            if weights is None
            else torch.as_tensor(weights, dtype=torch.float32)
        )
        weights = self._weights
        if weights.numel() != self.size:
            raise ValueError(f"weights length ({weights.numel()}) must match size ({self.size})")
        if torch.any(weights <= 0):
            raise ValueError("weights must be positive")

        self._counts = torch.zeros(self.size, dtype=torch.float32)
        self._inv_weights = (1.0 / self._weights).tolist()
        self._scores = [0.0 for _ in range(self.size)]
        self._heap = [(0.0, idx) for idx in range(self.size)]
        heapq.heapify(self._heap)

    def _select_idx(self) -> int:
        while self._heap:
            score, idx = heapq.heappop(self._heap)
            if score == self._scores[idx]:
                return int(idx)
        # Heap should never be empty; rebuild defensively.
        self._heap = [(score, idx) for idx, score in enumerate(self._scores)]
        heapq.heapify(self._heap)
        score, idx = heapq.heappop(self._heap)
        return int(idx)

    def get(self) -> int:
        if self._pending_idx is None:
            self._pending_idx = self._select_idx()
        return self._pending_idx

    def update(self):
        idx = self._pending_idx if self._pending_idx is not None else self._select_idx()
        self._pending_idx = None
        self._counts[idx] += 1
        self._scores[idx] += self._inv_weights[idx]
        heapq.heappush(self._heap, (self._scores[idx], idx))
        self.data_idx += 1

    def state_dict(self):
        return dict(
            size=self.size,
            data_idx=self.data_idx,
            counts=self._counts.tolist(),
            rng_state=self._rng.get_state().tolist(),
        )

    def load_state_dict(self, state_dict):
        if "size" in state_dict:
            assert self.size == state_dict["size"]
        self.data_idx = state_dict["data_idx"]
        counts = state_dict.get("counts")
        if counts is None:
            self._counts = torch.zeros(self.size, dtype=torch.float32)
        else:
            if len(counts) != self.size:
                raise ValueError(f"counts length ({len(counts)}) must match size ({self.size})")
            self._counts = torch.tensor(counts, dtype=torch.float32)
        self._scores = [float(self._counts[i]) * self._inv_weights[i] for i in range(self.size)]
        self._heap = [(self._scores[i], i) for i in range(self.size)]
        heapq.heapify(self._heap)
        rng_state = state_dict.get("rng_state")
        if rng_state is None:
            self._rng.manual_seed(int(self.base_seed))
        else:
            self._rng.set_state(torch.tensor(rng_state, dtype=torch.uint8))
        self._pending_idx = None
