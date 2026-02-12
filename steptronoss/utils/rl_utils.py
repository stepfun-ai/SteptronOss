import contextlib
import copy
import math
import os
import re
import shutil
import threading
import time
from collections import OrderedDict, defaultdict, deque
from functools import lru_cache, reduce
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from steptronoss.data.nextable import Nextable
from steptronoss.exp.rl import PackedPPOSamples, PPOCheckpointCfg, PPOSample

from .dist_utils import all_gather_object


class RunningMoments:
    def __init__(self):
        """
        Calculates the running mean and standard deviation of a data stream. Modified version of
        https://github.com/DLR-RM/stable-baselines3/blob/a6f5049a99a4c21a6f0bcce458ca3306cef310e0/stable_baselines3/common/running_mean_std.py
        """
        self.mean = 0
        self.std = 1
        self.var = 1
        self.count = 1e-24

    def update(self, xs: torch.Tensor):
        from steptronoss.core.parallel_state import PM

        # get xs_mean, xs_var, xs_count
        xs_count = torch.as_tensor(xs.numel(), device=xs.device)
        torch.distributed.all_reduce(xs_count, group=PM.group_of("DP"))
        xs_mean = xs.sum()
        torch.distributed.all_reduce(xs_mean, group=PM.group_of("DP"))
        xs_mean /= xs_count
        xs_var = ((xs - xs_mean) ** 2).sum()
        torch.distributed.all_reduce(xs_var, group=PM.group_of("DP"))
        xs_var /= xs_count

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta**2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += delta * xs_count / tot_count
        self.var = tot_sum / tot_count
        self.std = (self.var * tot_count / (tot_count - 1)).sqrt()
        self.count = tot_count

        return xs_mean, (xs_var * xs_count / (xs_count - 1)).sqrt()


class DistributedDictList:
    def __init__(self, dist_group=None):
        self.local_data = defaultdict(list)
        self.global_data = defaultdict(list)

        self._in_local = False

        self.group = dist_group

    @contextlib.contextmanager
    def in_local(self):
        self._in_local = True
        yield
        self.synchronize()
        self._in_local = False

    def __setitem__(self, key, value):
        return (self.local_data if self._in_local else self.global_data).__setitem__(key, value)

    def __getitem__(self, key):
        return (self.local_data if self._in_local else self.global_data).__getitem__(key)

    def synchronize(self):
        new_data = all_gather_object(self.local_data, group=self.group)
        self.local_data = defaultdict(list)
        for d in new_data:
            for k in d:
                self.global_data[k] += d[k]


class RaggedPPOSampleDumper:
    """
    unpack PackedPPOSamples to PPOSample, and then dump to msgpack file
    """

    def __init__(self, use_single_float: bool = True):
        self.use_single_float = use_single_float

    def _unpack_samples(self, packed_data: PackedPPOSamples, dump_sample_keys: list[str]) -> list[PPOSample]:
        """
        NOTE: cannot fully trust packed_data.samples:
        * change in ref_logprobs, actor_logprobs, advantages, returns, values are not reflected in packed_data.samples
        * however, we assume other attributes are good in this function
        """
        # TODO: hardcode packed_keys here
        packed_keys = [
            "ref_logprobs",
            "actor_logprobs",
            "advantages",
            "returns",
            "values",
        ]
        samples: list[PPOSample] = packed_data.samples
        if len(samples) == 0:
            logger.warning("No samples to unpack, return empty list")
            return []

        dump_packed_keys = [key for key in dump_sample_keys if key in packed_keys]
        if len(dump_packed_keys) == 0:
            return samples

        # unpack packed_keys logic
        action_seqlens: list[int] = []
        cu_seqlens: torch.Tensor = packed_data.cu_seqlens
        is_gen_mask: torch.Tensor = packed_data.is_gen_mask
        for item_idx in range(len(cu_seqlens) - 1):
            start_pos_idx, end_pos_idx = (
                cu_seqlens[item_idx].item(),
                cu_seqlens[item_idx + 1].item(),
            )
            # NOTE: assume is_gen_mask is of shape (1, S_t)
            action_seqlen = is_gen_mask[0][start_pos_idx:end_pos_idx].sum().item()
            action_seqlens.append(action_seqlen)

        action_seqlens_tensor = torch.tensor(action_seqlens)
        for key in dump_packed_keys:
            packed_item_tensor = packed_data[key]
            if packed_item_tensor is None:
                # NOTE: we just silently skip
                continue
            for item_idx in range(len(action_seqlens)):
                action_seqlen = action_seqlens_tensor[item_idx]
                start_action_pos_idx = action_seqlens_tensor[:item_idx].sum().item()
                end_action_pos_idx = start_action_pos_idx + action_seqlen
                item_tensor = packed_item_tensor[start_action_pos_idx:end_action_pos_idx]
                samples[item_idx][key] = item_tensor

        return samples

    def _convert_to_data_for_saving(self, samples: list[PPOSample], dump_sample_keys: list[str]) -> list[dict]:
        data = []
        for sample in samples:
            data_item = {}
            for key in dump_sample_keys:
                assert key in sample.__dict__, f"key {key} not found in sample!"
                # TODO: only assume one non python basic class: torch tensor
                data_item[key] = sample.__dict__[key]
            data_item["meta"] = sample.meta
            data.append(data_item)
        return data

    def __call__(
        self,
        samples: list[PackedPPOSamples],
        dump_sample_keys: list[str],
    ):
        unpacked_samples: list[PPOSample] = reduce(
            lambda acc, item: acc + self._unpack_samples(item, dump_sample_keys),
            samples,
            [],
        )
        unpacked_samples_to_save: list[dict] = self._convert_to_data_for_saving(unpacked_samples, dump_sample_keys)
        return unpacked_samples_to_save


def compute_gae(values: torch.Tensor, rewards: torch.Tensor, lambd=0.95, gamma=1.0):
    lastgaelam = 0
    advantages_reversed = []
    T = values.shape[-1]

    for t in reversed(range(T)):
        nextvalues = values[..., t + 1] if t < T - 1 else 0.0
        delta = rewards[..., t] + gamma * nextvalues - values[..., t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], -1)
    returns = advantages + values
    return advantages, returns


def compute_gae_opt(values, rewards, lambd=0.95, gamma=1.0, gae_coeff=None):
    nextvalues = torch.cat([values[..., 1:], torch.zeros_like(values[..., :1])], dim=-1)
    delta = rewards + gamma * nextvalues - values
    T = values.shape[-1]
    if gae_coeff is None:
        # do the for-loop computation on CPU to avoid cuda kernel launch
        delta = delta.cpu()
        advantages = torch.empty_like(delta, device="cpu")
        for t in reversed(range(T)):
            advantages[..., t] = delta[..., t] + gamma * lambd * (advantages[..., t + 1] if t < T - 1 else 0.0)
        advantages = advantages.cuda()
    else:
        # if the pre-computed gae coefficient matrix is provided, use matmul
        advantages = torch.matmul(delta, gae_coeff[:T, :T].cuda())
    returns = advantages + values
    return advantages, returns


class PersistentQueue:
    """
    Basically just a Queue with rollback support:
    - put(0, 1, 2): [2, 1, 0]
    - get(): [2, 1] -> 0
    - get(): [2] -> 1
    - ack(1)
    - kill()
    - restore(): [2, 0] # un-ack 0 return to queue.

    """

    def __init__(self):
        self.data = deque()
        # Must use OrderedDict here, when rollback, data return to queue with same order.
        self.pending = OrderedDict()

    def __repr__(self):
        data_rep, pending_rep = [], []
        irep, iclass = 0, type(None)
        for item in list(self.data) + [None]:
            if type(item) is not iclass:
                if irep:
                    data_rep.append(f"{iclass.__name__} x{irep}")
                iclass, irep = type(item), 0
            irep += 1

        irep, iclass = 0, type(None)
        for item in list(reversed(self.pending.values())) + [None]:
            if type(item) is not iclass:
                if irep:
                    pending_rep.append(f"{iclass.__name__} x{irep}")
                iclass, irep = type(item), 0
            irep += 1

        return f"{self.__class__.__name__}[{', '.join(data_rep)}]~[{', '.join(pending_rep)}]~>"

    def pop(self) -> Any:
        while not self.data:
            time.sleep(0.5)
        return self.data.pop()

    def get(self) -> tuple[str, Any]:
        sample = self.pop()
        sample_id = str(uuid4())
        self.pending[sample_id] = sample
        return sample_id, sample

    def put(self, *data):
        return self.data.extendleft(data)

    def ack(self, data_id):
        self.pending.pop(data_id, None)

    def state_dict(self):
        # data: [3, 2] -> 1, 0 # right early
        # pending: {'xx': 0, 'yy': 1} # left early
        return list(self.data) + list(self.pending.values())[::-1]

    def load_state_dict(self, state_dict):
        self.data = deque(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        self = cls()
        self.load_state_dict(state_dict)
        return self

    def __len__(self):
        return len(self.data)

    def empty(self):
        return len(self) == 0

    def all_done(self):
        """NOTE: this does not equal to len(self)==0, since we might have un-ack samples."""
        return not (self.data or self.pending)


class PersistentSource(PersistentQueue):
    """PersistentQueue with a Nextable as source, and disable put()"""

    def __init__(self, nextable: Nextable):
        super().__init__()
        self._source = nextable

    def pop(self):
        if self.data:
            sample = self.data.pop()
        else:
            sample = next(self._source)
        return sample

    def put(self, data):
        raise RuntimeError("Cannot put to Source!")

    def state_dict(self):
        return {
            "source": self._source.state_dict(),
            "data": super().state_dict(),
        }

    def load_state_dict(self, state_dict):
        self._source.load_state_dict(state_dict["source"])
        super().load_state_dict(state_dict["data"])


class PersistentDumper(PersistentQueue):
    def __init__(self, dump_dir: str = "./", dump_interval=1024):
        super().__init__()
        self.dump_dir = dump_dir
        self.dump_interval = dump_interval
        self.dump_counter = 0

        if os.path.exists(self.dump_dir):
            hist_dumps = [x for x in os.listdir(self.dump_dir) if x.startswith("pdump_")]
            if hist_dumps:
                hist_dump_ids = [int(re.search(r"pdump_(\d+)\.pt", x).group(1)) for x in hist_dumps]
                self.dump_counter = max(hist_dump_ids) + 1

    def get(self):
        raise RuntimeError("Cannot get from Dumper!")

    def put(self, data):
        self.data.appendleft(data)
        if len(self.data) >= self.dump_interval:
            torch.save(
                list(self.data),
                os.path.join(self.dump_dir, f"pdump_{self.dump_counter}.pt"),
            )
            self.dump_counter += 1
            self.data.clear()

    def ack(self, data_id):
        raise RuntimeError("Cannot ack sample to Dumper!")

    def state_dict(self):
        return list(self.data)

    def load_state_dict(self, state_dict):
        self.data = deque(state_dict)


class PersistentFlow(dict[str, PersistentQueue]):
    class Signal(dict):
        def __init__(self, name: str, **data):
            self.name = self["name"] = name
            super().__init__(**data)

    def __init__(
        self,
        meta: dict | None = None,
        **stages: dict[str, PersistentQueue],
    ):
        """Data flow that supports checkpointing. NOTE: One sample must be marked as finish
        by call ack().
        e.g.
        ```
        pflow = PersistentFlow(
            source = PersistentSource(nextable=dataloader),
            gen_ed = PersistentQueue(),
            dumper = PersistentDumper(dump_path="./"),
        )

        sample_id, sample = pflow["source"].get()
        trajectories = generate(sample)

        with pflow.lock:
            pflow["gen_ed"].put(*trajectories)
            pflow["source"].ack(sample_id)

        sample_id, sample = pflow["gen_ed"].get()
        train_on(sample)

        with pflow.lock:
            pflow["dumper"].put(sample)
            pflow["gen_ed"].ack(sample_id)

        ```

        """
        super().__init__(**stages)

        self.lock = threading.Lock()
        self.meta = meta or {}

    def state_dict(self):
        return {k: self[k].state_dict() for k in self} | {"meta": self.meta}

    def load_state_dict(self, state_dict: dict):
        self.meta = state_dict.pop("meta", {})
        return {k: self[k].load_state_dict(v) for k, v in state_dict.items()}
