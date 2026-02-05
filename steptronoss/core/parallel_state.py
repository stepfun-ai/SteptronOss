# Copyright (c) 2025, STEPFUN CORPORATION. All rights reserved.

"""Model and data parallel groups."""

import atexit
import operator
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed
import torch.distributed as dist

from steptronoss.exp.base_exp import ParallelConfig

from .utils import GlobalMemoryBuffer, _get_rng_state, _set_rng_seed, _set_rng_state

_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = 1

_GLOBAL_MEMORY_BUFFER = None


@dataclass
class ParallelGroups:
    """A group of process group under same parallel setting."""

    groups: list[torch.distributed.ProcessGroup]
    ranks: list[list[int]]
    valid_replicas: int

    group: torch.distributed.ProcessGroup
    my_group_rank: int


class ParallelManager:
    def __init__(self):

        self.world_size: int = 1
        self.world_rank: int = 0

        self.timeout_min = 10

        self.parallels: dict[str, ParallelGroups] = {}
        self.all_parallels: dict[str, dict[str, ParallelGroups]] = {}
        self.rng_states: dict[str, tuple] = {}

        self._all_groups: dict[str, torch.distributed.ProcessGroup] = {}

        self._stack = []
        self._cur_cfg: ParallelConfig = None
        self._rng_seeds: dict[str, int] = {}

    def _initialize_torch_dist(self, backend="nccl"):
        if not dist.is_initialized():
            # Manually set the device ids.
            if (n := torch.cuda.device_count()) > 0:
                torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")) % n)
            # set default
            os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "127.0.0.1")
            os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "23456")
            dist.init_process_group(
                backend=backend,
                init_method=None,
                world_size=int(os.getenv("WORLD_SIZE", "1")),
                rank=int(os.getenv("RANK", "0")),
                timeout=timedelta(minutes=self.timeout_min),
            )
            atexit.register(torch.distributed.destroy_process_group)

    def _new_group(self, ranks: list[int], **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = timedelta(minutes=self.timeout_min)
        key = str(list(ranks)) + str(kwargs.items())
        if key not in self._all_groups:
            self._all_groups[key] = torch.distributed.new_group(ranks, **kwargs)
        return self._all_groups[key]

    def initialize(self, backend="nccl", timeout_min: int = 300):
        self.timeout_min = timeout_min

        self._initialize_torch_dist(backend=backend)

        self.world_size = dist.get_world_size()
        self.world_rank = dist.get_rank()

    def define_parallel(self, pattern: str, **kwargs) -> list[list[int]]:
        from functools import reduce

        from einops import rearrange
        from einops.parsing import ParsedExpression

        src, dst = [x.strip() for x in pattern.split("->")]

        src_comp = ParsedExpression(src).composition
        dst_comp = ParsedExpression(dst).composition
        assert len(src_comp) == 1, "Source must be 1 dimensional!"
        assert len(dst_comp) == 2, "Destination must be 2 dimensional!"

        def _flatten(comp):
            return reduce(operator.iadd, [_flatten(i) for i in comp], []) if isinstance(comp, list) else [comp]

        src_div = reduce(lambda a, b: a * b, [kwargs.get(k, 1) for k in _flatten(src_comp)], 1)

        world_size = self.world_size

        world_size = (world_size // src_div) * src_div

        assert world_size, (
            f"Cannot define '{pattern}' with {kwargs}! At least {src_div} ranks required, got {self.world_size}"
        )
        kwargs = {k: v for k, v in kwargs.items() if k in _flatten(src_comp)}

        ranks = rearrange(torch.arange(world_size), f"{src} -> {dst}", **kwargs).tolist()
        if world_size != self.world_size:
            ranks.append(torch.arange(world_size, self.world_size).tolist())

        return ranks

    def new_parallel(self, name: str, ranks: list[list[int]], **kwargs):
        if name in self.parallels:
            assert ranks == self.parallels[name].ranks
        else:
            groups, my_group_rank, full_size, valid_replicas = [], None, None, 0
            for gid, r in enumerate(ranks):
                full_size = full_size or len(r)
                if full_size == len(r):
                    valid_replicas += 1
                g = self._new_group(r, use_local_synchronization=True, **kwargs)
                if self.world_rank in r:
                    my_group_rank = gid
                groups.append(g)
            self.parallels[name] = ParallelGroups(
                groups=groups,
                ranks=ranks,
                group=groups[my_group_rank],
                my_group_rank=my_group_rank,
                valid_replicas=valid_replicas,
            )
        dist.barrier()
        return self.parallels[name]

    def set_mesh(self, parallel_cfg: ParallelConfig):
        self._cur_cfg = parallel_cfg
        identity = str(self._cur_cfg)
        if identity not in self.all_parallels:
            self.parallels = {}
            parallel_dict = parallel_cfg.build_parallel()
            for name, ranks in parallel_dict.items():
                self.new_parallel(name, ranks)
            self.all_parallels[identity] = self.parallels
        self.parallels = self.all_parallels[identity]

        # Vritual Parallels
        if hasattr(parallel_cfg, "virtual_pipeline_model_parallel_size"):
            set_vpp_size(parallel_cfg.virtual_pipeline_model_parallel_size)
        return self.parallels

    @contextmanager
    def use_mesh(self, parallel_cfg: ParallelConfig):
        self._stack.append(self._cur_cfg)
        self.set_mesh(parallel_cfg)
        yield
        self.set_mesh(self._stack.pop(-1))

    def register_rng(self, name: str, seed: int, diff_across: list[str]):
        if diff_across is None:
            diff_across = []
        offset = 0
        stride = 1
        for pname in diff_across:
            assert pname in self.parallels, f"parallel group '{pname}' not initialized"
            offset += self.rank_in(pname) * stride
            stride *= self.size_of(pname)
        rng_seed = int(seed) + offset
        if name in self._rng_seeds:
            if self._rng_seeds[name] != rng_seed:
                raise RuntimeError(f"RNG '{name}' already registered with a different seed")
            return
        self._rng_seeds[name] = rng_seed

        # Store old
        old_state = _get_rng_state()

        # Set New
        _set_rng_seed(rng_seed)

        # Store New
        self.rng_states[name] = _get_rng_state()

        # Restore old
        _set_rng_state(old_state)

    @contextmanager
    def use_rng(self, name: str):
        if name not in self.rng_states:
            raise RuntimeError(f"RNG '{name}' is not registered")

        raw_rng_state = _get_rng_state()

        _set_rng_state(self.rng_states[name])

        yield
        self.rng_states[name] = _get_rng_state()

        _set_rng_state(raw_rng_state)

    def get_all_rng(self):
        current_state = _get_rng_state()
        stashed_state = self.rng_states.copy()
        return current_state, stashed_state

    def set_all_rng(self, saved: tuple[tuple, dict]):
        current_state, stashed_state = saved
        _set_rng_state(current_state)
        self.rng_states.clear()
        self.rng_states.update(stashed_state)

    # Parallel APIs
    def size_of(self, name: str) -> int:
        assert name in self.parallels
        return dist.get_world_size(self.parallels[name].group)

    def rank_in(self, name: str) -> int:
        assert name in self.parallels
        groups = self.parallels[name]
        return groups.ranks[groups.my_group_rank].index(self.world_rank)

    def i_am(self, name: str, rank: int) -> bool:
        rank = rank % self.size_of(name)
        return rank == self.rank_in(name)

    def ranks_of(self, name: str) -> list[int]:
        assert name in self.parallels
        parallel = self.parallels[name]
        return parallel.ranks[parallel.my_group_rank]

    def group_of(self, name: str) -> dist.ProcessGroup:
        assert name in self.parallels
        return self.parallels[name].group

    def rank_of(self, name: str) -> int:
        assert name in self.parallels
        return self.parallels[name].my_group_rank

    def replica_of(self, name: str) -> list[int]:
        assert name in self.parallels
        return self.parallels[name].valid_replicas

    def world_ranks_of(self, name: str) -> list[list[int]]:
        assert name in self.parallels
        return self.parallels[name].ranks

    def __repr__(self):
        text = ["ParallelManager("]
        for pname in self.parallels:
            text.append(
                f"    {pname} = I({self.rank_in(pname)}/{self.size_of(pname)}) E({self.rank_of(pname)}/{self.replica_of(pname)})"
            )
        text.append(")")
        return "\n".join(text)


PM = ParallelManager()


def is_unitialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return "DP" not in PM.parallels


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    return "TP" in PM.parallels and "PP" in PM.parallels and "DP" in PM.parallels


def set_vpp_size(world_size):
    """Set the pipeline model parallel size"""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_pipeline_model_parallel_world_size(ignore_virtual=True):
    """Return world size for the pipeline model parallel group."""
    pp_size = PM.size_of("PP")
    if ignore_virtual:
        return pp_size
    else:
        vp_size = get_vpp_size() or 1
        return vp_size * pp_size


def get_pipeline_model_parallel_rank(ignore_virtual=True):
    """Return my rank for the pipeline model parallel group."""
    rank = PM.rank_in("PP")
    if ignore_virtual:
        return rank
    else:
        vp_rank = get_vpp_rank() or 0
        return vp_rank * get_pipeline_model_parallel_world_size() + rank


def get_pipeline_model_parallel_next_rank():
    """Return the global rank that follows the caller in the pipeline"""
    ranks = PM.ranks_of("PP")
    return ranks[(ranks.index(PM.world_rank) + 1) % PM.size_of("PP")]


def get_pipeline_model_parallel_prev_rank():
    """Return the global rank that preceeds the caller in the pipeline"""
    ranks = PM.ranks_of("PP")
    return ranks[(ranks.index(PM.world_rank) - 1) % PM.size_of("PP")]


def is_pipeline_first_stage(ignore_virtual=False):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        if get_vpp_size() is not None and get_vpp_rank() != 0:
            return False
    return get_pipeline_model_parallel_rank() == 0


def is_pipeline_last_stage(ignore_virtual=False):
    """Return True if in the last pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        virtual_pipeline_model_parallel_world_size = get_vpp_size()
        if virtual_pipeline_model_parallel_world_size is not None and get_vpp_rank() != (
            virtual_pipeline_model_parallel_world_size - 1
        ):
            return False
    return get_pipeline_model_parallel_rank() == (get_pipeline_model_parallel_world_size() - 1)


# def get_tp_info_for_param(param):
#     """Return (group, world_size, rank) for the tensor-parallel of a parameter.

#     If the parameter is an expert parameter (expert_model_parallel=True),
#     this returns the expert tensor-parallel (ETP) info; otherwise it returns
#     the standard tensor-parallel (TP) info.
#     """
#     is_expert = getattr(param, "expert_model_parallel", False)
#     if is_expert:
#         return (
#             get_expert_tensor_parallel_group(),
#             get_expert_tensor_parallel_world_size(),
#             get_expert_tensor_parallel_rank(),
#         )
#     else:
#         return (
#             get_tensor_model_parallel_group(),
#             get_tensor_model_parallel_world_size(),
#             get_tensor_model_parallel_rank(),
#         )


def get_vpp_rank() -> int:
    """Return the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK


def set_vpp_rank(rank):
    """Set the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_vpp_size() -> int:
    """Return the virtual pipeline-parallel world size."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE


def set_vpp_size(world_size):
    """Set the virtual pipeline-parallel world size"""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_data_world_size():
    return PM.size_of("DP") // PM.size_of("CP")


def get_data_rank():
    """
    Data sharding example for tp=1, ep=2, dp=4
    gpu_id 0   1   2   3   4   5   6   7
    dataid 0   1   2   3   4   5   6   7  (cp=1)
    dataid 0   0'  1   1'  2   2'  3   3' (cp=2)
    (e.g., 0' and 0 represent the same data but different token slices)
    """
    return PM.rank_in("DP") // PM.size_of("CP")


def _set_global_memory_buffer():
    """Initialize global buffer"""
    pass


def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object"""
    global _GLOBAL_MEMORY_BUFFER
    if _GLOBAL_MEMORY_BUFFER is None:
        _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()
    return _GLOBAL_MEMORY_BUFFER


def destroy_global_memory_buffer():
    """Sets the global memory buffer to None"""
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None
