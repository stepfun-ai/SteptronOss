import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, TypedDict
from weakref import ref

import torch
from loguru import logger
from torch.nn import Parameter

from steptronoss.core.parallel_state import PM
from steptronoss.utils.general import balanced_list_split


class SteptronParameter(Parameter):
    main_grad: torch.FloatTensor

    expert_model_parallel: Optional[bool]
    manual_grad_bucket_prefix: Optional[str]
    muon_bucket_signature: Optional[str]

    _log_name: str


@dataclass(frozen=True)
class ParamBucketKey:
    """
    Composite identifier for bucketing parameters.

    The legacy implementation keyed everything by dtype, which breaks for Muon
    because multiple logical parameter buckets can share the same dtype. We
    extend the key with:
    - allreduce_group: DP / EDP
    - signature: optional extra discriminator to split buckets that still
      look identical (e.g., same dtype and manual_prefix).
    """

    dtype: torch.dtype
    allreduce_group: str = "DP"  # DP / EDP
    signature: int | str | None = None


class Range:
    """
    A range represents a start and end points for indexing a shard
    from a full tensor.
    """

    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.size = end - start

    def normalize(self, start=0):
        return Range(start, start + self.size)

    def __str__(self):
        return "%d,%d [%d]" % (self.start, self.end, self.size)


class ParamInfo(TypedDict):
    fp32_buffer: ref[torch.Tensor]
    dtype_buffer: ref[torch.Tensor]
    range: slice
    bucket_key: ParamBucketKey
    in_this_dp: bool


def get_bucket_key_from_param_attrs(param: SteptronParameter):
    """Return provisional GradBucketKey plus flags for reduction decisions."""
    if getattr(param, "expert_model_parallel", False):
        allreduce_group = "EDP"
    else:
        allreduce_group = "DP"

    signature = f"MUON:{getattr(param, 'is_muon_param', False)}"

    bucket_key = ParamBucketKey(
        dtype=param.dtype,
        allreduce_group=allreduce_group,
        signature=signature,
    )
    return bucket_key


def build_grad_buffers(
    module: torch.nn.Module,
) -> tuple[
    dict[ParamBucketKey, torch.Tensor],
    dict[ParamBucketKey, torch.Tensor],
    dict[SteptronParameter, ParamInfo],
]:
    """```
    Build buffers for a Module.
    - fp32_buffer will be in fp32.
    - buffer in param dtype will alse be created by reuse fp32 buffer.
    - buffer will be split into DP_SIZE(by param.allreduce_group) sections.
    - param allocation ensures param wont be cut @ DP_SPLIT. i.e.

                         ↓ no split here
    DP2    : [[BUFFER1  ], [BUFFER2    ]]
    tensor : [[T1, T2...], [Tn, Tn+1...]]

    - Returns:
        - fp32_buffers: dict[BucketKey, Buffer]
        - dtype_buffers: dict[BucketKey, Buffer]
        - param_info: dict[Tensor, ParamInfo]

    ```"""

    bucket_params: dict[ParamBucketKey, list[SteptronParameter]] = defaultdict(list)
    fp32_buffers: dict[ParamBucketKey, torch.Tensor] = {}
    raw_typed_buffers: dict[ParamBucketKey, torch.Tensor] = {}
    param_info: dict[SteptronParameter, ParamInfo] = {}

    for param in module.parameters():
        if param.requires_grad:
            bucket_key = get_bucket_key_from_param_attrs(param)
            bucket_params[bucket_key].append(param)

    # Allocate the buffer.

    for bucket_key, params in bucket_params.items():
        dp_size = PM.size_of(bucket_key.allreduce_group)
        cur_dp_rank = PM.rank_in(bucket_key.allreduce_group)

        param_ids = list(range(len(params)))
        param_sizes = [p.numel() for p in params]
        splited_params_ids = balanced_list_split(param_ids, sizes=param_sizes, split=dp_size)
        splited_sizes = [sum(param_sizes[x] for x in ids) for ids in splited_params_ids]
        max_split_size = max(splited_sizes)

        num_elements_padded = max_split_size * dp_size

        # Allocate grad buffer.
        grad_buffer = torch.zeros(
            num_elements_padded,
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        )
        param_buffer = torch.tensor(
            grad_buffer.untyped_storage(),
            dtype=bucket_key.dtype,
            device=grad_buffer.device,
            requires_grad=False,
        )
        param_buffer = param_buffer[: grad_buffer.numel()]

        # Save to buffers
        fp32_buffers[bucket_key] = grad_buffer
        raw_typed_buffers[bucket_key] = param_buffer

        # allocate range & save to param_info
        for dp_rank, param_ids in enumerate(splited_params_ids):
            dp_offset = max_split_size * dp_rank
            in_dp_offset = 0
            for param_id in param_ids:
                param = params[param_id]
                param_info[param] = ParamInfo(
                    bucket_key=bucket_key,
                    fp32_buffer=ref(fp32_buffers[bucket_key]),
                    dtype_buffer=ref(raw_typed_buffers[bucket_key]),
                    range=slice(
                        dp_offset + in_dp_offset,
                        dp_offset + in_dp_offset + param.numel(),
                    ),
                    in_this_dp=dp_rank == cur_dp_rank,
                )
                in_dp_offset += param.numel()

        overhead_ratio = num_elements_padded / sum(param_sizes)
        logger.info(
            f"Gradbuffer: {bucket_key}, elements: {len(params)}, dp_world_size: {dp_size}, "
            f"mem_overhead: {overhead_ratio}"
        )

    return fp32_buffers, raw_typed_buffers, param_info
