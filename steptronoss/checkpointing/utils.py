import os
import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from megfile import smart_listdir
from torch import distributed as dist

from steptronoss.core.parallel_state import PM
from steptronoss.utils.dist_utils import broadcast_tensors


def recur_stat(obj: object) -> int:
    """Recursively estimate the total byte size of tensors/arrays in a container."""
    total_bytes = 0
    data_bytes = lambda x: 2 if x in [torch.float16, torch.bfloat16] else 4
    if isinstance(obj, torch.Tensor):
        total_bytes += obj.numel() * data_bytes(obj.dtype)
    elif isinstance(obj, np.ndarray):
        total_bytes += obj.size * obj.dtype.itemsize
    elif isinstance(obj, dict):
        for j in obj.values():
            total_bytes += recur_stat(j)
    elif isinstance(obj, list):
        for j in obj:
            total_bytes += recur_stat(j)
    return total_bytes


def move_tensor_to_memory(tensor: torch.Tensor) -> torch.Tensor:
    """Copy a tensor to CPU memory without modifying the original object."""
    if tensor.device == torch.device("cpu"):
        return tensor.clone()
    else:
        return tensor.detach().cpu()


def move_to_memory(state_dict: Any) -> Any:
    """Recursively move tensors in nested containers to CPU memory."""
    if type(state_dict) is list:
        new_state_dict = state_dict.__class__()
        for i in range(len(state_dict)):
            if isinstance(state_dict[i], torch.Tensor):
                new_state_dict.append(move_tensor_to_memory(state_dict[i]))
            else:
                new_state_dict.append(move_to_memory(state_dict[i]))
    elif type(state_dict) is dict:
        new_state_dict = state_dict.__class__()
        for k in state_dict:
            if isinstance(state_dict[k], torch.Tensor):
                new_state_dict[k] = move_tensor_to_memory(state_dict[k])
            else:
                new_state_dict[k] = move_to_memory(state_dict[k])
    elif isinstance(state_dict, torch.Tensor):
        new_state_dict = state_dict.detach().cpu()
    else:
        new_state_dict = deepcopy(state_dict)
    return new_state_dict


def analyze_dir(path: str) -> tuple[int, int, int]:
    """Infer tensor/pipeline/data parallel sizes from checkpoint filenames."""
    files = smart_listdir(path)
    tp, pp, dp = set(), set(), set()
    for f in files:
        tp_id = re.findall(r"TP(\d+)", f)
        pp_id = re.findall(r"PP(\d+)", f)
        dp_id = re.findall(r"DP(\d+)", f)
        if tp_id:
            tp.add(int(tp_id[0]))
        if pp_id:
            pp.add(int(pp_id[0]))
        if dp_id:
            dp.add(int(dp_id[0]))
    tp = len(tp) or 1
    pp = len(pp) or 1
    dp = len(dp) or 1
    return tp, pp, dp


def get_rank_specific_name() -> str:
    """Return a rank-tagged filename stem for the current TP/PP/DP ranks."""
    # Use both the tensor and pipeline MP rank.
    tp_rank = PM.rank_in("TP")
    pp_rank = PM.rank_in("PP")
    dp_rank = PM.rank_in("DP")

    return f"TP{tp_rank:02d}_PP{pp_rank:03d}_DP{dp_rank:03d}"


def find_rank_specific_path(path: str, use_dp0: bool = False) -> str:
    """Build the rank-specific checkpoint path for this process."""
    tp = PM.size_of("TP")
    pp = PM.size_of("PP")

    tp_rank = PM.rank_in("TP")
    pp_rank = PM.rank_in("PP")
    dp_rank = PM.rank_in("DP")
    if use_dp0:
        dp_rank = 0

    ftp, fpp, fdp = analyze_dir(path)

    assert (tp == (ftp or 1)) and (pp == (fpp or 1)), "Ckpt split mismatch!"

    found = os.path.join(
        path,
        (
            f"TP{tp_rank:02d}"
            + (f"_PP{pp_rank:03d}" if fpp is not None else "")
            + (f"_DP{dp_rank:03d}" if fdp is not None else "")
            + ".pt"
        ),
    )
    return found


def split_state_dict(
    model_state_dict: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split model state dict into expert and non-expert parts based on parameter names.

    Expert parameters are identified by naming patterns:
    - Contains ".experts." or ".experts["
    - Contains ".moe." or ".moe_"
    """
    expert_state_dict = {k: v for k, v in model_state_dict.items() if not k.startswith("model")}
    non_expert_state_dict = {k: v for k, v in model_state_dict.items() if not k.startswith("model")}

    # Check if model states exist (either "model" or "model0", "model1", etc.)
    model_keys = [k for k in model_state_dict if k.startswith("model")]
    if not model_keys:
        return expert_state_dict, non_expert_state_dict

    # Initialize model dicts for all model keys
    for model_key in model_keys:
        expert_state_dict[model_key] = {}
        non_expert_state_dict[model_key] = {}

    def is_expert_param(param_name: str) -> bool:
        """Determine if a parameter is an expert parameter based on its name."""
        # Check for common expert parameter patterns
        expert_patterns = [
            ".experts.",
        ]
        return any(pattern in param_name for pattern in expert_patterns)

    # Split the model state dict based on parameter name patterns
    for model_key in model_keys:
        for key, value in model_state_dict[model_key].items():
            if is_expert_param(key):
                expert_state_dict[model_key][key] = value
            else:
                non_expert_state_dict[model_key][key] = value

    return expert_state_dict, non_expert_state_dict


def chunked_broadcast_bytes(
    data: bytes,
    src: int,
    group: dist.ProcessGroup,
    chunk_size: int = 5 * 10**9,
) -> bytes:
    """Broadcast a large byte buffer in chunks and reassemble it."""
    chunk_num = None
    if dist.get_rank() == src:
        chunk_num = int((len(data) + chunk_size - 1) // chunk_size)

    chunk_num = broadcast_tensors(chunk_num, src_rank=src, group=group)

    merged_state = []
    for chunk_id in range(int(chunk_num)):
        chunk_serialized = None
        if dist.get_rank() == src:
            chunk_serialized = data[chunk_id * chunk_size : min((chunk_id + 1) * chunk_size, len(data))]
        chunk_serialized = broadcast_tensors(chunk_serialized, src_rank=src, group=group)
        merged_state.append(chunk_serialized)

    merged_state = b"".join(merged_state)
    return merged_state
