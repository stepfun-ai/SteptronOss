import random
import re
from copy import deepcopy

import numpy as np
import torch
from megfile import smart_listdir

from steptronoss.core.parallel_state import PM


def recur_stat(state_dict):
    total_bytes = 0
    data_bytes = lambda x: 2 if x in [torch.float16, torch.bfloat16] else 4
    for i in state_dict:
        if isinstance(i, torch.Tensor):
            total_bytes += i.numel() * data_bytes(i.dtype)
        elif isinstance(i, np.ndarray):
            total_bytes += i.size * i.dtype.itemsize
        elif isinstance(i, dict):
            total_bytes += recur_stat(i.values())
        elif isinstance(i, list):
            total_bytes += recur_stat(i)
    return total_bytes


def move_tensor_to_memory(tensor: torch.Tensor):
    if tensor.device == torch.device("cpu"):
        return tensor.clone()
    else:
        return tensor.detach().cpu()


def move_to_memory(state_dict):
    if type(state_dict) is list:
        new_state_dict = state_dict.__class__()
        for i in range(len(state_dict)):
            if isinstance(state_dict[i], torch.Tensor):
                new_state_dict.append(move_tensor_to_memory(state_dict[i]))
            else:
                new_state_dict.append(move_to_memory(state_dict[i]))
    elif type(state_dict) is dict:
        new_state_dict = state_dict.__class__()
        for k in state_dict.keys():
            if isinstance(state_dict[k], torch.Tensor):
                new_state_dict[k] = move_tensor_to_memory(state_dict[k])
            else:
                new_state_dict[k] = move_to_memory(state_dict[k])
    elif isinstance(state_dict, torch.Tensor):
        new_state_dict = state_dict.detach().cpu()
    else:
        new_state_dict = deepcopy(state_dict)
    return new_state_dict


def analyze_dir(path):
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
    tp = len(tp) or None
    pp = len(pp) or None
    dp = len(dp)
    return tp, pp, dp
