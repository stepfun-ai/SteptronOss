# Copyright (c) 2026, STEPFUN CORPORATION. All rights reserved.

"""General utilities."""

import json
import os
from typing import Optional

import msgpack
import torch
import torch.distributed
from loguru import logger
from tabulate import tabulate
from torch.nn.parallel import DistributedDataParallel as torchDDP

from steptronoss.core.parallel_state import PM, get_vpp_rank
from steptronoss.model.module import MegatronModule, ModelWrapperBase

from .general import convert_num


def unwrap_model(model, module_instances=(torchDDP, ModelWrapperBase)) -> MegatronModule:
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def calc_params_l2_norm(model, is_bf16=False):
    """Calculate l2 norm of parameters"""
    from steptronoss.core.tensor_parallel import (
        param_is_not_expert_parallel_duplicate,
        param_is_not_tensor_parallel_duplicate,
    )
    from steptronoss.core.utils import multi_tensor_applier, multi_tensor_l2_norm
    from steptronoss.model.module import param_is_not_shared

    if not isinstance(model, list):
        model = [model]
    # Remove duplicate params.
    params_data = []
    for model_ in model:
        for param in model_.parameters():
            is_not_shared = param_is_not_shared(param)
            is_moe_param = getattr(param, "expert_model_parallel", False)
            if is_moe_param:
                is_not_duplicate = param_is_not_expert_parallel_duplicate(param)
            else:
                is_not_duplicate = param_is_not_tensor_parallel_duplicate(param)
            if is_not_shared and is_not_duplicate:
                if is_bf16:
                    params_data.append(param.data.float())
                else:
                    params_data.append(param.data)
    # Calculate norm
    dummy_overflow_buf = torch.tensor([0], device="cuda", dtype=torch.int)
    norm, _ = multi_tensor_applier(
        multi_tensor_l2_norm,
        dummy_overflow_buf,
        [params_data],
        False,  # no per-parameter norm
    )
    norm_2 = norm * norm
    # Sum across all model-parallel GPUs.
    torch.distributed.all_reduce(norm_2, op=torch.distributed.ReduceOp.SUM, group=PM.group_of("MP"))
    return norm_2.item() ** 0.5


def get_mem_brief(name="", unit="G"):
    """Simple GPU memory report."""
    giga_bytes = 1024**3
    m_bytes = 1024**2
    unit_bytes = m_bytes if unit == "M" else giga_bytes
    string = (
        f"{name} MemReport "
        f"Allocated: {torch.cuda.memory_allocated() / unit_bytes:.1f} "
        f"(max: {torch.cuda.max_memory_allocated() / unit_bytes:.1f}) {unit} | "
        f"Reserved: {torch.cuda.memory_reserved() / unit_bytes:.1f} "
        f"(max: {torch.cuda.max_memory_reserved() / unit_bytes:.1f}) {unit} | "
    )
    return string


def force_clear_mem():
    """NOTE: 这个函数会将所有gc管理的cuda tensor挪到CPU, 但并不负责将它们搬回CUDA"""
    import gc

    logger.info(get_mem_brief("Before Force Clear: "))
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if hasattr(obj, "data") and torch.is_tensor(obj.data) and obj.data.is_cuda:
                    obj.data = obj.data.to("cpu")
        except:
            pass
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    logger.info(get_mem_brief("After Force Clear: "))


def print_n_params(model: list[torch.nn.Module]):
    # Print number of parameters.
    pp_size = PM.size_of("PP")
    tp_size = PM.size_of("TP")
    etp_size = PM.size_of("ETP")
    ep_size = PM.size_of("EP")

    # ---------- 辅助函数 ----------
    def _flag(p, name):
        return bool(getattr(p, name, False))

    def global_equiv_numel(p: torch.nn.Parameter):
        """将该rank参数换算为全局唯一贡献值"""
        n = p.numel()

        # Expert Tensor parallel
        if _flag(p, "tensor_model_parallel") and _flag(p, "expert_model_parallel"):
            # ETP 分片：不聚合时，需要乘回ETP_SIZE
            n *= ep_size * etp_size

        # Expert parallel
        elif _flag(p, "expert_model_parallel"):
            # 专家分片：不聚合时，需要乘回EP_SIZE
            n *= ep_size
        # Tensor parallel
        elif _flag(p, "tensor_model_parallel"):
            # 专家分片：不聚合时，需要乘回EP_SIZE
            n *= tp_size

        return n

    # ---------- 1) 本地统计 ----------
    local_raw_numel = 0
    local_global_equiv = 0
    local_expert_equiv = 0
    local_nonexpert_equiv = 0
    local_tensor_parallel_equiv = 0

    for module in model:
        for key, p in module.named_parameters():
            n_raw = p.numel()
            n_global = global_equiv_numel(p)
            local_raw_numel += n_raw
            local_global_equiv += n_global
            if _flag(p, "expert_model_parallel"):
                local_expert_equiv += n_global
            else:
                local_nonexpert_equiv += n_global
                if _flag(p, "tensor_model_parallel"):
                    local_tensor_parallel_equiv += n_global

    all_tp_params = [None for i in range(tp_size)]

    torch.distributed.all_gather_object(all_tp_params, local_raw_numel, group=PM.group_of("TP"))

    if PM.rank_in("DP") == 0 and PM.rank_in("TP") == 0:
        all_tp_params = [convert_num(all_tp_params[i]) for i in range(tp_size)]
        all_node_params = [None for i in range(PM.size_of("PP"))]
        torch.distributed.all_gather_object(
            all_node_params,
            all_tp_params,
            group=PM.group_of("PP"),
        )
        logger.info(
            " > # params distribution:\n{}".format(
                str(
                    tabulate(
                        all_node_params,
                        headers=[f"TP{i}" for i in range(tp_size)],
                        tablefmt="simple_grid",
                        showindex=True,
                    )
                ),
            ),
            at=0,
        )

    # 在所有pp rank上求和
    total_nums = torch.tensor(
        [
            local_global_equiv,
            local_expert_equiv,
            local_nonexpert_equiv,
            local_tensor_parallel_equiv,
        ],
        device="cuda",
    )
    torch.distributed.all_reduce(total_nums, group=PM.group_of("PP"))

    if torch.distributed.get_rank() == 0:
        logger.info(f"total params: {convert_num(total_nums[0].item())}")
        logger.info(f"total expert params: {convert_num(total_nums[1].item())}")
        logger.info(f"total non-expert params: {convert_num(total_nums[2].item())}")
        logger.info(f"total tensor-parallel params: {convert_num(total_nums[3].item())}")


def print_layer_map(layer_map):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:

        def _format_layers(vp_layers: dict[int, dict]) -> str:
            if not vp_layers:
                return "-"
            entries = []
            for layer_id in sorted(vp_layers.keys()):
                cfg = vp_layers[layer_id] or {}
                recompute = cfg.get("recompute", None)
                if recompute is not None:
                    entries.append(f"{layer_id}(recompute:{recompute})")
                else:
                    entries.append(f"{layer_id}")
            return " ".join(entries) if entries else "-"

        data = []
        max_vp = max((len(v) for v in layer_map.values()), default=0)
        for k, v in sorted(layer_map.items()):
            row = [k]
            for _, vp_layers in sorted(v.items()):
                row.append(_format_layers(vp_layers))
            while len(row) - 1 < max_vp:
                row.append("-")
            data.append(row)

        logger.info(
            " > # layer map:\n{}".format(
                str(
                    tabulate(
                        data,
                        headers=["PP", *[f"LAYERS@VP{i}" for i in range(max_vp)]],
                        tablefmt="simple_outline",
                    )
                ),
            ),
            at=0,
        )


def get_path_tree_repr(paths: list[str]) -> str:
    """
    Convert a list of file paths into a tree structure string.

    Args:
        paths: List of file path strings

    Returns:
        A string representing the tree structure
    """
    if not paths:
        return ""

    BRANCH = "|-- "
    LAST_BRANCH = "+-- "
    VERTICAL = "|   "
    EMPTY = "    "

    # Build tree structure
    tree = {}
    for path in paths:
        parts = path.strip().split("/")
        current = tree
        for part in parts:
            if part:  # Skip empty parts (from leading slash or double slashes)
                if part not in current:
                    current[part] = {}
                current = current[part]

    # List to collect all lines
    lines = []

    def build_tree(node, prefix="", is_last=True, is_root=True):
        """Recursively build the tree structure."""
        items = list(node.items())

        for i, (name, children) in enumerate(items):
            is_last_item = i == len(items) - 1

            # Add current node
            if is_root:
                # For root level, just add the name
                lines.append(name)
                if children:
                    build_tree(children, "", is_last_item, False)
            else:
                # Determine the branch character
                if is_last_item:
                    lines.append(f"{prefix}{LAST_BRANCH}{name}")
                    extension = EMPTY
                else:
                    lines.append(f"{prefix}{BRANCH}{name}")
                    extension = VERTICAL

                # Recursively build children
                if children:
                    new_prefix = prefix + extension
                    build_tree(children, new_prefix, is_last_item, False)

    # Build the tree
    build_tree(tree)

    # Join all lines with newlines and return
    return "\n".join(lines)


def profile_allreduce():
    import os
    import time

    import torch
    import torch.distributed as dist

    assert dist.is_initialized()

    n = 1_000_000_000 // 4

    x = torch.randn(n).cuda(torch.cuda.current_device())

    logger.info("Profiling AllReduce... NCCL ENVs:", at=0)
    for k in os.environ:
        if k.startswith("NCCL"):
            logger.info(f"{k} = {os.environ[k]}", at=0)
    stats = []
    for _ in range(10):
        dist.barrier()
        t = time.time()
        dist.all_reduce(x)
        torch.cuda.synchronize()
        MBpS = n * 4 * 2 / (time.time() - t) / 1e9
        stats.append(MBpS)
    stats = ", ".join([f"{i:.2f}" for i in stats])
    logger.info(f"Global allreduce (10 tries):\n>> [{stats}] GB/s", at=0)

    for g_name, parallel in PM.parallels.items():
        group = parallel.group
        stats = []
        logger.debug(f"start all reduce of group {g_name}")
        for i in range(10):
            dist.barrier()
            t = time.time()
            dist.all_reduce(x, group=group)
            torch.cuda.synchronize()
            dist.barrier()
            MBpS = torch.tensor(n * 4 * 2 / (time.time() - t) / 1e9, device="cuda")

            MBpS_mean = MBpS.clone()
            dist.all_reduce(MBpS_mean, op=dist.ReduceOp.SUM)
            MBpS_mean = MBpS_mean / dist.get_world_size()

            MBpS_min = MBpS.clone()
            dist.all_reduce(MBpS_min, op=dist.ReduceOp.MIN)

            stats.append(
                (
                    MBpS_mean.item(),
                    MBpS_min.item(),
                )
            )
        stats_mean = ", ".join([f"{i[0]:.2f}" for i in stats])
        stats_min = ", ".join([f"{i[1]:.2f}" for i in stats])
        logger.info(
            f"Group allreduce [{g_name}] (10 tries):" f"\n>> Avg [{stats_mean}] GB/s\n>> Min [{stats_min}] GB/s",
            at=0,
        )


def check_nan(tensors, input_tensor):
    if isinstance(tensors, torch.Tensor):
        tensor = tensors
    elif isinstance(tensors, tuple) and isinstance(tensors[0], torch.Tensor):
        tensor = tensors[0]
    else:
        return
    if tensor.isnan().any():
        N_nan = tensor.isnan().sum().item()
        N_total = tensor.numel()
        input_is_nan = 0
        if isinstance(input_tensor, torch.Tensor):
            input_is_nan = input_tensor.isnan().sum().item()

        if input_is_nan == 0:
            logger.warning(
                f"NaN detected [{N_nan}/{N_total}] on TP={PM.rank_in('TP')}/"
                f"PP={PM.rank_in('PP')}/"
                f"VP={get_vpp_rank()}"
            )


def get_exp_id() -> Optional[str]:
    """Get EXP_ID for this worker."""
    exp_id = os.environ.get("EXP_ID", None)
    if exp_id is not None:
        exp_id = exp_id.split("/")[0]
    return exp_id


def get_normalizer(x: torch.Tensor):
    n = torch.as_tensor(x.numel(), device=x.device)
    torch.distributed.all_reduce(n, group=PM.group_of("DP"))
    # n = n.float()
    m = x.sum()
    torch.distributed.all_reduce(m, group=PM.group_of("DP"))
    m /= n
    v = ((x - m) ** 2).sum()
    torch.distributed.all_reduce(v, group=PM.group_of("DP"))
    v /= n
    s = torch.sqrt(v)
    return m, s


# jsonl file saving related
def safe_dump_jsonl(records: list, file_path: str):
    try:
        with open(file_path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.error(f"Error dumping jsonl to {file_path}: {e}")


def load_jsonl(file_path: str):
    with open(file_path, "r") as f:
        for line in f:
            yield json.loads(line.strip())


# msgpack file saving related
def safe_dump_msgpack(
    data,
    file_path: str,
    use_single_float: bool = True,
):
    """
    Dump data to msgpack file
    - use_single_float=True means will save all float to fp32; if False, will save in fp64
    """
    try:
        with open(file_path, "wb") as f:
            f.write(msgpack.packb(data, use_single_float=use_single_float))
    except Exception as e:
        logger.error(f"Error dumping msgpack to {file_path}: {e}")


def load_msgpack(file_path: str):
    with open(file_path, "rb") as f:
        res = msgpack.unpackb(f.read())
    return res


# magic load function from zhy
def _load_tar(path: str):
    import tarfile

    import megfile

    fileobj = megfile.smart_open(path, "rb")
    stream = tarfile.open(fileobj=fileobj, mode="r|*")
    output = {}
    for tarinfo in stream:
        fname = tarinfo.name
        try:
            if not tarinfo.isreg():
                continue
            if fname is None:
                continue
            data = stream.extractfile(tarinfo).read()
            output[fname] = data
            stream.members = []
        except Exception as exn:
            if hasattr(exn, "args") and len(exn.args) > 0:
                exn.args = (str(exn.args[0]) + " @ " + str(fileobj),) + exn.args[1:]
                break
    del stream
    return output


def load(path: str, mode=None, **kwargs):
    import pickle

    import megfile

    if path.endswith(".json") or mode == "json":
        with megfile.smart_open(path, "rb") as f:
            data = json.load(f, **kwargs)
    elif path.endswith(".jsonl") or mode == "jsonl":
        with megfile.smart_open(path, "r") as f:
            data = [json.loads(t, **kwargs) for t in f.readlines()]
    elif path.endswith(".hjson") or mode == "hjson":
        import hjson

        with megfile.smart_open(path, "rb") as f:
            data = hjson.load(f, **kwargs)
    elif path.endswith(".pt") or path.endswith(".pth") or mode == "torch":
        with megfile.smart_open(path, "rb") as f:
            data = torch.load(f, **kwargs, weights_only=False, map_location="cpu")
    elif path.endswith(".pkl") or mode == "pickle":
        with megfile.smart_open(path, "rb") as f:
            data = pickle.load(f, **kwargs)
    elif path.endswith(".yaml") or mode == "yaml":
        import yaml

        with megfile.smart_open(path, "rb") as f:
            data = yaml.load(f, yaml.BaseLoader, **kwargs)
    elif path.endswith(".tar") or mode == "tar":
        data = _load_tar(path)
    elif path.endswith(".safetensors") or mode == "safetensors":
        import safetensors.torch

        data = safetensors.torch.load_file(path)
    elif path.endswith(".msgpack") or mode == "msgpack":
        data = load_msgpack(path)
    else:
        data = megfile.smart_open(path, "rb")
    return data


def moving_iter(data: list, active="cuda", default="cpu"):
    """Same like iter(), but move object to active device when yield, and
    move back to default device when done.

    Args:
        data (list): list of data.
        active (str, optional): active device. Defaults to "cuda".
        default (str, optional): default device. Defaults to "cpu".

    Yields:
        data: data item moved to active device.
    """
    from steptronoss.utils.general import recur_to

    for i in data:
        i = recur_to(i, active)
        yield i
        i = recur_to(i, default)


def cos_sim(a: torch.Tensor, b: torch.Tensor):

    target_device = a.device if a.is_cuda else "cpu"
    b = b.to(target_device)

    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


class RepeatedDict:
    def __init__(self):
        self.dict: dict[str, list] = {}

    def pop(self, key):
        if key not in self.dict:
            return None
        value_list = self.dict[key]
        if len(value_list) == 1:
            self.dict.pop(key)
        else:
            self.dict[key] = value_list[1:]
        return value_list[0]

    def update(self, another_dict: dict, repeat: int = 1):
        for key, value in another_dict.items():
            if key not in self.dict:
                self.dict[key] = [value] * repeat
            else:
                self.dict[key].extend([value] * repeat)

    def clear(self):
        self.dict.clear()

    def __len__(self):
        return sum(len(value) for value in self.dict.values())

    def __getitem__(self, key):
        return self.dict[key][0]

    def __setitem__(self, key, value):
        if key not in self.dict:
            self.dict[key] = [value]
        else:
            self.dict[key].append(value)

    def __delitem__(self, key):
        raise NotImplementedError("RepeatedDict does not support del")

    def state_dict(self):
        return self.dict

    def load_state_dict(self, states: dict):
        if states is None:
            self.dict = {}
        else:
            self.dict = states
