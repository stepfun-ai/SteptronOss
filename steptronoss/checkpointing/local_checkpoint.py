# Copyright (c) 2024, StepFun CORPORATION. All rights reserved.

"""Input/output checkpointing."""

import gc
import os
import pickle
from os.path import join
from threading import Thread
from typing import Optional, TypedDict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from configurize import recur_to_allowed_types
from loguru import logger
from megfile import (
    smart_exists,
    smart_listdir,
    smart_makedirs,
    smart_open,
    smart_rename,
)

from steptronoss.checkpointing.hf_checkpoint import dump_safetensors
from steptronoss.checkpointing.utils import (
    analyze_dir,
    move_to_memory,
    recur_stat,
)
from steptronoss.core.parallel_state import PM, get_vpp_size, set_vpp_rank
from steptronoss.exp.checkpointing import CheckpointConfig
from steptronoss.utils import broadcast_tensors, unwrap_model
from steptronoss.utils.weight_loader import HFWeights

SIG = {
    True: "[√]",
    False: "[×]",
}


def _get_rank_specific_name():
    # Use both the tensor and pipeline MP rank.
    tp_rank = PM.rank_in("TP")
    pp_rank = PM.rank_in("PP")
    dp_rank = PM.rank_in("DP")

    return f"TP{tp_rank:02d}_PP{pp_rank:03d}_DP{dp_rank:03d}"


def _find_rank_specific_path(path, use_dp0=False):
    tp = PM.size_of("TP")
    pp = PM.size_of("PP")

    tp_rank = PM.rank_in("TP")
    pp_rank = PM.rank_in("PP")
    dp_rank = PM.rank_in("DP")
    if use_dp0:
        dp_rank = 0

    ftp, fpp, fdp = analyze_dir(path)

    assert (tp == (ftp or 1)) and (pp == (fpp or 1)), "Ckpt split mismatch!"

    found = join(
        path,
        (
            f"TP{tp_rank:02d}"
            + (f"_PP{pp_rank:03d}" if fpp is not None else "")
            + (f"_DP{dp_rank:03d}" if fdp is not None else "")
            + ".pt"
        ),
    )
    return found


def uploader(
    tasks: list,
    world_size: int,
    rank: int,
    success_path: str,
    update_latest: tuple[str, str],
):
    role = f"<Uploader:{rank}>"
    for obj, dest in tasks:
        logger.info(f"{role}: Started dumping to {dest}", at=0)
        with smart_open(dest, "wb") as f:
            if dest.endswith(".pkl"):
                pickle.dump(obj, f)
            else:
                torch.save(obj, f)
        del obj
        gc.collect()
    with smart_open(os.path.join(success_path, f"{rank}.success"), "w") as f:
        pass
    if tasks:
        successes = [f for f in smart_listdir(success_path) if f.endswith(".success")]
        logger.info(f"Finished uploading: [{len(successes)} / {world_size}]", at=0)
        if len(successes) == world_size:
            logger.info(f"All Rank Finished!", at="all")
            if update_latest:
                to_write, write_path = update_latest
                latest_tmp = os.path.join(write_path, f"latest_ckpt.rank{rank}")
                with smart_open(latest_tmp, "w") as f:
                    f.write(to_write)
                try:
                    smart_rename(latest_tmp, join(write_path, "latest_ckpt"))
                except:
                    logger.error(f"{role}: Renaming Fail!")


def dump_ckpt(
    main_path,
    cfg: CheckpointConfig,
    sub_name=None,
    mark_latest=True,
    # objects
    iteration=0,
    model=None,
    optimizer=None,
    opt_param_scheduler=None,
    dataloader=None,
    extra_info: Optional[dict] = {},
):
    """
    Async save a model checkpoint. Return the handle of thread or None.

    Old implementation is mixed with new implementation using checkpoint engine.
    The flag 'disable_checkpoint_engine' controls whether to use new implementation.
    """
    if sub_name is not None:
        final_path = join(main_path, sub_name)
    else:
        final_path = main_path
    if not dist.is_initialized() or dist.get_rank() == 0:
        smart_makedirs(final_path, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()

    comps = " | ".join(
        [
            n
            for x, n in zip(
                [model, optimizer, opt_param_scheduler, dataloader],
                ["model", "optimizer", "scheduler", "data"],
            )
            if x is not None and cfg.save_option[n]
        ]
    )
    logger.info(f"Saving to {final_path}:", at=0)
    logger.info(f"Savings: {comps}", at=0)

    node_specific_path = join(final_path, f"{_get_rank_specific_name()}.pt")
    states_path = join(final_path, f"{_get_rank_specific_name()}_OPT.pt")

    # Collect args, model, RNG.
    model_state_dict = {}
    # Save safetensors if configured
    if cfg.save_safetensors:
        if model is not None:
            model_unwrapped = unwrap_model(model)
            dump_safetensors(
                save_path=join(final_path, "hf"),
                model_reference_path=cfg.model_config_path,
                tokenizer_reference_path=cfg.tokenizer_path,
                models=model_unwrapped,
            )

    if cfg.save_option.model:
        # Determine whether this rank should save model
        if PM.i_am("DP", 0) or PM.i_am("EDP", 0):
            # Arguments, iteration, and model.
            if model is not None:
                # Only rank zero of the data parallel writes to the disk.
                model = unwrap_model(model)
                if len(model) == 1:
                    model_state_dict["model"] = model[0].state_dict()
                else:
                    for i in range(len(model)):
                        set_vpp_rank(i)
                        model_state_dict["model%d" % i] = model[i].state_dict()

    model_state_dict["extra"] = recur_to_allowed_types(extra_info)
    model_state_dict["checkpoint_version"] = 4.0
    model_state_dict["iteration"] = iteration
    # RNG states.
    if cfg.save_option.rng_state:
        rng_state = PM.get_all_rng()
        model_state_dict["rng_state"] = rng_state

    if cfg.save_option.scheduler and opt_param_scheduler is not None:
        model_state_dict["opt_param_scheduler"] = opt_param_scheduler.state_dict()

    # Collect optimizer state. (Optimizer is saved separately from the model, due
    # to the conflicting data pattern when using the distributed optimizer.)
    optim_state_dict = {}
    # Determine whether this rank should save optimizer

    if cfg.use_distributed_optimizer or PM.i_am("DP", 0) or PM.i_am("EDP", 0):
        # Optimizer stuff.
        if cfg.save_option.optimizer and optimizer is not None:
            optim_state_dict["optimizer"] = optimizer.state_dict()

    # Save model and optimizer together.
    # state_dict = {**model_state_dict, **optim_state_dict}
    model_g_bytes = recur_stat(model_state_dict.values()) / 1024**3
    state_g_bytes = recur_stat(optim_state_dict.values()) / 1024**3
    async_tasks = []

    size_info = ""
    if model_state_dict:  # only saves if populated (i.e., inherits conditions above)
        size_info += f"Model: {model_g_bytes:.1f} G  |  "
        model_state_dict = move_to_memory(model_state_dict)
        async_tasks.append((model_state_dict, node_specific_path))
    if optim_state_dict:
        size_info += f"States: {state_g_bytes:.1f} G"
        optim_state_dict = move_to_memory(optim_state_dict)
        async_tasks.append((optim_state_dict, states_path))
    if size_info:
        logger.info(size_info, at=0)

    # Since the dataloader are the same on all rank, save only at rank 0
    if cfg.save_option.data and dataloader is not None:
        if dist.get_rank() == 0:
            if hasattr(dataloader, "state_dict"):
                data_state = move_to_memory(dataloader.state_dict())
                data_path = os.path.join(final_path, "data.pkl")
                async_tasks.append((data_state, data_path))

    if mark_latest:
        update_latest = (main_path, cfg.save_path)
    else:
        update_latest = None

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if not cfg.async_dump:
        for obj, dest in async_tasks:
            with smart_open(dest, "wb") as f:
                if dest.endswith(".pkl"):
                    pickle.dump(obj, f)
                else:
                    torch.save(obj, f)
            del obj
            gc.collect()
        if update_latest and rank == 0:
            with open(join(update_latest[1], "latest_ckpt"), "w") as f:
                f.write(update_latest[0])
        return
    else:
        if async_tasks:
            if cfg.async_dump == "thread" or cfg.async_dump is True:
                uploading_thread = Thread(
                    target=uploader,
                    args=(async_tasks, world_size, rank, final_path, update_latest),
                )
            elif cfg.async_dump == "mp":
                uploading_thread = mp.get_context("fork").Process(
                    target=uploader,
                    args=(async_tasks, world_size, rank, final_path, update_latest),
                )
            else:
                raise NotImplementedError
            uploading_thread.start()
        else:
            with smart_open(os.path.join(final_path, f"{rank}.success"), "w") as f:
                pass
            uploading_thread = None

        logger.info(f"Started thread uploading to {final_path}", at=0)
        return uploading_thread


def split_state_dict(model_state_dict):
    """Split model state dict into expert and non-expert parts based on parameter names.

    Expert parameters are identified by naming patterns:
    - Contains ".experts." or ".experts["
    - Contains ".moe." or ".moe_"
    """
    expert_state_dict = {k: v for k, v in model_state_dict.items() if not k.startswith("model")}
    non_expert_state_dict = {k: v for k, v in model_state_dict.items() if not k.startswith("model")}

    # Check if model states exist (either "model" or "model0", "model1", etc.)
    model_keys = [k for k in model_state_dict.keys() if k.startswith("model")]
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


def chunked_broadcast_bytes(data: bytes, src, group, chunk_size=5 * 10**9):
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


class CheckpointDict(TypedDict):
    iteration: int
    model: list[dict]
    optimizer: dict
    scheduler: dict
    data: dict


def load_ckpt(path, cfg: CheckpointConfig) -> tuple[CheckpointDict, dict]:
    """Load a model checkpoint and return the iteration."""
    state_dicts = CheckpointDict()
    hf_weights = None
    if cfg.load_safetensors:
        if isinstance(cfg.load_safetensors, str):
            safetensor_path = cfg.load_safetensors
        else:
            safetensor_path = join(path, "hf")
        hf_weights = HFWeights(safetensor_path)
        logger.info(
            f"Loading safetensors from {safetensor_path}, pt weight will be skipped.",
            at=0,
        )
        if hf_weights:
            state_dicts["model"] = safetensor_path
        cfg.load_option.model = False

    if not path:
        return state_dicts, {}

    no_optim = not cfg.load_option.optimizer

    old_tp, old_pp, old_dp = analyze_dir(path)
    old_tp_size = old_tp or 1
    old_pp_size = old_pp or 1
    old_dp_size = old_dp or 1
    new_dp_size = PM.size_of("DP")
    opt_probe_path = join(path, "TP00_PP000_DP000_OPT.pt")
    has_optimizer_files = smart_exists(opt_probe_path)
    need_reshard = (
        getattr(cfg, "reshard_optimizer_state", False)
        and cfg.use_distributed_optimizer
        and cfg.load_option.optimizer
        and has_optimizer_files
        and old_dp_size != new_dp_size
    )

    model_path = _find_rank_specific_path(path, use_dp0=not cfg.broadcast_from_dp0)

    optim_path = model_path.replace(".pt", "_OPT.pt")

    def load(load_path):
        logger.info(f"Loading from {load_path}")
        return torch.load(
            smart_open(load_path, "rb"),
            map_location="cpu",
            weights_only=False,  # Default to True in PyTorch 2.6.0
        )

    # Check if expert data parallel group is separate
    # When expert_model_parallel_size > 1, we have a separate expert data parallel group
    separate_expert_dp = PM.size_of("EP") > 1

    if PM.rank_in("DP") == 0:
        # dp0 get model, optim if dst
        this_rank_data = load(model_path)
        if not cfg.use_distributed_optimizer:
            # we all use dp0
            if smart_exists(optim_path) and not no_optim:
                this_rank_data.update(load(optim_path))
        dp_0_data = {k: v for k, v in this_rank_data.items()}
    else:
        # dp x load own, meta only
        this_rank_data = {}
        if smart_exists(model_path):
            this_rank_data = load(model_path)
        dp_0_data = None

    if cfg.use_distributed_optimizer and not need_reshard:
        # use my rank, not dp0
        if smart_exists(optim_path) and not no_optim:
            this_rank_data.update(load(optim_path))

    if cfg.broadcast_from_dp0:
        # every rank load own ckpt, use broadcasted dp0 for default

        if PM.size_of("DP") > 1 or PM.size_of("EDP") > 1:
            if separate_expert_dp:
                # Separate expert and non-expert states based on parameter naming patterns
                is_dp_rank_0 = PM.i_am("DP", 0)
                is_expert_dp_rank_0 = PM.i_am("EDP", 0)

                # Split data based on whether this rank is a source for broadcasting
                if is_dp_rank_0 or is_expert_dp_rank_0:
                    # Use dp_0_data if this is regular DP rank 0, otherwise use this_rank_data
                    data_to_split = dp_0_data if is_dp_rank_0 else this_rank_data
                    if data_to_split is not None:
                        expert_state_dict, non_expert_state_dict = split_state_dict(data_to_split)
                    else:
                        expert_state_dict, non_expert_state_dict = {}, {}
                else:
                    expert_state_dict, non_expert_state_dict = None, None

                # Prepare states for broadcasting
                if is_dp_rank_0:
                    _non_expert_state_dict = non_expert_state_dict
                else:
                    _non_expert_state_dict = None

                if is_expert_dp_rank_0:
                    _expert_state_dict = expert_state_dict
                else:
                    _expert_state_dict = None

                # Broadcast non-expert states through regular data parallel group
                if PM.size_of("DP") > 1:
                    logger.info(f"Broadcasting non-expert states from {PM.ranks_of('DP')[0]}")
                    _non_expert_state_dict = broadcast_tensors(
                        _non_expert_state_dict,
                        src_rank=PM.ranks_of("DP")[0],
                        group=PM.group_of("DP"),
                    )

                # Broadcast expert states through expert data parallel group
                if PM.size_of("EDP") > 1:
                    logger.info(f"Broadcasting expert states from {PM.ranks_of('EDP')[0]}")
                    _expert_state_dict = broadcast_tensors(
                        _expert_state_dict,
                        src_rank=PM.ranks_of("EDP")[0],
                        group=PM.group_of("EDP"),
                    )

                # Merge the two parts back together
                dp_0_data = _non_expert_state_dict if _non_expert_state_dict is not None else {}
                expert_states = _expert_state_dict
                if isinstance(expert_states, dict):
                    for k in expert_states:
                        if k.startswith("model"):
                            dp_0_data.setdefault(k, {})
                            dp_0_data[k].update(expert_states[k])
            else:
                # broadcast in DP group (fallback when no separate expert DP)
                logger.info(f"Broadcast model from {PM.ranks_of('DP')[0]}")
                dp_0_data = broadcast_tensors(
                    dp_0_data,
                    src_rank=PM.ranks_of("DP")[0],
                    group=PM.group_of("DP"),
                )
    else:
        dp0_path = _find_rank_specific_path(path, use_dp0=True)
        if PM.rank_in("DP") != 0:
            dp_0_data = load(dp0_path)
            if not cfg.use_distributed_optimizer:
                dp0_optim_path = dp0_path.replace(".pt", "_OPT.pt")
                if smart_exists(dp0_optim_path) and not no_optim:
                    dp_0_data.update(load(dp0_optim_path))

    # use local first
    for k, v in dp_0_data.items():
        if k not in this_rank_data:
            logger.info(f"Use dp0: {k}")
            this_rank_data[k] = v
        elif k.startswith("model") and isinstance(v, dict) and isinstance(this_rank_data[k], dict):
            # For model dict, use local first for each key
            for model_key, model_val in v.items():
                this_rank_data[k].setdefault(model_key, model_val)

    # this shall never be None
    assert this_rank_data is not None

    # Set iteration.
    if cfg.load_option.iter:
        state_dicts["iteration"] = this_rank_data["iteration"]
    else:
        state_dicts["iteration"] = -1

    # Check arguments.
    extra_info: dict = this_rank_data.get("extra", {})
    if extra_info:
        logger.info(f"Extra info the checkpoint: {extra_info.keys()}", at=0)

    # Model.
    logger.info(f"Loading modules from {path}...", at=0)
    logger.info(f"([required] [in_ckpt])", at=0)
    model_exists = "model" in this_rank_data or "model0" in this_rank_data
    logger.info(
        f"model: [{'√' if cfg.load_option.model else '×'}] [{'√' if model_exists else '×'}]",
        at=0,
    )
    if cfg.load_option.model:
        if "model" in this_rank_data:
            state_dicts["model"] = [this_rank_data["model"]]
        else:
            state_dicts["model"] = [this_rank_data[f"model{vp}"] for vp in range(get_vpp_size())]

    # Optimizer.
    optim_exists = "optimizer" in this_rank_data
    if need_reshard and has_optimizer_files:
        optim_exists = True
    logger.info(
        f"optim: [{'√' if cfg.load_option.optimizer else '×'}] [{'√' if optim_exists else '×'}]",
        at=0,
    )
    if need_reshard:
        state_dicts["optimizer_reshard"] = {
            "path": path,
            "old_dp_size": old_dp_size,
            "old_tp_size": old_tp_size,
            "old_pp_size": old_pp_size,
            # "ckpt_exp": extra_info,
            "strict": getattr(cfg, "reshard_optimizer_strict", True),
        }
    elif cfg.load_option.optimizer and "optimizer" in this_rank_data:
        state_dicts["optimizer"] = this_rank_data["optimizer"]

    # Scheduler
    sched_exists = "opt_param_scheduler" in this_rank_data
    logger.info(
        f"sched: [{'√' if cfg.load_option.scheduler else '×'}] [{'√' if sched_exists else '×'}]",
        at=0,
    )
    if cfg.load_option.scheduler and sched_exists:
        state_dicts["scheduler"] = this_rank_data["opt_param_scheduler"]

    # load data dump w/ torch is extremely slow, use pickle later, but keep backward compatibility
    data_path = os.path.join(path, "data.pkl")
    if not smart_exists(data_path):
        data_path = os.path.join(path, "data.pt")
    loader_exists = smart_exists(data_path) or "data_loader" in this_rank_data
    logger.info(
        f"data : [{'√' if cfg.load_option.data else '×'}] [{'√' if loader_exists else '×'}]",
        at=0,
    )

    if cfg.load_option.data and loader_exists:
        data_state_serialized = smart_open(data_path, "rb").read() if dist.get_rank() == 0 else None
        data_state_serialized = chunked_broadcast_bytes(data_state_serialized, src=0, group=None)
        data_state_dict = pickle.loads(data_state_serialized)

        state_dicts["data"] = data_state_dict

    # rng states.
    if cfg.load_option.rng_state:
        rng_state = this_rank_data["rng_state"]
        PM.set_all_rng(rng_state)

    # Some utilities want to load a checkpoint without distributed being initialized
    if dist.is_initialized():
        dist.barrier()

    logger.info(
        f"Successfully loaded checkpoint from {path} at iteration {state_dicts['iteration']}",
        at=0,
    )
    return move_to_memory(state_dicts), extra_info
