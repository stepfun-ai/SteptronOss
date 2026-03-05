# Copyright (c) 2024, StepFun CORPORATION. All rights reserved.

from __future__ import annotations

import gc
import os
import pickle
import time
from os.path import join
from threading import Thread
from typing import TYPE_CHECKING, TypedDict

import torch
import torch.distributed as dist
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
    chunked_broadcast_bytes,
    find_rank_specific_path,
    get_rank_specific_name,
    move_to_memory,
    recur_stat,
    split_state_dict,
)
from steptronoss.core.parallel_state import PM, get_vpp_size, set_vpp_rank
from steptronoss.exp.checkpointing import CheckpointConfig
from steptronoss.utils import broadcast_tensors, unwrap_model
from steptronoss.utils.weight_loader import HFWeights

if TYPE_CHECKING:
    from steptronoss.model.module import MegatronModule
    from steptronoss.optimizer.base_gradient_manager import GradientManager
    from steptronoss.optimizer.hparam_scheduler import Scheduler


FileObjects = dict[str, object]
"""{"a.pkl": dict(...), "b.pt": dict(...)}"""


class CheckpointDict(TypedDict):
    iteration: int
    model: list[dict]
    optimizer: dict
    scheduler: dict
    data: dict
    extra_info: dict


class Checkpointer:
    def __init__(self):
        self.dump_tasks: list[tuple] = []

        Thread(target=self._dumper, daemon=True).start()

    def submit_dump_task(
        self,
        file_objects: FileObjects,
        save_path: str,
        mark_latest: tuple[str, str] | None = None,
        async_dump: bool = True,
    ):
        """
        file_objects: {"a.pkl": dict(...), "b.pt": dict(...)}
        save_path: "/mnt/some_path/"
        finish_callback: Callable[[], ] (call by last finish rank, you dont know which rank)
        async_dump: if True, return a thread.
        """
        if PM.world_rank == 0:
            smart_makedirs(save_path, exist_ok=True)
        dist.barrier()

        if not async_dump:
            self._dump_with_callback(file_objects, save_path, mark_latest)
        else:
            self.dump_tasks.append((file_objects, save_path, mark_latest))

    def join_dumping_thread(self):
        """ensure all dumping threads are finished"""

        if self.dump_tasks:
            logger.warning("Last uploading not finished. Consider increasing your save_interval!")
            while self.dump_tasks:
                time.sleep(0.5)

        torch.distributed.barrier()

    def _dumper(self):
        while 1:
            if self.dump_tasks:
                file_objects, save_path, mark_latest = self.dump_tasks.pop(0)
                self._dump_with_callback(file_objects, save_path, mark_latest)
            else:
                time.sleep(1)

    @staticmethod
    def size_stat(file_objects: FileObjects):
        size_info = []

        for file_name, obj in file_objects.items():
            size_g_bytes = recur_stat(obj) / 1024**3
            if size_g_bytes:
                size_info.append(f"{file_name}: {size_g_bytes:.1f} G")
            else:
                size_info.append(f"{file_name}: NOTENSOR")

        if size_info:
            size_info = " | ".join(size_info)
            logger.info(size_info, at=0)

    def make_ckpt(
        self,
        cfg: CheckpointConfig,
        model: list[MegatronModule] | None = None,
        optimizer: GradientManager | None = None,
        scheduler: Scheduler | None = None,
        dataloader: object = None,
        extra_info: dict | None = {},
    ) -> FileObjects:

        file_dicts: FileObjects = {}
        # Collect args, model, RNG.
        model_state_dict, optim_state_dict = {}, {}

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
                            model_state_dict[f"model{i}"] = model[i].state_dict()

        model_state_dict["extra"] = recur_to_allowed_types(extra_info)
        # RNG states.
        if cfg.save_option.rng_state:
            rng_state = PM.get_all_rng()
            model_state_dict["rng_state"] = rng_state

        if cfg.save_option.scheduler and scheduler is not None:
            model_state_dict["scheduler"] = scheduler.state_dict()

        # Collect optimizer state. (Optimizer is saved separately from the model, due
        # to the conflicting data pattern when using the distributed optimizer.)
        # Determine whether this rank should save optimizer

        if cfg.use_distributed_optimizer or PM.i_am("DP", 0) or PM.i_am("EDP", 0):
            # Optimizer stuff.
            if cfg.save_option.optimizer and optimizer is not None:
                optim_state_dict["optimizer"] = optimizer.state_dict()

        # Since the dataloader are the same on all rank, save only at rank 0
        if cfg.save_option.data and dataloader is not None:
            if PM.world_rank == 0:
                if hasattr(dataloader, "state_dict"):
                    file_dicts["data.pkl"] = move_to_memory(dataloader.state_dict())

        file_dicts[f"{get_rank_specific_name()}.pt"] = move_to_memory(model_state_dict)
        file_dicts[f"{get_rank_specific_name()}_OPT.pt"] = move_to_memory(optim_state_dict)
        return file_dicts

    def dump_ckpt(
        self,
        cfg: CheckpointConfig,
        iter_path: str | None = None,
        sub_name="",
        mark_latest=True,
        # objects
        iteration=0,
        model=None,
        optimizer=None,
        opt_param_scheduler=None,
        dataloader=None,
        extra_info: dict | None = None,
    ):
        """
        Async save a model checkpoint. Return the handle of thread or None.
        - Files(TPxxx.pt) will save to "iter_path/sub_name/"
        - After success, "cfg.save_path/latest_ckpt" will be updated, point to "iter_path/"
        """
        extra_info = extra_info or {}
        iter_path = iter_path or join(cfg.save_path, f"it{iteration}")
        path_with_subname = join(iter_path, sub_name)

        extra_info["iteration"] = iteration

        # Save safetensors if configured
        if cfg.save_safetensors:
            dump_safetensors(
                save_path=join(path_with_subname, "hf"),
                model_reference_path=cfg.model_config_path,
                tokenizer_reference_path=cfg.tokenizer_path,
                models=model,
            )

        file_dicts = self.make_ckpt(
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            scheduler=opt_param_scheduler,
            dataloader=dataloader,
            extra_info=extra_info,
        )

        self.size_stat(file_dicts)

        if mark_latest:
            mark_info = (cfg.save_path, iter_path)
        else:
            mark_info = None

        self.submit_dump_task(
            file_objects=file_dicts,
            save_path=path_with_subname,
            mark_latest=mark_info,
            async_dump=cfg.async_dump,
        )

    def load_ckpt(self, path: str, cfg: CheckpointConfig) -> CheckpointDict:
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
            return state_dicts

        no_optim = not cfg.load_option.optimizer

        old_tp_size, old_pp_size, old_dp_size = analyze_dir(path)
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

        model_path = find_rank_specific_path(path, use_dp0=not cfg.broadcast_from_dp0)

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
            dp_0_data = dict(this_rank_data)
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
            dp0_path = find_rank_specific_path(path, use_dp0=True)
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

        # Check arguments.
        extra_info: dict = this_rank_data.get("extra", {})
        if extra_info:
            logger.info(f"Extra info the checkpoint: {extra_info.keys()}", at=0)
        state_dicts["extra_info"] = extra_info

        # Set iteration.
        if cfg.load_option.iter:
            state_dicts["iteration"] = extra_info.get("iteration", -1)
        else:
            state_dicts["iteration"] = -1

        # Model.
        logger.info(f"Loading modules from {path}...", at=0)
        logger.info("([required] [in_ckpt])", at=0)
        model_exists = "model" in this_rank_data or "model0" in this_rank_data
        logger.info(
            f"model: [{'√' if cfg.load_option.model else '×'}] [{'√' if model_exists else '×'}]",
            at=0,
        )
        if cfg.load_option.model and model_exists:
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
        sched_exists = "scheduler" in this_rank_data
        logger.info(
            f"sched: [{'√' if cfg.load_option.scheduler else '×'}] [{'√' if sched_exists else '×'}]",
            at=0,
        )
        if cfg.load_option.scheduler and sched_exists:
            state_dicts["scheduler"] = this_rank_data["scheduler"]

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

        dist.barrier()

        logger.info(
            f"Successfully loaded checkpoint from {path} at iteration {state_dicts['iteration']}",
            at=0,
        )
        return move_to_memory(state_dicts)

    @staticmethod
    def _dump_with_callback(
        file_objects: FileObjects,
        save_path: str,
        mark_latest: tuple[str, str] | None = None,
    ):
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        for filename, obj in file_objects.items():
            filename = join(save_path, filename)
            logger.info(f"<Uploader:{rank}>: Started dumping to {filename}", at=0)
            with smart_open(filename, "wb") as f:
                if filename.endswith(".pkl"):
                    pickle.dump(obj, f)
                else:
                    torch.save(obj, f)
            del obj
            gc.collect()

        with smart_open(os.path.join(save_path, f"{rank}.success"), "w") as f:
            pass

        successes = [f for f in smart_listdir(save_path) if f.endswith(".success")]
        logger.info(f"Finished uploading: [{len(successes)} / {world_size}]", at=0)
        if len(successes) == world_size:
            logger.info("All Rank Finished!", at="all")
            if mark_latest:
                mark_dir, latest_path = mark_latest
                latest_tmp = join(mark_dir, f"latest_ckpt.rank{PM.world_rank}")
                with smart_open(latest_tmp, "w") as f:
                    f.write(latest_path)
                try:
                    smart_rename(latest_tmp, join(mark_dir, "latest_ckpt"))
                except:
                    logger.error(f"Rank {PM.world_rank}: Renaming Fail!")
