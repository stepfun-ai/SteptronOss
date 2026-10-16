"""Synchronous, rank-local LoRA checkpoints with an atomic completion manifest.

First version requires the same world size on resume and a shared local
filesystem. Exported PEFT adapters remain portable to other world sizes.
"""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from steptronoss.core.parallel_state import PM
from steptronoss.optimizer.fsdp2_gradient_manager import pack_local_state

FORMAT = "steptron-fsdp2-lora-v1"


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + f".rank{dist.get_rank()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_write_text(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


class FSDP2Checkpointer:
    def join_dumping_thread(self):
        pass  # Intentionally synchronous; publish latest only after all ranks finish.

    def dump_ckpt(self, cfg, iteration, model, optimizer, opt_param_scheduler, dataloader, extra_info):
        directory = Path(cfg.save_path) / f"it{iteration}"
        directory.mkdir(parents=True, exist_ok=True)
        rank, world = dist.get_rank(), dist.get_world_size()
        options = cfg.save_option
        state = {
            "format": FORMAT,
            "world_size": world,
            "rank": rank,
            "extra_info": dict(extra_info, iteration=iteration),
        }
        if options.model:
            state["model"] = [pack_local_state(model[0].state_dict())]
        if options.optimizer:
            state["optimizer"] = optimizer.state_dict()
        if options.scheduler:
            state["scheduler"] = opt_param_scheduler.state_dict()
        if options.data:
            # Native data sync constructs a loader only on CP rank zero. Save
            # the shared logical cursor on every rank without duplicating loaders.
            data_state = [dataloader.state_dict() if dataloader is not None else None]
            if getattr(model[0], "cp_size", 1) > 1:
                dist.broadcast_object_list(data_state, src=PM.ranks_of("CP")[0], group=PM.group_of("CP"))
            if data_state[0] is None:
                raise ValueError("Missing data cursor on the CP data-source rank")
            state["data"] = data_state[0]
        if options.rng_state:
            state["rng_state"] = PM.get_all_rng()
        path = directory / f"rank{rank}.pt"
        atomic_torch_save(state, path)
        sizes = [None] * world
        dist.all_gather_object(sizes, path.stat().st_size)
        if rank == 0:
            manifest = {"format": FORMAT, "world_size": world, "iteration": iteration, "rank_file_sizes": sizes}
            # CP does not change the parameter shard identity. Full resume must
            # keep the data/optimizer topology; model-only imports may change CP.
            cp_size = getattr(model[0], "cp_size", 1)
            manifest["context_parallel_size"] = cp_size
            manifest["data_parallel_size"] = world // cp_size
            atomic_write_text(json.dumps(manifest), directory / "manifest.json")
        dist.barrier()

    def publish_latest(self, cfg, iteration):
        if dist.get_rank() == 0:
            directory = Path(cfg.save_path) / f"it{iteration}"
            atomic_write_text(str(directory.resolve()), Path(cfg.save_path) / "latest_ckpt")
        dist.barrier()

    def load_ckpt(self, path, cfg):
        if not path:
            return {}
        directory = Path(path)
        manifest = json.loads((directory / "manifest.json").read_text())
        world, rank = dist.get_world_size(), dist.get_rank()
        if manifest.get("format") != FORMAT or manifest.get("world_size") != world:
            raise ValueError("FSDP2 checkpoint format/world size mismatch; resharding is not implemented")
        old_cp = manifest.get("context_parallel_size", 1)
        new_cp = PM.size_of("CP") if "CP" in PM.parallels else 1
        options = cfg.load_option
        if old_cp != new_cp and any(
            getattr(options, k) for k in ("optimizer", "scheduler", "data", "rng_state", "iter")
        ):
            raise ValueError("Full FSDP2 resume requires the same context_parallel_size; use model-only loading")
        sizes = manifest.get("rank_file_sizes", [])
        if len(sizes) != world or any(
            not (directory / f"rank{r}.pt").is_file() or (directory / f"rank{r}.pt").stat().st_size != sizes[r]
            for r in range(world)
        ):
            raise ValueError("Incomplete FSDP2 checkpoint")
        state = torch.load(directory / f"rank{rank}.pt", map_location="cpu", weights_only=False)
        if state.get("format") != FORMAT or state.get("rank") != rank or state.get("world_size") != world:
            raise ValueError("FSDP2 rank checkpoint identity mismatch")
        options = cfg.load_option
        result = {"extra_info": state["extra_info"], "iteration": manifest["iteration"] if options.iter else -1}
        for key in ("model", "optimizer", "scheduler", "data"):
            if getattr(options, key):
                if key not in state:
                    raise ValueError(f"Requested {key} is missing from FSDP2 checkpoint")
                result[key] = state[key]
        if options.rng_state:
            PM.set_all_rng(state["rng_state"])
        return result
