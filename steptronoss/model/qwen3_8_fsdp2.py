"""Shard the HF model once per decoder block; only rank zero loads base weights."""

import torch
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, fully_shard

from steptronoss.optimizer.fsdp2_gradient_manager import require_fsdp2_version


def reshard_for_checkpoint(model):
    """Restore DTensor parameters after a forward that did not run backward.

    Call only between training steps, when no pending graph needs the gathered
    adapter parameters. FSDPModule.reshard() is not recursive.
    """
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()


def _build_meshes(device, world, cp_size):
    """Return (param_mesh, cp_mesh): a 1-D mesh over every rank, plus the CP axis.

    `fully_shard` shards across exactly the ranks of the mesh it is given, and the
    sequence axis is not a parameter-sharding axis -- LoRA weights are replicated
    across CP, since CP partitions tokens rather than parameters. So parameters go
    on a flattened mesh spanning all ranks. Keeping it 1-D also means
    fsdp2_gradient_manager keeps working untouched: its `pack/unpack_local_state`
    placements, `self.mesh.get_group()` and `norm.full_tensor()` all assume one dim.

    Flattening ("dp","cp") preserves rank order 0..world-1, so the checkpoint shard
    layout is identical whether or not CP is on.
    """
    if cp_size <= 1:
        return init_device_mesh(device.type, (world,), mesh_dim_names=("fsdp",)), None
    if world % cp_size:
        raise ValueError(f"world size {world} is not divisible by context_parallel_size {cp_size}")
    mesh = init_device_mesh(device.type, (world // cp_size, cp_size), mesh_dim_names=("dp", "cp"))
    return mesh["dp", "cp"]._flatten(), mesh["cp"]


def _assert_cp_groups_agree(cp_mesh):
    """PM and the torch mesh must group the same ranks, or CP talks to the wrong peers.

    Both orderings happen to agree -- PM factors "(p d c t) -> (p d t) c" so the flat
    index is d*cp + c, and init_device_mesh((dp, cp)) lays out rank = dp_idx*cp +
    cp_idx -- but a disagreement would corrupt results silently rather than crash,
    and the CP collectives in this repo are written against PM. So check it.
    """
    from steptronoss.core.parallel_state import PM

    pm_ranks = sorted(PM.ranks_of("CP"))
    mesh_ranks = sorted(cp_mesh.mesh.flatten().tolist())
    if pm_ranks != mesh_ranks:
        raise RuntimeError(
            f"CP group disagreement: PM says this rank's CP peers are {pm_ranks}, the device mesh says {mesh_ranks}"
        )


def build_fsdp2_qwen(cfg, device=None):
    require_fsdp2_version()
    if cfg.tuner != "lora":
        raise ValueError("The FSDP2 Qwen experiment currently supports LoRA only")
    if not torch.distributed.is_initialized():
        raise RuntimeError("Initialize the process group before building FSDP2")
    device = torch.device(device or f"cuda:{torch.cuda.current_device()}")
    rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    cp_size = int(cfg.parallel_cfg.context_parallel_size)
    model = cfg.build_model(meta_init=rank != 0)
    hf_model = model.hf_model
    # Retain CPU references before fully_shard replaces Parameter storage. This
    # avoids loading a full 27B CPU replica in every worker on an 8-GPU host.
    full_state = hf_model.state_dict() if rank == 0 else {}
    buffers = {name: b.detach().cpu().clone() for name, b in hf_model.named_buffers()} if rank == 0 else {}
    mesh, cp_mesh = _build_meshes(device, world, cp_size)
    if cp_mesh is not None:
        _assert_cp_groups_agree(cp_mesh)
    base = hf_model.get_base_model()
    cp_sync_groups = []
    for block in base.model.language_model.layers:
        # Older FSDP2 versions require a uniform original dtype per group.
        # Group the FP32 LoRA leaf modules separately from the BF16 frozen
        # decoder. Keep the small adapter group unsharded through forward to
        # avoid an all-gather on every tiny A/B matrix invocation.
        adapters = []
        for module in block.modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                adapters.extend(module.lora_A.values())
                adapters.extend(module.lora_B.values())
        if adapters:
            fully_shard(adapters, mesh=mesh, reshard_after_forward=False)
            if cp_size > 1:
                block._cp_adapter_groups = (adapters[0],)
                cp_sync_groups.append(tuple(adapters))
        fully_shard(block, mesh=mesh, reshard_after_forward=True)
    # The root includes the embedding, head and unused frozen vision tower.
    # Keep the vision tower frozen and never execute it for text-only SFT.
    fully_shard(hf_model, mesh=mesh, reshard_after_forward=True)
    if cp_size > 1:
        # Automatic look-ahead may unshard a previous layer on one rank while
        # its peer is still waiting for this layer's GDN state gradient. An
        # explicit self target disables default look-ahead; self is already
        # unsharded at pre-backward, so the explicit prefetch is a no-op.
        for module in hf_model.modules():
            if isinstance(module, FSDPModule):
                module.set_modules_to_backward_prefetch([module])
                module.set_requires_gradient_sync(False, recurse=False)
    if rank != 0:
        hf_model.to_empty(device=device)
    set_model_state_dict(
        hf_model,
        full_state,
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, strict=True),
    )
    del full_state
    # Non-persistent RoPE buffers are absent from state_dict and must not be
    # left uninitialized after meta -> to_empty. Preserve rank-zero dtypes too.
    for name, _target in list(hf_model.named_buffers()):
        metadata = [(tuple(buffers[name].shape), buffers[name].dtype) if rank == 0 else None]
        torch.distributed.broadcast_object_list(metadata, src=0)
        shape, dtype = metadata[0]
        value = buffers[name].to(device) if rank == 0 else torch.empty(shape, dtype=dtype, device=device)
        torch.distributed.broadcast(value, src=0)
        owner, _, key = name.rpartition(".")
        hf_model.get_submodule(owner)._buffers[key] = value
    model.cp_sync_groups = tuple(cp_sync_groups)
    model.fsdp_mesh = mesh
    model.cp_mesh = cp_mesh
    model.cp_size = cp_size
    # CP does not change the flat parameter layout. Full-resume topology is
    # checked separately in the checkpoint manifest, not adapter identity.
    model.checkpoint_identity = dict(model.checkpoint_identity, format="steptron-qwen38-fsdp2-v1", world_size=world)
    return model
