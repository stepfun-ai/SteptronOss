"""Context-parallel wiring for the FSDP2 Qwen3.8 path.

Context parallelism is the one model-parallel axis this path can support, because
it is the only one that needs no KV cache. The Gated DeltaNet state crosses ranks
as an ordinary argument, and `GradientCheckpointingLayer.__call__` only strips
`use_cache` / `past_key_values` / `layer_past` -- so per-layer recomputation and
activation CPU offload both stay on, which is what the memory budget depends on.

The mesh tests here run on CPU with gloo on purpose. They check wiring, not
kernels, so they need no GPU and can run while a training job holds all of them.
"""

import os

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from steptronoss.model.qwen3_8_fsdp2 import _build_meshes


def _dist_ready():
    return all(k in os.environ for k in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"))


@pytest.fixture
def single_rank_group():
    """A one-rank gloo group, so mesh construction can be exercised without GPUs.

    Mirrors the fake-process-group fixture in tests/test_qwen3_8_fsdp2.py: these
    tests check wiring, not collectives, so a socketless single rank is enough.
    """
    created = not dist.is_initialized()
    if created:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29591")
        dist.init_process_group("gloo", rank=0, world_size=1)
    yield
    if created:
        dist.destroy_process_group()


@pytest.mark.cpu
def test_cp_disabled_keeps_the_single_dim_mesh(single_rank_group):
    """With CP off the mesh must stay exactly as it was, so nothing changes for the
    runs already on record."""
    mesh, cp_mesh = _build_meshes(torch.device("cpu"), world=1, cp_size=1)
    assert cp_mesh is None
    assert mesh.ndim == 1
    assert mesh.size() == 1


@pytest.mark.cpu
def test_cp_size_must_divide_the_world(single_rank_group):
    with pytest.raises(ValueError, match="not divisible"):
        _build_meshes(torch.device("cpu"), world=6, cp_size=4)


@pytest.mark.node2
@pytest.mark.skipif(not _dist_ready(), reason="needs torchrun to provide RANK/WORLD_SIZE")
def test_param_mesh_spans_every_rank_while_cp_mesh_is_the_group():
    """The distinction this asserts is the one that would silently break FSDP2.

    `fully_shard` shards across exactly the ranks of the mesh it is handed. CP
    partitions tokens, not parameters, so parameters must go on a mesh spanning
    ALL ranks -- handing it the `dp` sub-mesh instead would shard each parameter
    only within a dp group and quietly change what every rank holds.

    Keeping that mesh 1-D is also what lets fsdp2_gradient_manager stay untouched:
    its placements, `self.mesh.get_group()` and `norm.full_tensor()` all assume a
    single dimension.
    """
    if not dist.is_initialized():
        dist.init_process_group("gloo")
    world = dist.get_world_size()
    cp = 2
    assert world % cp == 0, "run this with an even nproc-per-node"

    mesh, cp_mesh = _build_meshes(torch.device("cpu"), world=world, cp_size=cp)

    assert mesh.ndim == 1, "parameter mesh must be flat or the gradient manager breaks"
    assert mesh.size() == world, f"parameter mesh covers {mesh.size()} of {world} ranks"
    assert sorted(mesh.mesh.flatten().tolist()) == list(range(world))

    assert cp_mesh.size() == cp
    peers = sorted(cp_mesh.mesh.flatten().tolist())
    # init_device_mesh((dp, cp)) lays ranks out as dp_idx*cp + cp_idx, so a rank's
    # CP peers are the contiguous block it falls in.
    block = dist.get_rank() // cp
    assert peers == [block * cp + i for i in range(cp)], f"rank {dist.get_rank()} got peers {peers}"


@pytest.mark.node2
@pytest.mark.skipif(not _dist_ready(), reason="needs torchrun to provide RANK/WORLD_SIZE")
def test_flattening_preserves_rank_order_so_checkpoints_stay_compatible():
    """A checkpoint written with CP off must be loadable with CP on.

    That holds only because flattening ("dp","cp") yields ranks 0..world-1 in the
    same order as the plain 1-D mesh, so each rank owns the same parameter shard
    either way. If this ever stops being true, the checkpoint shard layout becomes
    CP-dependent and resuming across a change in degree would load wrong shards
    without erroring.
    """
    if not dist.is_initialized():
        dist.init_process_group("gloo")
    world = dist.get_world_size()
    flat_cp_off, _ = _build_meshes(torch.device("cpu"), world=world, cp_size=1)
    flat_cp_on, _ = _build_meshes(torch.device("cpu"), world=world, cp_size=2)
    assert flat_cp_off.mesh.flatten().tolist() == flat_cp_on.mesh.flatten().tolist()


@pytest.mark.node2
@pytest.mark.skipif(not _dist_ready(), reason="needs torchrun to provide RANK/WORLD_SIZE")
def test_pm_and_device_mesh_group_the_same_ranks():
    """PM and the torch mesh must agree on who a rank's CP peers are.

    The CP collectives in this repo are written against PM, while FSDP2 owns a
    torch DeviceMesh. They happen to agree -- PM factors "(p d c t) -> (p d t) c"
    so the flat index is d*cp + c, matching init_device_mesh's dp_idx*cp + cp_idx --
    but a disagreement would send the sequence comms to the wrong peers and corrupt
    results rather than raise, so it is asserted rather than assumed.
    """
    from steptronoss.core.parallel_state import PM
    from steptronoss.model.qwen3_8_fsdp2 import _assert_cp_groups_agree

    if not dist.is_initialized():
        dist.init_process_group("gloo")
    world = dist.get_world_size()
    cp = 2
    PM.world_size = world
    PM.world_rank = dist.get_rank()

    from steptronoss.exp.base_exp import ParallelConfig

    cfg = ParallelConfig()
    cfg.context_parallel_size = cp
    PM.set_mesh(cfg)

    assert PM.size_of("CP") == cp
    _, cp_mesh = _build_meshes(torch.device("cpu"), world=world, cp_size=cp)
    _assert_cp_groups_agree(cp_mesh)  # raises on disagreement


def _gdn_only_model(num_layers=4, seed=0, layer_types=None):
    """A local tiny hybrid model with configurable layer types."""
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    torch.manual_seed(seed)
    config = Qwen3_5Config(
        text_config=dict(
            vocab_size=96,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=num_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            layer_types=layer_types or ["linear_attention"] * num_layers,
            use_cache=False,
            rope_parameters=dict(
                rope_type="default", rope_theta=10000.0, partial_rotary_factor=1.0, mrope_section=[1, 1, 2]
            ),
        ),
        vision_config=dict(
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            out_hidden_size=32,
            patch_size=2,
            num_position_embeddings=16,
        ),
        image_token_id=90,
        video_token_id=91,
        vision_start_token_id=92,
        vision_end_token_id=93,
    )
    model = Qwen3_5ForConditionalGeneration(config)
    model.config.use_cache = False
    return model


def _grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


@pytest.mark.gpu
@pytest.mark.node2
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GDN kernels require CUDA")
@pytest.mark.skipif(not _dist_ready(), reason="needs torchrun to provide RANK/WORLD_SIZE")
def test_cp_gradients_match_the_unsplit_run():
    """The gate for the state hand-off: splitting the sequence must not change the gradient.

    The objective is a fixed linear functional of the hidden states rather than the
    real loss, because CP-aware loss scaling is a later step; this isolates the
    hand-off from it. Under CP each rank computes a partial gradient, so they are
    summed over the CP group before comparing -- in the real system FSDP2's reduce
    does that, here it is explicit.

    A control run measures run-to-run noise, because this path is not
    deterministic: flash-attention's backward accumulates with atomics, and two
    identical runs were measured diverging by iteration 3. So the bar is "CP
    differs from unsplit by no more than unsplit differs from itself", not equality.
    """
    from steptronoss.core.parallel_state import PM
    from steptronoss.exp.base_exp import ParallelConfig
    from steptronoss.model.optimizations.context_parallel_qwen38 import install_context_parallel

    if not dist.is_initialized():
        dist.init_process_group("nccl")
    world = dist.get_world_size()
    assert world == 2, "run with --nproc-per-node=2"
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    cp = 2
    seq = 256  # multiple of cp * 64

    PM.world_size, PM.world_rank = world, dist.get_rank()
    cfg = ParallelConfig()
    cfg.context_parallel_size = cp
    PM.set_mesh(cfg)

    torch.manual_seed(7)
    ids = torch.randint(0, 80, (1, seq), device="cuda")
    # Fixed functional, so "the objective" is identical whether or not it is split.
    weight = torch.randn(1, seq, 32, device="cuda", dtype=torch.float32)

    def run_unsplit():
        model = _gdn_only_model().cuda()
        out = model.model.language_model(input_ids=ids).last_hidden_state
        (out.float() * weight).sum().backward()
        return _grads(model)

    reference = run_unsplit()
    control = run_unsplit()  # same config twice: the noise floor

    model = _gdn_only_model().cuda()
    install_context_parallel(model)
    out = model.model.language_model(input_ids=ids).last_hidden_state
    block = seq // cp
    local_weight = weight.narrow(1, dist.get_rank() * block, block)
    (out.float() * local_weight).sum().backward()
    cp_grads = _grads(model)
    # CP ranks hold slices of ONE sample, so their gradients sum rather than average.
    for tensor in cp_grads.values():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    assert set(reference) == set(cp_grads), "parameter sets differ"
    worst_cp = worst_noise = 0.0
    worst_name = "none"
    for name, ref in reference.items():
        scale = max(ref.abs().max().item(), 1e-8)
        noise = (control[name] - ref).abs().max().item() / scale
        err = (cp_grads[name] - ref).abs().max().item() / scale
        worst_noise = max(worst_noise, noise)
        if err > worst_cp:
            worst_cp, worst_name = err, name
    print(f"\n  worst CP deviation {worst_cp:.3e} on {worst_name}; noise floor {worst_noise:.3e}")
    assert worst_cp <= max(10 * worst_noise, 2e-3), (
        f"CP gradient deviates {worst_cp:.3e} (worst on {worst_name}), "
        f"well above the {worst_noise:.3e} run-to-run noise floor"
    )
