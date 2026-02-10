import os

import pytest
import torch
import torch.distributed as dist

from steptronoss.core.parallel_state import PM
from steptronoss.core.trainers.packed_model import PackedModel
from steptronoss.exp.base_exp import GradientManagerConfig, Megatron3DParallelModelConfig, ParallelConfig
from steptronoss.exp.lr_schedulers import ConstantSchedulerConfig
from steptronoss.exp.ntp import NTPTrainerConfig

pytestmark = pytest.mark.gpu


def _init_dist_and_mesh():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for PackedModel tests")
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")

    did_init = False
    if dist.is_initialized():
        if dist.get_backend() != "nccl":
            pytest.skip("PackedModel tests require NCCL backend")
    else:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        torch.cuda.set_device(0)
        dist.init_process_group(backend="nccl", world_size=1, rank=0)
        did_init = True

    if dist.get_world_size() != 1:
        pytest.skip("PackedModel tests assume WORLD_SIZE=1")

    torch.cuda.set_device(0)

    PM.initialize(backend="nccl")
    parallel_cfg = ParallelConfig()
    parallel_cfg.tensor_model_parallel_size = 1
    parallel_cfg.pipeline_model_parallel_size = 1
    parallel_cfg.context_parallel_size = 1
    parallel_cfg.expert_model_parallel_size = 1
    parallel_cfg.expert_tensor_parallel_size = 1
    parallel_cfg.virtual_pipeline_model_parallel_size = 1
    PM.set_mesh(parallel_cfg)

    return did_init


@pytest.fixture(scope="function")
def dist_and_mesh():
    did_init = _init_dist_and_mesh()
    yield
    if did_init and dist.is_initialized():
        dist.destroy_process_group()
        PM._all_groups.clear()
        PM.parallels.clear()
        PM.all_parallels.clear()
        PM._stack.clear()
        PM._cur_cfg = None
        PM._rng_seeds.clear()
        PM.rng_states.clear()


class DummyModelConfig(Megatron3DParallelModelConfig):
    def __init__(self):
        super().__init__()
        self.hidden_size = 1000
        self.params_dtype = torch.bfloat16

    def build_model(self) -> torch.nn.Module:
        model = torch.nn.Linear(1000, 1000, bias=False)
        model.is_pipeline_first_stage = lambda: True
        model.is_pipeline_last_stage = lambda: True
        return model


def _build_packed_model(use_distributed_optimizer: bool):
    trainer_cfg = NTPTrainerConfig()
    trainer_cfg.micro_batch_size = 1
    trainer_cfg.global_batch_size = 1
    trainer_cfg.global_seq_length = 4

    model_cfg = DummyModelConfig()

    grad_cfg = GradientManagerConfig()
    grad_cfg.use_distributed_optimizer = use_distributed_optimizer
    grad_cfg.optimizer_cfg.lr = 0.01
    grad_cfg.optimizer_cfg.weight_decay = 0.0
    grad_cfg.clip_grad = 0.0
    grad_cfg.log_num_zeros_in_grad = False

    scheduler_cfg = ConstantSchedulerConfig()
    scheduler_cfg.warmup_schedule = 0
    scheduler_cfg.lr = 10

    return PackedModel(
        trainer_config=trainer_cfg,
        model_config=model_cfg,
        grad_manager_config=grad_cfg,
        scheduler_config=scheduler_cfg,
        training=True,
    )


def assert_memory_around(expect: float):
    assert expect * 0.9 - 1024 < torch.cuda.memory_allocated() < expect * 1.1 + 1024


def _run_one_step(packed: PackedModel, x: torch.Tensor, y: torch.Tensor):
    packed.grad_manager.zero_grad()
    out = packed.models[0](x)
    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()
    for param in packed.models[0].parameters():
        if param.requires_grad:
            assert param.grad is None
            assert param.main_grad is not None
            assert torch.count_nonzero(param.main_grad).item() > 0
    packed.optimizer_step()


@pytest.mark.parametrize("use_distributed_optimizer", [False, True])
def test_packed_model_offload_backload_buffers(dist_and_mesh, use_distributed_optimizer):
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    packed = _build_packed_model(use_distributed_optimizer)

    packed.optimizer_step()
    packed.optimizer_step()

    torch.cuda.synchronize()
    # 18 GiB = 2 + 4 + 12

    assert_memory_around(base + 18e6)

    packed._offload_grad_buffer()
    assert_memory_around(base + 14e6)

    packed._offload_param()
    assert_memory_around(base + 12e6)

    packed._offload_optimizer_state()
    assert_memory_around(base + 0e6)

    packed._backload_param()
    assert_memory_around(base + 2e6)

    packed._backload_grad_buffer()
    assert_memory_around(base + 6e6)

    packed._backload_optimizer_state()
    assert_memory_around(base + 18e6)


@pytest.mark.parametrize("use_distributed_optimizer", [False, True])
def test_packed_model_offload_keep_grad(dist_and_mesh, use_distributed_optimizer):
    torch.manual_seed(0)
    packed_a = _build_packed_model(use_distributed_optimizer)
    packed_b = _build_packed_model(use_distributed_optimizer)

    packed_b.models[0].load_state_dict(packed_a.models[0].state_dict())
    packed_b.grad_manager.reload_model_params()

    x = torch.randn(2, 1000, device="cuda")
    y = torch.randn(2, 1000, device="cuda")

    def fwbw(packed, x, y):
        out = packed.models[0](x)
        loss = torch.nn.functional.mse_loss(out, y)
        loss.backward()

    fwbw(packed_a, x, y)
    fwbw(packed_b, x, y)

    packed_b._offload_optimizer_state()
    packed_b._backload_optimizer_state()

    weight_a_before = packed_a.models[0].module.weight.detach().clone()
    weight_b_before = packed_b.models[0].module.weight.detach().clone()

    fwbw(packed_a, x, y)
    fwbw(packed_b, x, y)
    packed_a.optimizer_step()
    packed_b.optimizer_step()

    assert not torch.allclose(packed_a.models[0].module.weight, weight_a_before)
    assert not torch.allclose(packed_b.models[0].module.weight, weight_b_before)
    assert torch.allclose(packed_a.models[0].module.weight, packed_b.models[0].module.weight, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("use_distributed_optimizer", [False, True])
def test_packed_model_offload_backload_preserves_step(dist_and_mesh, use_distributed_optimizer):
    torch.manual_seed(0)
    packed_a = _build_packed_model(use_distributed_optimizer)
    packed_b = _build_packed_model(use_distributed_optimizer)

    packed_b.models[0].load_state_dict(packed_a.models[0].state_dict())
    packed_b.grad_manager.reload_model_params()

    x = torch.randn(2, 1000, device="cuda")
    y = torch.randn(2, 1000, device="cuda")

    packed_b._offload_grad_buffer()
    packed_b._backload_grad_buffer()

    packed_b._offload_param()
    packed_b._backload_param()

    packed_b._offload_optimizer_state()
    packed_b._backload_optimizer_state()

    weight_a_before = packed_a.models[0].module.weight.detach().clone()
    weight_b_before = packed_b.models[0].module.weight.detach().clone()

    _run_one_step(packed_a, x, y)
    _run_one_step(packed_b, x, y)

    assert not torch.allclose(packed_a.models[0].module.weight, weight_a_before)
    assert not torch.allclose(packed_b.models[0].module.weight, weight_b_before)
    assert torch.allclose(packed_a.models[0].module.weight, packed_b.models[0].module.weight, rtol=1e-5, atol=1e-6)


def test_packed_model_distributed_optimizer_matches_non_distributed(dist_and_mesh):
    torch.manual_seed(0)
    packed_a = _build_packed_model(use_distributed_optimizer=True)
    packed_b = _build_packed_model(use_distributed_optimizer=False)

    packed_b.models[0].load_state_dict(packed_a.models[0].state_dict())
    packed_b.grad_manager.reload_model_params()

    x = torch.randn(2, 1000, device="cuda")
    y = torch.randn(2, 1000, device="cuda")

    _run_one_step(packed_a, x, y)
    _run_one_step(packed_b, x, y)

    assert torch.allclose(
        packed_a.models[0].module.weight,
        packed_b.models[0].module.weight,
        rtol=1e-5,
        atol=1e-6,
    )
