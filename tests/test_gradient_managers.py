import copy
import os

import pytest
import torch
import torch.distributed as dist
from torch import nn

from steptronoss.core.parallel_state import PM
from steptronoss.exp.base_exp import GradientManagerConfig, ParallelConfig

pytestmark = pytest.mark.gpu


def _init_dist_and_mesh():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for gradient manager tests")
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")

    did_init = False
    if dist.is_initialized():
        if dist.get_backend() != "nccl":
            pytest.skip("gradient manager tests require NCCL backend")
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
        pytest.skip("gradient manager tests assume WORLD_SIZE=1")

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
        # Clear cached process groups so a re-init doesn't reuse invalid groups.
        PM._all_groups.clear()
        PM.parallels.clear()
        PM.all_parallels.clear()
        PM._stack.clear()
        PM._cur_cfg = None
        PM._rng_seeds.clear()
        PM.rng_states.clear()


def _make_model(dtype: torch.dtype) -> nn.Module:
    torch.manual_seed(0)
    return nn.Linear(4, 3, bias=True).to(device="cuda", dtype=dtype)


def _make_data(dtype: torch.dtype):
    torch.manual_seed(1)
    x = torch.randn(2, 4, device="cuda", dtype=dtype)
    y = torch.randn(2, 3, device="cuda", dtype=dtype)
    return x, y


def _run_step(manager, model, x, y):
    manager.zero_grad()
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()
    manager.step()
    return loss.item()


def _assert_main_grad(model: nn.Module):
    for param in model.parameters():
        assert hasattr(param, "main_grad")
        assert param.main_grad.shape == param.shape
        assert param.main_grad.dtype == torch.float32
        assert param.main_grad.is_cuda


def _assert_models_close(model1: nn.Module, model2: nn.Module, dtype: torch.dtype):
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 1e-2)
    assert torch.allclose(model1.weight.float(), model2.weight.float(), rtol=rtol, atol=atol)
    assert torch.allclose(model1.bias.float(), model2.bias.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gradient_manager_step_and_state_restore(dist_and_mesh, dtype):
    model1 = _make_model(dtype)
    model2 = _make_model(dtype)

    cfg = GradientManagerConfig()
    cfg.use_distributed_optimizer = False
    cfg.optimizer_cfg.lr = 0.1
    cfg.optimizer_cfg.weight_decay = 0.0
    cfg.clip_grad = 0.0
    cfg.log_num_zeros_in_grad = True

    gm1 = cfg.build_gradient_manager(model1)
    gm2 = cfg.build_gradient_manager(model2)

    _assert_main_grad(model1)

    x, y = _make_data(dtype)

    weight_before = model1.weight.detach().clone()
    bias_before = model1.bias.detach().clone()

    _run_step(gm1, model1, x, y)

    assert not torch.allclose(model1.weight, weight_before)
    assert not torch.allclose(model1.bias, bias_before)

    state = copy.deepcopy(gm1.state_dict())
    model2.load_state_dict(model1.state_dict())
    gm2.reload_model_params()
    gm2.load_state_dict(state)

    _run_step(gm1, model1, x, y)
    _run_step(gm2, model2, x, y)

    _assert_models_close(model1, model2, dtype)


@pytest.mark.skipif(
    not hasattr(torch.distributed, "reduce_scatter_tensor") or not hasattr(torch.distributed, "all_gather_into_tensor"),
    reason="reduce_scatter_tensor/all_gather_into_tensor missing",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero1_gradient_manager_step_and_state_restore(dist_and_mesh, dtype):
    model1 = _make_model(dtype)
    model2 = _make_model(dtype)

    cfg = GradientManagerConfig()
    cfg.use_distributed_optimizer = True
    cfg.optimizer_cfg.lr = 0.1
    cfg.optimizer_cfg.weight_decay = 0.0
    cfg.clip_grad = 0.0
    cfg.log_num_zeros_in_grad = True

    gm1 = cfg.build_gradient_manager(model1)
    gm2 = cfg.build_gradient_manager(model2)

    _assert_main_grad(model1)

    x, y = _make_data(dtype)

    weight_before = model1.weight.detach().clone()
    bias_before = model1.bias.detach().clone()

    _run_step(gm1, model1, x, y)

    assert not torch.allclose(model1.weight, weight_before)
    assert not torch.allclose(model1.bias, bias_before)

    state = copy.deepcopy(gm1.state_dict())
    model2.load_state_dict(model1.state_dict())
    gm2.reload_model_params()
    gm2.load_state_dict(state)

    _run_step(gm1, model1, x, y)
    _run_step(gm2, model2, x, y)

    _assert_models_close(model1, model2, dtype)
