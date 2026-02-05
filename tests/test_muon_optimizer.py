import copy

import pytest
import torch
from torch import nn

from steptronoss.exp.optimizer import MuonConfig
from steptronoss.optimizer.muon import Muon


def _make_model() -> nn.Module:
    torch.manual_seed(0)
    model = nn.Linear(4, 3, bias=True)
    for param in model.parameters():
        param.merge_op = _IdentityReshape()
    return model


def _make_data():
    torch.manual_seed(1)
    x = torch.randn(2, 4)
    y = torch.randn(2, 3)
    return x, y


def _step(model: nn.Module, optimizer, x: torch.Tensor, y: torch.Tensor) -> float:
    optimizer.zero_grad()
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()


@pytest.mark.cpu
def test_muon_optimizer_step_and_state_restore():

    x, y = _make_data()

    model1 = _make_model()
    model2 = _make_model()

    cfg = MuonConfig()
    cfg.lr = 0.1
    cfg.weight_decay = 0.0

    opt1 = cfg.build_optimizer(model1)
    opt2 = cfg.build_optimizer(model2)

    assert getattr(model1.weight, "is_muon_param", False) is True
    assert getattr(model1.bias, "is_muon_param", False) is False

    weight_before = model1.weight.detach().clone()
    bias_before = model1.bias.detach().clone()

    _step(model1, opt1, x, y)

    assert not torch.allclose(model1.weight, weight_before)
    assert not torch.allclose(model1.bias, bias_before)

    state = copy.deepcopy(opt1.state_dict())
    model2.load_state_dict(model1.state_dict())
    opt2.load_state_dict(state)

    _step(model1, opt1, x, y)
    _step(model2, opt2, x, y)

    assert torch.allclose(model1.weight, model2.weight)
    assert torch.allclose(model1.bias, model2.bias)


def _make_model_cuda(dtype: torch.dtype) -> nn.Module:
    torch.manual_seed(0)
    model = nn.Linear(4, 3, bias=True).to(device="cuda", dtype=dtype)
    for param in model.parameters():
        param.merge_op = _IdentityReshape()
    return model


def _make_data_cuda(dtype: torch.dtype):
    torch.manual_seed(1)
    x = torch.randn(2, 4, device="cuda", dtype=dtype)
    y = torch.randn(2, 3, device="cuda", dtype=dtype)
    return x, y


class _IdentityReshape:
    def forward(self, piece: dict) -> dict:
        return piece

    def backward(self, piece: dict) -> dict:
        return piece


def _build_optimizer_cuda(model: nn.Module) -> Muon:
    return Muon(
        [
            {
                "params": [model.weight],
                "is_muon_param": True,
                "lr": 0.1,
                "weight_decay": 0.0,
                "matched_adamw_rms": 0.2,
                "momentum": 0.95,
                "nesterov": True,
                "ns_steps": 2,
                "adamw_betas": (0.9, 0.95),
                "adamw_eps": 1e-8,
            },
            {
                "params": [model.bias],
                "is_muon_param": False,
                "lr": 0.1,
                "weight_decay": 0.0,
                "adamw_betas": (0.9, 0.95),
                "adamw_eps": 1e-8,
            },
        ],
        run_ns_in_fp16=True,
        newtonschulz_fn="default",
    )


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_muon_optimizer_step_and_state_restore_gpu(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Muon optimizer test")

    x, y = _make_data_cuda(dtype)

    model1 = _make_model_cuda(dtype)
    model2 = _make_model_cuda(dtype)

    opt1 = _build_optimizer_cuda(model1)
    opt2 = _build_optimizer_cuda(model2)

    weight_before = model1.weight.detach().clone()
    bias_before = model1.bias.detach().clone()

    _step(model1, opt1, x, y)

    assert not torch.allclose(model1.weight, weight_before)
    assert not torch.allclose(model1.bias, bias_before)

    assert "muon_buffer" in opt1.state[model1.weight]
    assert "adamw_exp_avg" in opt1.state[model1.bias]
    assert "adamw_exp_avg_sq" in opt1.state[model1.bias]

    state = copy.deepcopy(opt1.state_dict())
    model2.load_state_dict(model1.state_dict())
    opt2.load_state_dict(state)

    _step(model1, opt1, x, y)
    _step(model2, opt2, x, y)

    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 1e-2)
    assert torch.allclose(model1.weight.float(), model2.weight.float(), rtol=rtol, atol=atol)
    assert torch.allclose(model1.bias.float(), model2.bias.float(), rtol=rtol, atol=atol)
