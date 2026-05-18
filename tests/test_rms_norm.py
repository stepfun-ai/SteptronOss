import pytest
import torch

from steptronoss.model.common.rms_norm import RMSNorm

pytestmark = pytest.mark.cpu


def manual_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, bias: float = 0.0) -> torch.Tensor:
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return y.to(x.dtype) * (weight + bias)


def test_rms_norm_uses_configured_eps():
    x = torch.tensor([[1.0, 2.0, 4.0], [0.5, -1.5, 3.0]], dtype=torch.float32)
    module = RMSNorm(dim=3, eps=1e-3, use_fp32=False)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([1.0, 0.5, 1.5]))

    got = module(x)
    expected = manual_rms_norm(x, module.weight, eps=1e-3)

    assert torch.allclose(got, expected, atol=1e-6, rtol=0)


def test_rms_norm_bfloat16_path_uses_configured_eps():
    x = torch.tensor([[1.0, -2.0, 4.0], [0.25, 1.5, -3.0]], dtype=torch.bfloat16)
    module = RMSNorm(dim=3, eps=1e-6, use_fp32=True)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([1.0, 1.0, 1.0]))

    got = module(x)
    expected = manual_rms_norm(x, module.weight, eps=1e-6)

    assert torch.allclose(got, expected, atol=1e-3, rtol=0)
