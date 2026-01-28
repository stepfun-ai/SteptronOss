import pytest
import torch

pytestmark = pytest.mark.gpu

from steptronoss.model.utils import grouped_gemm


def _ref_gmm(a: torch.Tensor, b: torch.Tensor, batch_sizes: torch.Tensor, trans_b: bool) -> torch.Tensor:
    batch_sizes_list = batch_sizes.tolist()
    outputs = []
    start = 0
    for i, size in enumerate(batch_sizes_list):
        rhs = b[i].t() if trans_b else b[i]
        outputs.append(a[start : start + size] @ rhs)
        start += size
    if outputs:
        return torch.cat(outputs, dim=0)
    return a.new_zeros((0, b.shape[1] if trans_b else b.shape[2]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped_gemm requires CUDA")
@pytest.mark.skipif(not torch.cuda.is_bf16_supported(), reason="bf16 not supported")
@pytest.mark.parametrize("trans_b", [False, True])
def test_gmm_forward_backward(trans_b: bool):
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch_sizes = torch.tensor([3, 1, 4], dtype=torch.int64)
    k = 8
    n = 6
    a = torch.randn(batch_sizes.sum().item(), k, device=device, dtype=torch.bfloat16)
    if trans_b:
        b = torch.randn(batch_sizes.numel(), n, k, device=device, dtype=torch.bfloat16)
    else:
        b = torch.randn(batch_sizes.numel(), k, n, device=device, dtype=torch.bfloat16)

    a.requires_grad_(True)
    b.requires_grad_(True)
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)

    out = grouped_gemm(a, b, batch_sizes, trans_b=trans_b)
    expected = _ref_gmm(a_ref, b_ref, batch_sizes, trans_b)
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)

    out.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(a.grad, a_ref.grad, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(b.grad, b_ref.grad, rtol=1e-2, atol=1e-2)
