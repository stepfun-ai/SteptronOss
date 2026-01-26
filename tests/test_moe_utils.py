import pytest
import torch

from steptronoss.model.utils.moe_utils import (
    histogram,
    moe_scatter,
    moe_weighted_gather,
)

pytestmark = pytest.mark.cpu


def test_expert_histogram_basic():
    top_k_rank = torch.tensor([0, 1, 1, 3, 3, 3], dtype=torch.int64)
    out = histogram(top_k_rank, expert_num=4)
    assert out.dtype == torch.int32
    assert out.tolist() == [1, 2, 0, 3]


def test_expert_histogram_empty():
    top_k_rank = torch.tensor([], dtype=torch.int32)
    out = histogram(top_k_rank, expert_num=3)
    assert out.tolist() == [0, 0, 0]


def test_expert_histogram_negative_indices():
    top_k_rank = torch.tensor([0, -1, 2, -3, 2], dtype=torch.int64)
    out = histogram(top_k_rank, expert_num=4)
    assert out.tolist() == [1, 0, 2, 0]


def test_expert_histogram_out_of_range():
    top_k_rank = torch.tensor([0, 2], dtype=torch.int64)
    with pytest.raises(ValueError, match="out-of-range"):
        histogram(top_k_rank, expert_num=2)


def test_moe_weighted_gather_forward():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    idx = torch.tensor([[2, 0], [1, 2]], dtype=torch.int64)
    w = torch.tensor([[0.2, 0.8], [1.0, 0.5]])
    out = moe_weighted_gather(x, idx, w)
    expected = torch.tensor([[1.8, 2.8], [5.5, 7.0]])
    assert torch.allclose(out, expected)


def test_moe_weighted_gather_negative_indices():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    idx = torch.tensor([[2, -1], [1, -2]], dtype=torch.int64)
    w = torch.tensor([[0.2, 0.8], [1.0, 0.5]])
    out = moe_weighted_gather(x, idx, w)
    expected = torch.tensor([[1.0, 1.2], [3.0, 4.0]])
    assert torch.allclose(out, expected)


def test_moe_weighted_gather_backward():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    idx = torch.tensor([[2, 0], [1, 2]], dtype=torch.int64)
    w = torch.tensor([[0.2, 0.8], [1.0, 0.5]], requires_grad=True)

    out = moe_weighted_gather(x, idx, w)
    loss = out.sum()
    loss.backward()

    assert torch.allclose(x.grad, torch.tensor([[0.8, 0.8], [1.0, 1.0], [0.7, 0.7]]))
    assert torch.allclose(w.grad, torch.tensor([[11.0, 3.0], [7.0, 11.0]]))


def test_moe_weighted_gather_backward_negative_indices():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    idx = torch.tensor([[2, -1], [1, -2]], dtype=torch.int64)
    w = torch.tensor([[0.2, 0.8], [1.0, 0.5]], requires_grad=True)

    out = moe_weighted_gather(x, idx, w)
    out.sum().backward()

    assert torch.allclose(x.grad, torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.2, 0.2]]))
    assert torch.allclose(w.grad, torch.tensor([[11.0, 0.0], [7.0, 0.0]]))


def test_moe_scatter_forward():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    idx = torch.tensor([[0, 2], [1, 3]], dtype=torch.int64)
    out = moe_scatter(x, idx)
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [3.0, 4.0]])
    assert torch.allclose(out, expected)


def test_moe_scatter_forward_negative_indices():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    idx = torch.tensor([[0, -1], [1, -2]], dtype=torch.int64)
    out = moe_scatter(x, idx)
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert torch.allclose(out, expected)


def test_moe_scatter_backward():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    idx = torch.tensor([[0, 2], [1, 3]], dtype=torch.int64)
    out = moe_scatter(x, idx)
    loss = out.sum()
    loss.backward()
    assert torch.allclose(x.grad, torch.tensor([[2.0, 2.0], [2.0, 2.0]]))


def test_moe_scatter_backward_negative_indices():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    idx = torch.tensor([[0, -1], [1, -2]], dtype=torch.int64)
    out = moe_scatter(x, idx)
    out.sum().backward()
    assert torch.allclose(x.grad, torch.tensor([[1.0, 1.0], [1.0, 1.0]]))


def test_moe_scatter_gather_symmetry():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    idx = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64)
    scattered = moe_scatter(x, idx)
    weight = torch.ones_like(idx, dtype=torch.float32)
    gathered = moe_weighted_gather(scattered, idx, weight)
    expected = x * idx.shape[1]
    assert torch.allclose(gathered, expected)


def test_moe_scatter_gather_symmetry_with_invalid():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    idx = torch.tensor([[0, -1, 1], [2, -2, 3]], dtype=torch.int64)
    scattered = moe_scatter(x, idx)
    weight = torch.ones_like(idx, dtype=torch.float32)
    gathered = moe_weighted_gather(scattered, idx, weight)
    valid_count = (idx >= 0).sum(dim=1, keepdim=True).to(x.dtype)
    expected = x * valid_count
    assert torch.allclose(gathered, expected)
