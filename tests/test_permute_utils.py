import pytest
import torch

from steptronoss.model.utils.permute_utils import (
    moe_permute,
    moe_permute_with_probs,
    moe_sort_chunks_by_index,
    moe_sort_chunks_by_index_with_probs,
    moe_unpermute,
)

pytestmark = pytest.mark.cpu


def _mask_expected_indices(routing_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, num_experts = routing_map.shape
    indices = []
    row_ids = []
    for expert_id in range(num_experts):
        for token_id in range(num_tokens):
            if routing_map[token_id, expert_id]:
                indices.append(token_id)
                row_ids.append(expert_id * num_tokens + token_id)
    device = routing_map.device
    return (
        torch.tensor(indices, device=device, dtype=torch.long),
        torch.tensor(row_ids, device=device, dtype=torch.long),
    )


def test_moe_permute_mask_roundtrip():
    tokens = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    routing_map = torch.tensor(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 1],
        ],
        dtype=torch.bool,
    )
    probs = torch.tensor(
        [
            [0.6, 0.0, 0.4, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.5],
        ],
        dtype=torch.float32,
    )

    permuted, permuted_probs, row_id_map = moe_permute_with_probs(tokens, probs, routing_map)
    expected_indices, expected_row_ids = _mask_expected_indices(routing_map)
    expected_permuted = tokens.index_select(0, expected_indices)
    expected_probs = probs.T.contiguous().reshape(-1).index_select(0, expected_row_ids)

    assert torch.equal(row_id_map, expected_row_ids)
    assert torch.allclose(permuted, expected_permuted)
    assert torch.allclose(permuted_probs, expected_probs)

    restored = moe_unpermute(
        permuted,
        row_id_map,
        merging_probs=probs,
        restore_shape=tokens.shape,
        map_type="mask",
    )
    assert torch.allclose(restored, tokens)


def test_moe_permute_mask_padding():
    tokens = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    routing_map = torch.tensor(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 1],
        ],
        dtype=torch.bool,
    )
    probs = torch.tensor(
        [
            [0.6, 0.0, 0.4, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.5],
        ],
        dtype=torch.float32,
    )

    expected_indices, _ = _mask_expected_indices(routing_map)
    num_out_tokens = expected_indices.numel() + 2
    permuted, row_id_map = moe_permute(tokens, routing_map, num_out_tokens=num_out_tokens, map_type="mask")
    assert permuted.shape[0] == num_out_tokens
    assert torch.equal(row_id_map[-2:], torch.tensor([-1, -1]))
    assert torch.allclose(permuted[-2:], torch.zeros_like(permuted[-2:]))

    restored = moe_unpermute(
        permuted,
        row_id_map,
        merging_probs=probs,
        restore_shape=tokens.shape,
        map_type="mask",
    )
    assert torch.allclose(restored, tokens)


def test_moe_permute_index_roundtrip():
    tokens = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    routing_map = torch.tensor([[1, 0], [0, 2], [2, -1]], dtype=torch.int64)
    probs = torch.tensor(
        [
            [0.6, 0.4],
            [0.5, 0.5],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    permuted, row_id_map = moe_permute(tokens, routing_map, map_type="index")

    flat = routing_map.reshape(-1)
    flat_pos = torch.arange(flat.numel())
    valid = flat >= 0
    sort_key = flat[valid] * flat.numel() + flat_pos[valid]
    order = torch.argsort(sort_key)
    expected_row_ids = flat_pos[valid][order]
    expected_tokens = tokens.index_select(0, expected_row_ids // routing_map.size(1))

    assert torch.equal(row_id_map, expected_row_ids)
    assert torch.allclose(permuted, expected_tokens)

    restored = moe_unpermute(
        permuted,
        row_id_map,
        merging_probs=probs,
        restore_shape=tokens.shape,
        map_type="index",
    )
    assert torch.allclose(restored, tokens)


def test_moe_sort_chunks_by_index():
    inp = torch.arange(12, dtype=torch.float32).view(6, 2)
    split_sizes = torch.tensor([2, 1, 3], dtype=torch.int64)
    sorted_index = torch.tensor([2, 0, 1], dtype=torch.int64)

    out = moe_sort_chunks_by_index(inp, split_sizes, sorted_index)
    expected = torch.cat([inp[3:6], inp[0:2], inp[2:3]], dim=0)
    assert torch.allclose(out, expected)


def test_moe_sort_chunks_by_index_with_probs():
    inp = torch.arange(12, dtype=torch.float32).view(6, 2)
    probs = torch.arange(6, dtype=torch.float32)
    split_sizes = torch.tensor([2, 1, 3], dtype=torch.int64)
    sorted_index = torch.tensor([2, 0, 1], dtype=torch.int64)

    out, out_probs = moe_sort_chunks_by_index_with_probs(inp, probs, split_sizes, sorted_index)
    expected = torch.cat([inp[3:6], inp[0:2], inp[2:3]], dim=0)
    expected_probs = torch.cat([probs[3:6], probs[0:2], probs[2:3]], dim=0)

    assert torch.allclose(out, expected)
    assert torch.allclose(out_probs, expected_probs)
