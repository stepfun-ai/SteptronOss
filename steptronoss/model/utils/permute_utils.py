"""Pure PyTorch implementations of TE MoE permutation helpers."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

__all__ = [
    "moe_permute",
    "moe_permute_with_probs",
    "moe_sort_chunks_by_index",
    "moe_sort_chunks_by_index_with_probs",
    "moe_unpermute",
]


def _pad_or_trim_indices(indices: torch.Tensor, num_out_tokens: int) -> Tuple[torch.Tensor, int]:
    if num_out_tokens < 0:
        return indices, indices.numel()
    if num_out_tokens == indices.numel():
        return indices, indices.numel()
    if num_out_tokens < indices.numel():
        return indices[:num_out_tokens], num_out_tokens
    pad = torch.full(
        (num_out_tokens - indices.numel(),),
        -1,
        device=indices.device,
        dtype=indices.dtype,
    )
    return torch.cat([indices, pad], dim=0), num_out_tokens


def _gather_by_token_indices(inp: torch.Tensor, token_indices: torch.Tensor, out_len: int) -> torch.Tensor:
    hidden = inp.shape[1]
    out = inp.new_zeros((out_len, hidden))
    if out_len == 0:
        return out
    valid = token_indices >= 0
    if valid.any():
        out[valid] = inp.index_select(0, token_indices[valid].to(torch.long))
    return out


def _permute_mask_map(
    inp: torch.Tensor,
    routing_map: torch.Tensor,
    num_out_tokens: int,
    probs: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if inp.numel() == 0:
        empty = torch.tensor([], device=inp.device, dtype=torch.long)
        return inp, empty, None if probs is None else probs.new_empty((0,))

    if routing_map.dim() != 2:
        raise ValueError("routing_map must be 2D [num_tokens, num_experts]")
    if inp.dim() != 2:
        raise ValueError("inp must be 2D [num_tokens, hidden_size]")
    if inp.size(0) != routing_map.size(0):
        raise ValueError("inp and routing_map must have the same num_tokens")

    num_tokens, num_experts = routing_map.shape
    routing_mask = routing_map.to(torch.bool)
    expert_ids, token_ids = routing_mask.T.nonzero(as_tuple=True)
    if expert_ids.numel() == 0:
        empty = torch.tensor([], device=inp.device, dtype=torch.long)
        row_id_map, out_len = _pad_or_trim_indices(empty, num_out_tokens)
        output = inp.new_zeros((out_len, inp.size(1)))
        permuted_probs = None if probs is None else probs.new_zeros((out_len,))
        return output, row_id_map, permuted_probs

    row_id_map = expert_ids.to(torch.long) * num_tokens + token_ids.to(torch.long)
    row_id_map, out_len = _pad_or_trim_indices(row_id_map, num_out_tokens)
    valid = row_id_map >= 0
    if row_id_map.numel() == 0:
        token_ids = row_id_map
    else:
        token_ids = row_id_map % num_tokens
        token_ids = token_ids.masked_fill(~valid, -1)
    output = _gather_by_token_indices(inp, token_ids, out_len)

    permuted_probs = None
    if probs is not None:
        if probs.shape != routing_map.shape:
            raise ValueError("probs must have shape [num_tokens, num_experts]")
        probs_flat = probs.T.contiguous().reshape(-1)
        permuted_probs = probs.new_zeros((out_len,))
        valid = row_id_map >= 0
        if valid.any():
            permuted_probs[valid] = probs_flat.index_select(0, row_id_map[valid].to(torch.long))
    return output, row_id_map, permuted_probs


def _permute_index_map(
    inp: torch.Tensor,
    routing_map: torch.Tensor,
    num_out_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if inp.numel() == 0:
        empty = torch.tensor([], device=inp.device, dtype=torch.long)
        return inp, empty

    if routing_map.dim() != 2:
        raise ValueError("routing_map must be 2D [num_tokens, top_k]")
    if inp.dim() != 2:
        raise ValueError("inp must be 2D [num_tokens, hidden_size]")
    if inp.size(0) != routing_map.size(0):
        raise ValueError("inp and routing_map must have the same num_tokens")

    num_tokens, top_k = routing_map.shape
    flat_expert = routing_map.reshape(-1)
    total_num = flat_expert.numel()
    flat_pos = torch.arange(total_num, device=routing_map.device, dtype=torch.long)
    valid = flat_expert >= 0
    if valid.any():
        valid_expert = flat_expert[valid].to(torch.long)
        valid_pos = flat_pos[valid]
        sort_key = valid_expert * total_num + valid_pos
        order = torch.argsort(sort_key)
        sorted_pos = valid_pos[order]
    else:
        sorted_pos = flat_pos.new_empty((0,))

    row_id_map = sorted_pos.to(torch.long)
    row_id_map, out_len = _pad_or_trim_indices(row_id_map, num_out_tokens)
    token_ids = row_id_map // top_k
    output = _gather_by_token_indices(inp, token_ids, out_len)
    return output, row_id_map


def moe_permute(
    inp: torch.Tensor,
    routing_map: torch.Tensor,
    num_out_tokens: int = -1,
    max_token_num: int = -1,
    map_type: str = "mask",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Permute tokens based on routing_map using pure PyTorch."""
    del max_token_num
    if map_type == "index":
        return _permute_index_map(inp, routing_map, num_out_tokens)
    if map_type == "mask":
        output, row_id_map, _ = _permute_mask_map(inp, routing_map, num_out_tokens, None)
        return output, row_id_map
    raise ValueError("map_type should be one of 'mask' or 'index'")


def moe_permute_with_probs(
    inp: torch.Tensor,
    probs: torch.Tensor,
    routing_map: torch.Tensor,
    num_out_tokens: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Permute tokens and probs based on routing_map using pure PyTorch."""
    output, row_id_map, permuted_probs = _permute_mask_map(inp, routing_map, num_out_tokens, probs)
    return output, permuted_probs, row_id_map


def moe_unpermute(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    merging_probs: Optional[torch.Tensor] = None,
    restore_shape: Optional[torch.Size] = None,
    map_type: str = "mask",
    probs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Unpermute tokens based on row_id_map using pure PyTorch."""
    if probs is not None:
        if merging_probs is not None:
            raise ValueError("Both merging_probs and probs are provided. probs is deprecated.")
        merging_probs = probs

    if inp.numel() == 0:
        return inp

    if map_type == "mask":
        if merging_probs is not None:
            num_tokens, _ = merging_probs.shape
        elif restore_shape is not None:
            num_tokens = restore_shape[0]
        else:
            raise ValueError("restore_shape or merging_probs must be provided for map_type='mask'.")

        hidden = inp.size(1)
        out_shape = restore_shape if restore_shape is not None else (num_tokens, hidden)
        output = inp.new_zeros(out_shape)
        valid = row_id_map >= 0
        if not valid.any():
            return output

        row_id_valid = row_id_map[valid].to(torch.long)
        token_ids = row_id_valid % num_tokens
        inp_valid = inp[valid]

        if merging_probs is not None:
            probs_flat = merging_probs.T.contiguous().reshape(-1)
            weights = probs_flat.index_select(0, row_id_valid)
            inp_valid = inp_valid * weights.unsqueeze(-1)

        output.scatter_add_(0, token_ids.unsqueeze(1).expand(-1, hidden), inp_valid)
        return output

    if map_type == "index":
        if merging_probs is not None:
            num_tokens, top_k = merging_probs.shape
        elif restore_shape is not None:
            num_tokens = restore_shape[0]
            valid = row_id_map >= 0
            if valid.any():
                top_k = int(row_id_map[valid].max().item()) // max(num_tokens, 1) + 1
            else:
                top_k = 1
        else:
            num_tokens = row_id_map.size(0)
            top_k = 1

        hidden = inp.size(1)
        out_shape = restore_shape if restore_shape is not None else (num_tokens, hidden)
        output = inp.new_zeros(out_shape)
        valid = row_id_map >= 0
        if not valid.any():
            return output

        row_id_valid = row_id_map[valid].to(torch.long)
        token_ids = row_id_valid // top_k
        inp_valid = inp[valid]

        if merging_probs is not None:
            probs_flat = merging_probs.reshape(-1)
            weights = probs_flat.index_select(0, row_id_valid)
            inp_valid = inp_valid * weights.unsqueeze(-1)

        output.scatter_add_(0, token_ids.unsqueeze(1).expand(-1, hidden), inp_valid)
        return output

    raise ValueError("map_type should be one of 'mask' or 'index'")


def moe_sort_chunks_by_index(
    inp: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_index: torch.Tensor,
) -> torch.Tensor:
    """Split inp by split_sizes and reorder chunks by sorted_index."""
    if inp.numel() == 0:
        return inp
    if split_sizes.dim() != 1:
        raise ValueError("split_sizes must be 1D")
    if sorted_index.dim() != 1:
        raise ValueError("sorted_index must be 1D")
    sizes = split_sizes.tolist()
    chunks = list(inp.split(sizes, dim=0))
    reordered = [chunks[i] for i in sorted_index.tolist()]
    return torch.cat(reordered, dim=0) if reordered else inp.new_empty((0,) + inp.shape[1:])


def moe_sort_chunks_by_index_with_probs(
    inp: torch.Tensor,
    probs: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split inp and probs by split_sizes and reorder chunks by sorted_index."""
    if inp.numel() == 0:
        return inp, probs
    if probs.size(0) != inp.size(0):
        raise ValueError("probs and inp must have the same leading dimension")
    if split_sizes.dim() != 1:
        raise ValueError("split_sizes must be 1D")
    if sorted_index.dim() != 1:
        raise ValueError("sorted_index must be 1D")
    sizes = split_sizes.tolist()
    inp_chunks = list(inp.split(sizes, dim=0))
    prob_chunks = list(probs.split(sizes, dim=0))
    reordered_inp = [inp_chunks[i] for i in sorted_index.tolist()]
    reordered_probs = [prob_chunks[i] for i in sorted_index.tolist()]
    out_inp = torch.cat(reordered_inp, dim=0) if reordered_inp else inp.new_empty((0,) + inp.shape[1:])
    out_probs = torch.cat(reordered_probs, dim=0) if reordered_probs else probs.new_empty((0,) + probs.shape[1:])
    return out_inp, out_probs
