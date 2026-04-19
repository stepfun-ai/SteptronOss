from __future__ import annotations

import torch

from steptronoss.model.optimizations.glm5_dsa import (
    tilelang_lighting_indexer,
    tilelang_sparse_mla,
)
from steptronoss.utils.optimizable import optimizable


def generate_varlen_mask_params(cu_seqlens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = int(cu_seqlens[-1].item())
    q_indices = torch.arange(0, seq_len, device=cu_seqlens.device, dtype=torch.int32)
    seq_indices = torch.searchsorted(cu_seqlens.to(torch.int32), q_indices, right=True) - 1
    starts = cu_seqlens.to(torch.int32)[seq_indices]
    ends = q_indices + 1
    return starts, ends


def _extract_topk_scores(logits: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
    valid_mask = topk_indices != -1
    safe_indices = topk_indices.clamp(min=0).to(torch.int64)
    scores = torch.gather(logits, dim=-1, index=safe_indices)
    return torch.where(valid_mask, scores, torch.full_like(scores, float("-inf")))


@optimizable(alternatives={"tilelang": tilelang_lighting_indexer})
def lighting_indexer(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk: int,
    topk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if index_q.dim() != 3:
        raise ValueError("index_q must be [seq_len, heads, dim]")
    if index_k.dim() != 2:
        raise ValueError("index_k must be [seq_len_kv, dim]")
    if weights.dim() == 3:
        if weights.shape[-1] != 1:
            raise ValueError("weights with 3 dims must end with singleton channel")
        weights = weights.squeeze(-1)
    if weights.dim() != 2:
        raise ValueError("weights must be [seq_len, heads] or [seq_len, heads, 1]")
    if weights.shape != index_q.shape[:2]:
        raise ValueError("weights must match index_q leading dimensions")

    seq_len, heads, _ = index_q.shape
    seq_len_kv = index_k.shape[0]
    if cu_seqlen_ks.shape[0] != seq_len or cu_seqlen_ke.shape[0] != seq_len:
        raise ValueError("cu_seqlen_ks/cu_seqlen_ke must have shape [seq_len]")

    logits = torch.zeros((seq_len, seq_len_kv), device=index_q.device, dtype=torch.float32)
    index_k_fp32 = index_k.float()
    weight_fp32 = weights.float()
    for head_idx in range(heads):
        head_scores = torch.einsum("qd,kd->qk", index_q[:, head_idx].float(), index_k_fp32).relu_()
        logits.add_(head_scores * weight_fp32[:, head_idx].unsqueeze(-1))

    token_positions = torch.arange(seq_len_kv, device=index_q.device, dtype=torch.int32).view(1, seq_len_kv)
    starts = cu_seqlen_ks.to(device=index_q.device, dtype=torch.int32).view(seq_len, 1)
    ends = cu_seqlen_ke.to(device=index_q.device, dtype=torch.int32).view(seq_len, 1)
    valid_mask = (token_positions >= starts) & (token_positions < ends)
    logits = logits.masked_fill(~valid_mask, float("-inf"))

    if topk_indices is None:
        topk = min(topk, seq_len_kv)
        topk_scores, topk_indices = torch.topk(logits, topk, dim=-1)
        topk_indices = topk_indices.masked_fill(~torch.isfinite(topk_scores), -1).to(torch.int32)
        return topk_scores, topk_indices

    return _extract_topk_scores(logits, topk_indices), topk_indices.to(torch.int32)


@optimizable(alternatives={"tilelang": tilelang_sparse_mla})
def sparse_mla(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scaling: float,
    d_v: int | None = None,
    query_chunk_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.dim() != 3:
        raise ValueError("q must be [seq_len, heads, dim_plus_tail]")
    if kv.dim() != 3:
        raise ValueError("kv must be [seq_len_kv, kv_group, dim_plus_tail]")
    if indices.dim() != 3:
        raise ValueError("indices must be [seq_len, kv_group, topk]")
    if kv.shape[1] != indices.shape[1]:
        raise ValueError("kv_group of kv and indices must match")
    if kv.shape[1] != 1:
        raise NotImplementedError("python sparse_mla reference currently supports kv_group == 1 only")

    seq_len, _, dim_plus_tail = q.shape
    seq_len_kv = kv.shape[0]
    topk = indices.shape[-1]
    if d_v is None:
        raise ValueError("d_v must be provided for sparse_mla")
    if not (0 < d_v <= dim_plus_tail):
        raise ValueError("d_v must be within (0, dim_plus_tail]")

    query_chunk_size = query_chunk_size or seq_len
    kv_flat = kv[:, 0]
    value_flat = kv_flat[:, :d_v]
    outputs = []
    lses = []

    for chunk_start in range(0, seq_len, query_chunk_size):
        chunk_end = min(chunk_start + query_chunk_size, seq_len)
        q_chunk = q[chunk_start:chunk_end]
        idx_chunk = indices[chunk_start:chunk_end, 0]
        valid = idx_chunk.ge(0)
        safe_idx = idx_chunk.clamp(min=0, max=seq_len_kv - 1).to(torch.int64)

        gather_shape = (chunk_end - chunk_start, topk, dim_plus_tail)
        gathered_k = kv_flat.index_select(0, safe_idx.reshape(-1)).reshape(gather_shape)
        gathered_v = value_flat.index_select(0, safe_idx.reshape(-1)).reshape(chunk_end - chunk_start, topk, d_v)

        scores = torch.einsum("qhd,qtd->qht", q_chunk.float(), gathered_k.float()) * scaling
        scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
        has_any = valid.any(dim=-1, keepdim=True)
        safe_scores = torch.where(has_any.unsqueeze(1), scores, torch.zeros_like(scores))
        lse_chunk = torch.logsumexp(safe_scores, dim=-1)
        lse_chunk = torch.where(
            has_any.expand_as(lse_chunk),
            lse_chunk,
            torch.full_like(lse_chunk, float("-inf")),
        )

        probs = torch.softmax(safe_scores, dim=-1, dtype=torch.float32)
        probs = torch.where(has_any.unsqueeze(1), probs, torch.zeros_like(probs))
        probs = probs.masked_fill(~valid.unsqueeze(1), 0.0)
        outputs.append(torch.einsum("qht,qtm->qhm", probs.to(gathered_v.dtype), gathered_v))
        lses.append(lse_chunk)

    return torch.cat(outputs, dim=0), torch.cat(lses, dim=0)
