from __future__ import annotations

from typing import Optional

import torch
from configurize import Config, Ref

from steptronoss.core import tensor_parallel
from steptronoss.core.parallel_state import PM
from steptronoss.exp.base_exp import MegatronTPConfig
from steptronoss.model.common.attention_core import AttentionCore


class MultiHeadAttentionConfig(Config):
    """Multi-head attention config."""

    tp_cfg: MegatronTPConfig = Ref("..tp_cfg")
    """Tensor-parallel config."""
    hidden_size: int
    """Model hidden size."""
    num_attention_heads: int
    """Number of attention heads."""
    head_dim: int | None
    """Per-head dim (defaults to hidden_size / num_attention_heads)."""
    attention_dropout: float
    """Attention dropout."""
    use_qkv_bias: bool
    """Use bias for QKV projection."""
    use_out_proj_bias: bool
    """Use bias for output projection."""

    def __init__(self):
        super().__init__()
        self.tp_cfg = Ref("..tp_cfg")
        self.hidden_size = Ref("..hidden_size")
        self.num_attention_heads = 0
        self.head_dim = None
        self.attention_dropout = 0.0
        self.use_qkv_bias = True
        self.use_out_proj_bias = True

    def build_model(self, layer_id: int):
        return MultiHeadAttention(cfg=self, layer_id=layer_id)


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, cfg: MultiHeadAttentionConfig, layer_id: int):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id

        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_attention_heads
        self.tp_size = PM.size_of("TP")
        assert self.num_heads % self.tp_size == 0, "num_attention_heads must be divisible by TP size"
        self.num_local_heads = self.num_heads // self.tp_size

        if cfg.head_dim is None:
            assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_attention_heads"
            self.head_dim = self.hidden_size // self.num_heads
        else:
            self.head_dim = cfg.head_dim

        self.wqkv = tensor_parallel.ColumnParallelLinear(
            self.hidden_size,
            3 * self.hidden_size,
            bias=cfg.use_qkv_bias,
            gather_output=False,
            async_tensor_model_parallel_allreduce=cfg.tp_cfg.async_tensor_model_parallel_allreduce,
            **cfg.tp_cfg.get_tp_kwargs(),
        )
        self.wo = tensor_parallel.RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=cfg.use_out_proj_bias,
            input_is_parallel=True,
            **cfg.tp_cfg.get_tp_kwargs(),
        )
        self.attn_dropout = cfg.attention_dropout
        self.sequence_parallel = cfg.tp_cfg.sequence_parallel
        self.core_attention = AttentionCore(
            causal=False,
            attention_dropout=self.attn_dropout,
            sliding_window=-1,
        )

    def forward_attention_core(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        if not self.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                output = self.core_attention(
                    xq,
                    xk,
                    xv,
                    cu_seqlens=cu_seqlens,
                    max_seq_len=max_seqlen,
                )
        else:
            output = self.core_attention(
                xq,
                xk,
                xv,
                cu_seqlens=cu_seqlens,
                max_seq_len=max_seqlen,
            )
        return output

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seq_len: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        qkv = self.wqkv(x)[0]
        seq_len, batch, _ = qkv.shape

        if self.sequence_parallel:
            seq_len *= self.tp_size

        qkv = qkv.view(seq_len, batch, 3, self.num_local_heads, self.head_dim)
        xq, xk, xv = qkv.unbind(dim=2)
        xq = xq.contiguous()
        xk = xk.contiguous()

        if cu_seqlens is None:
            assert batch == 1, "cu_seqlens is not supported for bsz > 1!"
            cu_seqlens = torch.arange(
                0,
                (xq.shape[0] + 1) * xq.shape[1],
                step=seq_len,
                dtype=torch.int32,
                device=xq.device,
            )
            max_seq_len = torch.max(cu_seqlens[1:] - cu_seqlens[:-1]).cpu()
        elif max_seq_len is None:
            max_seq_len = torch.max(cu_seqlens[1:] - cu_seqlens[:-1]).cpu()

        xq = xq.permute(1, 0, 2, 3).contiguous()
        xk = xk.permute(1, 0, 2, 3).contiguous()
        xv = xv.permute(1, 0, 2, 3).contiguous()
        output = self.forward_attention_core(xq, xk, xv, cu_seqlens=cu_seqlens, max_seqlen=max_seq_len)
        output = output.permute(1, 0, 2, 3).contiguous().view(seq_len, batch, -1)
        return self.wo(output)[0]
