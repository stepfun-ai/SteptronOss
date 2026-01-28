"""Attention core implementations for SteptronOss."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from steptronoss.utils.optimizable import optimizable


@torch.no_grad()
def parse_cu_seqlens(cu_seqlens, max_seq_len=None):
    if isinstance(max_seq_len, dict):
        max_q_len = max_seq_len["q"]
        max_k_len = max_seq_len["k"]
    else:
        max_q_len = max_k_len = max_seq_len

    if isinstance(cu_seqlens, dict):
        cu_seqlens_q = torch.zeros_like(cu_seqlens["q"], dtype=torch.int32)
        cu_seqlens_q[1:] = cu_seqlens["q"][1:]
        cu_seqlens_k = cu_seqlens["k"].to(torch.int32)
        if max_q_len is None:
            max_q_len = torch.max(cu_seqlens_q[1:] - cu_seqlens_q[:-1])
        if max_k_len is None:
            max_k_len = torch.max(cu_seqlens_k[1:] - cu_seqlens_k[:-1])
    else:
        cu_seqlens = cu_seqlens.to(torch.int32)
        cu_seqlens_q = cu_seqlens
        cu_seqlens_k = cu_seqlens
        if max_q_len is None or max_k_len is None:
            max_q_len = max_k_len = torch.max(cu_seqlens[1:] - cu_seqlens[:-1])
    return cu_seqlens_q, cu_seqlens_k, max_q_len, max_k_len


class FlashAttention(nn.Module):
    """Flash Attention implementation wrapper.

    This wraps flash_attn for efficient attention computation.
    """

    def __init__(
        self,
        causal: bool = True,
        attention_dropout: float = 0.0,
        sliding_window: int = -1,
        **kwargs,
    ):
        super().__init__()
        self.causal = causal
        self.attention_dropout = attention_dropout
        self.sliding_window = (sliding_window, sliding_window)  # for fa3

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute flash attention.

        Args:
            q: Query tensor of shape [batch, seq, heads, head_dim]
            k: Key tensor of shape [batch, seq, kv_heads, head_dim]
            v: Value tensor of shape [batch, seq, kv_heads, head_dim]
            cu_seqlens: Cumulative sequence lengths for variable length sequences
            max_seq_len: Maximum sequence length
            cpu_offload_info: CPU offload configuration

        Returns:
            Attention output of shape [batch, seq, heads, head_dim]
        """
        from flash_attn import flash_attn_func, flash_attn_varlen_func

        batch_size, seq_len, num_heads, head_dim = q.shape

        if cu_seqlens is not None:
            # Variable length attention
            cu_seqlens_q, cu_seqlens_k, max_q_len, max_k_len = parse_cu_seqlens(cu_seqlens, max_seq_len)

            # Reshape for varlen attention: [batch, seq, heads, dim] -> [total, heads, dim]
            q = q.reshape(-1, num_heads, head_dim)
            k = k.reshape(-1, k.shape[2], head_dim)
            v = v.reshape(-1, v.shape[2], head_dim)

            output = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_q_len,
                max_seqlen_k=max_k_len,
                dropout_p=self.attention_dropout if self.training else 0.0,
                causal=self.causal,
                window_size=self.sliding_window,
            )
            output = output.reshape(batch_size, seq_len, num_heads, head_dim)
        else:
            # Standard flash attention
            output = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attention_dropout if self.training else 0.0,
                causal=self.causal,
                window_size=self.sliding_window,
            )

        return output


@optimizable(alternatives={"flash-attn": FlashAttention})
class AttentionCore(nn.Module):
    """Scaled Dot-Product Attention (SDPA) implementation with FlashAttention-compatible API."""

    def __init__(
        self,
        causal: bool = True,
        attention_dropout: float = 0.0,
        sliding_window: int = -1,
        **kwargs,
    ):
        super().__init__()
        self.causal = causal
        self.attention_dropout = attention_dropout
        self.sliding_window = (sliding_window, sliding_window)  # keep API parity with FlashAttention

    @staticmethod
    def _maybe_expand_kv(k: torch.Tensor, v: torch.Tensor, num_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
        kv_heads = k.shape[2]
        if kv_heads == num_heads:
            return k, v
        if num_heads % kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by kv_heads ({kv_heads})")
        repeat = num_heads // kv_heads
        k = k.repeat_interleave(repeat, dim=2)
        v = v.repeat_interleave(repeat, dim=2)
        return k, v

    @staticmethod
    def _build_local_mask(q_len: int, k_len: int, window: int, causal: bool, device) -> torch.Tensor:
        q_idx = torch.arange(q_len, device=device).unsqueeze(1)
        k_idx = torch.arange(k_len, device=device).unsqueeze(0)
        if window < 0:
            if causal:
                allowed = k_idx <= q_idx
            else:
                allowed = torch.ones((q_len, k_len), dtype=torch.bool, device=device)
        else:
            if causal:
                allowed = (k_idx <= q_idx) & (k_idx >= (q_idx - window))
            else:
                allowed = (k_idx - q_idx).abs() <= window
        return ~allowed  # True means masked for SDPA

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute SDPA attention with FlashAttention-compatible inputs/outputs."""
        batch_size, seq_len, num_heads, head_dim = q.shape
        k, v = self._maybe_expand_kv(k, v, num_heads)

        window = self.sliding_window[0]
        use_mask = window is not None and window >= 0

        if cu_seqlens is not None:
            cu_seqlens_q, cu_seqlens_k, max_q_len, max_k_len = parse_cu_seqlens(cu_seqlens, max_seq_len)

            q_flat = q.reshape(-1, num_heads, head_dim)
            k_flat = k.reshape(-1, num_heads, head_dim)
            v_flat = v.reshape(-1, num_heads, head_dim)

            outputs = []
            for b in range(cu_seqlens_q.numel() - 1):
                q_start = int(cu_seqlens_q[b].item())
                q_end = int(cu_seqlens_q[b + 1].item())
                k_start = int(cu_seqlens_k[b].item())
                k_end = int(cu_seqlens_k[b + 1].item())

                q_seq = q_flat[q_start:q_end].transpose(0, 1).unsqueeze(0)  # [1, h, q, d]
                k_seq = k_flat[k_start:k_end].transpose(0, 1).unsqueeze(0)  # [1, h, k, d]
                v_seq = v_flat[k_start:k_end].transpose(0, 1).unsqueeze(0)  # [1, h, k, d]

                attn_mask = None
                is_causal = self.causal and not use_mask
                if use_mask:
                    attn_mask = self._build_local_mask(
                        q_seq.shape[-2],
                        k_seq.shape[-2],
                        window,
                        self.causal,
                        device=q_seq.device,
                    )
                    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

                out = self._sdpa(q_seq, k_seq, v_seq, is_causal=is_causal, attn_mask=attn_mask)
                outputs.append(out.squeeze(0).transpose(0, 1))  # [q, h, d]

            output = torch.cat(outputs, dim=0)
            output = output.reshape(batch_size, seq_len, num_heads, head_dim)
        else:
            q_t = q.transpose(1, 2)  # [b, h, s, d]
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)

            attn_mask = None
            is_causal = self.causal and not use_mask
            if use_mask:
                attn_mask = self._build_local_mask(seq_len, k_t.shape[-2], window, self.causal, device=q_t.device)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

            output = self._sdpa(q_t, k_t, v_t, is_causal=is_causal, attn_mask=attn_mask)
            output = output.transpose(1, 2)

        return output
