"""Attention core implementations for SteptronOss."""

from typing import Optional

import torch
import torch.nn as nn

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
        try:
            from flash_attn import flash_attn_varlen_func, flash_attn_func
        except ImportError:
            # Fallback to standard attention if flash_attn is not available
            return self._standard_attention(q, k, v)

        batch_size, seq_len, num_heads, head_dim = q.shape

        if cu_seqlens is not None:
            # Variable length attention
            cu_seqlens_q, cu_seqlens_k, max_q_len, max_k_len = parse_cu_seqlens(
                cu_seqlens, max_seq_len
            )

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

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Fallback standard attention implementation."""
        batch_size, seq_len, num_heads, head_dim = q.shape
        kv_heads = k.shape[2]

        # Handle GQA: expand k, v to match q heads
        if kv_heads != num_heads:
            repeat_factor = num_heads // kv_heads
            k = k.repeat_interleave(repeat_factor, dim=2)
            v = v.repeat_interleave(repeat_factor, dim=2)

        # Transpose for attention: [batch, heads, seq, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / (head_dim**0.5)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Build attention mask
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=q.device)

        if self.causal:
            # Causal mask: cannot attend to future positions
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device),
                diagonal=1,
            )
            mask = mask | causal_mask

        # Sliding window mask
        window_left, window_right = self.sliding_window
        if window_left >= 0 or window_right >= 0:
            # Create position indices
            rows = torch.arange(seq_len, device=q.device).unsqueeze(1)
            cols = torch.arange(seq_len, device=q.device).unsqueeze(0)

            # Mask positions outside the window
            if window_left >= 0:
                # Cannot attend to positions more than window_left steps before
                left_mask = cols < (rows - window_left)
                mask = mask | left_mask

            if window_right >= 0 and not self.causal:
                # Cannot attend to positions more than window_right steps after
                # (only applies for non-causal, causal already masks future)
                right_mask = cols > (rows + window_right)
                mask = mask | right_mask

        if mask.any():
            attn_weights.masked_fill_(mask, float("-inf"))

        attn_weights = torch.softmax(attn_weights.float(), dim=-1).type_as(q)

        if self.training and self.attention_dropout > 0:
            attn_weights = torch.dropout(attn_weights, self.attention_dropout, True)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2)  # [batch, seq, heads, dim]

        return output
