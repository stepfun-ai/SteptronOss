import math

import pytest
import torch

from steptronoss.model.common.attention_core import AttentionCore, FlashAttention


def _manual_sliding_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window: int,
    causal: bool,
) -> torch.Tensor:
    # q, k, v: [batch, heads, seq, head_dim]
    _, _, q_len, head_dim = q.shape
    k_len = k.shape[-2]
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    mask = torch.ones((q_len, k_len), dtype=torch.bool, device=q.device)
    for q_idx in range(q_len):
        if window < 0:
            start = 0
            end = q_idx if causal else k_len - 1
        else:
            if causal:
                start = max(0, q_idx - window)
                end = q_idx
            else:
                start = max(0, q_idx - window)
                end = min(k_len - 1, q_idx + window)
        mask[q_idx, start : end + 1] = False

    scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


@pytest.mark.parametrize("causal,window", [(True, 2), (False, 2)])
def test_attention_core_sliding_window_matches_manual(causal, window):
    torch.manual_seed(0)
    batch, seq_len, num_heads, head_dim = 1, 6, 2, 8
    q = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.float32)
    k = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.float32)
    v = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.float32)

    core = AttentionCore(causal=causal, attention_dropout=0.0, sliding_window=window).eval()
    with torch.no_grad():
        out = core(q, k, v)

    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    expected = _manual_sliding_window_attention(q_t, k_t, v_t, window, causal)
    expected = expected.transpose(1, 2)

    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "batch,seq_len,num_heads,head_dim",
    [
        (1, 16, 2, 16),
        (2, 32, 4, 16),
        (2, 64, 8, 32),
        (4, 128, 8, 64),
    ],
)
def test_flash_attention_matches_sdpa(batch, seq_len, num_heads, head_dim):
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for flash_attn")
    pytest.importorskip("flash_attn")

    device = torch.device("cuda")
    q = torch.randn(batch, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)

    flash = FlashAttention(causal=True, attention_dropout=0.0).to(device).eval()
    sdpa = AttentionCore(causal=True, attention_dropout=0.0).to(device).eval()

    with torch.no_grad():
        out_flash = flash(q, k, v)
        out_sdpa = sdpa(q, k, v)

    torch.testing.assert_close(out_sdpa, out_flash, rtol=1e-3, atol=1e-3)


@pytest.mark.gpu
@pytest.mark.parametrize("window", [2, 8])
def test_flash_attention_matches_sdpa_sliding_window(window):
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for flash_attn")
    pytest.importorskip("flash_attn")

    device = torch.device("cuda")
    batch, seq_len, num_heads, head_dim = 2, 64, 8, 32
    q = torch.randn(batch, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)

    flash = FlashAttention(causal=True, attention_dropout=0.0, sliding_window=window).to(device).eval()
    sdpa = AttentionCore(causal=True, attention_dropout=0.0, sliding_window=window).to(device).eval()

    with torch.no_grad():
        out_flash = flash(q, k, v)
        out_sdpa = sdpa(q, k, v)

    torch.testing.assert_close(out_sdpa, out_flash, rtol=1e-3, atol=1e-3)
