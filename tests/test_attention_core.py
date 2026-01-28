import pytest
import torch

from steptronoss.model.common.attention_core import AttentionCore, FlashAttention


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
