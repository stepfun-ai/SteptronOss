import pytest
import torch

from steptronoss.model.common.rope import YARNRoPE
from steptronoss.model.glm5 import Glm5YARNRoPE


def test_yarn_rope_cache_stays_fp32_under_default_bf16():
    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        rope = YARNRoPE(dim=8, max_position_embeddings=16)
        assert rope._cos_cache.dtype == torch.float32
        assert rope._sin_cache.dtype == torch.float32

        rope = rope.to(torch.bfloat16)
        assert rope._cos_cache.dtype == torch.float32
        assert rope._sin_cache.dtype == torch.float32

        feature = torch.randn(1, 4, 1, 8, dtype=torch.bfloat16)
        position_id = torch.arange(4, dtype=torch.int32)
        output = rope(feature, position_id)

        assert output.dtype == torch.bfloat16
        assert rope._cos_cache.dtype == torch.float32
        assert rope._sin_cache.dtype == torch.float32
    finally:
        torch.set_default_dtype(old_default_dtype)


def test_glm5_rope_can_reproduce_glm_precision_bug():
    rope = Glm5YARNRoPE(dim=16, max_position_embeddings=32, theta=1_000_000.0)
    freqs = rope._get_frequencies(device="cpu")
    exact = 1.0 / (rope.theta ** torch.linspace(0, 1, steps=rope.dim // 2 + 1, dtype=torch.float32)[:-1])

    assert torch.allclose(freqs, exact.to(torch.bfloat16).to(torch.float32), atol=0, rtol=0)
    assert not torch.allclose(freqs, exact, atol=0, rtol=0)


def test_glm5_rope_applies_rotation_in_input_dtype():
    torch.manual_seed(1234)
    feature = torch.randn(1, 32, 1, 16, dtype=torch.bfloat16)
    position_id = torch.arange(32, dtype=torch.int32)
    kwargs = dict(dim=16, max_position_embeddings=64, theta=1_000_000.0)

    base_rope = YARNRoPE(**kwargs)
    glm_rope = Glm5YARNRoPE(**kwargs)

    base_output = base_rope(feature, position_id)
    glm_output = glm_rope(feature, position_id)

    cos_cache, sin_cache = glm_rope._check_set_cos_sin_cache(glm_rope.max_position_embeddings, "cpu")
    cos = cos_cache[None, position_id, None, :].to(feature.dtype)
    sin = sin_cache[None, position_id, None, :].to(feature.dtype)
    expected = feature * cos + glm_rope.rotate_half(feature) * sin

    assert torch.equal(glm_output, expected)
    assert not torch.equal(base_output, glm_output)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="RoPE cuda cache test requires CUDA")
@pytest.mark.skipif(not torch.cuda.is_bf16_supported(), reason="bf16 not supported")
def test_yarn_rope_cuda_bfloat16_keeps_fp32_cache():
    rope = YARNRoPE(dim=8, max_position_embeddings=16).cuda().bfloat16()

    assert rope._cos_cache.device.type == "cuda"
    assert rope._sin_cache.device.type == "cuda"
    assert rope._cos_cache.dtype == torch.float32
    assert rope._sin_cache.dtype == torch.float32

    feature = torch.randn(1, 4, 1, 8, device="cuda", dtype=torch.bfloat16)
    position_id = torch.arange(4, device="cuda", dtype=torch.int32)
    output = rope(feature, position_id)

    assert output.dtype == torch.bfloat16
    assert rope._cos_cache.device.type == "cuda"
    assert rope._sin_cache.device.type == "cuda"
    assert rope._cos_cache.dtype == torch.float32
    assert rope._sin_cache.dtype == torch.float32
