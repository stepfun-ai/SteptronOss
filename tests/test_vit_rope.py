from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from steptronoss.core.parallel_state import PM
from steptronoss.exp.base_exp import ParallelConfig
from steptronoss.model.common.vit import EncoderRope2D, VisionAttention, VisionTransformerConfig


@pytest.fixture()
def single_rank_gloo_dist(tmp_path):
    created_group = False
    if not dist.is_initialized():
        init_file = tmp_path / "dist_init"
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=0,
            world_size=1,
        )
        created_group = True

    PM.initialize(backend="gloo")
    parallel_cfg = ParallelConfig()
    parallel_cfg.tensor_model_parallel_size = 1
    parallel_cfg.pipeline_model_parallel_size = 1
    parallel_cfg.context_parallel_size = 1
    parallel_cfg.expert_model_parallel_size = 1
    parallel_cfg.expert_tensor_parallel_size = 1
    parallel_cfg.virtual_pipeline_model_parallel_size = 1
    PM.set_mesh(parallel_cfg)
    yield

    if created_group and dist.is_initialized():
        dist.destroy_process_group()


def _vision_attention_cfg(*, use_rope2d: bool) -> VisionTransformerConfig:
    cfg = VisionTransformerConfig()
    cfg.image_size = 4
    cfg.patch_size = 2
    cfg.hidden_size = 8
    cfg.num_attention_heads = 2
    cfg.attention_dropout = 0.0
    cfg.use_cls_token = False
    cfg.use_rope2d = use_rope2d
    return cfg


def test_encoder_rope2d_applies_cached_grid_positions() -> None:
    rope = EncoderRope2D(dim=8, max_grid_height=4, max_grid_width=4)
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)

    q_out, k_out = rope(q, k, grid_hw=(2, 2))

    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    assert torch.isfinite(q_out).all().item()
    assert torch.isfinite(k_out).all().item()
    torch.testing.assert_close(q_out[:, :, 0], q[:, :, 0])
    torch.testing.assert_close(k_out[:, :, 0], k[:, :, 0])
    assert not torch.allclose(q_out[:, :, 1:], q[:, :, 1:])


def test_vision_attention_rope_changes_attention_result(single_rank_gloo_dist) -> None:
    torch.manual_seed(1234)
    plain_attn = VisionAttention(_vision_attention_cfg(use_rope2d=False))
    rope_attn = VisionAttention(_vision_attention_cfg(use_rope2d=True))
    rope_attn.load_state_dict(plain_attn.state_dict(), strict=True)

    x = torch.randn(1, 4, 8)

    plain_out = plain_attn(x, grid_hw=(2, 2))
    rope_out = rope_attn(x, grid_hw=(2, 2))

    assert plain_out.shape == rope_out.shape
    assert torch.isfinite(rope_out).all().item()
    assert not torch.allclose(plain_out, rope_out)
