import os

import pytest
import torch
import torch.distributed as dist

from steptronoss.core.parallel_state import PM
from steptronoss.exp.base_exp import ParallelConfig
from steptronoss.initialize import set_mpu_random_seed
from steptronoss.model.vision.perception_encoders import (
    PerceptionEmbedConfig,
    PerceptionEncoderConfig,
    PerceptionPoolConfig,
    VisionAttentionConfig,
    VisionMLPConfig,
)


class TinyAttnConfig(VisionAttentionConfig):
    def __init__(self):
        super().__init__()
        self.num_attention_heads = 4
        self.num_attention_groups = 4
        self.head_dim = 8


class TinyEmbedConfig(PerceptionEmbedConfig):
    def __init__(self):
        super().__init__()
        self.patch_size = 4
        self.image_size = 32
        self.use_abs_posemb = False
        self.use_cls_token = False


class TinyPoolConfig(PerceptionPoolConfig):
    def __init__(self):
        super().__init__()
        self.pool_type = "none"


class TinyMLPConfig(VisionMLPConfig):
    def __init__(self):
        super().__init__()
        self.mlp_ratio = 2.0
        self.ffn_hidden_size = 64


class TinyPerceptionConfig(PerceptionEncoderConfig):
    attn_cfg = TinyAttnConfig
    embed_cfg = TinyEmbedConfig
    pool_cfg = TinyPoolConfig
    ffn_cfg = TinyMLPConfig
    params_dtype: torch.dtype
    """Params dtype for TP config refs."""

    def __init__(self):
        super().__init__()
        self.params_dtype = torch.float32
        self.hidden_size = 32
        self.num_layers = 2


@pytest.mark.node2
@pytest.mark.xdist_group("torchrun")
def test_perception_encoder_forward():
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        pytest.skip("TP attention/FFN requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        PM.initialize(backend="nccl")
    if not getattr(PM, "parallels", None):
        PM.set_mesh(ParallelConfig(tensor_model_parallel_size=2))
    set_mpu_random_seed(1234)
    cfg = TinyPerceptionConfig()
    cfg.sanity_check()
    model = cfg.build_model().to(device)
    x = torch.randn(2, 3, 32, 32, device=device)
    assert x.is_cuda
    assert next(model.parameters()).is_cuda
    out = model(x)
    assert out.shape == (2, 4, 128)
