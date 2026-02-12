import os

import pytest
import torch
import torch.distributed as dist

from steptronoss.core.parallel_state import PM
from steptronoss.exp.abstract import ModelConfig
from steptronoss.exp.base_exp import MegatronTPConfig, ParallelConfig
from steptronoss.initialize import set_mpu_random_seed
from steptronoss.model.common.encoder_as_embedding import (
    ImageForInsert,
    StepEncoderInputEmbedding,
    WithEncoderInputEmbeddingConfig,
)


class TinyEncoder(ModelConfig):
    out_len: int
    """Output sequence length."""
    out_dim: int
    """Output hidden size."""

    def __init__(self):
        super().__init__()
        self.out_len = 2
        self.out_dim = 32

    def build_model(self):
        return TinyEncoderModule(cfg=self)


class TinyEncoderModule(torch.nn.Module):
    def __init__(self, cfg: TinyEncoder):
        super().__init__()
        self.cfg = cfg

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.shape[0]
        return (
            torch.zeros(
                batch,
                self.cfg.out_len,
                self.cfg.out_dim,
                device=images.device,
                dtype=images.dtype,
            )
            + 7
        )


class TinyEncoderEmbeddingConfig(WithEncoderInputEmbeddingConfig):
    encoder_cfg = TinyEncoder
    tp_cfg = MegatronTPConfig

    def __init__(self):
        super().__init__()
        self.vocab_size = 128
        self.hidden_size = 32
        self.embedding_weights_in_fp32 = False
        self.fp32_residual_connection = False
        self.tp_cfg.params_dtype = torch.float32

    def build_adapter(self):
        return torch.nn.Identity()


@pytest.mark.node2
@pytest.mark.xdist_group("torchrun")
def test_encoder_as_embedding_insert_feature():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        pytest.skip("encoder as embedding requires CUDA")

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    if not dist.is_initialized():
        PM.initialize(backend="nccl")
    if not getattr(PM, "parallels", None):
        PM.set_mesh(ParallelConfig(tensor_model_parallel_size=2))
    set_mpu_random_seed(1234)

    cfg = TinyEncoderEmbeddingConfig()
    cfg.sanity_check()
    model = StepEncoderInputEmbedding(cfg).to(device)
    model.eval()

    flag = 99
    input_ids = torch.tensor([[1, 2, flag, 3, 4, 5]], device=device)
    image_features = torch.full((1, 2, cfg.hidden_size), 7.0, device=device)
    images = [
        ImageForInsert(
            insert_start_token=flag,
            image_features=image_features,
        )
    ]

    with torch.no_grad():
        output = model(input_ids, images=images)

    assert output.shape == (input_ids.shape[1], input_ids.shape[0], cfg.hidden_size)
    inserted = output[3:5, 0]
    assert torch.allclose(inserted, image_features[0], atol=0, rtol=0)


@pytest.mark.node2
@pytest.mark.xdist_group("torchrun")
def test_encoder_as_embedding_insert_image():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        pytest.skip("encoder as embedding requires CUDA")

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    if not dist.is_initialized():
        PM.initialize(backend="nccl")
    if not getattr(PM, "parallels", None):
        PM.set_mesh(ParallelConfig(tensor_model_parallel_size=2))
    set_mpu_random_seed(1234)

    cfg = TinyEncoderEmbeddingConfig()
    cfg.sanity_check()
    model = StepEncoderInputEmbedding(cfg).to(device)
    model.eval()

    flag = 99
    input_ids = torch.tensor([[1, 2, flag, 3, 4, 5]], device=device)
    image_features = torch.full((1, 2, cfg.hidden_size), 7.0, device=device)
    images = [
        ImageForInsert(
            insert_start_token=flag,
            images=torch.zeros(1, 3, 8, 8, device=device),
        )
    ]

    with torch.no_grad():
        output = model(input_ids, images=images)

    assert output.shape == (input_ids.shape[1], input_ids.shape[0], cfg.hidden_size)
    inserted = output[3:5, 0]
    assert torch.allclose(inserted, image_features[0], atol=0, rtol=0)
