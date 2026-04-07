import pytest
import torch
import torch.distributed as dist

from steptronoss.core.parallel_state import PM
from steptronoss.model.common.parallel_embedding import ImageInsertEmbedding
from steptronoss.model.common.vit import VisionTransformerConfig


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
    yield

    if created_group and dist.is_initialized():
        dist.destroy_process_group()


def test_insert_features_replaces_tokens_after_flag():
    input_embeddings = torch.zeros(6, 1, 2)
    input_ids = torch.tensor([[9, 1, 2, 3, 4, 5]])
    image_features = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    output = ImageInsertEmbedding.insert_features(
        input_embeddings=input_embeddings,
        image_features=image_features,
        input_ids=input_ids,
        flag=9,
    )

    assert torch.equal(output[1:3, 0], image_features[0])
    assert torch.equal(output[0, 0], torch.zeros(2))


def test_insert_features_accepts_variable_length_feature_lists():
    input_embeddings = torch.zeros(7, 1, 1)
    input_ids = torch.tensor([[7, 0, 7, 0, 0, 0, 0]])
    image_features = [
        torch.tensor([[1.0], [2.0]]),
        torch.tensor([[3.0]]),
    ]

    output = ImageInsertEmbedding.insert_features(
        input_embeddings=input_embeddings,
        image_features=image_features,
        input_ids=input_ids,
        flag=7,
    )

    assert torch.equal(output[1:3, 0], image_features[0])
    assert torch.equal(output[3:4, 0], image_features[1])


def test_vision_transformer_forward_shape(single_rank_gloo_dist):
    cfg = VisionTransformerConfig()
    cfg.in_channels = 3
    cfg.image_size = 28
    cfg.patch_size = 14
    cfg.hidden_size = 32
    cfg.ffn_hidden_size = 64
    cfg.num_layers = 2
    cfg.num_attention_heads = 4
    cfg.vit_downsampler_hidden_dim = 24
    cfg.output_dim = 16
    cfg.layernorm_epsilon = 1e-5
    cfg.attention_dropout = 0.0
    cfg.layer_scale_init_value = None

    PM.set_mesh(cfg.parallel_cfg)
    model = cfg.build_model()
    pixel_values = torch.randn(2, 3, 28, 28)
    with PM.use_mesh(cfg.parallel_cfg):
        output = model(pixel_values)

    assert output.shape == (2, 1, 16)
