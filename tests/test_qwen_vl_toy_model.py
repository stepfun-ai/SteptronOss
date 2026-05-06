from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
from torch import nn

from playground.pretrain.step3v.step3v_10b import Step3V10BConfig
from steptronoss.core.parallel_state import PM
from steptronoss.model.common.mesh_connector import MeshConnector
from steptronoss.model.common.parallel_embedding import ImageForInsert, ImageInsertEmbedding
from steptronoss.model.qwen_vl import QwenImageInsertModel


class _ToyStep3VModelConfig(Step3V10BConfig):
    """Tiny Step3-VL config used by the CPU-only Qwen-VL unit test."""

    def __init__(self):
        super().__init__()
        self.params_dtype = torch.float32

        self.num_layers = 1
        self.hidden_size = 64
        self.layernorm_epsilon = 1e-6
        self.recompute = False

        self.attn_cfg.hidden_size = self.hidden_size
        self.attn_cfg.num_attention_heads = 4
        self.attn_cfg.num_attention_groups = 1
        self.attn_cfg.head_dim = 16
        self.attn_cfg.max_position_embeddings = 1024

        self.ffn_cfg.hidden_size = self.hidden_size
        self.ffn_cfg.ffn_hidden_size = 128

        self.tok_embed_cfg.hidden_size = self.hidden_size
        self.tok_embed_cfg.vocab_size = 512
        self.tok_embed_cfg.img_start_token = 7
        self.tok_embed_cfg.image_token_id = 8
        self.tok_embed_cfg.img_end_token = 9
        self.tok_embed_cfg.patch_start_token = 10
        self.tok_embed_cfg.patch_end_token = 11
        self.tok_embed_cfg.patch_new_line_token = 12

        self.out_embed_cfg.hidden_size = self.hidden_size
        self.out_embed_cfg.vocab_size = 512

        self.parallel_cfg.tensor_model_parallel_size = 1
        self.parallel_cfg.pipeline_model_parallel_size = 1
        self.parallel_cfg.virtual_pipeline_model_parallel_size = 1
        self.parallel_cfg.context_parallel_size = 1
        self.parallel_cfg.expert_model_parallel_size = 1
        self.parallel_cfg.expert_tensor_parallel_size = 1

        self.tp_cfg.sequence_parallel = False
        self.tp_cfg.async_tensor_model_parallel_allreduce = False
        self.variable_seq_lengths = True

        encoder_cfg = self.tok_embed_cfg.encoder_cfg
        encoder_cfg.image_size = 28
        encoder_cfg.patch_size = 14
        encoder_cfg.hidden_size = 32
        encoder_cfg.ffn_hidden_size = 64
        encoder_cfg.num_layers = 1
        encoder_cfg.num_attention_heads = 4
        encoder_cfg.vit_downsampler_hidden_dim = 48
        encoder_cfg.output_dim = self.hidden_size
        encoder_cfg.layer_scale_init_value = None
        encoder_cfg.parallel_cfg.tensor_model_parallel_size = 1
        encoder_cfg.parallel_cfg.pipeline_model_parallel_size = 1
        encoder_cfg.parallel_cfg.virtual_pipeline_model_parallel_size = 1
        encoder_cfg.parallel_cfg.context_parallel_size = 1
        encoder_cfg.parallel_cfg.expert_model_parallel_size = 1
        encoder_cfg.parallel_cfg.expert_tensor_parallel_size = 1


class _CpuToyEmbedding(nn.Module):
    def __init__(self, cfg: _ToyStep3VModelConfig):
        super().__init__()
        hidden_size = cfg.hidden_size
        vocab_size = cfg.tok_embed_cfg.vocab_size

        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.align_projector = nn.Linear(cfg.tok_embed_cfg.encoder_cfg.output_dim, hidden_size, bias=False)

    def forward(self, input_ids: torch.Tensor, images: list[ImageForInsert] | None = None, **kwargs) -> torch.Tensor:
        del kwargs
        input_embeddings = self.word_embeddings(input_ids).transpose(0, 1).contiguous()
        if not images:
            return input_embeddings

        for image in images:
            if image.image_features is None:
                raise ValueError("CPU toy embedding expects precomputed image_features")
            projected = self.align_projector(image.image_features.to(input_embeddings.dtype))
            input_embeddings = ImageInsertEmbedding.insert_features(
                input_embeddings=input_embeddings,
                image_features=projected,
                input_ids=input_ids,
                flag=image.insert_start_token,
            )
        return input_embeddings


class _CpuToyQwenVLModel(QwenImageInsertModel):
    def __init__(self, cfg: _ToyStep3VModelConfig):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.pp_rank = 0
        self.pp_size = 1
        self.layers = nn.ModuleList()
        self.tok_embeddings = _CpuToyEmbedding(cfg)
        with PM.use_mesh(cfg.tok_embed_cfg.encoder_cfg.parallel_cfg):
            self.encoder = cfg.tok_embed_cfg.encoder_cfg.build_model().cpu()
        self.mesh_connector = MeshConnector(
            src_mesh=cfg.parallel_cfg,
            dst_mesh=cfg.tok_embed_cfg.encoder_cfg.parallel_cfg,
            is_data_source=True,
        )

    def forward_chunk(self, last_tensor, **kwargs):
        del kwargs
        return last_tensor

    def forward_tail(self, last_tensor, **kwargs):
        del kwargs
        return last_tensor


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


def test_qwen_vl_forward_runs_decoupled_vit_then_inserts_features_on_cpu(single_rank_gloo_dist):
    cfg = _ToyStep3VModelConfig()
    PM.set_mesh(cfg.parallel_cfg)

    model = _CpuToyQwenVLModel(cfg)
    model.eval()
    vision_mesh_seen = []

    raw_encoder_forward = model.encoder.forward

    def wrapped_encoder_forward(*args, **kwargs):
        vision_mesh_seen.append(PM._cur_cfg is cfg.tok_embed_cfg.encoder_cfg.parallel_cfg)
        return raw_encoder_forward(*args, **kwargs)

    model.encoder.forward = wrapped_encoder_forward

    with torch.no_grad():
        model.tok_embeddings.word_embeddings.weight.zero_()
        model.tok_embeddings.align_projector.weight.copy_(torch.eye(cfg.hidden_size))

    input_ids = torch.tensor([[1, cfg.tok_embed_cfg.img_start_token, 2, 3, 4]], dtype=torch.long)
    image_tensor = torch.linspace(
        0.0,
        1.0,
        steps=cfg.tok_embed_cfg.encoder_cfg.in_channels
        * cfg.tok_embed_cfg.encoder_cfg.image_size
        * cfg.tok_embed_cfg.encoder_cfg.image_size,
        dtype=torch.float32,
    ).reshape(
        1,
        cfg.tok_embed_cfg.encoder_cfg.in_channels,
        cfg.tok_embed_cfg.encoder_cfg.image_size,
        cfg.tok_embed_cfg.encoder_cfg.image_size,
    )
    expected_features = model.tok_embeddings.align_projector(raw_encoder_forward(image_tensor))
    images = [
        ImageForInsert(
            insert_start_token=cfg.tok_embed_cfg.img_start_token,
            images=image_tensor,
        )
    ]

    hidden_states = model.abstract_forward(input_ids=input_ids, images=images)

    assert hidden_states.shape == (input_ids.shape[1], input_ids.shape[0], cfg.hidden_size)
    torch.testing.assert_close(hidden_states[2 : 2 + expected_features.shape[1], 0], expected_features[0])
    torch.testing.assert_close(hidden_states[0, 0], torch.zeros(cfg.hidden_size))
    torch.testing.assert_close(hidden_states[1, 0], torch.zeros(cfg.hidden_size))
    assert vision_mesh_seen == [True]
    assert PM._cur_cfg is cfg.parallel_cfg

    reshaper = model.build_reshaper()
    assert any(script.src == "vision_model.*" for script in reshaper.scripts)
    assert any(script.dst == "encoder.*" for script in reshaper.scripts)
