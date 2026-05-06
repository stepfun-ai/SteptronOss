from __future__ import annotations

import torch
from configurize import Ref

from playground.pretrain.qwen3.qwen3_8b import Qwen3_8BConfig
from steptronoss.model.common.parallel_embedding import ImageInsertInputEmbeddingConfig
from steptronoss.model.common.vit import VisionTransformerConfig


class Step3VVisionConfig(VisionTransformerConfig):
    """PE-Lang G/14 728px vision encoder used by Step3-VL-10B."""

    def __init__(self):
        super().__init__()
        self.in_channels = 3
        self.image_size = 728
        self.patch_size = 14

        self.hidden_size = 1536
        self.ffn_hidden_size = 8960
        self.num_layers = 47
        self.num_attention_heads = 16

        self.vit_downsampler_hidden_dim = 3072
        self.output_dim = self.hidden_size * 4

        self.layernorm_epsilon = 1e-5
        self.attention_dropout = 0.0
        self.use_rope2d = True
        self.rope_theta = 10000
        self.rope_max_freq = 10
        self.rope_num_freqs = 1
        self.rope_theta_rescale_factor = 1.0
        self.layer_scale_init_value = 0.1
        self.use_cls_token = False
        self.use_ln_pre = True
        self.patch_embed_bias = False
        self.vit_downsampler1_kernel_size = 3
        self.vit_downsampler1_padding = 1
        self.vit_downsampler2_kernel_size = 3
        self.vit_downsampler2_padding = 1


class Step3VInputEmbeddingConfig(ImageInsertInputEmbeddingConfig):
    """Qwen3 token embedding with Step3-VL image feature insertion."""

    encoder_cfg = Step3VVisionConfig
    """Vision encoder config for raw image tensors."""

    image_token_id: int = 151679
    """Placeholder token id reserved for image feature slots."""

    img_end_token: int = 151681
    """Image end token id used by Step3-VL tokenizer data."""

    patch_start_token: int = 151689
    """Patch start token id kept for shared multimodal data helpers."""

    patch_end_token: int = 151690
    """Patch end token id kept for shared multimodal data helpers."""

    patch_new_line_token: int = 151691
    """Patch newline token id kept for shared multimodal data helpers."""

    def __init__(self):
        super().__init__()
        self.vocab_size = 151936
        self.hidden_size = Ref("..hidden_size")
        self.embedding_weights_in_fp32 = False
        self.fp32_residual_connection = False

        self.img_start_token = 151680
        self.encoder_no_grad = True
        self.projector_bias = False


class Step3V10BConfig(Qwen3_8BConfig):
    """Step3-VL-10B model config: Qwen3-8B text stack plus PE-G/14 vision."""

    tok_embed_cfg = Step3VInputEmbeddingConfig

    def __init__(self):
        super().__init__()
        self.params_dtype = torch.bfloat16
        self.attn_cfg.max_position_embeddings = 40960

    def build_model(self):
        from steptronoss.model.qwen_vl import QwenImageInsertModel

        return QwenImageInsertModel(cfg=self, layer_map=self.build_layer_map())


Step3VL10BConfig = Step3V10BConfig
