import torch
from configurize import Ref

from playground.pretrain.step3p5.step3p5_flash import Step3p5FlashModelConfig
from steptronoss.model.common.parallel_embedding import ImageInsertInputEmbeddingConfig
from steptronoss.model.common.vit import VisionTransformerConfig


class Step3p5vFlashVisionConfig(VisionTransformerConfig):
    """Vision encoder paired with the Step3p5v multimodal text stack."""

    def __init__(self):
        super().__init__()
        self.in_channels = 3
        self.image_size = 728
        self.patch_size = 14

        self.hidden_size = 1792
        self.ffn_hidden_size = 15360
        self.num_layers = 63
        self.num_attention_heads = 16

        self.vit_downsampler_hidden_dim = 4096
        self.output_dim = 8192

        self.layernorm_epsilon = 1e-5
        self.attention_dropout = 0.0
        self.layer_scale_init_value = None


class Step3p5vFlashInputEmbeddingConfig(ImageInsertInputEmbeddingConfig):
    """Text embedding with image insertion support."""

    encoder_cfg = Step3p5vFlashVisionConfig

    def __init__(self):
        super().__init__()
        self.vocab_size = 128896
        self.hidden_size = Ref("..hidden_size")
        self.embedding_weights_in_fp32 = False
        self.fp32_residual_connection = False

        self.img_start_token = 128000
        self.encoder_no_grad = True
        self.projector_bias = False


class Step3p5vFlashModelConfig(Step3p5FlashModelConfig):
    """Step3p5v multimodal model built on top of Step3.5 Flash."""

    tok_embed_cfg = Step3p5vFlashInputEmbeddingConfig

    def __init__(self):
        super().__init__()
        self.params_dtype = torch.bfloat16

    def build_model(self):
        from steptronoss.model.step3p5v import Step3p5vModel

        return Step3p5vModel(cfg=self, layer_map=self.build_layer_map())
