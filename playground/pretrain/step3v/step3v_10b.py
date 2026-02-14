import torch

from playground.pretrain.qwen3.qwen3_8 import Qwen3_8BConfig
from steptronoss.exp.base_exp import BaseExp, GradientManagerConfig, TokenizerConfig
from steptronoss.exp.checkpointing import CheckpointConfig
from steptronoss.exp.lr_schedulers import ConstantSchedulerConfig
from steptronoss.exp.ntp import NTPTrainerConfig
from steptronoss.model.common.encoder_as_embedding import (
    StepEncoderInputEmbedding,
    WithEncoderInputEmbeddingConfig,
)
from steptronoss.model.vision.perception_encoders import PE_LANG_G14_728_TP


class Step3VEncoderEmbeddingConfig(WithEncoderInputEmbeddingConfig):
    """Tokenizer embedding config with vision encoder insertion."""

    encoder_cfg = PE_LANG_G14_728_TP
    """Vision encoder config."""

    def __init__(self):
        super().__init__()
        self.vocab_size = 151936
        self.hidden_size = 4096
        self.embedding_weights_in_fp32 = False
        self.fp32_residual_connection = False
        self.encoder_no_grad = True

    def build_adapter(self):
        linear_layer = torch.nn.Linear(
            self.encoder_cfg.hidden_size * 4,
            self.hidden_size,
            bias=False,
        )
        torch.nn.init.kaiming_normal_(
            linear_layer.weight,
            mode="fan_in",
            nonlinearity="relu",
        )
        return linear_layer

    def build_model(self):
        return StepEncoderInputEmbedding(cfg=self)


class Step3VTokenizerConfig(TokenizerConfig):
    """Tokenizer config for Step3-VL-10B."""

    def __init__(self):
        super().__init__()
        self.tokenizer_path = "/mnt/shared-storage/tenant/zhy/Step3-VL-10B/"
        self.vocab_size = 151936

    def build_tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)


class Step3V_10BConfig(Qwen3_8BConfig):
    """Qwen3 8B + PE_LANG_G14_728_TP encoder."""

    tok_embed_cfg = Step3VEncoderEmbeddingConfig
    """Embedding config with vision encoder insertion."""


class Exp(BaseExp):
    model_cfg = Step3V_10BConfig
    grad_manager_cfg = GradientManagerConfig
    scheduler_cfg = ConstantSchedulerConfig
    trainer_cfg = NTPTrainerConfig
    tokenizer_cfg = Step3VTokenizerConfig
    checkpoint_cfg = CheckpointConfig

    def __init__(self):
        super().__init__()
        self.grad_manager_cfg.params_dtype = torch.bfloat16
        self.trainer_cfg.global_seq_length = 4096
        self.trainer_cfg.global_batch_size = 8
        self.resource_cfg.gpu = 4
        self.checkpoint_cfg.load_safetensors = "/mnt/shared-storage/tenant/zhy/Step3-VL-10B/"
        self.checkpoint_cfg.tokenizer_path = "/mnt/shared-storage/tenant/zhy/Step3-VL-10B/"

    def train(self):
        self.update_from_args()
        self.sanity_check()
        self.trainer_cfg.get_trainer_cls()(self).train()


if __name__ == "__main__":
    Exp().train()
