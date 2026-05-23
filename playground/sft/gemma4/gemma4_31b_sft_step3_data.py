"""Gemma4 31B SFT on the shared 0311 Step3 dialog data.

This experiment keeps the training weights on the base Gemma4-31B checkpoint
while using the Gemma4-31B-it tokenizer/chat template to format SFT dialogs.
That split is intentional: the Step3 data pipeline needs a chat template, but
the actual fine-tuning is expected to start from the base text model.
"""

from playground.data.sft.oss260312.step_sft_data_config0311_gemma_tokenizer import (
    Recipe0311GemmaCompiledSFTDataConfig,
)
from playground.pretrain.gemma4.gemma4_31b import Gemma4_31BConfig
from playground.sft.qwen3.qwen3_sft_base import Exp as BaseExp
from steptronoss.exp.base_exp import TokenizerConfig
from steptronoss.exp.lr_schedulers import CosineSchedulerConfig


class Gemma4TokenizerConfig(TokenizerConfig):
    tokenizer_path: str = "/oss/opensources_model/gemma-4-31B-it"
    """Gemma4-31B-it tokenizer path used for chat-template SFT formatting."""

    vocab_size: int = 262144
    """Gemma4 tokenizer vocabulary size."""

    def build_tokenizer(self):
        from steptronoss.tokenizer.hf_compat_tokenizer import load_hf_tokenizer

        return load_hf_tokenizer(self.tokenizer_path)


class Exp(BaseExp):
    tokenizer_cfg = Gemma4TokenizerConfig

    scheduler_cfg = CosineSchedulerConfig
    model_cfg = Gemma4_31BConfig
    data_cfg = Recipe0311GemmaCompiledSFTDataConfig

    def __init__(self):
        super().__init__()
        self.trainer_cfg.micro_batch_size = 1
        self.trainer_cfg.global_batch_size = 32
        self.trainer_cfg.global_seq_length = 128 * 1024
        self.trainer_cfg.train_iters = None
        self.trainer_cfg.log_interval = 1

        # Keep LR schedule aligned with the Step3.5 Flash SFT recipe even
        # though Gemma starts from a different checkpoint family.
        self.scheduler_cfg.lr = 1e-5
        self.scheduler_cfg.min_lr = 5e-6
        self.scheduler_cfg.warmup_schedule = 140
        self.scheduler_cfg.scheduler_unit = "iter"
        self.scheduler_cfg.weight_decay = 0.1
        self.scheduler_cfg.total_schedule = None

        self.model_cfg.parallel_cfg.tensor_model_parallel_size = 4
        self.model_cfg.recompute = True

        self.checkpoint_cfg.load_option.none(but=["model"])
        self.checkpoint_cfg.load_safetensors = "/oss/opensources_model/gemma-4-31B"
        self.checkpoint_cfg.model_config_path = "/oss/opensources_model/gemma-4-31B"
        self.checkpoint_cfg.tokenizer_path = "/oss/opensources_model/gemma-4-31B-it"
        self.checkpoint_cfg.save_safetensors = True
        self.checkpoint_cfg.save_dir = "/oss/checkpoints/"
        self.checkpoint_cfg.save_option.all()
        self.checkpoint_cfg.save_interval = 100

    def configure_optimizable(self):
        # Gemma4 currently uses a Gemma-specific attention wrapper instead of the
        # common optimizable `AttentionCore` path.
        pass


if __name__ == "__main__":
    Exp().train()
