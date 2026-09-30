"""
Single-node SFT debugging experiment for the GLM-5 runtime path.

Launch from the repo root with 8 GPUs on one machine:

    torchrun --standalone --nproc-per-node=8 playground/sft/glm5/glm5_toy_sft_step3_data.py

This experiment keeps the GLM-5 block structure intact but reduces the layer
count to make single-node point-to-point checks practical.
"""

from playground.data.sft.oss260312.step_sft_data_config0311_glm5_tokenizer import (
    Recipe0311Glm5CompiledSFTDataConfig,
)
from playground.pretrain.glm5.glm5_toy import GLM5ToyConfig
from playground.sft.qwen3.qwen3_sft_base import Exp as BaseExp
from steptronoss.exp.ntp import MoePretrainMetricConfig
from steptronoss.exp.resources import TorchrunResourceConfig


class OneNodeResourceConfig(TorchrunResourceConfig):
    def __init__(self):
        super().__init__()
        self.replica = 1
        self.gpu = 8


class Exp(BaseExp):
    """Toy-model SFT debug config that mirrors the single-node GLM-5 path."""

    resource_cfg = OneNodeResourceConfig
    model_cfg = GLM5ToyConfig
    data_cfg = Recipe0311Glm5CompiledSFTDataConfig
    metric_cfg = MoePretrainMetricConfig

    def __init__(self):
        super().__init__()
        self.trainer_cfg.micro_batch_size = 1
        self.trainer_cfg.global_batch_size = 8
        # The current GLM-5 DSA path materializes dense scores/masks, so keep
        # the toy context small enough for reliable single-node debugging.
        self.trainer_cfg.global_seq_length = 1024
        self.trainer_cfg.log_interval = 1

        self.scheduler_cfg.lr = 5e-6
        self.scheduler_cfg.min_lr = 5e-7
        self.scheduler_cfg.warmup_schedule = 20

        self.checkpoint_cfg.load_option.none(but=["model"])
        self.checkpoint_cfg.load_safetensors = "/oss/opensources_model/GLM-5/"
        self.checkpoint_cfg.tokenizer_path = self.checkpoint_cfg.load_safetensors
        self.checkpoint_cfg.save_safetensors = False
        self.checkpoint_cfg.save_dir = "/oss/checkpoints/"
        self.checkpoint_cfg.save_option.none()
        self.checkpoint_cfg.save_interval = 100
        self.checkpoint_cfg.auto_resume = False

        self.model_cfg.recompute = False
        self.model_cfg.parallel_cfg.context_parallel_size = 1
        self.model_cfg.tp_cfg.sequence_parallel = False


if __name__ == "__main__":
    Exp().train()
