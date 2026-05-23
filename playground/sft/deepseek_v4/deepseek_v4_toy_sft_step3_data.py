"""
Single-node DeepSeek V4 toy SFT run on the existing Step3 data pipeline.

Launch from the repo root:

    torchrun --standalone --nproc-per-node=8 playground/sft/deepseek_v4/deepseek_v4_toy_sft_step3_data.py

This is the first Step3-data integration target for the new DeepSeek V4
runtime path. It keeps model parallelism at TP=EP=1, so an 8-GPU launch runs as
data parallelism while the architecture parity path is still being hardened.
"""

from playground.data.sft.oss260312.step_sft_data_config0311_step3p5_tokenizer import Recipe0311CompiledSFTDataConfig
from playground.pretrain.deepseek_v4.deepseek_v4_toy import DeepseekV4ToyModelConfig
from playground.sft.qwen3.qwen3_sft_base import Exp as BaseExp


class DeepseekV4ToyStep3DataModelConfig(DeepseekV4ToyModelConfig):
    def __init__(self):
        super().__init__()
        # The current Step3 data recipe uses the Step3.5 tokenizer whose vocab fits
        # under the DeepSeek V4 vocab size. Keep the toy hidden size, but expose the
        # production-size vocab so raw token ids are in range.
        self.vocab_size = 129280
        self.tok_embed_cfg.vocab_size = self.vocab_size
        self.out_embed_cfg.vocab_size = self.vocab_size


class Exp(BaseExp):
    model_cfg = DeepseekV4ToyStep3DataModelConfig
    data_cfg = Recipe0311CompiledSFTDataConfig

    def __init__(self):
        super().__init__()
        self.trainer_cfg.micro_batch_size = 1
        self.trainer_cfg.global_batch_size = 8
        self.trainer_cfg.global_seq_length = 4096
        self.trainer_cfg.log_interval = 1

        self.scheduler_cfg.lr = 1e-5
        self.scheduler_cfg.min_lr = 1e-6
        self.scheduler_cfg.warmup_schedule = 100

        self.checkpoint_cfg.load_option.none()
        self.checkpoint_cfg.save_safetensors = False
        self.checkpoint_cfg.save_dir = "/oss/checkpoints/"
        self.checkpoint_cfg.save_path = None
        self.checkpoint_cfg.save_option.none()
        self.checkpoint_cfg.save_interval = 0
        self.checkpoint_cfg.auto_resume = False

    def configure_optimizable(self):
        # DeepSeek V4 has a 512-dim production head and attention sinks, so the
        # first toy route stays on eager attention math for HF parity.
        pass


if __name__ == "__main__":
    Exp().train()
