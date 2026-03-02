from playground.data.sft.step_recipe0226_qwen_tokenized import (
    Step3SFTDataQwenTokenizedConfig,
)
from playground.pretrain.qwen3.qwen3_1p7b import Qwen3_1p7BConfig
from playground.sft.qwen3.qwen3_sft_base import Exp as BaseExp


class Exp(BaseExp):
    model_cfg = Qwen3_1p7BConfig

    data_cfg = Step3SFTDataQwenTokenizedConfig

    def __init__(self):
        super().__init__()
        self.trainer_cfg.log_interval = 1
        self.trainer_cfg.micro_batch_size = 1
        self.trainer_cfg.global_batch_size = 32
        self.trainer_cfg.global_seq_length = 128 * 1024
        # very long, use TP8
        self.model_cfg.parallel_cfg.tensor_model_parallel_size = 8

        self.scheduler_cfg.lr = 1e-5
        self.scheduler_cfg.min_lr = 1e-6
        self.scheduler_cfg.warmup_schedule = 100

        self.checkpoint_cfg.load_option.none(but=["model"])
        self.checkpoint_cfg.load_safetensors = "/oss/opensources_model/Qwen3-1.7B-Base/"
        self.checkpoint_cfg.save_safetensors = True
        self.checkpoint_cfg.save_dir = "/oss/checkpoints/"
        self.checkpoint_cfg.save_option.all()
        self.checkpoint_cfg.save_interval = 1000

    def configure_optimizable(self):
        from steptronoss.utils.optimizable import set_optimization

        set_optimization(AttentionCore="flash-attn")


if __name__ == "__main__":
    Exp().train()
