from playground.data.sft.oss260312.step_sft_data_config0311_glm5_tokenizer import (
    Recipe0311Glm5CompiledSFTDataConfig,
)
from playground.pretrain.glm5.glm5 import GLM5Config
from playground.sft.glm5.glm5_sft_base import Exp as BaseExp


class Exp(BaseExp):
    model_cfg = GLM5Config

    data_cfg = Recipe0311Glm5CompiledSFTDataConfig

    def __init__(self):
        super().__init__()
        self.trainer_cfg.micro_batch_size = 1
        self.trainer_cfg.global_batch_size = 16
        # Initial DSA path still materializes a dense mask, so start from a
        # shorter context and scale up after a dedicated sparse kernel lands.
        self.trainer_cfg.global_seq_length = 8192
        self.trainer_cfg.log_interval = 1

        self.scheduler_cfg.lr = 5e-6
        self.scheduler_cfg.min_lr = 5e-7
        self.scheduler_cfg.warmup_schedule = 100

        self.checkpoint_cfg.load_option.none(but=["model"])
        self.checkpoint_cfg.load_safetensors = "/oss/opensources_model/GLM-5/"
        self.checkpoint_cfg.tokenizer_path = self.checkpoint_cfg.load_safetensors
        self.checkpoint_cfg.save_safetensors = True
        self.checkpoint_cfg.save_dir = "/oss/checkpoints/"
        self.checkpoint_cfg.save_option.all()
        self.checkpoint_cfg.save_interval = 500

        self.model_cfg.recompute = True
        self.model_cfg.parallel_cfg.context_parallel_size = 1
        self.model_cfg.tp_cfg.sequence_parallel = False


if __name__ == "__main__":
    Exp().train()
