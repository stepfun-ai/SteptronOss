from playground.data.sft.step_recipe0226 import (
    Step3SFTDataStep3TokenizedConfig,
)
from playground.pretrain.step3p5.step3p5_toy import Step3p5ToyModelConfig
from playground.sft.qwen3.qwen3_sft_base import Exp as BaseExp
from playground.sft.step3.muon_optimizer import Step3p5MuonConfig
from steptronoss.exp.base_exp import GradientManagerConfig

# sys.excepthook = lambda a, b, c: [print("HoldOneError"), time.sleep(3600)]


class MuonGradientManagerConfig(GradientManagerConfig):
    optimizer_cfg = Step3p5MuonConfig
    """Use Muon for 2D params with AdamW fallback."""


class Exp(BaseExp):
    model_cfg = Step3p5ToyModelConfig

    data_cfg = Step3SFTDataStep3TokenizedConfig

    optimizer_cfg = MuonGradientManagerConfig

    def __init__(self):
        super().__init__()
        self.trainer_cfg.micro_batch_size = 1
        self.trainer_cfg.global_batch_size = 8
        self.trainer_cfg.global_seq_length = 65536

        self.trainer_cfg.train_iters = 1000
        self.trainer_cfg.log_interval = 1

        self.checkpoint_cfg.load_option.none(but=["model"])
        # self.checkpoint_cfg.load_safetensors = "/oss/opensources_model/Qwen3-30B-A3B-Base"
        self.checkpoint_cfg.save_safetensors = True
        self.checkpoint_cfg.save_dir = "/oss/checkpoints/"
        self.checkpoint_cfg.save_option.all()
        self.checkpoint_cfg.save_interval = 100
        self.profiler_cfg.timing_log_level = 2
        self.model_cfg.recompute = True

    def configure_optimizable(self):
        from steptronoss.utils.optimizable import set_optimization

        set_optimization(
            # grouped_gemm="nv_grouped_gemm",
            AttentionCore="flash-attn",
            default=None,
        )


if __name__ == "__main__":
    Exp().train()
