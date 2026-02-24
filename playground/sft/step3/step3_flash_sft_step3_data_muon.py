from playground.data.sft.reasoning_GCMKSTIDF_sft_stage1_1203_compile_step3 import (
    Step3SFTDataStep3TokenizedConfig,
)
from playground.pretrain.step3p5.step3p5_flash import Step3p5FlashModelConfig
from playground.sft.qwen3.qwen3_sft_base import Exp as BaseExp
from playground.sft.step3.muon_optimizer import Step3p5MuonConfig
from steptronoss.exp.base_exp import GradientManagerConfig
from steptronoss.exp.resources import TorchrunResourceConfig


# sys.excepthook = lambda a, b, c: [print("HoldOneError"), time.sleep(3600)]
class Step3F128kSFTResourceConfig(TorchrunResourceConfig):
    def __init__(self):
        super().__init__()
        self.replica = 8
        self.gpu = 8
        self.mounts.extend([
            "juicefs+s3://oss.i.shaipower.com/step2-alignment-jfs:/mnt/step2-alignment-jfs",
            "juicefs+s3://oss.i.shaipower.com/tenant:/mnt/shared-storage/tenant",
        ])


class MuonGradientManagerConfig(GradientManagerConfig):
    optimizer_cfg = Step3p5MuonConfig
    """Use Muon for 2D params with AdamW fallback."""


class Exp(BaseExp):
    resource_cfg = Step3F128kSFTResourceConfig

    model_cfg = Step3p5FlashModelConfig

    data_cfg = Step3SFTDataStep3TokenizedConfig

    optimizer_cfg = MuonGradientManagerConfig

    def __init__(self):
        super().__init__()
        self.trainer_cfg.micro_batch_size = 1
        self.trainer_cfg.global_batch_size = 64
        self.trainer_cfg.global_seq_length = 1024 * 128

        self.trainer_cfg.train_iters = None  # use data-dependent
        self.trainer_cfg.log_interval = 1

        self.checkpoint_cfg.load_option.none(but=["model"])
        self.checkpoint_cfg.save_safetensors = True
        self.checkpoint_cfg.save_dir = "/mnt/shared-storage/tenant/tmp/zhy/tmp/"
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
