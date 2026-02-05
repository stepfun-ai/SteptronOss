from playground.data.sft.reasoning_GCMKSTIDF_sft_stage1_1203_compile_step3 import (
    Step3SFTDataStep3TokenizedConfig,
)
from playground.pretrain.step3p5.step3p5_toy import Step3p5ToyModelConfig
from playground.sft.qwen3.qwen3_sft_base import Exp as BaseExp

# sys.excepthook = lambda a, b, c: [print("HoldOneError"), time.sleep(3600)]


class Exp(BaseExp):
    model_cfg = Step3p5ToyModelConfig

    data_cfg = Step3SFTDataStep3TokenizedConfig

    def __init__(self):
        super().__init__()
        self.trainer_cfg.micro_batch_size = 1
        self.trainer_cfg.global_batch_size = 8
        self.trainer_cfg.global_seq_length = 65536

        self.trainer_cfg.train_iters = 1000
        self.trainer_cfg.log_interval = 1

        self.checkpoint_cfg.load_option.none(but=["model"])
        # self.checkpoint_cfg.load_safetensors = "/mnt/step2-alignment-jfs/zane/opensources_model/Qwen3-30B-A3B-Base"
        self.checkpoint_cfg.save_safetensors = True
        self.checkpoint_cfg.save_dir = "/mnt/shared-storage/tenant/tmp/zhy/tmp/"
        self.checkpoint_cfg.save_option.all()
        self.checkpoint_cfg.save_interval = 100
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
