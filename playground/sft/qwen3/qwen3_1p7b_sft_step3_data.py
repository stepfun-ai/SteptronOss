from playground.data.sft.reasoning_GCMKSTIDF_sft_stage1_1203_compile_qwen import (
    Step3SFTDataQwenTokenizedConfig,
)
from playground.pretrain.qwen3.qwen3_1p7b import Qwen3_1p7BConfig
from playground.sft.qwen3.qwen3_sft_base import Exp as BaseExp


class Exp(BaseExp):
    model_cfg = Qwen3_1p7BConfig

    data_cfg = Step3SFTDataQwenTokenizedConfig

    def __init__(self):
        super().__init__()
        self.trainer_cfg.micro_batch_size = 1
        self.trainer_cfg.global_batch_size = 8
        self.trainer_cfg.global_seq_length = 65536

        # self.trainer_cfg.train_iters = 1000 # NOTE disable
        self.trainer_cfg.log_interval = 1

        self.checkpoint_cfg.load_option.none(but=["model"])
        self.checkpoint_cfg.load_safetensors = "/mnt/step2-alignment-jfs/zane/opensources_model/Qwen3-1.7B-Base/"
        self.checkpoint_cfg.save_safetensors = True
        self.checkpoint_cfg.save_dir = "/mnt/shared-storage/tenant/tmp/zhy/tmp/"
        self.checkpoint_cfg.save_option.all()
        self.checkpoint_cfg.save_interval = 100

    def configure_optimizable(self):
        return


if __name__ == "__main__":
    Exp().train()
