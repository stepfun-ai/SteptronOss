from playground.data.sft.step_recipe0226_qwen_tokenized import (
    Step3SFTDataQwenTokenizedConfig,
)
from playground.pretrain.qwen3.qwen3_30a3b import Qwen3_30A3BConfig
from playground.sft.qwen3.qwen3_sft_base import Exp as BaseExp


class Exp(BaseExp):
    model_cfg = Qwen3_30A3BConfig

    data_cfg = Step3SFTDataQwenTokenizedConfig

    def __init__(self):
        super().__init__()
        self.trainer_cfg.micro_batch_size = 1
        self.trainer_cfg.global_batch_size = 8
        self.trainer_cfg.global_seq_length = 65536

        # self.trainer_cfg.train_iters = 1000
        self.trainer_cfg.log_interval = 1

        self.checkpoint_cfg.load_option.none(but=["model"])
        self.checkpoint_cfg.load_safetensors = "/oss/opensources_model/Qwen3-30B-A3B-Base"
        self.checkpoint_cfg.save_safetensors = True
        self.checkpoint_cfg.save_dir = "/oss/checkpoints/"
        self.checkpoint_cfg.save_option.all()
        self.checkpoint_cfg.save_interval = 100
        self.model_cfg.recompute = True

    def configure_optimizable(self):
        from steptronoss.utils.optimizable import set_optimization

        set_optimization(
            # grouped_gemm="nv_grouped_gemm",
            AttentionCore="flash-attn",
            default="torch_compile",
        )


if __name__ == "__main__":
    Exp().train()
