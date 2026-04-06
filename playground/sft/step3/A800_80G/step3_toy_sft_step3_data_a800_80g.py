"""
2-node A800 80G tuning variant of ``step3_toy_sft_step3_data``.
"""

from playground.sft.step3.step3_toy_sft_step3_data import Exp as BaseExp
from steptronoss.exp.resources import TorchrunResourceConfig


class TwoNodeA80080GResourceConfig(TorchrunResourceConfig):
    def __init__(self):
        super().__init__()
        self.replica = 1
        self.gpu = 8


class Exp(BaseExp):
    resource_cfg = TwoNodeA80080GResourceConfig

    def __init__(self):
        super().__init__()
        self.trainer_cfg.global_seq_length = 64 * 1024
        self.trainer_cfg.offload_optimizer_state = True

        self.model_cfg.parallel_cfg.context_parallel_size = 1
        self.model_cfg.parallel_cfg.tensor_model_parallel_size = 8
        self.model_cfg.tp_cfg.sequence_parallel = True

    def configure_optimizable(self):
        from steptronoss.utils.optimizable import set_optimization

        set_optimization(
            grouped_gemm="nv_grouped_gemm",
            AttentionCore="flash-attn",
        )


if __name__ == "__main__":
    Exp().train()
