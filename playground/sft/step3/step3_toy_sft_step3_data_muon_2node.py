"""
2-node variant of step3_toy_sft_step3_data_muon.
Runs on 2 nodes (16 GPUs total) with data parallelism.
"""

from playground.sft.step3.step3_toy_sft_step3_data_muon import Exp as BaseExp
from steptronoss.exp.resources import TorchrunResourceConfig


class TwoNodeResourceConfig(TorchrunResourceConfig):
    def __init__(self):
        super().__init__()
        self.replica = 2
        self.gpu = 8


class Exp(BaseExp):
    resource_cfg = TwoNodeResourceConfig

    def __init__(self):
        super().__init__()
        # Scale global batch size for 2 nodes (2x data parallelism)
        self.trainer_cfg.global_batch_size = 16


if __name__ == "__main__":
    Exp().train()
