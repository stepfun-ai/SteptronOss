from playground.rlvr.qwen3_1p5b_rlvr_math import Exp as BaseExp
from playground.rlvr.qwen3_1p5b_rlvr_math import RLVRTrainerConfig
from steptronoss.exp.rl import FullyAsyncFlowControllerConfig


class FullyAsyncMathFlowControllerConfig(FullyAsyncFlowControllerConfig):
    def __init__(self):
        super().__init__()
        self.prompt_per_iter = 16
        self.max_untrained_prompts = 32
        self.max_staleness = 2


class FullyAsyncRLVRTrainerConfig(RLVRTrainerConfig):
    flow_cfg = FullyAsyncMathFlowControllerConfig


class Exp(BaseExp):
    trainer_cfg: FullyAsyncRLVRTrainerConfig = FullyAsyncRLVRTrainerConfig

    def __init__(self):
        super().__init__()
        self.checkpoint_cfg.save_path = "/oss/checkpoints/qwen3_1p5b_rlvr_math_fully_async"


if __name__ == "__main__":
    Exp().entrypoint()
