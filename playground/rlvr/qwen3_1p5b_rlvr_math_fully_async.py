"""Fully-async RLVR variant for flow-controller A/B experiments.

Historical note
---------------
Before removing the temporary synthetic-latency harness from the shared RLVR
experiment code, a balanced-base stress test showed that fully-async can beat
one-step-off once the remaining variance is concentrated in a stronger long
tail. In that harness, `prompt_per_iter=10`, infer/train were balanced, and
the fully-async point `max_untrained_prompts=22`, `max_staleness=2`,
`max_concurrent_genables=10` improved iteration time by roughly:

- `~5-8%` under the milder long-tail setting
- `~20-23%` after further increasing the long-tail strength

Those numbers are retained here as design context for future fully-async A/B
work, but the temporary synthetic-sleep code used to produce them has been
removed from the main RLVR experiment files.
"""

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
