from steptronoss.exp.base_exp import (
    GradientManagerConfig,
    ProfilerConfig,
)
from steptronoss.exp.checkpointing import CheckpointConfig
from steptronoss.exp.lr_schedulers import ConstantSchedulerConfig
from steptronoss.exp.ntp import MoePretrainMetricConfig, NTPTrainerConfig
from steptronoss.exp.resources import TorchrunResourceConfig
from steptronoss.exp.sft import SFTExp


class Glm5SFTResourceConfig(TorchrunResourceConfig):
    def __init__(self):
        super().__init__()
        self.replica = 13
        self.gpu = 8
        self.envs |= {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }


class Exp(SFTExp):
    log_dir = "/oss/logs/"

    resource_cfg = Glm5SFTResourceConfig

    trainer_cfg = NTPTrainerConfig
    optimizer_cfg = GradientManagerConfig
    scheduler_cfg = ConstantSchedulerConfig
    checkpoint_cfg = CheckpointConfig
    metric_cfg = MoePretrainMetricConfig
    profiler_cfg = ProfilerConfig

    def __init__(self):
        super().__init__()
        self.model_cfg.tp_cfg.gradient_accumulation_fusion = False

    def configure_optimizable(self):
        pass

    def train(self):
        self.update_from_args()
        self.sanity_check()
        trainer_cls = self.trainer_cfg.get_trainer_cls()
        trainer = trainer_cls(exp=self)

        self.configure_optimizable()
        trainer.train()


if __name__ == "__main__":
    Exp().train()
