from steptronoss.exp.base_exp import (
    GradientManagerConfig,
    ProfilerConfig,
)
from steptronoss.exp.checkpointing import CheckpointConfig
from steptronoss.exp.lr_schedulers import ConstantSchedulerConfig
from steptronoss.exp.ntp import NTPTrainerConfig, PretrainMetricConfig
from steptronoss.exp.sft import SFTExp


class Exp(SFTExp):
    log_dir = "./tensorboard_logs/"

    trainer_cfg = NTPTrainerConfig
    optimizer_cfg = GradientManagerConfig
    scheduler_cfg = ConstantSchedulerConfig
    checkpoint_cfg = CheckpointConfig
    metric_cfg = PretrainMetricConfig
    profiler_cfg = ProfilerConfig

    def __init__(self):
        super().__init__()
        # turn on if apex installed
        self.model_cfg.tp_cfg.gradient_accumulation_fusion = False

    def configure_optimizable(self):
        from steptronoss.utils.optimizable import set_optimization

        set_optimization(AttentionCore="flash-attn", default="torch_compile")

    def train(self):
        self.update_from_args()
        self.sanity_check()
        trainer_cls = self.trainer_cfg.get_trainer_cls()
        trainer = trainer_cls(exp=self)

        self.configure_optimizable()
        trainer.train()


if __name__ == "__main__":
    Exp().train()
