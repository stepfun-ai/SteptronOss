from steptronoss.exp.base_exp import SchedulerConfig


class LinearSchedulerConfig(SchedulerConfig):
    scheduler_unit: str = "iter"
    """How to update scheduler counter ('iter'|'sample'|'token')"""

    def build_scheduler(self, optimizer, *args):
        from steptronoss.optimizer.hparam_scheduler import (
            Scheduler,
            FuncLinear,
        )

        scheduler = Scheduler(
            optimizer=optimizer,
            base_lr=self.lr,
            base_wd=self.weight_decay,
            lr_func=FuncLinear(
                n=self.total_schedule,
                warmup=self.warmup_schedule,
                min_scale=self.min_lr / self.lr,
            ),
            wd_func=lambda x: 1.0,  # constant is ok
        )

        return scheduler

class CosineSchedulerConfig(SchedulerConfig):
    def build_scheduler(self, optimizer, *args):
        from steptronoss.optimizer.hparam_scheduler import (
            Scheduler,
            FuncCosineDecr,
        )

        scheduler = Scheduler(
            optimizer=optimizer,
            base_lr=self.lr,
            base_wd=self.weight_decay,
            lr_func=FuncCosineDecr(
                n=self.total_schedule,
                warmup=self.warmup_schedule,
                min_scale=self.min_lr / self.lr,
            ),
            wd_func=lambda x: 1.0,  # constant is ok
        )

        return scheduler

class TokenBasedSchedulerConfig(SchedulerConfig):
    scheduler_unit = "token"