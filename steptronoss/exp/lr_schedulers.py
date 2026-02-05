from __future__ import annotations

from typing import Any, Literal

from steptronoss.exp.abstract import SchedulerConfig as AbstractSchedulerConfig


class SchedulerConfig(AbstractSchedulerConfig):
    lr: float
    """Initial learning rate. Depending on decay style and initial warmup,
    the learing rate at each iteration would be different."""

    min_lr: float
    """Minumum value for learning rate. The scheduler clip values below this threshold."""

    weight_decay: float
    """Weight decay coefficient for L2 regularization."""

    total_schedule: float
    """Scheduled count for lr scheduler"""

    warmup_schedule: float
    """Scheduled count for warm up"""

    scheduler_unit: Literal["iter", "sample", "token"]
    """How to update scheduler counter ('iter'|'sample'|'token')"""

    def sanity_check(self) -> None:
        assert self.min_lr <= self.lr
        assert self.scheduler_unit in ["iter", "sample", "token"]

    def build_scheduler(self, optimizer: Any, *args: Any) -> Any:
        raise NotImplementedError


class ConstantSchedulerConfig(SchedulerConfig):
    lr: float
    """Initial learning rate. Depending on decay style and initial warmup,
    the learing rate at each iteration would be different."""

    min_lr: float
    """Minumum value for learning rate. The scheduler clip values below this threshold."""

    weight_decay: float
    """Weight decay coefficient for L2 regularization."""

    total_schedule: float
    """Scheduled count for lr scheduler"""

    warmup_schedule: float
    """Scheduled count for warm up"""

    scheduler_unit: Literal["iter", "sample", "token"]
    """How to update scheduler counter ('iter'|'sample'|'token')"""

    def __init__(self):
        super().__init__()
        self.lr = 1e-4
        self.min_lr = 0.0
        self.weight_decay = 0.01
        self.total_schedule = 1000
        self.warmup_schedule = 100
        self.scheduler_unit = "iter"

    def build_scheduler(self, optimizer: Any, *args: Any) -> Any:
        from steptronoss.optimizer.hparam_scheduler import FuncConstant, Scheduler

        scheduler = Scheduler(
            optimizer=optimizer,
            base_lr=self.lr,
            base_wd=self.weight_decay,
            lr_func=FuncConstant(
                n=self.total_schedule,
                warmup=self.warmup_schedule,
                min_scale=self.min_lr / self.lr,
            ),
            wd_func=lambda x: 1.0,  # constant is ok
        )

        return scheduler


class LinearSchedulerConfig(SchedulerConfig):
    scheduler_unit: str
    """How to update scheduler counter ('iter'|'sample'|'token')"""

    def __init__(self):
        super().__init__()
        self.scheduler_unit = "iter"

    def build_scheduler(self, optimizer, *args):
        from steptronoss.optimizer.hparam_scheduler import (
            FuncLinear,
            Scheduler,
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
    def __init__(self):
        super().__init__()

    def build_scheduler(self, optimizer, *args):
        from steptronoss.optimizer.hparam_scheduler import (
            FuncCosineDecr,
            Scheduler,
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
