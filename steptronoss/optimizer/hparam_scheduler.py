# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Learning rate decay and weight decay incr functions."""

import math
from abc import abstractmethod
from typing import Callable

from loguru import logger

from steptronoss.utils import GlobalMetrics


class ScaleFunction:
    """A Scale Function that takes a CurrentTokenCount(x) and returns a lr scale

    class MyFunc(ScaleFunc):
        def __call__(x):
            return (self.n - x) / self.n
    """

    def __init__(self, n: int):
        self.n = n

    def __call__(self, x: int) -> float:
        raise NotImplementedError


class ScaleFnWithWarmUpAndClip(ScaleFunction):
    """A Scale Function with warm-up and value clip.

    class MyFunc(ScaleFnWithWarmUpAndClip):
        def get(x):
            return (self.n - x) / self.n
    schedule:
    - lr @ x = 0
    - lr / 2 + min_lr @ x = n/2
    - min_lr @ x >= n
    """

    def __init__(self, n=0, warmup=0, min_scale=0, clip_max=1, clip_min=0):
        assert warmup < n, "Warmup should less than total!"
        self.n = n - warmup
        self.warmup = warmup
        self.min_scale = min_scale
        self.clip_max = clip_max
        self.clip_min = clip_min

    @abstractmethod
    def get(self, x):
        return 1

    def __call__(self, x):
        # linear warmup
        if self.warmup and x <= self.warmup:
            return x / self.warmup

        scale = (1 - self.min_scale) * self.get(x - self.warmup) + self.min_scale
        # clip
        scale = max(scale, self.clip_min)
        scale = min(scale, self.clip_max)
        return scale


class FuncConstant(ScaleFnWithWarmUpAndClip):
    def get(self, x: int) -> float:
        return 1


class FuncLinear(ScaleFnWithWarmUpAndClip):
    def get(self, x: int) -> float:
        return 1 - (x / self.n)


class FuncCosineDecr(ScaleFnWithWarmUpAndClip):
    def get(self, x: int) -> float:
        return 0.5 * (math.cos(math.pi * (min(x, self.n) / self.n)) + 1.0)


class FuncCosineIncr(ScaleFnWithWarmUpAndClip):
    def get(self, x: int) -> float:
        return 0.5 * (math.cos(math.pi * (1 - min(x, self.n) / self.n)) + 1.0)


class FuncInvSqrt(ScaleFnWithWarmUpAndClip):
    def get(self, x: int) -> float:
        return self.warmup**0.5 / (x**0.5)


class CosineFuncWithReWarmup(object):
    def __init__(
        self,
        n,
        warmup=0,
        clip_min: float = 0,
        re_warmup=0,
        start_token=0,
        re_warmup_min: float = 0,
        re_warmup_max: float = None,
        start_bias: float = 0,
    ):
        self.end = n
        self.warmup = warmup + start_bias
        self.clip_min = clip_min
        self.re_warmup = re_warmup
        self.start_bias = start_bias
        if self.re_warmup > 0:
            self.start_token = start_token
            assert start_token > start_bias, "WARNING start_token must greater than start_bias"
            self.re_warmup_end = self.re_warmup + self.start_token
            self.re_warmup_max = max(
                (
                    self.cal_decay(
                        x=self.re_warmup_end,
                        warmup=self.warmup,
                        end=self.end,
                        clip_min=clip_min,
                        start_bias=self.start_bias,
                    )
                    if re_warmup_max is None
                    else re_warmup_max
                ),
                0,
            )
            self.re_warmup_min = max(re_warmup_min, 0)

    @staticmethod
    def cal_decay(x: int, warmup: int = 0, end: int = 0, clip_min: float = 0, start_bias: int = 0):
        if warmup and x < warmup:
            return max(0, (x - start_bias) / (warmup - start_bias)) if warmup > start_bias else 0
        else:
            decay_progress = min((x - warmup) / (end - warmup), 1)
            cos_decay = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            return max(clip_min + (1.0 - clip_min) * cos_decay, 0)

    def re_warmup_part(self, x):
        processed = min(max((x - self.start_token) / self.re_warmup, 0.0), 1)
        res = (1 - processed) * self.re_warmup_min + processed * self.re_warmup_max
        return res

    def __call__(self, x: int):
        if self.re_warmup > 0 and x < self.re_warmup_end and x >= self.start_token:
            return self.re_warmup_part(x)
        else:
            return self.cal_decay(
                x=x,
                warmup=self.warmup,
                end=self.end,
                clip_min=self.clip_min,
                start_bias=self.start_bias,
            )


class Scheduler(object):

    def __init__(
        self,
        optimizer,
        base_lr: float,
        base_wd: float,
        lr_func: Callable,
        wd_func: Callable,
        use_checkpointed=False,
    ):

        # Class values.
        self.optimizer = optimizer

        self.base_lr = base_lr
        self.base_wd = base_wd

        assert isinstance(lr_func, Callable), "lr_func should be a Callable [f(int) -> float]"
        assert isinstance(wd_func, Callable), "wd_func should be a Callable [f(int) -> float]"
        self.lr_func = lr_func
        self.wd_func = wd_func

        self.num_tokens = 0

        self.cur_lr = None
        self.cur_wd = None

        # Set the learning rate
        self.step(0)

        self.use_checkpointed = use_checkpointed

    def get_wd(self):
        wd = self.base_wd * self.wd_func(self.num_tokens)
        return wd

    def get_lr(self):
        lr = self.base_lr * self.lr_func(self.num_tokens)
        return lr

    def step(self, increment):
        """Set lr for all parameters groups."""
        self.num_tokens += increment
        self.cur_lr = self.get_lr()
        self.cur_wd = self.get_wd()
        GlobalMetrics.learning_rate.add(self.cur_lr)
        if self.optimizer:
            for group in self.optimizer.param_groups:
                group["lr"] = self.cur_lr * group.get("lr_mult", 1.0)
                group["weight_decay"] = self.cur_wd * group.get("wd_mult", 1.0)
        else:
            logger.warning("No optimizer passed, scheduler dry-running...")

    def state_dict(self):
        state_dict = {
            "base_lr": self.base_lr,
            "cur_lr": self.cur_lr,
            "base_wd": self.base_wd,
            "cur_wd": self.cur_wd,
            "num_tokens": self.num_tokens,
        }
        return state_dict

    def load_state_dict(self, sd):
        def _check_set(name, override):
            raw = getattr(self, name)
            new = sd.get(name)
            if override:
                if raw != new:
                    logger.info(f"Overriding with CKPT {name}: [{raw}] -> [{new}]")
                setattr(self, name, new)
            else:
                if raw != new:
                    logger.info(f"Diff: {name} - Cur(√) [{raw}] CKPT(×) [{new}]")

        # restore num_tokens
        _check_set("num_tokens", True)
        _check_set("base_lr", self.use_checkpointed)
        _check_set("base_wd", self.use_checkpointed)
        self.step(0)  # generate current lr & wd
        # Just warn
        _check_set("cur_lr", False)
        _check_set("cur_wd", False)
