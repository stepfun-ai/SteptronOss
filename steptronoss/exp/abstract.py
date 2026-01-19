# Copyright (c) StepFun Inc. All rights reserved.

import os
from functools import cached_property
from typing import TYPE_CHECKING, Iterable, NoReturn, TypedDict

from configurize import Config, writable_property
from torch.nn import Module

if TYPE_CHECKING:
    from transformers import AutoTokenizer

# Objects


class _DataSample(TypedDict):
    pass


class _Scheduler:
    def step(self, increment: int):
        pass


class _Optimizer:
    def step(self):
        pass


class _Trainer:
    def __init__(self, exp: "BaseExp"):
        pass


# Configs


class ModelConfig(Config):

    def build_model(self) -> Module:
        pass


class OptimizerConfig(Config):

    def build_optimizer(self, model: Module) -> _Optimizer:
        pass


class SchedulerConfig(Config):

    lr: float
    """Initial learning rate. Depending on decay style and initial warmup,
    the learing rate at each iteration would be different."""

    def build_scheduler(self, optimizer: _Optimizer, *args) -> _Scheduler:
        pass


class MetricConfig(Config):

    def to_dict(self, rep=False) -> dict[str, str]:
        return {k: repr(v) for k, v in self.items()}

    def register_metric(self) -> NoReturn:
        pass

    def register(self) -> NoReturn:
        pass


class TrainerConfig(Config):

    def get_trainer_cls(self) -> type[_Trainer]:
        raise NotImplementedError


class TokenizerConfig(Config):

    def build_tokenizer(self) -> "AutoTokenizer":
        pass


class ParallelConfig(Config):
    parallel_definition = {
        "TP": "(p d t) -> (p d) t",
        "PP": "(p d t) -> (d t) p",
        "DP": "(p d t) -> (p t) d",
        "MP": "(p d t) -> d (p t)",
    }
    pipeline_model_parallel_size = 8
    tensor_model_parallel_size = 8

    def build_parallel(self) -> dict[str, list[list[int]]]:
        from steptronoss.core.parallel_state import PM

        args = {
            "p": self.pipeline_model_parallel_size,
            "t": self.tensor_model_parallel_size,
        }

        parallel_groups = {k: PM.define_parallel(v, **args) for k, v in self.parallel_definition.items()}

        return parallel_groups


class DataConfig(Config):

    def build_dataloader(self, dp_rank: int = 0, dp_size: int = 1) -> Iterable[_DataSample]:
        """Build a Nextable that returns a dict when call next(dataloader)."""
        pass

    def preprocess(self, batch: _DataSample) -> _DataSample:
        """Process the dict returned by next(dataloader)"""
        pass


class TaskSpec(TypedDict, total=False):
    pass


class ResourceConfig(Config):

    command: str = "{TORCHRUN} {COMMAND}"
    """Actual Command. Given 'python tools/xxx_run.py my_exp.py arg1=1':
    - TORCHRUN: 'torchrun --nproc-per-node ...'
    - COMMAND : 'my_exp.py arg1=1'
    """

    envs: dict[str, str] = {}

    task_specs: dict[str, TaskSpec] = dict(default=TaskSpec())

    def __iter__(self) -> Iterable[tuple[str, "ResourceConfig"]]:
        assert isinstance(self.task_specs, dict), self.task_specs
        for task_name, task_spec in self.task_specs.items():
            task_spec["envs"] = self.envs | task_spec.get("envs", {})
            with self.modify(**task_spec):
                self.task_specs = None
                yield task_name, self


class BaseExp(Config):

    seed: int = 1234

    log_dir: str = "./"

    suffix: str = ""

    @cached_property
    def file_path(self) -> str:
        import inspect

        return inspect.getabsfile(self.__class__)

    @writable_property
    def exp_name(self) -> str:
        base_name = os.path.basename(self.file_path).split(".")[0]
        return f"{base_name}/{self.suffix}".rstrip("/")

    @writable_property
    def exp_log_dir(self) -> str:
        return os.path.join(
            self.log_dir,
            self.exp_name,
        )

    resource_cfg = ResourceConfig

    def update_from_args(self):
        from loguru import logger

        from steptronoss.utils.arguments import parse_args
        from steptronoss.utils.logger import setup_logger

        # apply and log diffs from arguments
        args = parse_args()
        diff = self.merge(args, exists_only=True)
        diff = "\n".join([f"{k}: {s} -> {t}" for k, s, t in diff])
        try:
            setup_logger(self.exp_log_dir)
        except:
            pass
        if diff:
            logger.info(f"Modified by Args:\n{diff}", at=0)
