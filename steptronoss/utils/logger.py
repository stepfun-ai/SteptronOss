#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Stepfun Inc. All rights reserved.

import inspect
import os
import re
import sys

from loguru import logger
from megfile import smart_exists, smart_open, smart_remove

try:
    import wandb
except ImportError:
    wandb = None

_LOGGER_INITIALIZED = False


class LoggerHijack:
    """A Hooked logger that disable stream to stderr on some ranks"""

    CONTROL_ATTR = "at"

    def __init__(self, logger, my_rank: int, world_size: int):
        self._rank = my_rank
        self._world_size = world_size
        self._logger = logger

        self._log_ranks = [0, world_size - 1]
        # format
        max_rank_len = len(str(world_size - 1))
        loguru_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            f"<yellow>Rank:{str(my_rank).zfill(max_rank_len)}</yellow> | "
            "<level>{level: <5}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )

        self.allow_stderr = True

        self._logger.add(
            sys.stderr,
            format=loguru_format,
            enqueue=True,
            level="INFO",
            filter=lambda x: x["level"].no > 20 or self.allow_stderr,
        )

    def hook(self, name):
        raw_func = getattr(self._logger.opt(depth=1), name)

        def hooked_func(*args, **kwargs):
            at_control = kwargs.pop("at", None)
            if isinstance(at_control, bool):
                self.allow_stderr = at_control
            elif isinstance(at_control, int):
                self.allow_stderr = self._rank == (at_control % self._world_size)
            elif isinstance(at_control, list):
                self.allow_stderr = self._rank in [i % self._world_size for i in at_control]
            elif at_control == "all":
                self.allow_stderr = True
            else:
                self.allow_stderr = self._rank in self._log_ranks
                if at_control is not None:
                    kwargs["at"] = at_control

            return raw_func(*args, **kwargs)

        setattr(self._logger, name, hooked_func)


def setup_logger(
    save_dir,
    distributed_rank=None,
    world_size=None,
    filename="log",
    mode="a",
):
    """setup logger for training and testing.
    Args:
        save_dir(str): location to save log file
        distributed_rank(int): device rank when multi-gpu environment
        filename (string): log save name.
        mode(str): log file write mode, `append` or `override`. default is `a`.
    Return:
        logger instance.
    """
    global _LOGGER_INITIALIZED
    if _LOGGER_INITIALIZED:
        return logger

    if distributed_rank is None:
        distributed_rank = int(os.getenv("RANK", "0"))
    if world_size is None:
        world_size = int(os.getenv("WORLD_SIZE", "1"))

    filename = filename + "_{}.txt".format(str(distributed_rank))
    save_file = os.path.join(save_dir, filename)
    if mode == "o" and smart_exists(save_file):
        smart_remove(save_file)
        mode = "w"

    logger.remove()

    hijacker = LoggerHijack(logger, distributed_rank, world_size)
    hijacker.hook("log")
    hijacker.hook("trace")
    hijacker.hook("debug")
    hijacker.hook("info")

    logger.add(smart_open(save_file, mode))

    _LOGGER_INITIALIZED = True
    return logger


class StepWriter:
    def __init__(self, log_dir, project_name, exp_name, backend, exp):

        self.project_name = project_name
        self.exp_name = exp_name
        self.log_dir = log_dir

        self._parse_project_name_for_wandb()

        self.tb_writer = None
        self.wandb_writer = None
        self.default_backends = backend
        # backward compatible
        if isinstance(self.default_backends, str):
            self.default_backends = [self.default_backends]
        if not isinstance(self.default_backends, list) or len(self.default_backends) == 0:
            raise ValueError(
                f"default_backends must be a non-empty list, got {type(self.default_backends)}: {self.default_backends}. "
                f"Check init_metrics_server call in remote_metrics.py"
            )

        try:
            from torch.utils.tensorboard import SummaryWriter

            logger.info("> setting tensorboard ...")
            self.tb_writer = SummaryWriter(
                log_dir=self.log_dir,
                max_queue=1000,
            )
        except ModuleNotFoundError:
            logger.warning(
                "TensorBoard writing requested but is not "
                "available (are you using PyTorch 1.1.0 or later?), "
                "no TensorBoard logs will be written.",
                flush=True,
            )

        try:
            assert wandb is not None
            logger.info("> setting wandb ...")
            if os.getenv("WANDB_API_KEY") is None:
                raise ValueError("WANDB_API_KEY is not set")
            wandb.login(
                key=os.getenv("WANDB_API_KEY"),
                host="https://wandb-cloud.stepfun-inc.com/",
            )
            self.wandb_writer = wandb.init(
                project=self.wandb_project_name,
                entity=self.wandb_entity,
                name=self.exp_name,
                tags=self.wandb_tag,
                id=self.exp_name,
                dir=self.log_dir,
                config=exp.to_dict(),
            )
            self.wandb_writer.define_metric("*", step_metric="wandb_step")
        except Exception as e:
            logger.warning(f"> failed to set wandb: {e}")

    def _parse_project_name_for_wandb(self):
        self.wandb_entity = None
        self.wandb_project_name = None
        self.wandb_tag = None
        # use re to parse project name like "wandb_entity/wandb_project_name:tag" where wandb_entity and tag are optional
        if self.project_name:
            match = re.match(r"^(?:([^/]+)/)?([^:]+)(?::([^:]+))?$", self.project_name)
            if match:
                self.wandb_entity = match.group(1)
                self.wandb_project_name = match.group(2)
                self.wandb_tag = match.group(3)
                if self.wandb_tag:
                    self.wandb_tag = self.wandb_tag.split(",")

    def add_scalar(self, key, value, global_step, backends=None):
        if not isinstance(backends, list) or len(backends) == 0:
            backends = self.default_backends

        if "tensorboard" in backends:
            if self.tb_writer:
                self.tb_writer.add_scalar(key, value, global_step)
        if "wandb" in backends:
            if self.wandb_writer:
                wandb_metrics_dict = {"wandb_step": global_step, key: value}
                self.wandb_writer.log(wandb_metrics_dict)

    def add_histogram(self, key, value, global_step, backends=None):
        if not isinstance(backends, list) or len(backends) == 0:
            backends = self.default_backends

        if "tensorboard" in backends:
            if self.tb_writer:
                self.tb_writer.add_histogram(key, value, global_step)
        if "wandb" in backends:
            if self.wandb_writer:
                wandb_metrics_dict = {
                    "wandb_step": global_step,
                    key: wandb.Histogram(value),
                }
                self.wandb_writer.log(wandb_metrics_dict)

    def add_text(self, key, text, global_step, backends=None):
        if not isinstance(backends, list) or len(backends) == 0:
            backends = self.default_backends

        if "tensorboard" in backends:
            if self.tb_writer:
                self.tb_writer.add_text(key, text, global_step)
        if "wandb" in backends:
            if self.wandb_writer:
                wandb_metrics_dict = {"wandb_step": global_step, key: text}
                try:  # dont sure
                    self.wandb_writer.log(wandb_metrics_dict)
                except Exception as e:
                    logger.warning(f"> failed to log text to wandb: {e}")

    def flush(self):
        """Flush all writers to ensure data is written to disk."""
        if self.tb_writer:
            self.tb_writer.flush()

    def __del__(self):
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_writer:
            self.wandb_writer.finish()
