from __future__ import annotations

import copy
import os
from typing import Literal, Optional, TypedDict

from configurize import Config
from loguru import logger


class TaskSpec(TypedDict, total=False):
    """Configuration for individual task"""

    # Override corresponding fields in ResourceConfig if set
    replica: int
    gpu: int
    command: str

    node_type: Literal["cpu", "gpu"]
    is_critical: bool

    mounts: list[str]
    envs: dict[str, str]
    """NOTE: Combined with ResourceConfig.envs (union) as the final envs"""
    image: str


class ResourceConfig(Config):
    replica: int
    """Should run on how many node."""

    gpu: int
    """Number of gpus for this node."""

    node_type: Literal["cpu", "gpu"]

    is_critical: bool = False
    """Mark a task as critical, if all 'critical tasks' of an exp terminate, kill all other tasks."""

    mounts: list[str] = []
    """File system mounting stuff"""

    envs: dict[str, str] = {}
    """NOTE: Combined with ResourceConfig.envs (union) as the final envs"""

    image: str | None = None
    """Specify docker image, none for no image."""

    command: str
    """Actual Command. Given 'python tools/xxx_run.py my_exp.py arg1=1':
    - TORCHRUN: 'torchrun --nproc-per-node ...'
    - COMMAND : 'my_exp.py arg1=1'
    """

    def __init__(self):
        super().__init__()
        self.task_specs = dict(default=TaskSpec(is_critical=True))

        self.replica = 1
        self.gpu = 0
        self.node_type = "cpu"
        self.command = "python {COMMAND}"

    def extract(self) -> dict[str, ResourceConfig]:
        """This util flatten ResourceCfg.task_specs to individual ResourceConfig.
        add extra attrs:
        - exp_name: str
        """
        assert isinstance(self.task_specs, dict), self.task_specs

        if "PYTHONPATH" not in self.envs:
            logger.info(
                "For not-installed steptronoss code, we need to set 'PYTHONPATH=.' to start EXPs, "
                "if you need this ENV be other value, please set it in resource_cfg.envs"
            )
            self.envs["PYTHONPATH"] = "."
        all_task_specs = self.find_leaf_task_specs()

        exp_id = os.environ["EXP_ID"]

        all_configs: dict[str, ResourceConfig] = {}

        for task_name, task_spec in all_task_specs.items():
            task_spec["envs"] = self.envs | task_spec.get("envs", {})
            with self.modify(**task_spec):
                task_cfg: ResourceConfig = self.__class__()
                for k, v in self.items():
                    if k != "task_specs":
                        setattr(task_cfg, k, copy.deepcopy(v))

                task_cfg._fix_ref()
                task_cfg._allow_set_new_attr = True

                task_cfg.envs["EXP_ID"] = f"{exp_id}/{task_name}"

                task_cfg.exp_name = f"{self.root().exp_name}/{task_name}"
                all_configs[task_name] = task_cfg

        return all_configs

    def find_leaf_task_specs(self) -> dict[str, TaskSpec]:
        """Experimental: find all task_specs on leaf nodes."""
        node = self
        while node.father() is not None:
            node = node.father()
            if hasattr(node, "resource_cfg") and isinstance(node.resource_cfg, ResourceConfig):
                break

        found_specs, found_names = {}, {}

        def recur_find_task_specs(node: Config):
            if hasattr(node, "task_specs"):
                prefix = node._get_node_name()
                for k, v in node.task_specs.items():
                    # Compat: ok if both exists but same
                    if k in found_specs and v != found_specs[k]:
                        raise RuntimeError(f"Conflict sub-task name '{k}' in {prefix} and {found_names[k]}!")
                    found_names[k] = prefix
                    found_specs[k] = v
            for _k, v in node.items():
                if isinstance(v, Config):
                    recur_find_task_specs(v)

        recur_find_task_specs(node)
        list_repr = "\n".join([f"- @{v}: {k}" for k, v in found_names.items()])
        logger.info(f"Get TaskSpecs from entire config tree:\n{list_repr}")
        return found_specs


class TorchrunResourceConfig(ResourceConfig):
    def __init__(self):
        super().__init__()
        self.node_type = "gpu"
        self.gpu = 8
        self.command = "{TORCHRUN} {COMMAND}"
