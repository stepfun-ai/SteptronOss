#!/usr/bin/env python3

import argparse
import os
import sys
from uuid import uuid4

sys.path.append(os.getcwd())

from configurize.mock_imports import mock_imports
from configurize.utils import get_object_from_file

from steptronoss.exp.base_exp import BaseExp, ResourceConfig

"""
This is a runnable platform adaptation template.
It keeps a simple submission flow and removes internal platform details.
Implement `launch_task(...)` for your scheduler.
"""


def launch_task(
    task_cfg: ResourceConfig,
    command: str,
    image: str | None = None,
    code_mount_point: str | None = None,
):
    """
    Submit one task to your scheduler and return a handle with:
    - wait(): block until completion
    - terminate(): best-effort stop
    Return None if your platform is fully async and you can't wait/terminate.
    """
    raise NotImplementedError("Implement launch_task(...) for your scheduler.")


def spawn_tasks(
    rsc_cfg: ResourceConfig,
    command: str,
    code_mount_point: str | None = None,
):
    exp_id = os.environ["EXP_ID"]
    rsc_cfg.envs["STEPTRON_MEET_DIR"] = os.environ["STEPTRON_MEET_DIR"]

    tasks = []
    critical_tasks = []
    rsc_cfgs = rsc_cfg.extract()
    for task_name, task_cfg in rsc_cfgs.items():
        task_cfg.envs["EXP_ID"] = f"{exp_id}/{task_name}"
        task_cfg.envs["NNODES"] = f"{task_cfg.replica}"
        print(task_name, task_cfg)
        t = launch_task(
            task_cfg=task_cfg,
            command=task_cfg.command.format(TORCHRUN="tools/smartrun", COMMAND=command),
            image=rsc_cfg.image,
            code_mount_point=code_mount_point,
        )
        if task_cfg.is_critical:
            critical_tasks.append(t)
        tasks.append(t)
    if not critical_tasks:
        critical_tasks = tasks

    for t in critical_tasks:
        t.wait()
    for t in tasks:
        t.terminate()


class ExampleSubmitter:
    def run(self):
        args, extra_args = self.parse_args()
        os.environ["EXP_ID"] = str(uuid4())[:5]

        with mock_imports():
            exp: BaseExp = get_object_from_file(args.exp, "Exp")()
            exp.update_from_args()
            try:
                exp.sanity_check()
            except Exception as err:
                input(f"⛔ Sanity Check Fail ⛔:\n{err}\n\npress enter to ignore and continue")

            command = " ".join([args.exp] + extra_args)

            spawn_tasks(rsc_cfg=exp.resource_cfg, command=command)

    def parse_args(self) -> tuple[argparse.Namespace, list[str]]:
        example = "Example:\n  python tools/example_submitter.py exp.py\n"
        note = (
            "Note:\n"
            "  Any unrecognized extra arguments will be passed directly to the exp.\n"
            "  For example, both\n"
            "    python tools/example_submitter.py exp.py --a -f file.c\n"
            "  and\n"
            "    python tools/example_submitter.py exp.py -- --a -f file.c\n"
            "  will pass `--a -f file.c` to the exp.\n"
        )
        epilog_text = example + "\n" + note
        formatter = argparse.RawTextHelpFormatter
        parser = argparse.ArgumentParser(
            description="Run Steptron exp with a platform submitter example.",
            epilog=epilog_text,
            formatter_class=formatter,
        )

        parser.add_argument("exp", help="relative or absolute path to your exp.")
        args, extra_args = parser.parse_known_args()

        return args, extra_args


if __name__ == "__main__":
    ExampleSubmitter().run()
