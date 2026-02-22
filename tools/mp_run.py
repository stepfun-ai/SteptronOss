#!/usr/bin/env python3

import argparse
import atexit
import os
import sys
import traceback
from subprocess import Popen
from uuid import uuid4

sys.path.append(os.getcwd())

from steptronoss.exp.base_exp import BaseExp, ResourceConfig
from steptronoss.utils.general import get_object_from_file


def validate_environment(rsc_cfg: ResourceConfig) -> list[str]:
    """Check environment is properly configured before spawning.

    Returns list of error messages (empty if valid).
    """
    errors = []

    # Check multi-node requirements
    if rsc_cfg.replica > 1:
        if "EXP_ID" not in os.environ:
            errors.append(
                "Multi-node training (replica > 1) requires EXP_ID to be set.\n"
                "  Run on ALL nodes: export EXP_ID=<same-id>"
            )

        meet_dir = os.environ.get("STEPTRON_MEET_DIR")
        if not meet_dir:
            errors.append(
                "Multi-node training requires STEPTRON_MEET_DIR for Redis rendezvous.\n"
                "  Run: export STEPTRON_MEET_DIR=/path/to/shared/filesystem"
            )

    return errors


def submit_one(command: str, envs: dict[str, str]):
    worker = Popen(command, shell=True, env=os.environ | envs, preexec_fn=os.setsid)  # noqa: S602

    def send_signal(sig):
        worker.poll()
        if worker.returncode is None:
            os.killpg(os.getpgid(worker.pid), sig)

    worker.send_signal = send_signal

    atexit.register(worker.terminate)

    return worker


def spawn_tasks(rsc_cfg: ResourceConfig, command: str):
    exp_id = os.environ["EXP_ID"]
    rsc_cfg.envs["EXP_ID"] = exp_id

    all_tasks = rsc_cfg.extract()

    tasks: list[Popen] = []
    critical_tasks: list[Popen] = []
    for task_name, task_cfg in all_tasks.items():
        print(task_name, task_cfg)
        t = submit_one(
            command=task_cfg.command.format(TORCHRUN="tools/smartrun", COMMAND=command),
            envs=task_cfg.envs | {"NNODES": str(task_cfg.replica)},
        )
        if task_cfg.is_critical:
            critical_tasks.append(t)
        tasks.append(t)

    if not critical_tasks:
        critical_tasks = tasks  # all critical

    for t in critical_tasks:
        t.wait()
    for t in tasks:  # kill all task
        t.terminate()


class MPRunner:
    def run(self):
        args, extra_args = self.parse_args()

        exp: BaseExp = get_object_from_file(args.exp, "Exp")()
        exp.update_from_args()

        # Validate environment before spawning
        errors = validate_environment(exp.resource_cfg)
        if errors:
            print("Environment validation failed:\n", file=sys.stderr)
            for err in errors:
                print(f"  - {err}\n", file=sys.stderr)
            sys.exit(1)

        # Use existing EXP_ID if set (for multi-node coordination), otherwise generate new one
        if "EXP_ID" not in os.environ:
            os.environ["EXP_ID"] = str(uuid4())[:5]

        err = None
        try:
            exp.sanity_check()
        except Exception as e:
            err = "\n".join(traceback.format_exception(e)[-2:])
        if err is not None:
            input(f"⛔ Sanity Check Fail ⛔:\n{err}\n\npress enter to ignore and continue")

        command = " ".join([args.exp] + extra_args)

        spawn_tasks(rsc_cfg=exp.resource_cfg, command=command)

    def parse_args(self) -> tuple[argparse.Namespace, list[str]]:
        example = "Example:\n  python tools/mp_run.py exp.py\n"
        note = (
            "Note:\n"
            "  Any unrecognized extra arguments will be passed directly to the exp.\n"
            "  For example, \n"
            "    python tools/mp_run.py exp.py suffix=test\n"
            "  will pass `suffix=test` to the exp.\n"
        )
        epilog_text = example + "\n" + note
        formatter = argparse.RawTextHelpFormatter
        parser = argparse.ArgumentParser(
            description="Run Steptron exp with given image via rlaunch.",
            epilog=epilog_text,
            formatter_class=formatter,
        )

        parser.add_argument("exp", help="relative or absolute path to your exp.")
        args, extra_args = parser.parse_known_args()

        return args, extra_args


if __name__ == "__main__":
    MPRunner().run()
