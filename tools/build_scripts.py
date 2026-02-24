#!/usr/bin/env python3

import argparse
import os
import shlex
import stat
import sys
import traceback
from dataclasses import dataclass
from uuid import uuid4

sys.path.append(os.getcwd())

from steptronoss.exp.base_exp import BaseExp, ResourceConfig
from steptronoss.utils.general import get_object_from_file


@dataclass(frozen=True)
class BuildScriptsResult:
    exp_id: str
    output_dir: str
    scripts: list[str]


def _format_command(task_cfg: ResourceConfig, command: str) -> str:
    return task_cfg.command.format(TORCHRUN="tools/smartrun", COMMAND=command)


def _format_exports(envs: dict[str, str]) -> list[str]:
    lines = []
    exp_id = envs.get("EXP_ID")
    if exp_id is not None:
        lines.append(f"export EXP_ID={shlex.quote(str(exp_id))}")
    for key in sorted(envs.keys()):
        if key == "EXP_ID":
            continue
        lines.append(f"export {key}={shlex.quote(str(envs[key]))}")
    return lines


def _write_script(path: str, envs: dict[str, str], command: str) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -e",
        *_format_exports(envs),
        command,
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    current_mode = os.stat(path).st_mode
    os.chmod(path, current_mode | stat.S_IXUSR)


def build_scripts(
    exp_path: str,
    output_root: str,
    extra_args: list[str],
    exp_id: str | None = None,
) -> BuildScriptsResult:
    exp_id = exp_id or str(uuid4())[:5]
    os.environ["EXP_ID"] = exp_id

    exp: BaseExp = get_object_from_file(exp_path, "Exp")()
    original_argv = sys.argv
    try:
        sys.argv = [exp_path, *extra_args]
        exp.update_from_args()
    finally:
        sys.argv = original_argv

    err = None
    try:
        exp.sanity_check()
    except Exception as e:
        err = "\n".join(traceback.format_exception(e)[-2:])
    if err is not None:
        input(f"Sanity Check Fail:\n{err}\n\npress enter to ignore and continue")

    command = " ".join([exp_path] + extra_args)

    rsc_cfg: ResourceConfig = exp.resource_cfg
    rsc_cfg.envs["EXP_ID"] = exp_id
    all_tasks = rsc_cfg.extract()

    output_dir = os.path.join(output_root, exp.exp_name, exp_id)
    os.makedirs(output_dir, exist_ok=True)

    scripts: list[str] = []
    node_indices: dict[str, int] = {}
    for _task_name, task_cfg in all_tasks.items():
        task_command = _format_command(task_cfg, command)
        envs = task_cfg.envs | {"NNODES": str(task_cfg.replica)}
        node_dir = os.path.join(output_dir, task_cfg.node_type)
        os.makedirs(node_dir, exist_ok=True)
        start_idx = node_indices.get(task_cfg.node_type, 0)
        for replica_idx in range(task_cfg.replica):
            script_idx = start_idx + replica_idx
            script_path = os.path.join(node_dir, f"{script_idx}.sh")
            _write_script(script_path, envs, task_command)
            scripts.append(script_path)
        node_indices[task_cfg.node_type] = start_idx + task_cfg.replica

    return BuildScriptsResult(exp_id=exp_id, output_dir=output_dir, scripts=scripts)


class BuildScriptsRunner:
    def run(self) -> None:
        args, extra_args = self.parse_args()
        result = build_scripts(args.exp, args.output_root, extra_args)
        print(f"Scripts written to: {result.output_dir}")
        for script_path in result.scripts:
            print(script_path)

    def parse_args(self) -> tuple[argparse.Namespace, list[str]]:
        example = "Example:\n  python tools/build_scripts.py exp.py /mnt/entrypoints/\n"
        note = (
            "Note:\n"
            "  Any unrecognized extra arguments will be passed directly to the exp.\n"
            "  For example,\n"
            "    python tools/build_scripts.py exp.py /mnt/entrypoints/ suffix=test\n"
            "  will pass `suffix=test` to the exp.\n"
        )
        epilog_text = example + "\n" + note
        formatter = argparse.RawTextHelpFormatter
        parser = argparse.ArgumentParser(
            description="Generate per-replica shell scripts for an exp.",
            epilog=epilog_text,
            formatter_class=formatter,
        )

        parser.add_argument("exp", help="relative or absolute path to your exp.")
        parser.add_argument("output_root", help="root directory to write scripts.")
        args, extra_args = parser.parse_known_args()

        return args, extra_args


if __name__ == "__main__":
    BuildScriptsRunner().run()
