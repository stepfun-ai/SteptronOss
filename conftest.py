import os
import subprocess
import sys

import pytest

RERUN_TAGS = ["node2"]
TAG_TO_COMMAND = {
    "node2": ["torchrun", "--nproc-per-node", "2"],
}


def _is_cov_run() -> bool:
    return any(arg.startswith("--cov") for arg in sys.argv[1:])


def _get_marker_expr() -> str | None:
    if "-m" not in sys.argv:
        return None
    try:
        idx = sys.argv.index("-m")
        return sys.argv[idx + 1]
    except (ValueError, IndexError):
        return None


def pytest_configure(config):
    rank = int(os.environ.get("RANK", "0"))
    config.addinivalue_line("markers", "xdist_group(name): group tests for xdist/torchrun")
    if os.environ.get("STEPTRON_TORCHRUN") == "1" and _is_cov_run():
        os.environ["COVERAGE_FILE"] = f".coverage.node2.{rank}"
    try:
        import atexit

        import torch.distributed as dist
        from torch.distributed import distributed_c10d as c10d

        original_destroy = c10d.destroy_process_group

        def _safe_destroy(pg=None):
            if not c10d.is_initialized():
                return
            if pg is None and c10d._get_default_group() is None:
                return
            return original_destroy(pg)

        c10d.destroy_process_group = _safe_destroy
        dist.destroy_process_group = _safe_destroy
        try:
            atexit.unregister(original_destroy)
        except Exception:
            pass
        atexit.register(_safe_destroy)
    except Exception:
        pass
    if rank != 0:
        config.option.quiet = True
        config.option.verbose = 0
        config.option.showcapture = "no"
        config.option.reportchars = ""
        config.option.tbstyle = "no"
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")


def pytest_collection_modifyitems(config, items):
    marker_expr = _get_marker_expr()
    config._steptron_rerun_tags = []

    if os.environ.get("STEPTRON_TORCHRUN") == "1":
        return

    if marker_expr is None:
        # Parent run: remove rerun-tagged tests and schedule them for rerun.
        config._steptron_rerun_tags = list(RERUN_TAGS)
        items[:] = [item for item in items if not any(tag in item.keywords for tag in RERUN_TAGS)]
        if int(os.environ.get("RANK", "0")) == 0 and config._steptron_rerun_tags:
            print(f"\n[pytest] rerun tags will run in individual torchrun passes: {config._steptron_rerun_tags}")


def pytest_sessionfinish(session, exitstatus):
    if os.environ.get("STEPTRON_TORCHRUN") == "1":
        return
    rerun_tags = getattr(session.config, "_steptron_rerun_tags", [])
    if not rerun_tags:
        return

    env = os.environ.copy()
    env["STEPTRON_TORCHRUN"] = "1"
    env.setdefault("TORCHRUN_LOG_LEVEL", "ERROR")
    env.setdefault("OMP_NUM_THREADS", "1")

    base_args = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg == "-m":
            skip_next = True
            continue
        base_args.append(arg)

    for tag in rerun_tags:
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"\n[pytest] starting torchrun pass for tag: {tag}")
        launcher = TAG_TO_COMMAND[tag]
        cmd = [*launcher, "-m", "pytest", "-m", tag, *base_args]
        result = subprocess.run(cmd, env=env, check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)

    if _is_cov_run():
        subprocess.run([sys.executable, "-m", "coverage", "combine"], check=False)
        subprocess.run(
            [sys.executable, "-m", "coverage", "xml", "-o", "coverage.xml"],
            check=False,
        )
