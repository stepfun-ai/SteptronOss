from __future__ import annotations

import sys
from pathlib import Path

_resource_root: Path | None = None


def _clear_dependent_caches() -> None:
    instructions_util = sys.modules.get("playground.eval.benchmarks.IFBench.official.instructions_util")
    if instructions_util is None:
        return
    for attr_name in ("get_nltk", "ensure_nltk_resources"):
        cached_fn = getattr(instructions_util, attr_name, None)
        if cached_fn is not None and hasattr(cached_fn, "cache_clear"):
            cached_fn.cache_clear()


def set_resource_root(path: str | Path) -> None:
    global _resource_root
    next_root = Path(path)
    if _resource_root == next_root:
        return
    _resource_root = next_root
    _clear_dependent_caches()


def get_resource_root() -> Path:
    if _resource_root is None:
        raise RuntimeError(
            "IFBench resource root is not configured. "
            "Call set_resource_root(...) before using the official IFBench helpers."
        )
    return _resource_root


def prompt_file_path() -> Path:
    return get_resource_root() / "IFBench_test.jsonl"


def nltk_data_path() -> Path:
    return get_resource_root() / "nltk_data"
