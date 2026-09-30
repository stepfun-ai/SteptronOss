from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu

REPO_ROOT = Path(__file__).resolve().parents[1]
GEMMA_SFT_EXP_PATH = REPO_ROOT / "playground/sft/gemma4/gemma4_31b_sft_step3_data.py"
GEMMA_COMPILED_DATA_PATH = REPO_ROOT / "playground/data/sft/oss260312/step_sft_data_config0311_gemma_tokenizer.py"

MACHINE_SPECIFIC_ABS_PATH_PREFIXES = (
    "/mnt/",
    "/usr/local/",
    "/tmp/",
    "/home/",
    "/data/",
)


def _reload_module(module_name: str):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_gemma4_sft_paths_use_standard_oss_locations():
    exp_module = _reload_module("playground.sft.gemma4.gemma4_31b_sft_step3_data")
    data_module = _reload_module("playground.data.sft.oss260312.step_sft_data_config0311_gemma_tokenizer")

    exp = exp_module.Exp()
    assert exp.trainer_cfg.micro_batch_size == 1
    assert exp.trainer_cfg.global_batch_size == 32
    assert exp.trainer_cfg.global_seq_length == 128 * 1024
    assert exp.scheduler_cfg.__class__.__name__ == "CosineSchedulerConfig"
    assert exp.scheduler_cfg.lr == 1e-5
    assert exp.scheduler_cfg.min_lr == 5e-6
    assert exp.scheduler_cfg.warmup_schedule == 140
    assert exp.scheduler_cfg.scheduler_unit == "iter"
    assert exp.scheduler_cfg.weight_decay == 0.1
    assert exp.scheduler_cfg.total_schedule is None
    assert exp.tokenizer_cfg.tokenizer_path == "/oss/opensources_model/gemma-4-31B-it"
    assert exp.checkpoint_cfg.load_safetensors == "/oss/opensources_model/gemma-4-31B"
    assert exp.checkpoint_cfg.model_config_path == "/oss/opensources_model/gemma-4-31B"
    assert exp.checkpoint_cfg.tokenizer_path == "/oss/opensources_model/gemma-4-31B-it"
    assert exp.checkpoint_cfg.save_dir == "/oss/checkpoints/"
    assert exp.data_cfg.__class__.__name__ == "Recipe0311GemmaCompiledSFTDataConfig"
    assert exp.data_cfg.dataset_cfg.__class__.__name__ == "Recipe0311GemmaCompiledDatasetsConfig"
    assert data_module.COMPILED_ROOT_0311_UNIFIED_GEMMA_TOKENIZER == "/oss/data/recipe_0311_compiled_gemma4_31b_it"


def test_gemma4_sft_source_files_do_not_embed_machine_specific_absolute_paths():
    for path in (GEMMA_SFT_EXP_PATH, GEMMA_COMPILED_DATA_PATH):
        offending_lines = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            path_literals = re.findall(r'/(?:[^"\'`\s)])+', line)
            if any(
                literal.startswith(prefix) for literal in path_literals for prefix in MACHINE_SPECIFIC_ABS_PATH_PREFIXES
            ):
                offending_lines.append(f"{path}:{line_no}: {line.strip()}")
        assert offending_lines == []
