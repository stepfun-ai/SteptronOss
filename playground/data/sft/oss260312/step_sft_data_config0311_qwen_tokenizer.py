"""0311 Qwen-tokenizer compiled data config.

This file contains:
- the Qwen compiled-datasets config
- the Qwen compiled SFT data config

How to use:
- first compile:
  instantiate `Recipe0311DatasetsConfig`, set the tokenizer path used by the
  actual experiment, and call
  `.compile(COMPILED_ROOT_0311_UNIFIED_QWEN_TOKENIZER)`
- or run:
  `python3 <this_file> --tokenizer-path /path/to/hf_tokenizer`
- then use:
  import `Recipe0311QwenCompiledSFTDataConfig` in experiments for training on
  the compiled shards
- for raw-json direct training, use `Recipe0311SFTDataConfig` from
  `step_sft_data_config0311.py`

Notes:
- this file is only the compiled variant for large-scale training
- compile should be semantically equivalent to raw-json training and only serve
  as an IO/throughput acceleration path
- the tokenizer used for compile must match the tokenizer configured in the
  actual experiment
"""

import argparse

from playground.data.sft.oss260312.step_sft_data_config0311 import (
    Recipe0311DatasetsConfig,
    Recipe0311SFTDataConfig,
)
from playground.tools.compile_recipe import CompiledDataRecipe, CompiledDatasetsConfig
import os

COMPILED_ROOT_0311_UNIFIED_QWEN_TOKENIZER = os.getenv("COMPILED_ROOT_0311_UNIFIED_QWEN_TOKENIZER", "/oss/data/recipe_0311_compiled_qwen")


# Datasets Configs
class Recipe0311QwenCompiledDatasetsConfig(CompiledDatasetsConfig):
    """Reads compiled shards, an acceleration-only form of the raw 0311 data."""

    compiled_recipe = CompiledDataRecipe(
        domains={
            "general": f"{COMPILED_ROOT_0311_UNIFIED_QWEN_TOKENIZER}/general",
        },
        epochs={
            "general": 1,
        },
    )


# Data Config ready for use
class Recipe0311QwenCompiledSFTDataConfig(Recipe0311SFTDataConfig):
    """Ready-to-use SFT config for the compiled large-scale training path."""

    dataset_cfg = Recipe0311QwenCompiledDatasetsConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer-path",
        required=True,
        help="HF tokenizer path. It should match the tokenizer used by the target experiment.",
    )
    args = parser.parse_args()

    data_cfg = Recipe0311DatasetsConfig()
    data_cfg.tokenizer_path = args.tokenizer_path
    data_cfg.compile(COMPILED_ROOT_0311_UNIFIED_QWEN_TOKENIZER)
