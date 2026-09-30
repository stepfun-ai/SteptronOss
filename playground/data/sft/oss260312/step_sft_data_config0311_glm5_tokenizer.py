"""0311 GLM-5-tokenizer compiled data config.

This file contains:
- the GLM-5 compiled-datasets config
- the GLM-5 compiled SFT data config

How to use:
- first compile:
  instantiate `Recipe0311DatasetsConfig`, set the tokenizer path used by the
  actual experiment, and call
  `.compile(COMPILED_ROOT_0311_UNIFIED_GLM5_TOKENIZER)`
- or run:
  `python3 <this_file> --tokenizer-path /path/to/hf_tokenizer`
- then use:
  import `Recipe0311Glm5CompiledSFTDataConfig` in experiments for training on
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

COMPILED_ROOT_0311_UNIFIED_GLM5_TOKENIZER = "/oss/data/recipe_0311_compiled_glm5"


class Recipe0311Glm5CompiledDatasetsConfig(CompiledDatasetsConfig):
    """Reads compiled shards for the GLM-5 tokenizer path."""

    compiled_recipe = CompiledDataRecipe(
        domains={
            "general": f"{COMPILED_ROOT_0311_UNIFIED_GLM5_TOKENIZER}/general",
        },
        epochs={
            "general": 1,
        },
    )


class Recipe0311Glm5CompiledSFTDataConfig(Recipe0311SFTDataConfig):
    """Ready-to-use compiled SFT data config for GLM-5."""

    dataset_cfg = Recipe0311Glm5CompiledDatasetsConfig


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
    data_cfg.compile(COMPILED_ROOT_0311_UNIFIED_GLM5_TOKENIZER)
