"""0311 Gemma-tokenizer compiled data config.

This file contains:
- the Gemma raw-json datasets config
- the Gemma compiled-datasets config
- the Gemma compiled SFT data config

How to use:
- first compile:
  instantiate `Recipe0311GemmaDatasetsConfig`, set the tokenizer path used by the
  actual experiment, and call
  `.compile(COMPILED_ROOT_0311_UNIFIED_GEMMA_TOKENIZER)`
- or run:
  `python3 <this_file> --tokenizer-path /path/to/hf_tokenizer`
- then use:
  import `Recipe0311GemmaCompiledSFTDataConfig` in experiments for training on
  the compiled shards
- for raw-json direct training, use `Recipe0311SFTDataConfig` from
  `step_sft_data_config0311.py`, or `Recipe0311GemmaSFTDataConfig` from this
  file when you need the Gemma-specific chat-template adapter

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

COMPILED_ROOT_0311_UNIFIED_GEMMA_TOKENIZER = "/oss/data/recipe_0311_compiled_gemma4_31b_it"


class Recipe0311GemmaDatasetsConfig(Recipe0311DatasetsConfig):
    """Raw-json 0311 config that adapts StepChat content for Gemma templates."""

    def get_template(self):
        from steptronoss.data.chat_templates.text_template import GemmaTemplate
        from steptronoss.tokenizer.hf_compat_tokenizer import load_hf_tokenizer

        tokenizer = load_hf_tokenizer(self.tokenizer_path)
        return GemmaTemplate(tokenizer=tokenizer)


class Recipe0311GemmaCompiledDatasetsConfig(CompiledDatasetsConfig):
    """Reads compiled shards, an acceleration-only form of the raw 0311 data."""

    compiled_recipe = CompiledDataRecipe(
        domains={
            "general": f"{COMPILED_ROOT_0311_UNIFIED_GEMMA_TOKENIZER}/general",
        },
        epochs={
            "general": 1,
        },
    )


class Recipe0311GemmaCompiledSFTDataConfig(Recipe0311SFTDataConfig):
    """Ready-to-use SFT config for the compiled large-scale training path."""

    dataset_cfg = Recipe0311GemmaCompiledDatasetsConfig


class Recipe0311GemmaSFTDataConfig(Recipe0311SFTDataConfig):
    """Ready-to-use Gemma raw-json SFT config over 0311 unified json files."""

    dataset_cfg = Recipe0311GemmaDatasetsConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer-path",
        required=True,
        help="HF tokenizer path. It should match the tokenizer used by the target experiment.",
    )
    args = parser.parse_args()

    data_cfg = Recipe0311GemmaDatasetsConfig()
    data_cfg.tokenizer_path = args.tokenizer_path
    data_cfg.compile(COMPILED_ROOT_0311_UNIFIED_GEMMA_TOKENIZER)
