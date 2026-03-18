#!/usr/bin/env python3
"""0311 unified recipe plus shared raw-json data config.

This file contains:
- the unified raw-data recipe
- the shared raw-json `Recipe0311DatasetsConfig`
- the shared raw-json `Recipe0311SFTDataConfig`
- the common base used by tokenizer-specific compiled data-config files

How to use:
- large-scale path:
  first compile with a tokenizer-specific file, then use the corresponding
  compiled `SFTDataConfig` in experiments
- direct path:
  import `Recipe0311SFTDataConfig` when you want to train directly from raw json

Notes:
- training directly from raw json is the reference path
- compile is an equivalent acceleration path for large-scale training; it should
  not change dataset semantics
- when compiling, always use the tokenizer path from the actual experiment
"""

from typing import Literal

import torch
from configurize import Ref

from playground.tools.compile_recipe import CompliableDatasetsConfig
from steptronoss.data.recipe import DataRecipe, DataSourceFile
from steptronoss.exp.sft import SFTDataConfig

DATA_ROOT_0311_UNIFIED = "/oss/data/step_sft_data/0312_rtu"


GENERAL_FILE_LIST = [
    DataSourceFile(f"{DATA_ROOT_0311_UNIFIED}/general/{basename}")
    for basename in [
        "chunk_0.json",
        "chunk_1.json",
        "chunk_2.json",
        "chunk_3.json",
        "chunk_4.json",
        "chunk_5.json",
        "chunk_6.json",
        "chunk_7.json",
        "chunk_8.json",
        "chunk_9.json",
        "chunk_10.json",
        "chunk_11.json",
        "chunk_12.json",
        "chunk_13.json",
        "chunk_14.json",
        "chunk_15.json",
        "chunk_16.json",
        "chunk_17.json",
        "chunk_18.json",
        "chunk_19.json",
        "chunk_20.json",
        "chunk_21.json",
        "chunk_22.json",
        "chunk_23.json",
        "chunk_24.json",
        "chunk_25.json",
        "chunk_26.json",
        "chunk_27.json",
        "chunk_28.json",
        "chunk_29.json",
        "chunk_30.json",
        "chunk_31.json",
        "chunk_32.json",
        "chunk_33.json",
        "chunk_34.json",
        "chunk_35.json",
        "chunk_36.json",
        "chunk_37.json",
        "chunk_38.json",
        "chunk_39.json",
        "chunk_40.json",
        "chunk_41.json",
        "chunk_42.json",
        "chunk_43.json",
        "chunk_44.json",
        "chunk_45.json",
        "chunk_46.json",
        "chunk_47.json",
        "chunk_48.json",
        "chunk_49.json",
        "chunk_50.json",
        "chunk_51.json",
        "chunk_52.json",
        "chunk_53.json",
        "chunk_54.json",
        "chunk_55.json",
        "chunk_56.json",
        "chunk_57.json",
        "chunk_58.json",
        "chunk_59.json",
        "chunk_60.json",
        "chunk_61.json",
        "chunk_62.json",
        "chunk_63.json",
        "chunk_64.json",
        "chunk_65.json",
        "chunk_66.json",
        "chunk_67.json",
        "chunk_68.json",
        "chunk_69.json",
        "chunk_70.json",
        "chunk_71.json",
        "chunk_72.json",
        "chunk_73.json",
        "chunk_74.json",
        "chunk_75.json",
        "chunk_76.json",
        "chunk_77.json",
        "chunk_78.json",
        "chunk_79.json",
        "chunk_80.json",
        "chunk_81.json",
        "chunk_82.json",
        "chunk_83.json",
        "chunk_84.json",
        "chunk_85.json",
        "chunk_86.json",
        "chunk_87.json",
        "chunk_88.json",
        "chunk_89.json",
        "chunk_90.json",
        "chunk_91.json",
        "chunk_92.json",
        "chunk_93.json",
        "chunk_94.json",
        "chunk_95.json",
        "chunk_96.json",
        "chunk_97.json",
        "chunk_98.json",
        "chunk_99.json",
    ]
]


STEP_DATA_RECIPE0311_UNIFIED = DataRecipe(
    domains={
        "general": GENERAL_FILE_LIST,
    },
    epochs={
        "general": 1,
    },
)

SFT_0311_UNIFIED_RECIPE = STEP_DATA_RECIPE0311_UNIFIED


# Datasets Configs
# `Recipe0311DatasetsConfig` is shared by tokenizer variants because raw-json
# loading and template construction are now both based on HF tokenizer paths.
# Use this path directly for debugging, smaller runs, or as the source of
# tokenizer-specific compile flows.
class Recipe0311DatasetsConfig(CompliableDatasetsConfig):
    """Dataset config that reads raw 0311 unified json files directly."""

    max_seq_len: int = 128 * 1024
    """Upper bound used while compiling raw dialogs."""

    tokenizer_path: str = Ref("...tokenizer_cfg.tokenizer_path")
    """Tokenizer path used by compile flow."""

    def get_recipe(self):
        return STEP_DATA_RECIPE0311_UNIFIED

    def get_dataset(self, filelist, template):
        from steptronoss.data.datasets.stepchat_dataset import StepChatJsonDataset

        return StepChatJsonDataset(filelist=filelist, template=template)

    def get_template(self):
        from steptronoss.data.chat_templates.text_template import HuggingFaceTemplate
        from steptronoss.tokenizer.hf_compat_tokenizer import load_hf_tokenizer

        tokenizer = load_hf_tokenizer(self.tokenizer_path)
        return HuggingFaceTemplate(tokenizer=tokenizer)


# Data Config ready for use
# `Recipe0311SFTDataConfig` is the shared raw-json training config.
# Tokenizer-specific compiled configs should inherit from this config so compile
# stays an acceleration-only transformation.
#
# Example:
# class MyRawJsonSFTDataConfig(Recipe0311SFTDataConfig):
#     dataset_cfg = Recipe0311DatasetsConfig
class Recipe0311SFTDataConfig(SFTDataConfig):
    """Ready-to-use SFT data config over raw 0311 unified json files."""

    dataset_cfg = Recipe0311DatasetsConfig

    oversize_policy: Literal["drop", "extend"] = "drop"
    """How to handle samples larger than the target pack length."""

    max_packing_seqlen = Ref("..trainer_cfg.global_seq_length")
    """Target packed sequence length provided by the trainer."""

    seqlen_divisible_by: int = 64
    """Pad packed sequences so lengths align with tensor-parallel needs."""

    global_data_keys = ["cu_seqlens", "position_id"]
    """Batch keys that must be broadcast globally."""

    def build_dataloader(self, dp_rank=0, dp_size=1):
        from steptronoss.data.dataloader.packed_dataloader import MixedPackedDataloader
        from steptronoss.data.nextable import DPMux, async_accelearte_slowfast

        datasets = self.dataset_cfg.build_datasets()
        dataloader = MixedPackedDataloader(
            datasets=[ds[0] for ds in datasets.values()],
            epochs=[ds[1] for ds in datasets.values()],
            max_length=self.max_packing_seqlen,
            oversize_policy=self.oversize_policy,
            transform=self.pack,
            dataset_sampling="sequential",
        )
        dataloader = DPMux(dataloader, dp_size=dp_size, dp_rank=dp_rank)
        dataloader = async_accelearte_slowfast(dataloader, num_workers=16)
        return dataloader

    def preprocess(self, batch: dict):
        cu_seqlens = batch["cu_seqlens"].to("cuda")
        position_id = batch["position_id"].to("cuda")
        max_seq_len = torch.max(cu_seqlens[1:] - cu_seqlens[:-1])

        if "tokens" in batch:
            tokens = batch["tokens"].to("cuda")
            labels = batch["labels"].to("cuda")
            loss_masks = batch["loss_mask"].to("cuda")

            return dict(
                input_ids=tokens[None].contiguous(),
                labels=labels[None].contiguous(),
                loss_masks=loss_masks[None].contiguous(),
                cu_seqlens=cu_seqlens,
                max_seq_len=max_seq_len,
                position_id=position_id,
            )
        else:
            return dict(
                cu_seqlens=cu_seqlens,
                max_seq_len=max_seq_len,
                position_id=position_id,
            )

    def pack(self, pieces: list):
        import numpy as np

        size = sum([len(s["tokens"]) - 1 for s in pieces])

        if size % self.seqlen_divisible_by != 0:
            padding_size = self.seqlen_divisible_by - size % self.seqlen_divisible_by
            padding_tensor = np.zeros(padding_size + 1)
            pieces.append({
                "tokens": padding_tensor,
                "loss_mask": padding_tensor,
            })

        sizes = torch.tensor([len(s["tokens"]) - 1 for s in pieces])
        from torch import tensor as T

        tokens = torch.cat([T(s["tokens"][:-1], dtype=torch.long) for s in pieces])
        labels = torch.cat([T(s["tokens"][1:], dtype=torch.long) for s in pieces])
        loss_mask = torch.cat([T(s["loss_mask"][1:], dtype=torch.float32) for s in pieces])

        cu_seqlens = torch.cat([
            torch.zeros(1),
            torch.cumsum(sizes, 0),
        ]).int()

        from steptronoss.utils.general import get_position_id_from_cu_seqlens

        return dict(
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            cu_seqlens=cu_seqlens,
            max_seq_len=sizes.max(),
            position_id=get_position_id_from_cu_seqlens(cu_seqlens),
        )
