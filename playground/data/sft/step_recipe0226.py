from typing import Literal

import torch
from configurize import Ref

from playground.tools.compile_recipe import (
    CompiledDataRecipe,
    CompiledDatasetsConfig,
    CompliableDatasetsConfig,
)
from steptronoss.exp.sft import SFTDataConfig


class Recipe0226DatasetsConfig(CompiledDatasetsConfig):
    compiled_recipe = CompiledDataRecipe(
        domains={
            "code": "/oss/data/step3p5_sft_data/step_data_0226/code",
            "math": "/oss/data/step3p5_sft_data/step_data_0226/math",
            "science": "/oss/data/step3p5_sft_data/step_data_0226/science",
            "swe_agentic": "/oss/data/step3p5_sft_data/step_data_0226/swe_agentic",
            "toolcall": "/oss/data/step3p5_sft_data/step_data_0226/toolcall",
            "general": "/oss/data/step3p5_sft_data/step_data_0226/general",
            "hot_fix": "/oss/data/step3p5_sft_data/step_data_0226/hot_fix",
            "logic": "/oss/data/step3p5_sft_data/step_data_0226/logic",
            "dr": "/oss/data/step3p5_sft_data/step_data_0226/dr",
            "long_context": "/oss/data/step3p5_sft_data/step_data_0226/long_context",
            "vc": "/oss/data/step3p5_sft_data/step_data_0226/vc",
        },
        epochs={
            "code": 3,
            "math": 2,
            "science": 2,
            "swe_agentic": 4,
            "toolcall": 2,
            "general": 2,
            "logic": 3,
            "hot_fix": 2,
            "dr": 3,
            "long_context": 1,
            "vc": 2,
        },
    )


class Step3SFTDataStep3TokenizedConfig(SFTDataConfig):
    dataset_cfg = Recipe0226DatasetsConfig

    oversize_policy: Literal["drop", "extend"] = "drop"

    max_packing_seqlen = Ref("..trainer_cfg.global_seq_length")

    seqlen_divisible_by: int = 64

    global_data_keys = ["cu_seqlens", "position_id"]

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
        )
        dataloader = DPMux(dataloader, dp_size=dp_size, dp_rank=dp_rank)

        dataloader = async_accelearte_slowfast(dataloader, num_workers=16)  # optional, remove for debug
        return dataloader

    def preprocess(self, batch: dict):
        cu_seqlens = batch["cu_seqlens"].to("cuda")
        position_id = batch["position_id"].to("cuda")
        max_seq_len = torch.max(cu_seqlens[1:] - cu_seqlens[:-1])

        if "tokens" in batch:  # on the head or tail of the pipeline parallel
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
            # padding to the tensor_model_parallel_size
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
