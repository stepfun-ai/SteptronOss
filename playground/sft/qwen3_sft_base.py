import os

import torch
from configurize import Ref

from playground.data.sft.reasoning_GCMKSTIDF_sft_stage1_1203_compile_qwen import (
    DatasetsConfig,
)
from playground.pretrain.qwen3.qwen3_8 import Qwen3_8BConfig_128K_80G
from playground.pretrain.qwen3.qwen3_30a3b import Qwen3_30A3BConfig
from steptronoss.exp.base_exp import (
    BaseExp,
    GradientManagerConfig,
    Megatron3DParallelModelConfig,
    ProfilerConfig,
)
from steptronoss.exp.checkpointing import CheckpointConfig, ConstantCheckpointConfig
from steptronoss.exp.lr_schedulers import ConstantSchedulerConfig, SchedulerConfig
from steptronoss.exp.ntp import NTPTrainerConfig, PretrainMetricConfig
from steptronoss.exp.sft import SFTDataConfig


class Qwen3_SFT_DataConfig(SFTDataConfig):
    dataset_cfg = DatasetsConfig

    max_packing_seqlen = Ref("..trainer_cfg.global_seq_length", 32000)

    seqlen_divisible_by: int = 64
    global_data_keys = ["cu_seqlens"]

    def build_dataloader(self, dp_rank=0, dp_size=1):
        from steptronoss.data.dataloader.packed_dataloader import MixedPackedDataloader
        from steptronoss.data.nextable import DPMux, async_accelearte_slowfast

        datasets = self.dataset_cfg.build_datasets()
        dataloader = MixedPackedDataloader(
            datasets=[ds[0] for ds in datasets.values()],
            epochs=[ds[1] for ds in datasets.values()],
            max_length=self.max_packing_seqlen,
            oversize_policy="drop",
            transform=self.pack,
        )
        dataloader = DPMux(dataloader, dp_size=dp_size, dp_rank=dp_rank)

        dataloader = async_accelearte_slowfast(dataloader)  # optional, remove for debug
        return dataloader

    def preprocess(self, batch: dict):
        cu_seqlens = batch["cu_seqlens"].to("cuda")
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
            )
        else:
            return dict(
                cu_seqlens=cu_seqlens,
                max_seq_len=max_seq_len,
            )

    def pack(self, pieces: list):
        import numpy as np

        size = sum([len(s["tokens"]) - 1 for s in pieces])

        if size % self.seqlen_divisible_by != 0:
            # padding to the tensor_model_parallel_size
            padding_size = self.seqlen_divisible_by - size % self.seqlen_divisible_by

            padding_tensor = np.zeros(padding_size + 1)
            pieces.append(
                {
                    "tokens": padding_tensor,
                    "loss_mask": padding_tensor,
                }
            )

        sizes = torch.tensor([len(s["tokens"]) - 1 for s in pieces])
        from torch import tensor as T

        tokens = torch.cat([T(s["tokens"][:-1], dtype=torch.long) for s in pieces])
        labels = torch.cat([T(s["tokens"][1:], dtype=torch.long) for s in pieces])
        loss_mask = torch.cat([T(s["loss_mask"][1:], dtype=torch.float32) for s in pieces])

        cu_seqlens = torch.cat(
            [
                torch.zeros(1),
                torch.cumsum(sizes, 0),
            ]
        ).int()

        return dict(
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            cu_seqlens=cu_seqlens,
            max_seq_len=sizes.max(),
        )


class SFTExp(BaseExp):
    trainer_cfg: NTPTrainerConfig
    model_cfg: Megatron3DParallelModelConfig
    optimizer_cfg: GradientManagerConfig
    scheduler_cfg: SchedulerConfig
    data_cfg: Qwen3_SFT_DataConfig
    checkpoint_cfg: CheckpointConfig

    metric_cfg: PretrainMetricConfig
    profiler_cfg: ProfilerConfig


import sys
import time

# sys.excepthook = lambda a, b, c: [print("Hold On Error"), time.sleep(3600)]


class Exp(SFTExp):
    log_dir = "./tensorboard_logs/"

    trainer_cfg = NTPTrainerConfig
    model_cfg = Qwen3_8BConfig_128K_80G
    # model_cfg: Qwen3_30A3BConfig = Qwen3_30A3BConfig
    optimizer_cfg: GradientManagerConfig = GradientManagerConfig
    scheduler_cfg = ConstantSchedulerConfig
    data_cfg = Qwen3_SFT_DataConfig
    checkpoint_cfg = ConstantCheckpointConfig
    metric_cfg = PretrainMetricConfig
    profiler_cfg = ProfilerConfig

    def __init__(self):
        super().__init__()
        self.trainer_cfg.micro_batch_size = 1
        self.trainer_cfg.global_batch_size = 8
        self.trainer_cfg.global_seq_length = 65536

        self.trainer_cfg.train_iters = 1000
        self.trainer_cfg.log_interval = 1

        self.checkpoint_cfg.load_option.none(but=["model"])
        self.checkpoint_cfg.load_safetensors = "/mnt/step2-alignment-jfs/zane/opensources_model/Qwen3-30B-A3B-Base"
        self.checkpoint_cfg.save_safetensors = True
        self.checkpoint_cfg.save_dir = "/mnt/shared-storage/tenant/tmp/zhy/tmp/"
        self.checkpoint_cfg.save_option.all()
        self.checkpoint_cfg.save_interval = 100
        self.model_cfg.recompute = True
        self.optimizer_cfg.use_distributed_optimizer = False

    def train(self):
        self.update_from_args()
        self.sanity_check()
        trainer_cls = self.trainer_cfg.get_trainer_cls()
        trainer = trainer_cls(exp=self)
        from steptronoss.utils.optimizable import OPTIMIZABLE_REGISTER

        for k, v in OPTIMIZABLE_REGISTER.items():
            if "torch_compile" in v["alternatives"]:
                v["use_optimize"] = "torch_compile"
        trainer.train()


if __name__ == "__main__":
    Exp().train()
