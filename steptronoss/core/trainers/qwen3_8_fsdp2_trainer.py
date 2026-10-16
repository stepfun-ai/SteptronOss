"""FSDP2 backend using SteptronOSS's original loop and FW/BW scheduler."""

import os

import torch
from loguru import logger

from steptronoss.checkpointing.fsdp2_checkpoint import FSDP2Checkpointer
from steptronoss.core.trainers.qwen3_8_sft_trainer import Qwen38SFTTrainer, Qwen38SFTTrainerConfig
from steptronoss.model.qwen3_8_fsdp2 import build_fsdp2_qwen


class Qwen38FSDP2TrainerConfig(Qwen38SFTTrainerConfig):
    def get_trainer_cls(self):
        return Qwen38FSDP2Trainer


class Qwen38FSDP2Trainer(Qwen38SFTTrainer):
    def __init__(self, exp):
        super().__init__(exp)
        self.checkpointer = FSDP2Checkpointer()

    def setup_model(self, model_config):
        model = build_fsdp2_qwen(model_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"FSDP2 Qwen3.8 trainable global parameters: {trainable:,}", at=0)
        return torch.nn.ModuleList([model])

    def train_step(self):
        torch.cuda.reset_peak_memory_stats()
        successful = super().train_step()
        if self.iteration % self.exp.trainer_cfg.log_interval == 0:
            allocated = torch.cuda.max_memory_allocated() / 2**30
            reserved = torch.cuda.max_memory_reserved() / 2**30
            logger.info(f"FSDP2 peak memory: allocated={allocated:.2f} GiB reserved={reserved:.2f} GiB", at="all")
        return successful

    def save_checkpoint(self):
        torch.cuda.reset_peak_memory_stats()
        logger.info("CP_PHASE save_start", at=0)
        self.checkpointer.dump_ckpt(
            cfg=self.exp.checkpoint_cfg,
            iteration=self.iteration,
            model=self.models,
            optimizer=self.grad_manager,
            opt_param_scheduler=self.opt_param_scheduler,
            dataloader=self.train_data_iterators[0],
            extra_info={"exp": self.exp.to_dict(), "history": self.history},
        )
        output = os.path.join(self.exp.checkpoint_cfg.save_path, f"it{self.iteration}", "hf_export")
        torch.cuda.synchronize()
        logger.info(
            f"CP_SAVE_MEMORY allocated={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB reserved={torch.cuda.max_memory_reserved() / 2**30:.2f} GiB",
            at="all",
        )
        torch.cuda.reset_peak_memory_stats()
        logger.info("CP_PHASE export_start", at=0)
        self.models[0].save_pretrained(output)
        torch.distributed.barrier()
        self.checkpointer.publish_latest(self.exp.checkpoint_cfg, self.iteration)
        torch.cuda.synchronize()
        logger.info(
            f"CP_EXPORT_MEMORY allocated={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB reserved={torch.cuda.max_memory_reserved() / 2**30:.2f} GiB",
            at="all",
        )
        logger.info("CP_PHASE save_export_done", at=0)
