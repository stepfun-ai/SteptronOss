"""Text LoRA defaults; override resources and hyperparameters with native Config."""

import os

from playground.sft.qwen3_8.qwen3_8_27b_lora import Exp as DPExp
from steptronoss.core.trainers.qwen3_8_fsdp2_trainer import Qwen38FSDP2TrainerConfig
from steptronoss.optimizer.fsdp2_gradient_manager import FSDP2GradientManagerConfig, require_fsdp2_version


class Exp(DPExp):
    trainer_cfg = Qwen38FSDP2TrainerConfig
    optimizer_cfg = FSDP2GradientManagerConfig

    def __init__(self):
        super().__init__()
        self.model_cfg.loss_backend = "chunked"
        self.model_cfg.loss_chunk_size = 128
        self.model_cfg.attn_implementation = "flash_attention_2"
        self.trainer_cfg.global_seq_length = 16384
        self.trainer_cfg.global_batch_size = 16
        self.trainer_cfg.log_num_zeros_in_grad = False
        self.optimizer_cfg.log_num_zeros_in_grad = False
        self.resource_cfg.envs |= {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    def sanity_check(self):
        super().sanity_check()
        require_fsdp2_version()
        if self.model_cfg.tuner != "lora":
            raise ValueError("FSDP2 experiment supports LoRA only")
        cp_size = int(self.model_cfg.parallel_cfg.context_parallel_size)
        if cp_size > 1:
            # Each rank owns one contiguous block, and every block boundary has to
            # land on a multiple of 64 -- the Gated DeltaNet kernel's internal chunk
            # -- because that is what makes resuming from a carried state exact.
            if self.trainer_cfg.global_seq_length % (cp_size * 64):
                raise ValueError(
                    f"global_seq_length {self.trainer_cfg.global_seq_length} must be a multiple of "
                    f"context_parallel_size * 64 = {cp_size * 64}"
                )
            world = int(os.environ["WORLD_SIZE"])
            if world % cp_size:
                raise ValueError(f"WORLD_SIZE {world} is not divisible by context_parallel_size {cp_size}")
        if self.optimizer_cfg.use_distributed_optimizer:
            raise ValueError("Native ZeRO-1 cannot be combined with FSDP2")
        if self.trainer_cfg.offload_optimizer_state or self.model_cfg.pipeline_activation_cpu_offload:
            raise ValueError("Native optimizer/pipeline offload is not supported by the FSDP2 backend")
        if self.checkpoint_cfg.async_dump or "://" in self.checkpoint_cfg.save_dir:
            raise ValueError("FSDP2 checkpoints require synchronous writes to a shared local filesystem")
        if self.checkpoint_cfg.load_path and "://" in self.checkpoint_cfg.load_path:
            raise ValueError("FSDP2 resume requires a shared local filesystem")
        if self.trainer_cfg.log_detailed_grad_norms:
            raise ValueError("Native detailed gradient norms do not support DTensor")

    def assert_critical_attrs_expected(self, extra):
        super().assert_critical_attrs_expected(extra)
        old, new = extra["exp"], self.to_dict()
        for key in (
            "loss_backend",
            "loss_chunk_size",
            "gradient_checkpointing",
            "attn_implementation",
            "fsdp2_activation_cpu_offload",
        ):
            if old["model_cfg"][key] != new["model_cfg"][key]:
                raise ValueError(f"Resume configuration changed: model_cfg.{key}")


if __name__ == "__main__":
    Exp().train()
