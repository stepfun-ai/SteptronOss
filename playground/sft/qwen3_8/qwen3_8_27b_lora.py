"""Pure text Qwen3.8-27B LoRA SFT using SteptronOSS native DP training."""

import os

from playground.sft.qwen3.qwen3_sft_base import Exp as BaseExp
from steptronoss.core.trainers.qwen3_8_sft_trainer import Qwen38SFTTrainerConfig
from steptronoss.data.qwen3_8_sft import Qwen38SFTDataConfig
from steptronoss.exp.lr_schedulers import CosineSchedulerConfig
from steptronoss.model.qwen3_8_hf import Qwen38HFConfig


class Exp(BaseExp):
    model_cfg = Qwen38HFConfig
    data_cfg = Qwen38SFTDataConfig
    trainer_cfg = Qwen38SFTTrainerConfig
    scheduler_cfg = CosineSchedulerConfig

    def __init__(self):
        super().__init__()
        self.model_cfg.tp_cfg.sequence_parallel = False
        self.model_cfg.tp_cfg.async_tensor_model_parallel_allreduce = False
        self.trainer_cfg.micro_batch_size = 1
        self.trainer_cfg.global_batch_size = 32
        self.trainer_cfg.global_seq_length = 2048
        self.scheduler_cfg.lr = 1e-4
        self.scheduler_cfg.min_lr = 1e-5
        self.scheduler_cfg.warmup_schedule = 0
        self.scheduler_cfg.scheduler_unit = "iter"
        self.scheduler_cfg.total_schedule = None
        self.scheduler_cfg.weight_decay = 0.0
        # Small LoRA state: native gradient manager's DP all-reduce is sufficient.
        self.optimizer_cfg.use_distributed_optimizer = False
        self.checkpoint_cfg.save_dir = "./checkpoints"
        self.checkpoint_cfg.save_interval = 100
        self.checkpoint_cfg.async_dump = False
        self.checkpoint_cfg.save_safetensors = False

    def configure_optimizable(self):
        pass  # HF selects SDPA/FLA kernels; native compile replacements don't apply.

    def sanity_check(self):
        super().sanity_check()
        world_size = int(os.environ["WORLD_SIZE"])
        if self.model_cfg.parallel_cfg.context_parallel_size > 1:
            from steptronoss.core.trainers.qwen3_8_fsdp2_trainer import Qwen38FSDP2Trainer

            if not issubclass(self.trainer_cfg.get_trainer_cls(), Qwen38FSDP2Trainer):
                raise ValueError("CP is supported only by the FSDP2 experiment")
        # Context parallelism splits one sample across its group, so the ranks in a
        # CP group consume a single sample between them. The data-parallel degree --
        # how many distinct samples are in flight -- is world_size / cp, and that is
        # what global_batch_size has to divide by.
        cp_size = int(self.model_cfg.parallel_cfg.context_parallel_size)
        if world_size % cp_size:
            raise ValueError(f"WORLD_SIZE {world_size} is not divisible by context_parallel_size {cp_size}")
        unit = world_size // cp_size * self.trainer_cfg.micro_batch_size
        if self.trainer_cfg.global_batch_size % unit:
            raise ValueError(
                "global_batch_size must be divisible by (WORLD_SIZE / context_parallel_size) "
                f"* micro_batch_size = {unit}"
            )
        if self.trainer_cfg.train_iters is not None and self.trainer_cfg.train_iters <= 0:
            raise ValueError("train_iters must be positive")
        if self.checkpoint_cfg.load_safetensors or self.checkpoint_cfg.save_safetensors:
            raise ValueError("Use model_cfg.model_path for base weights and the native checkpoint for resume")

    def assert_critical_attrs_expected(self, extra):
        previous = extra["exp"]
        current = self.to_dict()
        for section, keys in {
            "model_cfg": (
                "model_path",
                "revision",
                "tuner",
                "lora_rank",
                "lora_alpha",
                "lora_dropout",
                "lora_target_suffixes",
            ),
            "trainer_cfg": ("micro_batch_size", "global_batch_size", "global_seq_length"),
            "data_cfg": ("pad_token_id", "seed"),
            "optimizer_cfg": ("use_distributed_optimizer",),
        }.items():
            for key in keys:
                if previous[section][key] != current[section][key]:
                    raise ValueError(f"Resume configuration changed: {section}.{key}")


if __name__ == "__main__":
    Exp().train()
