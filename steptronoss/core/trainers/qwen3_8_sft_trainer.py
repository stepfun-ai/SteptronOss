"""Thin extension of the original SteptronOSS training loop."""

import os

import torch
from loguru import logger

from steptronoss.core import tensor_parallel
from steptronoss.core.parallel_state import PM
from steptronoss.core.trainers.lm_trainer import DecoderPretrainTrainer
from steptronoss.exp.ntp import NTPTrainerConfig
from steptronoss.utils import GlobalMetrics, print_n_params


class Qwen38SFTTrainerConfig(NTPTrainerConfig):
    def get_trainer_cls(self):
        return Qwen38SFTTrainer

    def loss_func(self, data, loss):
        if loss.numel() != 1:
            raise ValueError("Qwen38HFModel must return scalar loss")
        GlobalMetrics.lm_loss.add(loss, iop=lambda x: x.detach())
        GlobalMetrics.consumed_tokens.add(data["attention_mask"].sum() / PM.size_of("CP"), iop=lambda x: x.detach())
        return loss


class Qwen38SFTTrainer(DecoderPretrainTrainer):
    def load_checkpoint(self):
        state = super().load_checkpoint()
        cfg = self.exp.checkpoint_cfg
        self._resume_rng = PM.get_all_rng() if cfg.load_path and cfg.load_option.rng_state else None
        return state

    def before_train(self):
        super().before_train()
        # Parent loads RNG before model construction. Adapter initialization
        # consumes RNG, so restore the checkpoint state again after setup.
        if self._resume_rng is not None:
            PM.set_all_rng(self._resume_rng)

    def setup_model(self, model_config):
        model = model_config.build_model().cuda(torch.cuda.current_device())
        # Do not apply Float16Module's module.to(bf16): preserve HF's fp32 GDN
        # state parameters and PEFT's fp32 adapters.
        for param in model.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)
        if self.exp.trainer_cfg.log_detailed_grad_norms:
            model.name_parameters()
        print_n_params([model])
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Qwen3.8 trainable parameters: {trainable:,}", at=0)
        return torch.nn.ModuleList([model])

    def save_checkpoint(self):
        super().save_checkpoint()
        if PM.world_rank == 0:
            path = os.path.join(self.exp.checkpoint_cfg.save_path, f"it{self.iteration}", "hf_export")
            self.models[0].save_pretrained(path)
        torch.distributed.barrier()

    def after_train(self):
        # Native loop increments iteration AFTER the last update. Persist the
        # last completed index so parent resume's +1 does not skip an update.
        if self.iteration > self.start_iteration:
            self.iteration -= 1
            self.save_checkpoint()
            self.iteration += 1
        for hook in self._after_train_hooks:
            hook(self)
        del self.train_data_iterators
        logger.complete()
