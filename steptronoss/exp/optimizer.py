from __future__ import annotations

import torch
from configurize import Ref
from torch.nn import Module, Parameter

from steptronoss.exp.abstract import OptimizerConfig as AbstractOptimizerConfig


class OptimizerConfig(AbstractOptimizerConfig):

    weight_decay: float
    weight_decay_on_1d_params: bool

    lr: float

    def scale_lr_func(self, name: str, param: Parameter) -> float:
        if hasattr(param, "_lr_scale"):
            return param._lr_scale
        return 1.0

    def scale_wd_cond(self, name: str, param: Parameter) -> float:
        if name.endswith(".bias") or (len(param.shape) == 1 and not self.weight_decay_on_1d_params):
            return 0.0
        return 1.0

    def build_optimizer(self, model: Module) -> torch.optim.Optimizer:
        raise NotImplementedError

    def sanity_check(self) -> None:
        super().sanity_check()
        # if self.optimizer == "muon":
        #     assert int(self.muon_run_ns_in_fp16) + int(self.muon_run_ns_in_fp32) <= 1

        #     if self.muon_log_updates_grad_norms:
        #         assert self.log_detailed_grad_norms

        #     assert self.muon_auto_applier_attn_pack_param_strategy in [
        #         "split_by_head",
        #         "split_by_type",
        #         "no_split",
        #     ]
        #     assert self.muon_auto_applier_glu_pack_param_strategy in [
        #         "split_by_type",
        #         "no_split",
        #     ]


class AdamConfig(OptimizerConfig):
    lr: float = Ref("...scheduler_cfg.lr")
    weight_decay: float = Ref("...scheduler_cfg.weight_decay")
    weight_decay_on_1d_params: bool = False

    # adam-specific hyperparams
    adam_eps: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95

    def build_optimizer(self, model: Module) -> torch.optim.Optimizer:
        # Base optimizer.
        from torch.optim.adam import Adam

        from steptronoss.optimizer.utils import advanced_get_param_groups

        param_groups = advanced_get_param_groups(
            model,
            scale_lr_cond=self.scale_lr_func,
            scale_wd_cond=self.scale_wd_cond,
        )

        return Adam(
            param_groups,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_eps,
        )
