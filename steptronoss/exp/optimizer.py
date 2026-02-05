from __future__ import annotations

from typing import NoReturn

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

    # Build param groups once so scheduler scales stay consistent across Muon/AdamW.
    def build_optimizer(self, model: Module) -> torch.optim.Optimizer:
        raise NotImplementedError


class AdamConfig(OptimizerConfig):
    lr: float = Ref("...scheduler_cfg.lr")
    weight_decay: float = Ref("...scheduler_cfg.weight_decay")
    weight_decay_on_1d_params: bool = False

    # adam-specific hyperparams
    adam_eps: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95

    # Build param groups once so scheduler scales stay consistent across Muon/AdamW.
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


class MuonConfig(OptimizerConfig):
    # Muon for 2D weights, AdamW for others; selection is configurable below.
    lr: float = Ref("...scheduler_cfg.lr")
    """Base learning rate for both Muon and AdamW."""
    weight_decay: float = Ref("...scheduler_cfg.weight_decay")
    """Global weight decay applied to both optimizers."""
    weight_decay_on_1d_params: bool = False
    """Whether to apply weight decay to 1D parameters (e.g., biases/norms)."""

    muon_momentum: float = 0.95
    """Muon momentum coefficient."""
    muon_nesterov: bool = True
    """Enable Nesterov-style update in Muon."""
    muon_matched_adamw_rms: float = 0.2
    """Target AdamW update RMS used to scale Muon updates."""
    muon_ns_steps: int = 5
    """Number of Newton-Schulz steps for Muon."""
    muon_run_ns_in_fp16: bool = True
    """Run Newton-Schulz in bf16 for speed when True."""
    muon_newtonschulz_fn: str = "default"
    """Newton-Schulz kernel name (see ZEROPOWER_COEFFS)."""

    adam_eps: float = 1e-8
    """AdamW epsilon."""
    adam_beta1: float = 0.9
    """AdamW beta1."""
    adam_beta2: float = 0.95
    """AdamW beta2."""

    muon_param_attr_only: bool = False
    """Only use params tagged with is_muon_param when True."""
    muon_exclude_embeddings: bool = True
    """Exclude embedding weights from Muon when True."""
    muon_exclude_names: tuple[str, ...] = ()
    """Exclude params whose names contain any of these substrings."""

    def mark_muon_params(self, model: Module) -> NoReturn:
        """Set param.is_muon_param for all parameters using internal exclude rules."""
        muon_exclude_names = self.muon_exclude_names
        if muon_exclude_names is None:
            muon_exclude_names = ()

        embedding_params: set[Parameter] = set()
        if self.muon_exclude_embeddings:
            for module in model.modules():
                if isinstance(module, torch.nn.Embedding) or "Embedding" in module.__class__.__name__:
                    weight = getattr(module, "weight", None)
                    if isinstance(weight, Parameter):
                        embedding_params.add(weight)

        for name, param in model.named_parameters():
            if getattr(param, "is_muon_param", False):
                is_muon_param = True
            elif (
                self.muon_param_attr_only
                or (self.muon_exclude_embeddings and param in embedding_params)
                or (muon_exclude_names and any(tag in name for tag in muon_exclude_names))
            ):
                is_muon_param = False
            else:
                is_muon_param = param.ndim == 2

            param.is_muon_param = is_muon_param

    def mark_gather_ops(self, model: Module) -> NoReturn:
        """Attach merge_op for each parameter based on its TP partition info."""
        from steptronoss.checkpointing.reshape_ops import ColumnParallel, Inverse, KeepThisTP, RowParallel, Sequential

        identity = Sequential([])

        for _, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if getattr(param, "tensor_model_parallel", False):
                partition_dim = getattr(param, "partition_dim", -1)
                if partition_dim == 0:
                    merge_op = Inverse(ColumnParallel() + KeepThisTP())
                elif partition_dim == 1:
                    merge_op = Inverse(RowParallel() + KeepThisTP())
                else:
                    merge_op = identity
            else:
                merge_op = identity

            param.merge_op = merge_op

    # Build param groups once so scheduler scales stay consistent across Muon/AdamW.
    def build_optimizer(self, model: Module) -> torch.optim.Optimizer:
        from steptronoss.optimizer.muon import Muon
        from steptronoss.optimizer.utils import advanced_get_param_groups

        self.mark_muon_params(model)
        self.mark_gather_ops(model)
        missing_merge_op = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if not hasattr(param, "merge_op"):
                missing_merge_op.append(name)
        if missing_merge_op:
            preview = ", ".join(missing_merge_op[:10])
            total = len(missing_merge_op)
            raise RuntimeError(
                "Muon requires merge_op for all trainable params, "
                f"but found {total} missing (showing first 10): {preview}"
            )

        def is_muon_param(name: str, param: Parameter) -> bool:
            return getattr(param, "is_muon_param", False)

        param_groups = advanced_get_param_groups(
            model,
            scale_lr_cond=self.scale_lr_func,
            scale_wd_cond=self.scale_wd_cond,
            is_muon_param=is_muon_param,
        )

        return Muon(
            param_groups,
            lr=self.lr,
            weight_decay=self.weight_decay,
            matched_adamw_rms=self.muon_matched_adamw_rms,
            momentum=self.muon_momentum,
            nesterov=self.muon_nesterov,
            ns_steps=self.muon_ns_steps,
            adamw_betas=(self.adam_beta1, self.adam_beta2),
            adamw_eps=self.adam_eps,
            run_ns_in_fp16=self.muon_run_ns_in_fp16,
            newtonschulz_fn=self.muon_newtonschulz_fn,
        )
