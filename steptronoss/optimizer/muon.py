from __future__ import annotations

import math

import torch
from loguru import logger

from steptronoss.checkpointing.reshape_ops import Identity, ReshapeOp
from steptronoss.optimizer._helpers import get_zeropower_fn


# Adjust LR based on the Moonlight Muon recipe (scales by sqrt(max(A, B))).
def adjust_ratio_for_muon(matched_adamw_rms, param_shape):
    A, B = param_shape[-2:]
    adjusted_ratio = math.sqrt(max(A, B)) * matched_adamw_rms
    return adjusted_ratio


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        param_groups: The parameters to be optimized.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        matched_adamw_rms: The AdamW update RMS that Muon is designed to match. (0.2~0.4 recommended)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (5 is probably always enough)
        {0, 1}-D params or params not tagged as muon will be optimized by AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        param_groups,
        lr=2e-2,
        weight_decay=0.1,
        matched_adamw_rms=0.2,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        # When True, run Newton-Schulz in bf16 for speed; otherwise use fp32.
        run_ns_in_fp16=True,
        newtonschulz_fn="default",
        # adamw_use_nesterov=False,
    ):

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            matched_adamw_rms=matched_adamw_rms,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        super().__init__(param_groups, defaults)
        self.distributed_mode = False
        self.run_ns_in_fp16 = run_ns_in_fp16
        self.newtonschulz_fn = get_zeropower_fn(newtonschulz_fn, run_ns_in_fp16)

    @torch.no_grad()
    def step(self):
        has_muon_param = False

        for group_id, group in enumerate(self.param_groups):

            if not group.get("is_muon_param", False):
                continue

            has_muon_param = True

            # params = group["params"]
            momentum = group["momentum"]
            lr = group["lr"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            matched_adamw_rms = group["matched_adamw_rms"]

            for p in group["params"]:

                # ns_input = []
                # tp_split_dim = self.dist_metas[p].tp_split_dim
                # shard_cfg = p.sharded_tensor_config

                # merge_op is required to map sharded grads to per-matrix updates.
                merge_op: ReshapeOp = getattr(p, "merge_op", None)
                if merge_op is None:
                    logger.warning(f"A param with name shape {p.shape} has no merge_op, using identity (no gather).")
                    merge_op = Identity()

                # update muon momentum first
                g: torch.Tensor = p.grad
                assert g is not None

                # prepare muon buffer in state
                state: dict[str, torch.Tensor] = self.state[p]
                if not "muon_buffer" in state:
                    state["muon_buffer"] = torch.zeros_like(g)
                buf = state["muon_buffer"]
                buf.mul_(momentum).add_(g)

                # save to ns input
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                # Run Newton-Schulz in bf16 if requested.
                g = g.bfloat16() if self.run_ns_in_fp16 else g.float()

                merged_g = merge_op.forward({"grad": g})
                for k in list(merged_g):
                    v = merged_g[k]
                    adjusted_lr = lr * adjust_ratio_for_muon(matched_adamw_rms, v.shape)
                    merged_g[k] = self.newtonschulz_fn(v, steps=ns_steps) * -adjusted_lr

                update = merge_op.backward(merged_g)["grad"]
                # dist_meta = self.dist_metas[p]
                update = update.contiguous()
                # apply weight decay
                p.data.mul_(1 - lr * weight_decay)
                # apply update
                p.data.add_(update)

        if not hasattr(self, "_warned") and not has_muon_param:
            logger.warning(
                "Muon optimizer is used but no muon param is found. "
                "Make sure prepare_model_for_muon is called before instantiating LocalDDP."
            )
            self._warned = True

        # use adam for other params
        for group in self.param_groups:

            if group.get("is_muon_param", False):
                continue

            # init step
            if "step" in group:
                group["step"] += 1
            else:
                group["step"] = 1

            step = group["step"]
            params = group["params"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            use_nesterov = group.get("adamw_use_nesterov", False)

            for p in params:

                g = p.grad
                assert g is not None
                state = self.state[p]

                if len(state) == 0:
                    state["adamw_exp_avg"] = torch.zeros_like(g)
                    state["adamw_exp_avg_sq"] = torch.zeros_like(g)

                exp_avg = state["adamw_exp_avg"]
                exp_avg_sq = state["adamw_exp_avg_sq"]

                # decay the first and second moment running average coefficient
                exp_avg.lerp_(g, 1 - beta1)
                exp_avg_sq.lerp_(g.square(), 1 - beta2)

                # correct_bias
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                if use_nesterov:
                    bias_correction_nesterov = 1 - beta1 ** (step + 1)
                    momentum = beta1 * exp_avg / bias_correction_nesterov + (1 - beta1) * g / bias_correction1
                else:
                    # Use standard momentum, optionally with bias correction
                    momentum = exp_avg / bias_correction1

                # construct the denominator of the inner ADAM optimizer
                adam_second_moment = exp_avg_sq / bias_correction2
                adam_second_moment = adam_second_moment.sqrt() + eps
                update = momentum / adam_second_moment

                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(update, alpha=-lr)
