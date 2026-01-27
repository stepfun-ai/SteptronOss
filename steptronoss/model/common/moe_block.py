"""Copyright 2026 StepFun Inc. All Rights Reserved."""

import functools
from typing import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from configurize import Config, Ref
from loguru import logger
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import detach_variable

from steptronoss.core.parallel_state import PM
from steptronoss.core.tensor_parallel import checkpoint

# from steptron.core import parallel_state as mpu
# from steptron.core.tensor_parallel import get_custom_tp_communicator
# from steptron.core.tensor_parallel.custom_layers import (
#     pre_function_backward as activation_backward,
# )
from steptronoss.core.tensor_parallel.mappings import (
    _gather_along_first_dim,
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    slice_to_sequence_parallel_region,
    split_along_first_dim_with_padding,
)
from steptronoss.exp.base_exp import MegatronTPConfig
from steptronoss.exp.ntp import MoePretrainMetricConfig
from steptronoss.model.utils import (
    bind_aux_loss,
    grouped_gemm,
    histogram,
    index_compute,
    moe_scatter,
    moe_weighted_gather,
)

# from steptron.model.common.optimus_ops import moe_gather
from steptronoss.timers import get_timers

# from steptronoss.model.common.moe_logging_utils import calc_expert_output_avgnorm_naive
from steptronoss.utils.metrics import GlobalMetrics

# from steptron.model.common.optimus_op import moe_scatter as optimus_moe_scatter
# from steptronoss.model.deepep.token_dispatcher import MoEFlexTokenDispatcher


GlobalMetrics: MoePretrainMetricConfig


def activation_backward(pre_func_output, grad_input, detached_inputs):
    # calculate the backward of pre-function
    if isinstance(pre_func_output, torch.Tensor):
        pre_func_output = (pre_func_output,)
    torch.autograd.backward(pre_func_output, [grad_input])
    grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in detached_inputs)
    return grads


class MoEConfig(Config):
    tp_cfg: MegatronTPConfig
    """Tensor-parallel config; provides params dtype and sequence-parallel flags."""
    hidden_size: int
    """Token hidden size used for gate input and expert weight shapes."""
    activation: Callable
    """Activation function between expert projections (e.g., SwiGLU)."""
    # use_moe: bool = False
    # moe_every_n_layer: int = 1
    moe_num_experts: int
    """Total number of experts globally; gate outputs this many logits."""
    moe_top_k: int
    """Number of experts selected per token by the router."""
    moe_aux_loss_coef: float
    """Scale applied to the router auxiliary loss."""
    moe_hidden_size: int
    """Hidden size of the expert MLP (inner dimension)."""

    routed_scaling_factor: float
    """Final scaling applied to routed outputs after aux-loss binding."""
    enable_sigmoid_router: bool
    """Use sigmoid routing probabilities instead of softmax."""
    router_bias_update_rate: float
    """Update rate for router_balance_bias in aux-loss-free load balancing."""
    moe_enable_deepep: bool
    """Enable DeepEP expert-parallel dispatch when EP > 1."""
    # moe_deepep_num_sms: int
    # """Number of SMs reserved for DeepEP kernels/dispatcher."""

    # fuse_moescatter_and_moecolumn: bool
    # """Enable fused scatter+column expert kernel path when available."""
    enable_auxiliary_loss_free_load_balance: bool
    """Track local tokens and apply router_balance_bias for load balancing."""
    norm_expert_weight: bool
    """Normalize top-k expert weights after routing."""

    moe_layer_list: list
    """Layer ids that use MoE; used to compute moe_layer_id."""

    share_expert_dim: int
    """Hidden size of the shared expert FFN in MoeShareExpertFFN."""

    def get_aux_loss_calib_scale(self):
        return 1


class MoEGate(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        sequence_parallel=False,
    ):
        super().__init__()
        self.sequence_parallel = sequence_parallel
        self.weight = torch.nn.Parameter(torch.empty(num_experts, dim, device=torch.cuda.current_device()))
        setattr(self.weight, "sequence_parallel", sequence_parallel)

    def forward(self, x):
        logits = x @ self.weight.T
        return logits


class GroupedExperts(torch.nn.Module):
    def __init__(self, cfg: MoEConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id

        self.cfg = cfg

        self.activation = self.cfg.activation

        self.num_global_experts = cfg.moe_num_experts
        self.num_local_experts = cfg.moe_num_experts // PM.size_of("EP")

        self.w1 = torch.nn.Parameter(
            torch.empty(
                size=(
                    cfg.moe_num_experts // PM.size_of("EP"),
                    cfg.moe_hidden_size * 2 // PM.size_of("ETP"),
                    cfg.hidden_size,
                ),
                device=torch.cuda.current_device(),
                dtype=cfg.tp_cfg.params_dtype,
            )
        )
        self.w2 = torch.nn.Parameter(
            torch.empty(
                size=(
                    cfg.moe_num_experts // PM.size_of("EP"),
                    cfg.hidden_size,
                    cfg.moe_hidden_size // PM.size_of("ETP"),
                ),
                device=torch.cuda.current_device(),
                dtype=cfg.tp_cfg.params_dtype,
            )
        )

        self.w1.expert_model_parallel = True
        self.w2.expert_model_parallel = True

    def forward(self, x: torch.FloatTensor, token_expert_ids, token_weights):
        experts_histogram = histogram(token_expert_ids, self.num_local_experts)

        # When all expert counts are zero (e.g., all indices invalid after dispatch),
        # skip grouped_gemm to avoid backend asserts and return zeros while still
        # participating in the ETP reduction.
        if experts_histogram.numel() == 0 or int(experts_histogram.sum().item()) == 0:
            x = x.new_zeros(x.shape)
            x = reduce_from_tensor_model_parallel_region(x, group="ETP")
            return x

        scatter_index = index_compute(token_expert_ids, experts_histogram)
        x = moe_scatter(x, scatter_index)

        experts_histogram = experts_histogram.cpu().long()  # groupgemm need

        x = grouped_gemm(x, self.w1, batch_sizes=experts_histogram, trans_b=True)

        x = self.activation(x)

        x = grouped_gemm(x, self.w2, batch_sizes=experts_histogram, trans_b=True)

        x = moe_weighted_gather(x, scatter_index, token_weights)

        x = reduce_from_tensor_model_parallel_region(x, group="ETP")
        return x


class MoEBlock(nn.Module):
    def __init__(self, cfg: MoEConfig, layer_id=0):
        super().__init__()
        self.cfg = cfg
        self.recompute_dis_activation = cfg.tp_cfg.distribute_saved_activations

        self.moe_top_k = cfg.moe_top_k

        self.sequence_parallel = cfg.tp_cfg.sequence_parallel

        self.moe_layer_id = cfg.moe_layer_list.index(layer_id)

        self.moe_aux_loss_coef = cfg.moe_aux_loss_coef

        self.gate = MoEGate(
            dim=cfg.hidden_size,
            num_experts=cfg.moe_num_experts,
            sequence_parallel=cfg.tp_cfg.sequence_parallel,
        )

        self.experts = GroupedExperts(cfg=cfg, layer_id=layer_id)

        self.num_global_experts = cfg.moe_num_experts
        self.norm_expert_weight = cfg.norm_expert_weight

        # aux-loss free load balance
        self.router_balance_bias: torch.FloatTensor
        self.local_tokens_per_expert: torch.IntTensor
        if cfg.enable_auxiliary_loss_free_load_balance:
            self.register_buffer(
                "router_balance_bias",
                torch.zeros(cfg.moe_num_experts, dtype=torch.float32),
            )
            self.register_buffer(
                "local_tokens_per_expert",
                torch.zeros(cfg.moe_num_experts, dtype=torch.int32),
                persistent=False,
            )
        else:
            self.router_balance_bias = None
            self.local_tokens_per_expert = None

        self.micro_batch_count = 0
        self.indices_numel = None
        self.router_bias_update_rate = cfg.router_bias_update_rate

        self.use_sigmoid_router = self.cfg.enable_sigmoid_router

        # moe scaling factor
        self.routed_scaling_factor = self.cfg.routed_scaling_factor

    def forward_router(self, logits: torch.FloatTensor):
        logits = logits.float()  # S, global_experts. where S is ONE full sample

        GlobalMetrics.router_logits_max.add(
            logits,
            iop=lambda x: x.detach().abs().max(),
            subname=f"layer{self.moe_layer_id}",
        )
        GlobalMetrics.router_logits_std.add(
            logits,
            iop=lambda x: x.detach().std(1).mean(),
            subname=f"layer{self.moe_layer_id}",
        )

        # DeepSeekv3 balance-bias routing: biased selection, unbiased weights
        if self.use_sigmoid_router:
            gate_prob = F.sigmoid(logits)
        else:
            gate_prob = F.softmax(logits, dim=1)

        if self.cfg.enable_auxiliary_loss_free_load_balance:
            self.maintain_float32_router_balance_bias()
            # Use bias only for selecting indices
            _gate_prob_with_bias = gate_prob + self.router_balance_bias.unsqueeze(0)
            _sorted_prob, _sorted_indices = torch.sort(_gate_prob_with_bias, descending=True, stable=True)
        else:
            # Traditional: same probabilities for both selection and weights
            _sorted_prob, _sorted_indices = torch.sort(gate_prob, descending=True, stable=True)

        topk_prob, token_expert_ids = (
            _sorted_prob[:, : self.moe_top_k],
            _sorted_indices[:, : self.moe_top_k].contiguous(),
        )

        token_weights = topk_prob

        # Sigmoid router requires explicit normalization for stability
        if self.use_sigmoid_router:
            # assert self.norm_expert_weight, "must enable norm expert"
            token_weights = token_weights / (
                token_weights.sum(dim=-1, keepdim=True) + (1e-20 if self.router_balance_bias is not None else 0)
            )
            gate_prob = gate_prob / (
                gate_prob.sum(dim=-1, keepdim=True) + (1e-20 if self.router_balance_bias is not None else 0)
            )
        elif self.norm_expert_weight:
            token_weights = token_weights / (
                torch.sum(topk_prob, dim=-1, keepdim=True) + (1e-20 if self.router_balance_bias is not None else 0)
            )

        experts_histogram = histogram(token_expert_ids, self.num_global_experts)

        ce = experts_histogram.clone()
        dist.all_reduce(ce, group=PM.group_of("TP"))
        dist.all_reduce(ce, group=PM.group_of("EP"))

        me = torch.mean(gate_prob, dim=0)

        # Track local tokens for aux-loss-free load balancing (training only) to update router_balance_bias
        if self.local_tokens_per_expert is not None:
            with torch.no_grad():
                # If indices_numel is not initialized, set it once for consistency checks if needed later
                if self.indices_numel is None:
                    self.indices_numel = token_expert_ids.numel()
                self.local_tokens_per_expert += experts_histogram
                self.micro_batch_count += 1

        aux_loss = torch.sum(me * ce) * self.num_global_experts

        GlobalMetrics.moe_minp.add(
            gate_prob,
            iop=lambda x: x.detach().min(1).values.mean(),
            subname=f"layer{self.moe_layer_id}",
        )
        GlobalMetrics.moe_sumk.add(
            gate_prob,
            iop=lambda x: x.detach().gather(1, x.argsort(1, True)[:, : self.moe_top_k]).sum(1).mean(),
            subname=f"layer{self.moe_layer_id}",
        )
        GlobalMetrics.moe_std.add(
            gate_prob,
            iop=lambda x: x.detach().std(1).mean(),
            subname=f"layer{self.moe_layer_id}",
        )
        GlobalMetrics.moe_avg.add(
            gate_prob,
            iop=lambda x: x.detach().mean(1).mean(),
            subname=f"layer{self.moe_layer_id}",
        )
        GlobalMetrics.moe_max.add(
            gate_prob,
            iop=lambda x: x.detach().max(1).values.mean(),
            subname=f"layer{self.moe_layer_id}",
        )

        GlobalMetrics.expert_coef.add(
            experts_histogram,
            iop=lambda x: (x.max() / x.sum()).detach().item(),
            subname=f"layer{self.moe_layer_id}",
        )

        return token_expert_ids, token_weights, aux_loss

    def forward_experts_tp(self, x, token_expert_ids, token_weights):
        if self.sequence_parallel:
            x = gather_from_sequence_parallel_region(x, "ETP")
            token_expert_ids = gather_from_sequence_parallel_region(token_expert_ids, "ETP")
            token_weights = gather_from_sequence_parallel_region(token_weights, "ETP")

        x = self.experts(x, token_expert_ids, token_weights)

        if self.sequence_parallel:
            x = slice_to_sequence_parallel_region(x, group="ETP")
        return x

    def forward_experts_ep(self, x, token_expert_ids, token_weights):
        assert PM.size_of("ETP") == 1
        # scatter_index = index_compute(token_expert_ids, experts_histogram)
        from steptronoss.model.deepep.token_dispatcher import TorchA2ADispatcher

        dispatcher = TorchA2ADispatcher("EP", num_experts=self.cfg.moe_num_experts)

        x, token_expert_ids, token_weights = dispatcher.dispatch(x, token_expert_ids, token_weights)

        x = self.experts(x, token_expert_ids, token_weights)

        x = dispatcher.combine(x)

        return x

    def forward(self, x: torch.FloatTensor, recompute: bool = False) -> torch.FloatTensor:
        S, B, C = x.shape
        x = x.reshape(-1, C)  # token-wise moe needs 2D input
        logits = self.gate(x)
        token_expert_ids, token_weights, aux_loss = self.forward_router(logits)

        if PM.size_of("EP") > 1:
            # use deepep
            assert PM.size_of("ETP") == 1, "Expert tensor parallel size should be 1"
            if recompute:
                output = checkpoint(
                    self.forward_experts_ep,
                    self.recompute_dis_activation,
                    x,
                    token_expert_ids,
                    token_weights,
                )
            else:
                output = self.forward_experts_ep(x, token_expert_ids, token_weights)
            aux_loss = aux_loss * self.moe_aux_loss_coef / PM.size_of("TP")
        else:
            if recompute:
                output = checkpoint(
                    self.forward_experts_tp,
                    self.recompute_dis_activation,
                    x,
                    token_expert_ids,
                    token_weights,
                )
            else:
                output = self.forward_experts_tp(x, token_expert_ids, token_weights)
            aux_loss = aux_loss * self.moe_aux_loss_coef / PM.size_of("ETP")

        output = output.reshape(S, B, output.shape[-1])

        output = bind_aux_loss(output, aux_loss)
        output = output * self.routed_scaling_factor

        GlobalMetrics.moe_aux_loss.add(
            aux_loss,
            iop=lambda x: x.clone().detach(),
            subname=f"layer{self.moe_layer_id}",
        )
        return output

    # Hacky functions
    def maintain_float32_router_balance_bias(self):
        """
        Maintain the router_balance_bias in float32.

        When using bf16/fp16, the expert bias gets converted to lower precision in Float16Module.
        We keep it in float32 to avoid routing errors when updating the router_balance_bias.
        """
        if self.router_balance_bias is not None:
            if self.router_balance_bias.dtype != torch.float32:
                self.router_balance_bias.data = self.router_balance_bias.data.to(torch.float32)

    def _load_from_state_dict(self, *args, **kwargs):
        self.maintain_float32_router_balance_bias()  # switch to float32 before loading
        return super()._load_from_state_dict(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        self.maintain_float32_router_balance_bias()  # switch to float32 before saving
        return super().state_dict(*args, **kwargs)
