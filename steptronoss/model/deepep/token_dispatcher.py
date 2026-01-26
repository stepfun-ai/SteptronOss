# adopted from Megatron-Core with compatibility modifcations, megatron/core/transformer/moe/token_dispatcher.py
# commit: c78fb087

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as distnn

from steptronoss.core.parallel_state import PM
from steptronoss.model.utils.moe_utils import histogram
from steptronoss.timers import get_timers
from steptronoss.utils.dist_utils import all_gather_object, all_to_all_tensors

from .fused_a2a import (
    fused_combine,
    fused_dispatch,
    set_deepep_num_sms,
)
from .fused_indices_converter import fused_indices_to_multihot

if TYPE_CHECKING:
    from steptronoss.model.common.moe_block import MoEConfig

try:
    import transformer_engine as te  # type: ignore
    from transformer_engine.pytorch.permutation import (  # type: ignore
        moe_permute,
        moe_permute_with_probs,
        moe_sort_chunks_by_index,
        moe_sort_chunks_by_index_with_probs,
        moe_unpermute,
    )

    fused_permute = moe_permute
    fused_permute_with_probs = moe_permute_with_probs
    fused_sort_chunks_by_index = moe_sort_chunks_by_index
    fused_sort_chunks_by_index_with_probs = moe_sort_chunks_by_index_with_probs
    fused_unpermute = moe_unpermute

    HAVE_TE = True
except ImportError:
    fused_permute = None
    fused_permute_with_probs = None
    fused_sort_chunks_by_index = None
    fused_sort_chunks_by_index_with_probs = None
    fused_unpermute = None

    HAVE_TE = False


""" We use the following notation throughout this file:
     H: hidden size
     B: micro batch size
     S: sequence length
     TP: tensor model parallel size
     EP: expert model parallel size
     num_local_tokens: S/TP*B
     num_global_tokens: num_local_tokens*TP*EP
"""


def permute(
    tokens,
    routing_map,
    probs: Optional[torch.Tensor] = None,
    num_out_tokens: Optional[int] = None,
    fused: bool = False,
    drop_and_pad: bool = False,
):
    """Permute the tokens and probs based on the mask.
    Tokens with the same designated expert will be grouped together.
    The shape of mask is [tokens, num_experts], it indicates which experts were selected
    by each token.

    When drop_and_pad=True, in routing_map, the number of non-zeros in each column equals to
    expert capacity. This function exploits this feature to use ops that support cuda graph.

    Args:
        tokens (torch.Tensor): The input token tensor, [num_tokens, hidden].
        routing_map (torch.Tensor): The sparse token to expert mapping, [num_tokens, num_experts].
        probs (torch.Tensor, optional): The probs tensor, [num_tokens, num_experts].
        num_out_tokens (int, optional): The number of output tokens. If None, it's set to
                                        the number of input tokens.
        fused (bool, optional): Whether use the fused permute function.
        drop_and_pad (bool, optional): Whether or not the token dispatcher uses token-drop
                                       and pads the number of tokens to the expert capacity.
                                       If set to true, routing_map has a fixed number of non-zeros
                                       in each column.
    """
    if fused and probs is None:
        if not HAVE_TE or fused_permute is None:
            raise ValueError("fused_permute is not available. Please install TE >= 2.1.0.")
        permuted_input, sorted_indices = fused_permute(tokens, routing_map, num_out_tokens=num_out_tokens)
        return permuted_input, None, sorted_indices

    if fused and probs is not None:
        if not HAVE_TE or fused_permute_with_probs is None:
            raise ValueError("fused_permute_with_probs is not available. Please install TE >= 2.1.0.")
        return fused_permute_with_probs(tokens, probs, routing_map, num_out_tokens=num_out_tokens)

    num_tokens, hidden = tokens.shape
    num_experts = routing_map.shape[1]
    permuted_probs = None
    if drop_and_pad and not (num_out_tokens is None):
        capacity = num_out_tokens // num_experts
        assert not routing_map.requires_grad
        # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
        routing_map = routing_map.to(dtype=torch.int8).T.contiguous()
        # use argsort to put indices of all non-zeros in the beginning of list
        # and keep the first `capacity` number of indices
        sorted_indices = routing_map.argsort(dim=-1, descending=True, stable=True)[:, :capacity].contiguous()
        # flatten from [num_experts, capacity] to 1D
        sorted_indices = sorted_indices.view(-1)

        if probs is not None:
            # [num_tokens, num_experts] -> num_experts * num_tokens
            probs_T_1D = probs.T.contiguous().view(-1)
            # get 1D indices of the probs selected by routing_map
            indices_dim0 = torch.arange(num_experts, device=routing_map.device).unsqueeze(-1)
            indices_dim1 = sorted_indices.view(num_experts, capacity)
            indices_1D = (indices_dim0 * num_tokens + indices_dim1).view(-1)
            # get probs from indices
            permuted_probs = probs_T_1D.index_select(0, indices_1D)
    else:
        # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
        routing_map = routing_map.bool().T.contiguous()

        # Create a dense expert-to-token mapping from the sparse token-to-expert mapping
        token_indices = torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
        sorted_indices = token_indices.masked_select(routing_map)

        if probs is not None:
            permuted_probs = probs.T.contiguous().masked_select(routing_map)

    # use the mapping to permute the tokens
    permuted_input = tokens.index_select(0, sorted_indices)

    return permuted_input, permuted_probs, sorted_indices


def unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
    probs: torch.Tensor = None,
    routing_map: torch.Tensor = None,
    fused: bool = False,
    drop_and_pad: bool = False,
):
    """
    Restore the original order of tokens after permutation. If probs are provided, it
    will also apply them to the tokens before restoring the order.

    When drop_and_pad=True, the tensors will have the following properties:
      - In routing_map, the number of non-zeros in each column equals to expert capacity
      - The size of sorted_indices equals to num_experts * capacity, each split of `capacity`
        contains the indices of tokens routed to an expert.
    This function exploits these features to use ops that support cuda graph.

    Args:
        permuted_tokens (torch.Tensor): The permuted token tensor.
        sorted_indices (torch.Tensor): The indices used to sort the tokens.
        restore_shape (torch.Size): The shape of the unpermuted tensor.
        probs (torch.Tensor, optional): The unpermuted probs tensor,
        routing_map (torch.Tensor, optional): Token to expert mapping, shape
            [num_tokens, num_experts].
        fused (bool, optional): Whether use the fused unpermute function.
        drop_and_pad (bool, optional): Whether or not the token dispatcher uses token-drop
                                       and pads the number of tokens to the expert capacity.

    Returns:
        torch.Tensor: The tokens restored to their original order.
    """
    if fused:
        if not HAVE_TE or fused_unpermute is None:
            raise ValueError("fused_unpermute is not available. Please install TE >= 2.1.0.")
        return fused_unpermute(
            permuted_tokens,
            sorted_indices,
            merging_probs=probs,
            restore_shape=restore_shape,
        )

    _, hidden = restore_shape
    input_dtype = permuted_tokens.dtype

    if probs is not None:
        assert routing_map is not None, "Mask must be provided to permute the probs."
        if drop_and_pad:
            num_experts = routing_map.size(1)
            num_permuted_tokens = sorted_indices.size(0)
            capacity = num_permuted_tokens // num_experts
            num_unpermuted_tokens = probs.size(0)

            # [num_unpermuted_tokens, num_experts] -> num_experts * num_unpermuted_tokens
            probs_T_1D = probs.T.contiguous().view(-1)

            # get 1D indices of the probs selected by routing_map
            indices_dim0 = torch.arange(num_experts, device=routing_map.device).unsqueeze(-1)
            indices_dim1 = sorted_indices.view(num_experts, capacity)
            indices_1D = (indices_dim0 * num_unpermuted_tokens + indices_dim1).view(-1)

            # get probs from indices
            permuted_probs = probs_T_1D.index_select(0, indices_1D)
        else:
            permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())
        # Here may promote permuted_tokens to higher precision (fp32/fp64) if probs is in
        # higher precision due to moe_router_dtype being enabled. This can lead to
        # additional GPU memory usage. Use --moe-permute-fusion flag to avoid this extra memory
        # allocation.
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

    # Create an output tensor filled with zeros
    output_tokens = torch.zeros(restore_shape, dtype=permuted_tokens.dtype, device=permuted_tokens.device)
    # Scatter add the permuted_input back to the original positions
    output_tokens.scatter_add_(0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens)
    return output_tokens.to(dtype=input_dtype)


class _DispatchManager(ABC):
    """
    A manager class to handle dispatch and combine processes for MoE models.

    DispatcherManager handles token dispatching according to the routing_map of format
    [num_local_tokens, world_size, num_instances]. The routing_map is a 3D tensor where each
    element indicates whether a token should be sent to a specific rank.

    num_instances is the maximum number of tokens instances dispatched into a target rank, it
    can be the number of local experts, or the size of sub_group.
    """

    @abstractmethod
    def setup_metadata(self, routing_map: torch.Tensor, probs: torch.Tensor):
        """Set up metadata of routing_map and probs."""
        pass

    @abstractmethod
    def dispatch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Dispatch the hidden_states according to the routing_map."""
        pass

    @abstractmethod
    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Combine the hidden_states after expert processing."""
        pass

    @abstractmethod
    def get_dispached_metadata(self) -> torch.Tensor:
        """Get the metadata of the dispatched hidden_states."""
        pass

    @abstractmethod
    def get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get the permuted hidden states by instances."""
        pass

    @abstractmethod
    def get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get the restored hidden states by instances."""
        pass


class _DeepepManager(_DispatchManager):
    """
    A manager class to handle fused all-to-all communication processes for MoE models using
    DeepEP backend. See https://github.com/deepseek-ai/deepep for more details.

    The workflow of the DeepEP dispatcher is:
    (1) setup_metadata(): Process routing map and probabilities to prepare dispatch metadata
    (2) dispatch():
        - Use fused kernel to permute tokens and perform all-to-all communication in single step
    (3) get_permuted_hidden_states_by_instances():
        - Convert routing map and probabilities to multihot format
        - Permute tokens using fused kernel
    (4) get_restored_hidden_states_by_instances():
        - Reverse permutation using fused kernel
    (5) combine():
        - Reverse process using fused kernel to unpermute and perform all-to-all in single step

    This implementation uses fused communication kernels (fused_dispatch/fused_combine) that
    combine permutation and communication operations for improved efficiency compared to
    separate permute+alltoall steps.
    """

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        num_local_experts: int,
        router_topk: int,
        num_experts: int,
    ):
        """
        Initialize the DeepEP dispatcher.

        Args:
            group (torch.distributed.ProcessGroup): The process group to use for communication.
                This should be the ETPxEP group.
            num_local_experts (int): The number of local experts.
            router_topk (int): The number of experts for each token to select.
            num_experts (int): The total number of experts in the group.
        """
        self.group = group
        self.num_local_experts = num_local_experts
        self.router_topk = router_topk
        self.num_experts = num_experts
        self.permute_fusion = False

        # Metadata
        self.token_indices: Optional[torch.Tensor] = None
        self.token_probs: Optional[torch.Tensor] = None
        # Handle used for combine operation
        self.handle = None

        # if fused_dispatch is None:
        #     raise ImportError(
        #         "DeepEP is not installed. Please install DeepEP package from "
        #         "https://github.com/deepseek-ai/deepep."
        #     )

    def setup_metadata(self, token_indices: torch.Tensor, token_probs: torch.Tensor):
        # num_tokens = routing_map.shape[0]

        # routing_map = routing_map.reshape(num_tokens, self.num_experts)
        # probs = probs.reshape(num_tokens, self.num_experts)
        # # Convert the format of routing map from multihot to indices.
        # self.token_probs, self.token_indices = torch.topk(probs, self.router_topk, dim=-1)
        self.token_probs, self.token_indices = token_probs, token_indices

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> torch.Tensor:
        # DeepEP only supports float32 probs
        if self.token_probs.dtype != torch.float32:
            if self.token_probs.dtype in [torch.bfloat16, torch.float16]:
                print("DeepEP only supports float32 probs, please set --moe-router-dtype=fp32")
            self.token_probs = self.token_probs.float()  # downcast or upcast
        (
            hidden_states,
            dispatched_indices,
            dispatched_probs,
            num_tokens_per_expert,
            handle,
        ) = fused_dispatch(
            hidden_states,
            self.token_indices,
            self.token_probs,
            self.num_experts,
            self.group,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        self.handle = handle
        self.tokens_per_expert = num_tokens_per_expert
        self.dispatched_indices = dispatched_indices
        self.dispatched_probs = dispatched_probs

        return hidden_states

    def _indices_to_multihot(self, indices, probs):
        """
        Converts a tensor of indices to a multihot vector.

        Args:
            indices (torch.Tensor): [num_tokens, topk] token indices, where -1 means masked out.
            probs (torch.Tensor): [num_tokens, topk] token probabilities.

        Returns:
            A tuple of (routing_map, probs), where routing_map is the multihot vector
            and probs is the multihot probabilities.
        """
        batch_size = indices.shape[0]
        multihot_routing_map = torch.zeros(
            (batch_size, self.num_local_experts),
            dtype=torch.long,
            device=indices.device,
        )

        multihot_probs = torch.zeros(
            (batch_size, self.num_local_experts),
            dtype=torch.float,
            device=indices.device,
        )

        mask = indices != -1
        valid_indices = indices[mask]
        row_indices = torch.arange(batch_size, device=indices.device).repeat_interleave(mask.sum(dim=1))
        multihot_routing_map[row_indices, valid_indices] = 1
        multihot_probs[row_indices, valid_indices] = probs[mask]
        return multihot_routing_map.bool(), multihot_probs

    def get_dispached_metadata(self) -> torch.Tensor:
        return self.dispatched_indices, self.dispatched_probs

    def get_number_of_tokens_per_expert(self) -> torch.Tensor:
        """
        Get the number of tokens per expert.
        """
        return self.tokens_per_expert

    def combine(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> torch.Tensor:
        hidden_states, _ = fused_combine(
            hidden_states,
            self.group,
            self.handle,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        # Release the handle after combine operation
        self.handle = None
        return hidden_states


class TorchA2ADispatcher:
    """A dispatcher manager using torch.distributed collectives."""

    def __init__(self, parallel: str, num_experts: int):
        self.rank = PM.rank_in(parallel)
        self.world_size = PM.size_of(parallel)
        self.group = PM.group_of(parallel)

        self.num_local_experts = num_experts // PM.size_of(parallel)
        # Restore Info
        self.comm_record: tuple[list[int], list[int], int, torch.LongTensor] = None

    def dispatch(
        self,
        hidden_states: torch.FloatTensor,  # [S, C]
        token_expert_ids: torch.IntTensor,  # [S, topk]
        token_expert_weights: torch.FloatTensor,  # [S, topk]
    ) -> torch.Tensor:
        S, K = token_expert_ids.shape
        S, C = hidden_states.shape

        token_expert_ranks = token_expert_ids // self.num_local_experts

        token_ids = torch.arange(S, device=hidden_states.device, dtype=torch.int64)

        send_hidden, recv_hidden = [], []
        send_indices, recv_indices = [], []
        send_probs, recv_probs = [], []
        send_token_ids, recv_token_ids = [], []

        send_sizes: list[int] = []

        for rank in range(self.world_size):
            this_rank = (token_expert_ranks == rank).any(dim=1)  # [S, ]
            send_tokens = this_rank.sum().item()
            ranks_r = token_expert_ranks[this_rank]

            indices_r = token_expert_ids[this_rank] - rank * self.num_local_experts
            indices_r[ranks_r != rank] = -1

            probs_r = token_expert_weights[this_rank]
            probs_r[ranks_r != rank] = 0

            send_hidden.append(hidden_states[this_rank])
            send_indices.append(indices_r)
            send_probs.append(probs_r)
            send_token_ids.append(token_ids[this_rank])
            send_sizes.append(send_tokens)

        comm_sizes: list[list[int]] = all_gather_object(send_sizes, group=self.group)
        recv_sizes: list[int] = [comm_sizes[i][self.rank] for i in range(self.world_size)]

        for rank in range(self.world_size):
            recv_tokens = recv_sizes[rank]
            recv_hidden.append(hidden_states.new_empty((recv_tokens, C)))
            recv_indices.append(token_expert_ids.new_empty((recv_tokens, K)))
            recv_probs.append(token_expert_weights.new_empty((recv_tokens, K)))
            recv_token_ids.append(token_ids.new_empty((recv_tokens,)))

        distnn.all_to_all(recv_hidden, send_hidden, group=self.group)
        distnn.all_to_all(recv_indices, send_indices, group=self.group)
        distnn.all_to_all(recv_probs, send_probs, group=self.group)
        distnn.all_to_all(recv_token_ids, send_token_ids, group=self.group)

        recv_hidden = torch.cat(recv_hidden, dim=0)
        recv_indices = torch.cat(recv_indices, dim=0)
        recv_probs = torch.cat(recv_probs, dim=0)
        recv_token_ids = torch.cat(recv_token_ids, dim=0)

        self.comm_record = (send_sizes, recv_sizes, S, recv_token_ids)

        return recv_hidden, recv_indices, recv_probs

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # reverse dispatch

        if self.comm_record is None:
            raise RuntimeError("combine called before dispatch")
        S, C = hidden_states.shape

        send_sizes, recv_sizes, restore_S, token_ids = self.comm_record

        send_hidden, recv_hidden = [], []
        send_token_ids, recv_token_ids = [], []

        splitted_states = hidden_states.split(recv_sizes)
        splitted_token_ids = token_ids.split(recv_sizes)

        for rank in range(self.world_size):
            send_hidden.append(splitted_states[rank])
            send_token_ids.append(splitted_token_ids[rank])
            recv_hidden.append(hidden_states.new_empty((send_sizes[rank], C)))
            recv_token_ids.append(token_ids.new_empty((send_sizes[rank],)))

        distnn.all_to_all(recv_hidden, send_hidden, group=self.group)
        distnn.all_to_all(recv_token_ids, send_token_ids, group=self.group)

        recv_hidden = torch.cat(recv_hidden, dim=0)
        recv_token_ids = torch.cat(recv_token_ids, dim=0)

        output = hidden_states.new_zeros((restore_S, hidden_states.size(1)))
        if recv_hidden.numel() > 0:
            output.index_add_(0, recv_token_ids.to(torch.int64), recv_hidden)
        return output
