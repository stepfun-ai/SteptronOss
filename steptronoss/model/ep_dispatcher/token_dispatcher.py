from __future__ import annotations

import torch
import torch.distributed.nn.functional as distnn

from steptronoss.core.parallel_state import PM
from steptronoss.utils.dist_utils import all_gather_object


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

        recv_hidden = distnn.all_to_all(recv_hidden, send_hidden, group=self.group)
        recv_indices = distnn.all_to_all(recv_indices, send_indices, group=self.group)
        recv_probs = distnn.all_to_all(recv_probs, send_probs, group=self.group)
        recv_token_ids = distnn.all_to_all(recv_token_ids, send_token_ids, group=self.group)

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

        recv_hidden = distnn.all_to_all(recv_hidden, send_hidden, group=self.group)
        recv_token_ids = distnn.all_to_all(recv_token_ids, send_token_ids, group=self.group)

        recv_hidden = torch.cat(recv_hidden, dim=0)
        recv_token_ids = torch.cat(recv_token_ids, dim=0)

        output = hidden_states.new_zeros((restore_S, hidden_states.size(1)))
        if recv_hidden.numel() > 0:
            output.index_add_(0, recv_token_ids.to(torch.int64), recv_hidden)
        return output
