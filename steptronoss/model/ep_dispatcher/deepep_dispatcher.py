from __future__ import annotations

import torch

from steptronoss.core.parallel_state import PM

try:
    import deep_ep
    from deep_ep import Buffer

    HAVE_DEEP_EP = True
except Exception:
    deep_ep = None
    Buffer = None
    HAVE_DEEP_EP = False


def _get_hidden_bytes(x: torch.Tensor) -> int:
    # DeepEP uses at least bf16 (2 bytes) for tokens
    return x.size(1) * max(x.element_size(), 2)


class DeepEPDispatcher:
    """Dispatcher using DeepEP fused EP communication kernels."""

    def __init__(
        self,
        parallel: str,
        num_experts: int,
        num_sms: int | None = None,
        num_nvl_bytes: int = 0,
        num_rdma_bytes: int = 0,
        low_latency_mode: bool = False,
        allow_nvlink_for_low_latency_mode: bool = True,
        allow_mnnvl: bool = False,
        use_fabric: bool = False,
        explicitly_destroy: bool = False,
        enable_shrink: bool = False,
    ) -> None:
        if not HAVE_DEEP_EP:
            raise ImportError(
                "DeepEP is not installed. Build/install it from /data/DeepEP before using DeepEPDispatcher."
            )

        self.rank = PM.rank_in(parallel)
        self.world_size = PM.size_of(parallel)
        self.group = PM.group_of(parallel)

        self.num_experts = num_experts
        self.num_local_experts = num_experts // self.world_size

        if num_sms is not None:
            Buffer.set_num_sms(num_sms)

        self._buffer: Buffer | None = None
        self._buffer_nvl_bytes = num_nvl_bytes
        self._buffer_rdma_bytes = num_rdma_bytes
        self._low_latency_mode = low_latency_mode
        self._allow_nvlink_for_low_latency_mode = allow_nvlink_for_low_latency_mode
        self._allow_mnnvl = allow_mnnvl
        self._use_fabric = use_fabric
        self._explicitly_destroy = explicitly_destroy
        self._enable_shrink = enable_shrink

        self._handle = None
        self._cached_handle = None
        self._cached_recv_token_indices = None
        self._cached_recv_token_probs = None
        self._cached_token_sig = None

    def _should_query_rdma_size_hint(self) -> bool:
        # DeepEP intranode dispatch (world_size <= 8) uses only NVLink buffers
        # unless low-latency mode is enabled. Querying the RDMA hint there is
        # both unnecessary and can trip DeepEP builds compiled without NVSHMEM.
        return self._low_latency_mode or self.world_size > 8

    def _ensure_buffer(self, hidden_bytes: int) -> Buffer:
        # Use DeepEP size hints to grow buffers if needed.
        num_nvl_bytes = self._buffer_nvl_bytes
        use_rdma_hint = self._should_query_rdma_size_hint()
        num_rdma_bytes = self._buffer_rdma_bytes if use_rdma_hint else 0
        for config in (
            Buffer.get_dispatch_config(self.world_size),
            Buffer.get_combine_config(self.world_size),
        ):
            num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, self.world_size), num_nvl_bytes)
            if use_rdma_hint:
                num_rdma_bytes = max(
                    config.get_rdma_buffer_size_hint(hidden_bytes, self.world_size),
                    num_rdma_bytes,
                )

        if (
            self._buffer is None
            or self._buffer.group != self.group
            or self._buffer.num_nvl_bytes < num_nvl_bytes
            or self._buffer.num_rdma_bytes < num_rdma_bytes
        ):
            self._buffer = Buffer(
                self.group,
                num_nvl_bytes=num_nvl_bytes,
                num_rdma_bytes=num_rdma_bytes,
                low_latency_mode=self._low_latency_mode,
                allow_nvlink_for_low_latency_mode=self._allow_nvlink_for_low_latency_mode,
                allow_mnnvl=self._allow_mnnvl,
                use_fabric=self._use_fabric,
                explicitly_destroy=self._explicitly_destroy,
                enable_shrink=self._enable_shrink,
            )
            self._buffer_nvl_bytes = num_nvl_bytes
            self._buffer_rdma_bytes = num_rdma_bytes

        return self._buffer

    def dispatch(
        self,
        hidden_states: torch.Tensor,  # [S, C]
        token_expert_ids: torch.Tensor,  # [S, topk]
        token_expert_weights: torch.Tensor,  # [S, topk]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.numel() == 0:
            self._handle = None
            return hidden_states, token_expert_ids, token_expert_weights

        recv_x, recv_token_indices, recv_token_probs = _DeepEPDispatchFn.apply(
            self, hidden_states, token_expert_ids, token_expert_weights
        )
        return recv_x, recv_token_indices, recv_token_probs

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._handle is None:
            raise RuntimeError("combine called before dispatch")

        return _DeepEPCombineFn.apply(self, hidden_states)


class _DeepEPDispatchFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        dispatcher: DeepEPDispatcher,
        hidden_states: torch.Tensor,
        token_expert_ids: torch.Tensor,
        token_expert_weights: torch.Tensor,
    ):
        hidden_states = hidden_states.contiguous()
        token_expert_ids = token_expert_ids.contiguous()
        token_expert_weights = token_expert_weights.contiguous()

        buffer = dispatcher._ensure_buffer(_get_hidden_bytes(hidden_states))

        if token_expert_ids.dtype != deep_ep.topk_idx_t:
            token_expert_ids = token_expert_ids.to(dtype=deep_ep.topk_idx_t)
        if token_expert_weights.dtype != torch.float32:
            token_expert_weights = token_expert_weights.float()

        token_sig = (token_expert_ids.shape, token_expert_ids.dtype, token_expert_ids.device)
        if torch.is_grad_enabled() and dispatcher._cached_handle is not None:
            if dispatcher._cached_token_sig != token_sig:
                dispatcher._cached_handle = None
                dispatcher._cached_recv_token_indices = None
                dispatcher._cached_recv_token_probs = None
                dispatcher._cached_token_sig = None
            else:
                recv_x, _, _, _, _, _event = buffer.dispatch(hidden_states, handle=dispatcher._cached_handle)
                dispatcher._handle = dispatcher._cached_handle
                recv_token_indices = dispatcher._cached_recv_token_indices
                recv_token_probs = dispatcher._cached_recv_token_probs
                dispatcher._cached_handle = None
                dispatcher._cached_recv_token_indices = None
                dispatcher._cached_recv_token_probs = None
                dispatcher._cached_token_sig = None

                ctx.buffer = buffer
                ctx.handle = dispatcher._handle
                if recv_token_indices is not None:
                    ctx.mark_non_differentiable(recv_token_indices)
                return recv_x, recv_token_indices, recv_token_probs

        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = buffer.get_dispatch_layout(token_expert_ids, dispatcher.num_experts)

        (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            _num_recv_tokens_per_expert_list,
            handle,
            _event,
        ) = buffer.dispatch(
            hidden_states,
            topk_idx=token_expert_ids,
            topk_weights=token_expert_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=event,
        )

        dispatcher._handle = handle
        if not torch.is_grad_enabled():
            dispatcher._cached_handle = handle
            dispatcher._cached_recv_token_indices = recv_token_indices
            dispatcher._cached_recv_token_probs = recv_token_probs
            dispatcher._cached_token_sig = token_sig
        ctx.buffer = buffer
        ctx.handle = handle
        ctx.mark_non_differentiable(recv_token_indices)
        return recv_x, recv_token_indices, recv_token_probs

    @staticmethod
    def backward(ctx, grad_recv_x: torch.Tensor, grad_recv_token_indices, grad_recv_token_probs):
        buffer = ctx.buffer
        handle = ctx.handle

        if grad_recv_x is None:
            return None, None, None, None

        if grad_recv_x.dtype != torch.bfloat16:
            grad_recv_x = grad_recv_x.to(dtype=torch.bfloat16)
        grad_recv_x = grad_recv_x.contiguous()

        if grad_recv_token_probs is not None:
            if grad_recv_token_probs.dtype != torch.float32:
                grad_recv_token_probs = grad_recv_token_probs.float()
            grad_recv_token_probs = grad_recv_token_probs.contiguous()

        grad_hidden, grad_token_probs, _event = buffer.combine(
            grad_recv_x,
            handle=handle,
            topk_weights=grad_recv_token_probs,
        )
        return None, grad_hidden, None, grad_token_probs


class _DeepEPCombineFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dispatcher: DeepEPDispatcher, hidden_states: torch.Tensor):
        if dispatcher._handle is None:
            raise RuntimeError("combine called before dispatch")

        hidden_states = hidden_states.contiguous()
        buffer = dispatcher._ensure_buffer(_get_hidden_bytes(hidden_states))
        combined_x, _, _event = buffer.combine(hidden_states, handle=dispatcher._handle)
        ctx.buffer = buffer
        ctx.handle = dispatcher._handle
        dispatcher._handle = None
        return combined_x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        buffer = ctx.buffer
        handle = ctx.handle

        if grad_output is None:
            return None, None

        if grad_output.dtype != torch.bfloat16:
            grad_output = grad_output.to(dtype=torch.bfloat16)
        grad_output = grad_output.contiguous()

        grad_hidden, _, _, _, _, _event = buffer.dispatch(grad_output, handle=handle)
        return None, grad_hidden
