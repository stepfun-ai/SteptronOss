# Copyright (c) 2024, Stepfun Inc. All rights reserved.

from collections.abc import Callable, Sequence

import torch
from torch import distributed as dist

from steptronoss.core import parallel_state as mpu
from steptronoss.core.parallel_state import PM
from steptronoss.exp.base_exp import MegatronPPModelConfig
from steptronoss.timers import get_timers


def _communicate_shapes(
    tensor_send_next: torch.Tensor | None,
    tensor_send_prev: torch.Tensor | None,
    recv_prev: bool,
    recv_next: bool,
) -> tuple[list[int], list[int]]:
    """Communicate tensor shapes between stages. Used to communicate
    tensor shapes before the actual tensor communication happens.
    This is required when the sequence lengths across micro batches
    are not uniform.

    Takes the following arguments:
        tensor_send_next: tensor to send to next rank (no tensor sent if
                          set to None).
        tensor_send_prev: tensor to send to prev rank (no tensor sent if
                          set to None).
        recv_prev: boolean for whether tensor should be received from
                   previous rank.
        recv_next: boolean for whether tensor should be received from
                   next rank.
    Returns:
        (recv_prev_shape, recv_next_shape)
    """

    recv_prev_shape_tensor = None
    recv_next_shape_tensor = None
    send_prev_shape_tensor = None
    send_next_shape_tensor = None
    if recv_prev:
        recv_prev_shape_tensor = torch.empty((3), device=torch.cuda.current_device(), dtype=torch.int64)
    if recv_next:
        recv_next_shape_tensor = torch.empty((3), device=torch.cuda.current_device(), dtype=torch.int64)
    if tensor_send_prev is not None:
        send_prev_shape_tensor = torch.tensor(
            tensor_send_prev.size(),
            device=torch.cuda.current_device(),
            dtype=torch.int64,
        )
    if tensor_send_next is not None:
        send_next_shape_tensor = torch.tensor(
            tensor_send_next.size(),
            device=torch.cuda.current_device(),
            dtype=torch.int64,
        )

    ops = []
    if send_prev_shape_tensor is not None:
        send_prev_op = dist.P2POp(
            dist.isend,
            send_prev_shape_tensor,
            mpu.get_pipeline_model_parallel_prev_rank(),
        )
        ops.append(send_prev_op)
    if recv_prev_shape_tensor is not None:
        recv_prev_op = dist.P2POp(
            dist.irecv,
            recv_prev_shape_tensor,
            mpu.get_pipeline_model_parallel_prev_rank(),
        )
        ops.append(recv_prev_op)
    if send_next_shape_tensor is not None:
        send_next_op = dist.P2POp(
            dist.isend,
            send_next_shape_tensor,
            mpu.get_pipeline_model_parallel_next_rank(),
        )
        ops.append(send_next_op)
    if recv_next_shape_tensor is not None:
        recv_next_op = dist.P2POp(
            dist.irecv,
            recv_next_shape_tensor,
            mpu.get_pipeline_model_parallel_next_rank(),
        )
        ops.append(recv_next_op)
    if len(ops) > 0:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    # To protect against race condition when using batch_isend_irecv().
    # should take this out once the bug with batch_isend_irecv is resolved.
    torch.cuda.synchronize()

    recv_prev_shape = [0, 0, 0]
    if recv_prev_shape_tensor is not None:
        recv_prev_shape = recv_prev_shape_tensor.tolist()

    recv_next_shape = [0, 0, 0]
    if recv_next_shape_tensor is not None:
        recv_next_shape = recv_next_shape_tensor.tolist()

    return recv_prev_shape, recv_next_shape


def _batched_p2p_ops(
    *,
    tensor_send_prev: torch.Tensor | None,
    tensor_recv_prev: torch.Tensor | None,
    tensor_send_next: torch.Tensor | None,
    tensor_recv_next: torch.Tensor | None,
    group: dist.ProcessGroup,
) -> list[dist.Work]:
    ops = []
    if tensor_send_prev is not None:
        send_prev_op = dist.P2POp(
            dist.isend,
            tensor_send_prev,
            mpu.get_pipeline_model_parallel_prev_rank(),
            group,
        )
        ops.append(send_prev_op)
    if tensor_recv_prev is not None:
        recv_prev_op = dist.P2POp(
            dist.irecv,
            tensor_recv_prev,
            mpu.get_pipeline_model_parallel_prev_rank(),
            group,
        )
        ops.append(recv_prev_op)
    if tensor_send_next is not None:
        send_next_op = dist.P2POp(
            dist.isend,
            tensor_send_next,
            mpu.get_pipeline_model_parallel_next_rank(),
            group,
        )
        ops.append(send_next_op)
    if tensor_recv_next is not None:
        recv_next_op = dist.P2POp(
            dist.irecv,
            tensor_recv_next,
            mpu.get_pipeline_model_parallel_next_rank(),
            group,
        )
        ops.append(recv_next_op)
    if len(ops) > 0:
        reqs = dist.batch_isend_irecv(ops)
    else:
        reqs = []
    return reqs


def _p2p_ops(
    *,
    tensor_send_prev: torch.Tensor | None,
    tensor_recv_prev: torch.Tensor | None,
    tensor_send_next: torch.Tensor | None,
    tensor_recv_next: torch.Tensor | None,
    group: dist.ProcessGroup,
) -> list[dist.Work]:
    reqs = []
    _rank = mpu.get_pipeline_model_parallel_rank()
    if mpu.get_pipeline_model_parallel_rank() % 2 == 0:
        if tensor_send_next is not None:
            send_next_req = dist.isend(
                tensor=tensor_send_next,
                dst=mpu.get_pipeline_model_parallel_next_rank(),
                group=group,
            )
            reqs.append(send_next_req)

        if tensor_recv_prev is not None:
            recv_prev_req = dist.irecv(
                tensor=tensor_recv_prev,
                src=mpu.get_pipeline_model_parallel_prev_rank(),
                group=group,
            )
            reqs.append(recv_prev_req)

        if tensor_send_prev is not None:
            send_prev_req = dist.isend(
                tensor=tensor_send_prev,
                dst=mpu.get_pipeline_model_parallel_prev_rank(),
                group=group,
            )
            reqs.append(send_prev_req)

        if tensor_recv_next is not None:
            recv_next_req = dist.irecv(
                tensor=tensor_recv_next,
                src=mpu.get_pipeline_model_parallel_next_rank(),
                group=group,
            )
            reqs.append(recv_next_req)

    else:
        if tensor_recv_prev is not None:
            recv_prev_req = dist.irecv(
                tensor=tensor_recv_prev,
                src=mpu.get_pipeline_model_parallel_prev_rank(),
                group=group,
            )
            reqs.append(recv_prev_req)

        if tensor_send_next is not None:
            send_next_req = dist.isend(
                tensor=tensor_send_next,
                dst=mpu.get_pipeline_model_parallel_next_rank(),
                group=group,
            )
            reqs.append(send_next_req)

        if tensor_recv_next is not None:
            recv_next_req = dist.irecv(
                tensor=tensor_recv_next,
                src=mpu.get_pipeline_model_parallel_next_rank(),
                group=group,
            )
            reqs.append(recv_next_req)

        if tensor_send_prev is not None:
            send_prev_req = dist.isend(
                tensor=tensor_send_prev,
                dst=mpu.get_pipeline_model_parallel_prev_rank(),
                group=group,
            )
            reqs.append(send_prev_req)
    return reqs


def _communicate(
    tensor_send_next: torch.Tensor | None,
    tensor_send_prev: torch.Tensor | None,
    recv_prev: bool,
    recv_next: bool,
    tensor_shape: Sequence[int] | None,
    cfg: MegatronPPModelConfig,
    dtype_: torch.dtype | None = None,
    wait_on_reqs: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None, list[dist.Work] | None]:
    """Communicate tensors between stages. Used as helper method in other
    communication methods that are used in steptron/schedules.py.

    Takes the following arguments:
        tensor_send_next: tensor to send to next rank (no tensor sent if
                          set to None).
        tensor_send_prev: tensor to send to prev rank (no tensor sent if
                          set to None).
        recv_prev: boolean for whether tensor should be received from
                   previous rank.
        recv_next: boolean for whether tensor should be received from
                   next rank.
        tensor_shape: shape of tensor to receive (this method assumes that all
                      tensors sent and received in a single function call are
                      the same shape).
        dtype_: optional, this is used when the tensor that needs to be
                communicated is different from args.params_dtype.
    Returns:
        (tensor_recv_prev, tensor_recv_next)
    """

    # Create placeholder tensors for receive in forward and backward directions
    # if needed.
    tensor_recv_prev = None
    tensor_recv_next = None

    if not cfg.variable_seq_lengths:
        if tensor_shape is None:
            recv_prev_shape = cfg.pp_comm_shape
            recv_next_shape = cfg.pp_comm_shape
        else:
            recv_prev_shape = tensor_shape
            recv_next_shape = tensor_shape
    else:
        recv_prev_shape, recv_next_shape = _communicate_shapes(tensor_send_next, tensor_send_prev, recv_prev, recv_next)
    recv_prev_chunk_shape = recv_prev_shape
    recv_next_chunk_shape = recv_next_shape

    dtype = torch.float32 if cfg.fp32_residual_connection else cfg.params_dtype

    requires_grad = True
    if dtype_ is not None:
        dtype = dtype_
        requires_grad = False

    if recv_prev:
        tensor_recv_prev = torch.empty(
            recv_prev_chunk_shape,
            requires_grad=requires_grad,
            device=torch.cuda.current_device(),
            dtype=dtype,
        )
    if recv_next:
        tensor_recv_next = torch.empty(
            recv_next_chunk_shape,
            requires_grad=requires_grad,
            device=torch.cuda.current_device(),
            dtype=dtype,
        )

    # Send tensors in both the forward and backward directions as appropriate.
    if cfg.overlap_p2p_comm:
        p2p_func = _p2p_ops
    else:
        p2p_func = _batched_p2p_ops
    reqs = p2p_func(
        tensor_send_prev=tensor_send_prev,
        tensor_recv_prev=tensor_recv_prev,
        tensor_send_next=tensor_send_next,
        tensor_recv_next=tensor_recv_next,
        group=PM.group_of("PP"),
    )
    if wait_on_reqs and len(reqs) > 0:
        if len(reqs) > 0:
            for req in reqs:
                req.wait()
            reqs = None
        # To protect against race condition when using batch_isend_irecv().
        torch.cuda.synchronize()

    return tensor_recv_prev, tensor_recv_next, reqs


def recv_forward(
    cfg: MegatronPPModelConfig,
    tensor_shape: Sequence[int] | None = None,
    dtype_: torch.dtype | None = None,
) -> torch.Tensor | None:
    """Receive tensor from previous rank in pipeline (forward receive)."""
    timers = get_timers()

    if mpu.is_pipeline_first_stage():
        input_tensor = None
    else:
        with timers.record("forward-recv", log_level=2):
            input_tensor, _, _ = _communicate(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=True,
                recv_next=False,
                tensor_shape=tensor_shape,
                cfg=cfg,
                dtype_=dtype_,
            )
    return input_tensor


def recv_backward(cfg: MegatronPPModelConfig, tensor_shape: Sequence[int] | None = None) -> torch.Tensor | None:
    """Receive tensor from next rank in pipeline (backward receive)."""
    timers = get_timers()

    if mpu.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        with timers.record("backward-recv", log_level=2):
            _, output_tensor_grad, _ = _communicate(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=True,
                cfg=cfg,
                tensor_shape=tensor_shape,
            )
    return output_tensor_grad


def send_forward(
    cfg: MegatronPPModelConfig,
    output_tensor: torch.Tensor,
    tensor_shape: Sequence[int] | None = None,
    dtype_: torch.dtype | None = None,
) -> None:
    """Send tensor to next rank in pipeline (forward send)."""
    timers = get_timers()

    if not mpu.is_pipeline_last_stage():
        with timers.record("forward-send", log_level=2):
            _communicate(
                tensor_send_next=output_tensor,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=False,
                tensor_shape=tensor_shape,
                cfg=cfg,
                dtype_=dtype_,
            )


def send_backward(
    cfg: MegatronPPModelConfig,
    input_tensor_grad: torch.Tensor,
    tensor_shape: Sequence[int] | None = None,
) -> None:
    """Send tensor to previous rank in pipeline (backward send)."""
    timers = get_timers()

    if not mpu.is_pipeline_first_stage():
        with timers.record("backward-send", log_level=2):
            _communicate(
                tensor_send_next=None,
                tensor_send_prev=input_tensor_grad,
                recv_prev=False,
                recv_next=False,
                cfg=cfg,
                tensor_shape=tensor_shape,
            )


def send_forward_recv_backward(
    cfg: MegatronPPModelConfig,
    output_tensor: torch.Tensor,
    tensor_shape: Sequence[int] | None = None,
) -> torch.Tensor | None:
    """Batched send and recv with next rank in pipeline."""
    timers = get_timers()

    if mpu.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        with timers.record("forward-send-backward-recv", log_level=2):
            _, output_tensor_grad, _ = _communicate(
                tensor_send_next=output_tensor,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=True,
                cfg=cfg,
                tensor_shape=tensor_shape,
            )
    return output_tensor_grad


def send_backward_recv_forward(
    cfg: MegatronPPModelConfig,
    input_tensor_grad: torch.Tensor,
    tensor_shape: Sequence[int] | None = None,
) -> torch.Tensor | None:
    """Batched send and recv with previous rank in pipeline."""
    timers = get_timers()

    if mpu.is_pipeline_first_stage():
        input_tensor = None
    else:
        with timers.record("backward-send-forward-recv", log_level=2):
            input_tensor, _, _ = _communicate(
                tensor_send_next=None,
                tensor_send_prev=input_tensor_grad,
                recv_prev=True,
                recv_next=False,
                cfg=cfg,
                tensor_shape=tensor_shape,
            )
    return input_tensor


def send_forward_recv_forward(
    cfg: MegatronPPModelConfig,
    output_tensor: torch.Tensor,
    recv_prev: bool,
    tensor_shape: Sequence[int] | None = None,
    overlap_p2p_comm: bool = False,
) -> torch.Tensor | None | tuple[torch.Tensor | None, Callable[[], None]]:
    """Batched recv from previous rank and send to next rank in pipeline."""
    timers = get_timers()

    timers("forward-send-forward-recv", log_level=2).start()
    input_tensor, _, wait_handles = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=recv_prev,
        recv_next=False,
        cfg=cfg,
        wait_on_reqs=not (overlap_p2p_comm),
        tensor_shape=tensor_shape,
    )
    if overlap_p2p_comm:
        # sync timers when handlers are done
        def waiter():
            for req in wait_handles:
                req.wait()
            timers("forward-send-forward-recv").stop()

        return input_tensor, waiter
    else:
        timers("forward-send-forward-recv").stop()
        return input_tensor


def send_backward_recv_backward(
    cfg: MegatronPPModelConfig,
    input_tensor_grad: torch.Tensor,
    recv_next: bool,
    tensor_shape: Sequence[int] | None = None,
    overlap_p2p_comm: bool = False,
) -> torch.Tensor | None | tuple[torch.Tensor | None, Callable[[], None]]:
    """Batched recv from next rank and send to previous rank in pipeline."""
    timers = get_timers()

    if timers is not None:
        timers("backward-send-backward-recv", log_level=2).start()
    _, output_tensor_grad, wait_handles = _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=False,
        recv_next=recv_next,
        cfg=cfg,
        wait_on_reqs=not (overlap_p2p_comm),
        tensor_shape=tensor_shape,
    )
    if overlap_p2p_comm:
        # sync timers when handlers are done
        def waiter():
            for req in wait_handles:
                req.wait()
            timers("backward-send-backward-recv").stop()

        return output_tensor_grad, waiter
    else:
        if timers is not None:
            timers("backward-send-backward-recv").stop()
        return output_tensor_grad


def send_forward_backward_recv_forward_backward(
    cfg: MegatronPPModelConfig,
    output_tensor: torch.Tensor,
    input_tensor_grad: torch.Tensor,
    recv_prev: bool,
    recv_next: bool,
    tensor_shape: Sequence[int] | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Batched send and recv with previous and next ranks in pipeline."""
    timers = get_timers()

    with timers.record("forward-backward-send-forward-backward-recv", log_level=2):
        input_tensor, output_tensor_grad, _ = _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=input_tensor_grad,
            recv_prev=recv_prev,
            recv_next=recv_next,
            cfg=cfg,
            tensor_shape=tensor_shape,
        )
    return input_tensor, output_tensor_grad
