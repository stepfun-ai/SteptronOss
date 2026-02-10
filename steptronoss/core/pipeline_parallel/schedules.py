# Copyright (c) 2026, STEPFUN CORPORATION. All rights reserved.

from collections.abc import Callable, Iterator

import torch
from torch.autograd.variable import Variable

from steptronoss.core.parallel_state import (
    PM,
    get_vpp_rank,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
    set_vpp_rank,
)
from steptronoss.core.pipeline_parallel import p2p_communication as p2p_comm
from steptronoss.exp.base_exp import (
    MegatronPPModelConfig,
)
from steptronoss.model.module import MegatronModule
from steptronoss.timers import get_timers, timeit
from steptronoss.utils import check_nan, unwrap_model


def deallocate_output_tensor(out):
    """Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.
    """
    if out is None:
        return
    assert isinstance(out, torch.Tensor), f"expected Tensor, found {type(out).__name__}."
    assert out._base is None, "counter-productive to free a view of another tensor."
    out.data = torch.empty(
        (1,),
        device=out.device,
        dtype=out.dtype,
    )


def custom_backward(output, grad_output):
    """Directly call C++ autograd engine.

    To make the 'deallocate_output_tensor' (above) optimization work, the C++
    autograd engine must be called directly, bypassing Pytorch's
    torch.autograd.backward. Pytorch's 'backward' checks that the output and
    grad have the same shape, while C++'s 'backward' does not.
    """

    assert output.numel() == 1, "output should be pseudo-'freed' in schedule, to optimize memory"
    assert isinstance(output, torch.Tensor), f"output == '{type(output).__name__}'."
    assert isinstance(grad_output, (torch.Tensor, type(None))), f"grad_output == '{type(grad_output).__name__}'."

    # Handle scalar output
    if grad_output is None:
        assert output.numel() == 1, "implicit grad requires scalar output."
        grad_output = torch.ones_like(
            output,
            memory_format=torch.preserve_format,
        )

    # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
    Variable._execution_engine.run_backward(
        tensors=(output,),
        grad_tensors=(grad_output,),
        keep_graph=False,
        create_graph=False,
        inputs=tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )


def backward_step(input_tensor, output_tensor, output_tensor_grad):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    get_timers()("backward-step", log_level=2).start()
    custom_backward(output_tensor[0], output_tensor_grad[0])
    get_timers()("backward-step").stop()
    # logger.info(f"backward-step  {get_mem_brief()}")

    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    # Handle single skip connection if it exists (encoder_hidden_state in
    # model with encoder and decoder).
    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    return input_tensor_grad


class FWBWScheduler:
    def __init__(self, config: MegatronPPModelConfig) -> None:
        self.config = config

        # cache用于每个iteration预取的数据，结构：list[vp_rank][micro_batch_id] -> data
        self._prefetched_data: list[list] | None = None

        self._collected_outputs = []

    def configure(
        self,
        models: list[MegatronModule],
        data_iterators: list[Iterator],
        data_sync_fn=Callable[[Iterator], dict],
        data_proc_fn=Callable[[dict], dict],
        collect_output=False,
        # if training
        loss_fn: Callable | None = None,
        training=True,
    ):
        self.models = models
        self.data_iterators = data_iterators
        self.data_sync_fn = data_sync_fn
        self.data_proc_fn = data_proc_fn
        self.collect_output = collect_output

        self.loss_func = loss_fn

        self.training = training

    @timeit("batch-generator", level=2)
    def _prefetch_iteration_data(self, forward_num: int):
        """在每个iteration开始时预取本轮需要的全部数据，避免在forward_chunk里逐个取。"""
        # 记录当前vp rank，预取完再恢复，避免影响后续逻辑

        orig_vp_rank = get_vpp_rank()
        self._prefetched_data = [[] for _ in self.models]
        for vp_rank, data_iter in enumerate(self.data_iterators):
            set_vpp_rank(vp_rank)
            for _i in range(forward_num):
                # data_iter 可能为 None（非数据源rank），sync_get_data 会自行处理
                self._prefetched_data[vp_rank].append(self.data_sync_fn(data_iter))
                # logger.info(f"prefetch micro_batch {i} for vpp {vp_rank}")

        if orig_vp_rank is not None:
            set_vpp_rank(orig_vp_rank)

    def _clear_prefetched_data(self):
        self._prefetched_data = None

    def forward_chunk(self, vp_rank=0, input_tensor=None, loss_scale=1.0):
        """Forward step for passed-in model.

        If first stage, input tensor is obtained from data_iterator, otherwise
        passed-in input_tensor is used.

        Loss Scale already consider the bz warmup

        Returns output tensor."""

        set_vpp_rank(vp_rank)
        model = self.models[vp_rank]
        unwrap_model(model)._set_input_tensor(input_tensor)

        data = self._prefetched_data[vp_rank].pop(0)

        data = self.data_proc_fn(data)
        # ugly hack to get the grad_accumulation_steps in model
        data["loss_scale"] = loss_scale
        data["mtp_loss_scale"] = loss_scale

        with get_timers().record("forward-step", log_level=2):
            output = model(**data)

        # logger.info(f"forward-step [{vp_rank}] {get_mem_brief()}")

        if self.config.check_nan:
            check_nan(output, input_tensor)

        if is_pipeline_last_stage():
            if self.collect_output:
                self._collected_outputs.append(output)
            if self.loss_func is not None:
                loss = self.loss_func(data, output) * loss_scale
                return loss
        else:
            assert isinstance(output, torch.Tensor)
            return output

    def run(self, forward_num=1):
        self._collected_outputs.clear()
        assert len(self.models) == 1

        self._prefetch_iteration_data(forward_num)

        input_tensor, output_tensor_grad = None, None
        for _i in range(forward_num):
            output_tensor = self.forward_chunk(0, loss_scale=1 / forward_num)
            if self.training:
                backward_step(input_tensor, output_tensor, output_tensor_grad)

        self._clear_prefetched_data()
        return self._collected_outputs


class PPScheduler(FWBWScheduler):
    def run(self, forward_num=1):
        self._collected_outputs.clear()
        self._prefetch_iteration_data(forward_num)
        if self.training:
            self.input_tensors = []
            self.output_tensors = []

        num_warmup_microbatches = PM.size_of("PP") - PM.rank_in("PP") - 1
        num_warmup_microbatches = min(num_warmup_microbatches, forward_num)
        num_microbatches_remaining = forward_num - num_warmup_microbatches

        # Run warmup forward passes.
        for _i in range(num_warmup_microbatches):
            input_tensor = p2p_comm.recv_forward(self.config)
            output_tensor = self.forward_chunk(input_tensor=input_tensor, loss_scale=1 / forward_num)

            p2p_comm.send_forward(self.config, output_tensor)

            if self.training:
                self.input_tensors.append(input_tensor)
                self.output_tensors.append(output_tensor)
                deallocate_output_tensor(output_tensor)

        # Before running 1F1B, need to receive first forward tensor.
        # If all microbatches are run in warmup / cooldown phase, then no need to
        # receive this tensor here.
        if num_microbatches_remaining > 0:
            input_tensor = p2p_comm.recv_forward(self.config)

        # Run 1F1B in steady state.
        for i in range(num_microbatches_remaining):
            last_iteration = i == (num_microbatches_remaining - 1)

            output_tensor = self.forward_chunk(input_tensor=input_tensor, loss_scale=1 / forward_num)
            if not self.training:
                p2p_comm.send_forward(self.config, output_tensor)

                if not last_iteration:
                    input_tensor = p2p_comm.recv_forward(self.config)

            else:
                output_tensor_grad = p2p_comm.send_forward_recv_backward(self.config, output_tensor)

                # Add input_tensor and output_tensor to end of list.
                self.input_tensors.append(input_tensor)
                self.output_tensors.append(output_tensor)
                deallocate_output_tensor(output_tensor)

                # Pop input_tensor and output_tensor from the start of the list for
                # the backward pass.
                input_tensor = self.input_tensors.pop(0)
                output_tensor = self.output_tensors.pop(0)

                input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad)

                if last_iteration:
                    input_tensor = None
                    p2p_comm.send_backward(self.config, input_tensor_grad)
                else:
                    input_tensor = p2p_comm.send_backward_recv_forward(self.config, input_tensor_grad)

        # Run cooldown backward passes.
        if self.training:
            for _i in range(num_warmup_microbatches):
                input_tensor = self.input_tensors.pop(0)
                output_tensor = self.output_tensors.pop(0)

                output_tensor_grad = p2p_comm.recv_backward(self.config)

                input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad)

                p2p_comm.send_backward(self.config, input_tensor_grad)

        self._clear_prefetched_data()
        return self._collected_outputs


class VPPScheduler(PPScheduler):
    def run(self, forward_num=1):
        """Run interleaved 1F1B schedule (model split into model chunks), with
        communication between pipeline stages as needed.

        Returns dictionary with losses if the last stage, empty dict otherwise."""
        self._collected_outputs.clear()
        assert forward_num % PM.size_of("PP") == 0, (
            f"forward_num ({forward_num}) is not divisible by pp_size ({PM.size_of('PP')})"
        )

        self._prefetch_iteration_data(forward_num)
        input_tensors = [list() for i in self.models]
        output_tensors = [list() for i in self.models]

        if self.training:
            output_tensor_grads = [list() for i in self.models]
        tensor_shape = self.config.pp_comm_shape

        # Compute number of warmup and remaining microbatches.
        num_model_chunks = len(self.models)
        num_microbatches = forward_num * num_model_chunks
        all_warmup_microbatches = False
        if not self.training:
            num_warmup_microbatches = num_microbatches
        else:
            # Run all forward passes and then all backward passes if number of
            # microbatches is just the number of pipeline stages.
            # Otherwise, perform (num_model_chunks-1)*pipeline_parallel_size on
            # all workers, followed by more microbatches after depending on
            # stage ID (more forward passes for earlier stages, later stages can
            # immediately start with 1F1B).
            if forward_num == PM.size_of("PP"):
                num_warmup_microbatches = num_microbatches
                all_warmup_microbatches = True
            else:
                num_warmup_microbatches = (PM.size_of("PP") - PM.rank_in("PP") - 1) * 2
                num_warmup_microbatches += (num_model_chunks - 1) * PM.size_of("PP")
                num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
        num_microbatches_remaining = num_microbatches - num_warmup_microbatches

        def get_model_chunk_id(microbatch_id, forward):
            """Helper method to get the model chunk ID given the iteration number."""
            microbatch_id_in_group = microbatch_id % (PM.size_of("PP") * num_model_chunks)
            model_chunk_id = microbatch_id_in_group // PM.size_of("PP")
            if not forward:
                model_chunk_id = num_model_chunks - model_chunk_id - 1
            return model_chunk_id

        def forward_step_helper(microbatch_id):
            """Helper method to run forward step with model split into chunks
            (run set_virtual_pipeline_model_parallel_rank() before calling
            forward_step())."""
            model_chunk_id = get_model_chunk_id(microbatch_id, forward=True)
            set_vpp_rank(model_chunk_id)

            # forward step
            if is_pipeline_first_stage():
                if len(input_tensors[model_chunk_id]) == len(output_tensors[model_chunk_id]):
                    input_tensors[model_chunk_id].append(None)

            input_tensor = input_tensors[model_chunk_id][-1]
            output_tensor = self.forward_chunk(model_chunk_id, input_tensor, loss_scale=1 / forward_num)
            output_tensors[model_chunk_id].append(output_tensor)

            # if forward-only, no need to save tensors for a backward pass
            if not self.training:
                input_tensors[model_chunk_id].pop()
                output_tensors[model_chunk_id].pop()

            return output_tensor

        def backward_step_helper(microbatch_id):
            """Helper method to run backward step with model split into chunks
            (run set_virtual_pipeline_model_parallel_rank() before calling
            backward_step())."""
            model_chunk_id = get_model_chunk_id(microbatch_id, forward=False)
            set_vpp_rank(model_chunk_id)

            if is_pipeline_last_stage():
                if len(output_tensor_grads[model_chunk_id]) == 0:
                    output_tensor_grads[model_chunk_id].append(None)
            input_tensor = input_tensors[model_chunk_id].pop(0)
            output_tensor = output_tensors[model_chunk_id].pop(0)
            output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)
            input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad)

            return input_tensor_grad

        # Run warmup forward passes.
        set_vpp_rank(0)
        input_tensors[0].append(p2p_comm.recv_forward(self.config, tensor_shape))

        fwd_waiter = lambda: None
        bwd_waiter = lambda: None
        timers = get_timers()

        for k in range(num_warmup_microbatches):
            fwd_waiter()

            output_tensor = forward_step_helper(k)

            # Determine if tensor should be received from previous stage.
            next_forward_model_chunk_id = get_model_chunk_id(k + 1, forward=True)
            recv_prev = True
            if is_pipeline_first_stage(ignore_virtual=True):
                if next_forward_model_chunk_id == 0:
                    recv_prev = False
            if k == (num_microbatches - 1):
                recv_prev = False

            # Don't send tensor downstream if on last stage.
            if is_pipeline_last_stage():
                output_tensor = None

            # Send and receive tensors as appropriate (send tensors computed
            # in this iteration; receive tensors for next iteration).
            if not self.config.overlap_p2p_comm:
                if k == (num_warmup_microbatches - 1) and self.training and not all_warmup_microbatches:
                    input_tensor_grad = None
                    recv_next = True
                    if is_pipeline_last_stage(ignore_virtual=True):
                        recv_next = False

                    input_tensor, output_tensor_grad = p2p_comm.send_forward_backward_recv_forward_backward(
                        self.config,
                        output_tensor,
                        input_tensor_grad,
                        recv_prev=recv_prev,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                    )
                    output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
                else:
                    input_tensor = p2p_comm.send_forward_recv_forward(
                        self.config,
                        output_tensor,
                        recv_prev=recv_prev,
                        tensor_shape=tensor_shape,
                    )
                input_tensors[next_forward_model_chunk_id].append(input_tensor)
            else:
                input_tensor, fwd_waiter = p2p_comm.send_forward_recv_forward(
                    self.config,
                    output_tensor,
                    recv_prev=recv_prev,
                    tensor_shape=tensor_shape,
                    overlap_p2p_comm=True,
                )

                if k == (num_warmup_microbatches - 1) and self.training and not all_warmup_microbatches:
                    input_tensor_grad = None
                    recv_next = True
                    if is_pipeline_last_stage(ignore_virtual=True):
                        recv_next = False

                    output_tensor_grad, bwd_waiter = p2p_comm.send_backward_recv_backward(
                        self.config,
                        input_tensor_grad,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )

                    output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
                input_tensors[next_forward_model_chunk_id].append(input_tensor)

            deallocate_output_tensor(output_tensor)

        # Run 1F1B in steady state.
        for k in range(num_microbatches_remaining):
            # Forward pass.
            fwd_waiter()
            forward_k = k + num_warmup_microbatches
            if self.config.overlap_p2p_comm:
                # sync to reduce memory footprint due to p2p communication
                stream = torch.cuda.current_stream()
                stream.synchronize()

                deallocate_output_tensor(output_tensor)

                output_tensor = forward_step_helper(forward_k)

                # Determine if current stage has anything to send in either direction,
                # otherwise set tensor to None.
                forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
                set_vpp_rank(forward_model_chunk_id)

                # Last virtual stage no activation tensor to send
                if is_pipeline_last_stage():
                    output_tensor = None

                # Determine if peers are sending, and where in data structure to put
                # received tensors.
                recv_prev = True
                if is_pipeline_first_stage(ignore_virtual=True):
                    # First stage is ahead of last stage by (pipeline_parallel_size - 1).
                    next_forward_model_chunk_id = get_model_chunk_id(forward_k - (PM.size_of("PP") - 1), forward=True)
                    if next_forward_model_chunk_id == (num_model_chunks - 1):
                        recv_prev = False
                    next_forward_model_chunk_id += 1
                else:
                    next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1, forward=True)

                # If last iteration, don't receive; we already received one extra
                # before the start of the for loop.
                if k == (num_microbatches_remaining - 1):
                    recv_prev = False

                # Send activation tensor to the next stage and receive activation tensor from the
                # previous stage
                input_tensor, fwd_waiter = p2p_comm.send_forward_recv_forward(
                    self.config,
                    output_tensor,
                    recv_prev=recv_prev,
                    tensor_shape=tensor_shape,
                    overlap_p2p_comm=True,
                )
                # assert fwd_wait_handles is not None

                bwd_waiter()

                # Backward pass.
                backward_k = k
                input_tensor_grad = backward_step_helper(backward_k)

                backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
                set_vpp_rank(backward_model_chunk_id)

                # First virtual stage no activation gradient tensor to send
                if is_pipeline_first_stage():
                    input_tensor_grad = None

                # Determine if the current virtual stage has an activation gradient tensor to receive
                recv_next = True
                if is_pipeline_last_stage(ignore_virtual=True):
                    # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
                    next_backward_model_chunk_id = get_model_chunk_id(
                        backward_k - (PM.size_of("PP") - 1), forward=False
                    )
                    if next_backward_model_chunk_id == 0:
                        recv_next = False
                    next_backward_model_chunk_id -= 1
                else:
                    next_backward_model_chunk_id = get_model_chunk_id(backward_k + 1, forward=False)

                output_tensor_grad, bwd_waiter = p2p_comm.send_backward_recv_backward(
                    self.config,
                    input_tensor_grad,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    overlap_p2p_comm=True,
                )

            else:  # no p2p overlap
                output_tensor = forward_step_helper(forward_k)

                # Backward pass.
                backward_k = k
                input_tensor_grad = backward_step_helper(backward_k)

                # Send output_tensor and input_tensor_grad, receive input_tensor
                # and output_tensor_grad.

                # Determine if current stage has anything to send in either direction,
                # otherwise set tensor to None.
                forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
                set_vpp_rank(forward_model_chunk_id)
                if is_pipeline_last_stage():
                    output_tensor = None

                backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
                set_vpp_rank(backward_model_chunk_id)
                if is_pipeline_first_stage():
                    input_tensor_grad = None

                # Determine if peers are sending, and where in data structure to put
                # received tensors.
                recv_prev = True
                if is_pipeline_first_stage(ignore_virtual=True):
                    # First stage is ahead of last stage by (pipeline_parallel_size - 1).
                    next_forward_model_chunk_id = get_model_chunk_id(forward_k - (PM.size_of("PP") - 1), forward=True)
                    if next_forward_model_chunk_id == (num_model_chunks - 1):
                        recv_prev = False
                    next_forward_model_chunk_id += 1
                else:
                    next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1, forward=True)

                recv_next = True
                if is_pipeline_last_stage(ignore_virtual=True):
                    # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
                    next_backward_model_chunk_id = get_model_chunk_id(
                        backward_k - (PM.size_of("PP") - 1), forward=False
                    )
                    if next_backward_model_chunk_id == 0:
                        recv_next = False
                    next_backward_model_chunk_id -= 1
                else:
                    next_backward_model_chunk_id = get_model_chunk_id(backward_k + 1, forward=False)

                # If last iteration, don't receive; we already received one extra
                # before the start of the for loop.
                if k == (num_microbatches_remaining - 1):
                    recv_prev = False

                # Communicate tensors.
                input_tensor, output_tensor_grad = p2p_comm.send_forward_backward_recv_forward_backward(
                    self.config,
                    output_tensor,
                    input_tensor_grad,
                    recv_prev=recv_prev,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                )
                deallocate_output_tensor(output_tensor)

            # Put input_tensor and output_tensor_grad in data structures in the
            # right location.
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(input_tensor)
            if recv_next:
                output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)
        deallocate_output_tensor(output_tensor)

        bwd_waiter()
        fwd_waiter()
        # Run cooldown backward passes (flush out pipeline).
        if self.training:
            if all_warmup_microbatches:
                output_tensor_grads[num_model_chunks - 1].append(p2p_comm.recv_backward(self.config, tensor_shape))
            for k in range(num_microbatches_remaining, num_microbatches):
                input_tensor_grad = backward_step_helper(k)

                next_backward_model_chunk_id = get_model_chunk_id(k + 1, forward=False)
                recv_next = True
                if is_pipeline_last_stage(ignore_virtual=True):
                    if next_backward_model_chunk_id == (num_model_chunks - 1):
                        recv_next = False
                if k == (num_microbatches - 1):
                    recv_next = False

                if self.config.overlap_p2p_comm:
                    # We use asynchronous interface and then sync immediately. This way we keep
                    # the same pp behavior as before, while avoid synchronizing the entire device.
                    output_tensor_grad, bwd_waiter = p2p_comm.send_backward_recv_backward(
                        self.config,
                        input_tensor_grad,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                    output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)
                    bwd_waiter()
                else:
                    output_tensor_grads[next_backward_model_chunk_id].append(
                        p2p_comm.send_backward_recv_backward(
                            self.config,
                            input_tensor_grad,
                            recv_next=recv_next,
                            tensor_shape=tensor_shape,
                        )
                    )

        if self.config.overlap_p2p_comm:
            torch.cuda.synchronize()
            timers("backward-send-backward-recv").summarize_event_time()
            timers("forward-send-forward-recv").summarize_event_time()

        if self.training:
            torch.cuda.empty_cache()

        self._clear_prefetched_data()
        return self._collected_outputs
