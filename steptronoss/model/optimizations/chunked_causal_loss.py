"""Exact causal LM loss with a bounded logits allocation in forward AND backward.

The frozen LM head is not a LoRA target. Recompute its projection one token
chunk at a time instead of retaining a sequence-by-vocabulary autograd graph.
This torch implementation is a portable reference, not a fused Triton kernel.
"""

from types import MethodType

import torch
import torch.nn.functional as F


class _ChunkedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, chunk_size):
        if weight.requires_grad:
            raise ValueError("Chunked causal loss currently requires a frozen LM head")
        count = targets.numel()
        if count == 0:
            raise ValueError("Batch has no supervised next-token targets")
        ctx.save_for_backward(hidden, weight, targets)
        ctx.count = count
        ctx.chunk_size = chunk_size
        loss = torch.zeros((), dtype=torch.float32, device=hidden.device)
        for start in range(0, hidden.shape[0], chunk_size):
            target = targets[start : start + chunk_size]
            x = hidden[start : start + chunk_size]
            logits = F.linear(x.to(weight.dtype), weight).float()
            loss.add_(F.cross_entropy(logits, target, reduction="sum"))
        return loss / count

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        hidden, weight, targets = ctx.saved_tensors
        grad_hidden = torch.zeros_like(hidden)
        for start in range(0, hidden.shape[0], ctx.chunk_size):
            target = targets[start : start + ctx.chunk_size]
            with torch.enable_grad():
                x = hidden[start : start + ctx.chunk_size].detach().requires_grad_(True)
                logits = F.linear(x.to(weight.dtype), weight).float()
                loss = F.cross_entropy(logits, target, reduction="sum") / ctx.count
                (grad,) = torch.autograd.grad(loss, x, grad_outputs=grad_output)
            grad_hidden[start : start + ctx.chunk_size] = grad
        return grad_hidden, None, None, None


def chunked_causal_loss(hidden, weight, labels, chunk_size=128):
    """Labels are unshifted; masked prompt tokens still pass through the decoder."""
    if chunk_size <= 0:
        raise ValueError("loss_chunk_size must be positive")
    if hidden.shape[:2] != labels.shape:
        raise ValueError("Hidden states and labels must have the same batch/sequence shape")
    targets = labels[:, 1:].reshape(-1)
    supervised = targets != -100
    # Gather once, rather than synchronizing a CUDA boolean mask per chunk.
    # Autograd scatters the resulting gradients back to the full decoder output.
    selected_hidden = hidden[:, :-1].reshape(-1, hidden.shape[-1])[supervised]
    return _ChunkedLinearCE.apply(selected_hidden, weight, targets[supervised], chunk_size)


def cp_chunked_causal_loss(hidden, weight, labels, chunk_size=128):
    from steptronoss.model.optimizations.context_parallel_qwen38 import cp_info, split_contiguous

    size, rank, group = cp_info()
    # Shift BEFORE slicing: the final hidden of a block predicts the first token
    # of the following block. Only the final token of the complete sample is ignored.
    shifted = F.pad(labels[:, 1:], (0, 1), value=-100)
    targets = split_contiguous(shifted, size, rank).reshape(-1)
    selected = targets != -100
    denominator = (shifted != -100).sum()
    if not int(denominator):
        raise ValueError("Batch has no supervised next-token targets")
    count = int(selected.sum())
    if count:
        mean = _ChunkedLinearCE.apply(
            hidden.reshape(-1, hidden.shape[-1])[selected], weight, targets[selected], chunk_size
        )
        return mean * (size * count / denominator)
    # Retain every decoder/communication backward even on empty supervision ranks.
    return hidden.float().sum() * 0.0


def install_chunked_causal_loss(model, chunk_size):
    """Keep the model's normal forward entrypoint so FSDP root hooks execute."""
    original_forward = model.forward

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if labels is None:
            return original_forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        from transformers.modeling_outputs import CausalLMOutputWithPast

        # SFT-only loss path: callers cannot request a subset of logits or a
        # separate denominator; the native scheduler scales the scalar loss.
        for key in ("logits_to_keep", "num_items_in_batch", "return_dict", "use_cache"):
            kwargs.pop(key, None)
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True, **kwargs
        )
        from steptronoss.model.optimizations.context_parallel_qwen38 import cp_info

        loss_fn = cp_chunked_causal_loss if cp_info()[0] > 1 else chunked_causal_loss
        loss = loss_fn(outputs.last_hidden_state, self.lm_head.weight, labels, chunk_size)
        return CausalLMOutputWithPast(loss=loss)

    model.forward = MethodType(forward, model)
