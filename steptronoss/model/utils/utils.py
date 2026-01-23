"""Model utilities for SteptronOss."""

import torch
from loguru import logger


def init_weight_callback(cls, init_method="init_model_weight"):
    """this func is called after the model's __init__ method is executed"""

    def _recursive_init_model_weights(model, init_method):
        for child in model.children():
            _recursive_init_model_weights(child, init_method)
        if hasattr(model, init_method):
            getattr(model, init_method)()

    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Only execute the callback if the actual class of the instance is the class to which the decorator is applied
        if self.__class__ is cls:
            _recursive_init_model_weights(self, init_method)

    cls.__init__ = new_init
    return cls


def load_model_checkpoint(models, state_dict, strict_load_model=True):
    """Load model checkpoint."""
    from steptronoss.core.parallel_state import set_vpp_rank
    from steptronoss.utils.utils import unwrap_model

    assert "model" in state_dict, "state_dict must contain 'model' key."

    if isinstance(state_dict["model"], str):
        load_type = "safetensors"
        from steptronoss.utils.weight_loader import HFWeights

        state_dict["model"] = HFWeights(state_dict["model"])
    else:
        load_type = "pt"

    for vid, model in enumerate(models):
        model = unwrap_model(model)
        msg = None
        set_vpp_rank(vid)
        if load_type == "pt":
            msg = model.load_state_dict(state_dict["model"][vid], strict=strict_load_model)
        elif load_type == "safetensors":
            msg = model.load_hf_state_dict(state_dict["model"], strict=strict_load_model)
        else:
            raise NotImplementedError

        if msg is not None:
            if msg.missing_keys or msg.unexpected_keys:
                logger.warning(msg)
            else:
                # print "<All keys matched successfully>"
                logger.info(msg)
            # mark loaded. Error will be thrown if some parameter is not in ckpt when strict_load_model
            # is True. So we only need deal with missing_keys when strict_load_model is False.
            missing_keys_set = set(msg.missing_keys) if not strict_load_model else None
            for name, param in model.named_parameters():
                if strict_load_model or name not in missing_keys_set:
                    param.has_initialized = True


class AuxLossBackwardHook(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, aux_loss):
        # Keep aux_loss alive for backward and attach a gradient path.
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Inject a unit gradient for aux_loss while passing through grad_output.
        (aux_loss,) = ctx.saved_tensors
        scaled_aux_loss_grad = torch.ones_like(aux_loss)
        return grad_output, scaled_aux_loss_grad


def bind_aux_loss(output: torch.Tensor, aux_loss: torch.Tensor) -> torch.Tensor:
    """Bind an auxiliary loss to an output tensor for backward propagation.

    This keeps the forward value unchanged while ensuring gradients flow to
    ``aux_loss`` during backpropagation.
    """
    # Attach aux_loss to output without changing forward values; gradients will flow to aux_loss.
    return AuxLossBackwardHook.apply(output, aux_loss)
