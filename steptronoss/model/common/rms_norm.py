import torch
import torch.nn as nn


@torch.jit.script
def rms_foward(x):
    l2_norm_inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
    y = x * l2_norm_inv
    return y, l2_norm_inv


@torch.jit.script
def rms_backward(grad_y, y, l2_norm_inv):
    g = grad_y
    mean_gy = (g * y).mean(dim=-1, keepdim=True)
    gx = (g - y * mean_gy) * l2_norm_inv
    return (gx,)


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y, l2_norm_inv = rms_foward(x)
        ctx.save_for_backward(y, l2_norm_inv)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        y, l2_norm_inv = ctx.saved_variables
        gx = rms_backward(grad_y, y, l2_norm_inv)
        return gx


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,  # llama use 1e-5 by default
        sequence_parallel=False,
        use_fp32=True,
        use_zero_init=False,
    ):
        super().__init__()

        self.eps = eps
        self.dim = dim
        self.sequence_parallel = sequence_parallel
        self.use_fp32 = use_fp32
        self.use_zero_init = use_zero_init

        self.weight = nn.Parameter(torch.ones(dim)) if not use_zero_init else nn.Parameter(torch.zeros(dim))
        self.bias = 0 if not use_zero_init else 1
        self.weight.sequence_parallel = self.sequence_parallel

    def forward(self, x):
        weight = self.weight + self.bias

        if self.use_fp32:
            y = RMSNormFunction.apply(x.float()).type_as(x)
        else:
            y = RMSNormFunction.apply(x)
        return y * weight
