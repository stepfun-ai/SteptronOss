import torch
from configurize import Config, Ref
from torch.nn import functional as F

from steptronoss.core import tensor_parallel
from steptronoss.exp.base_exp import MegatronTPConfig


class FeedForwardConfig(Config):
    tp_cfg: MegatronTPConfig = Ref("..tp_cfg")

    hidden_size: int
    ffn_hidden_size: int

    layernorm_epsilon: float
    rms_norm_zero_gamma: bool

    swiglu_recompute_silu_out_proj: bool

    def activation(self, x, swiglu_limit=None):
        l, r = torch.chunk(x, 2, dim=-1)
        l = F.silu(l)
        if swiglu_limit is not None:
            l = l.clamp(min=None, max=swiglu_limit)
            r = r.clamp(min=-swiglu_limit, max=swiglu_limit)
        return l * r

    def build_model(self, layer_id: int):
        return FeedForward(cfg=self, layer_id=layer_id)


class FeedForward(torch.nn.Module):
    def __init__(self, cfg: FeedForwardConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id

        self.cfg = cfg

        self.distribute_saved_activations = cfg.tp_cfg.distribute_saved_activations

        self.activation = self.cfg.activation
        self.fuse_activation_w2 = cfg.swiglu_recompute_silu_out_proj

        self.w1 = tensor_parallel.ColumnParallelLinear(
            cfg.hidden_size,
            2 * cfg.ffn_hidden_size,
            bias=False,
            gather_output=False,
            async_tensor_model_parallel_allreduce=cfg.tp_cfg.async_tensor_model_parallel_allreduce,
            **self.cfg.tp_cfg.get_tp_kwargs(),
        )
        self.w2 = tensor_parallel.RowParallelLinear(
            cfg.ffn_hidden_size,
            cfg.hidden_size,
            bias=False,
            input_is_parallel=True,
            custom_pre_recompute_function=(self.activation if self.fuse_activation_w2 else None),
            **self.cfg.tp_cfg.get_tp_kwargs(),
        )

    def forward(self, x, recompute=False, **kwargs):
        if recompute:
            return tensor_parallel.checkpoint(self._forward, self.distribute_saved_activations, x)
        else:
            return self._forward(x)

    def _forward(self, x) -> torch.FloatTensor:
        if self.fuse_activation_w2:
            x = self.w1(x)[0]
            output = self.w2(x)[0]
        else:
            x = self.activation(self.w1(x)[0])
            output = self.w2(x)[0]
        return output
