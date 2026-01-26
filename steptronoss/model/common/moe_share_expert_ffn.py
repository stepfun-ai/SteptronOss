import torch
from configurize import Ref

# from steptronoss.core.tensor_parallel.layers import new_memory_multiple_paras
from steptronoss.core.parallel_state import PM
from steptronoss.model.common.feed_forward import FeedForward, FeedForwardConfig
from steptronoss.model.common.moe_block import MoEBlock, MoEConfig
from steptronoss.utils.metrics import GlobalMetrics


class MoEFeedForwardConfig(FeedForwardConfig):
    moe_cfg: MoEConfig = Ref("..moe_cfg")

    def build_model(self, layer_id: int):
        if layer_id in self.moe_cfg.moe_layer_list:
            return MoeShareExpertFFN(
                moe_cfg=self.moe_cfg,
                ffn_cfg=self,
                layer_id=layer_id,
            )
        else:
            return FeedForward(cfg=self, layer_id=layer_id)


class MoeShareExpertFFN(torch.nn.Module):
    def __init__(
        self,
        moe_cfg: MoEConfig,
        ffn_cfg: FeedForwardConfig,
        layer_id: int,
    ):
        super().__init__()
        self.moe_cfg = moe_cfg
        self.ffn_cfg = ffn_cfg

        self.enable_share_expert = moe_cfg.share_expert_dim > 0

        # swiglu_limit = None
        # if ffn_cfg.use_swiglu_limit:
        #     if isinstance(ffn_cfg.use_swiglu_limit, float):
        #         swiglu_limit = ffn_cfg.use_swiglu_limit
        #     elif isinstance(ffn_cfg.use_swiglu_limit, list):
        #         swiglu_limit = ffn_cfg.use_swiglu_limit[layer_id]

        # logger.info(f"Enable swiglu_limit : swiglu_limit={swiglu_limit}", at=0)
        # rewrite in this way to fix cfg.continuous_memory_ffn=False case

        self.moe = MoEBlock(
            cfg=moe_cfg,
            layer_id=layer_id,
        )
        if self.enable_share_expert:
            with ffn_cfg.modify(ffn_hidden_size=moe_cfg.share_expert_dim):

                # swiglu_limit_shared = None
                # if ffn_cfg.use_swiglu_limit_shared:
                #     if isinstance(ffn_cfg.use_swiglu_limit_shared, float):
                #         swiglu_limit_shared = ffn_cfg.use_swiglu_limit_shared
                #     elif isinstance(ffn_cfg.use_swiglu_limit_shared, list):
                #         swiglu_limit_shared = ffn_cfg.use_swiglu_limit_shared[layer_id]
                self.share_expert = FeedForward(
                    cfg=ffn_cfg,
                    layer_id=layer_id,
                )

    def forward(self, x, recompute: bool = False, **kwargs):
        output = self.moe.forward(x, recompute=recompute)

        GlobalMetrics.shared_routed_avgnorm.add(
            output,
            iop=lambda x: x.detach().norm(p=2, dim=-1).mean(),
            subname=f"layer{self.moe.moe_layer_id}.routed",
        )

        if self.enable_share_expert:
            share_output = self.share_expert(x, recompute=recompute)

            output = output + share_output

            GlobalMetrics.shared_routed_avgnorm.add(
                share_output,
                iop=lambda x: x.detach().norm(p=2, dim=-1).mean(),
                subname=f"layer{self.moe.moe_layer_id}.shared",
            )

        return output
