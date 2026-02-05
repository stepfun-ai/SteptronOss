from torch import nn

from steptronoss.core.parallel_state import PM, get_vpp_rank
from steptronoss.model.common.grouped_query_attention import AttentionConfig
from steptronoss.model.common.moe_share_expert_ffn import MoEFeedForwardConfig
from steptronoss.model.common.rms_norm import RMSNorm
from steptronoss.model.decoder_model import (
    DecoderLLMConfig,
    LlamaLikeModel,
    NoopTransformerBlock,
    TransformerBlock,
)


class Step3p5ModelConfig(DecoderLLMConfig):
    swa_cfg: AttentionConfig
    ffn_cfg: MoEFeedForwardConfig
    swa_layer_list: list[bool]

    def sanity_check(self):
        super().sanity_check()
        assert len(self.swa_layer_list) == self.num_layers


class Step3p5Block(TransformerBlock):
    """Extra support:
    - layer-specific SlidingWindowAttention & FullAttention
    """

    def __init__(self, cfg: Step3p5ModelConfig, layer_id, recompute=False):
        super(TransformerBlock, self).__init__()

        self.cfg = cfg
        self.layer_id = layer_id
        self.recompute = recompute
        if self.recompute is True:
            self.recompute = ["attention", "attn_norm", "feed_forward", "ffn_norm"]
        self.distribute_saved_activations = self.cfg.tp_cfg.distribute_saved_activations
        self.sequence_parallel = cfg.tp_cfg.sequence_parallel

        # Pre-attention LayerNorm
        self.attention_norm = RMSNorm(
            cfg.hidden_size,
            eps=cfg.layernorm_epsilon,
            sequence_parallel=cfg.tp_cfg.sequence_parallel,
            use_zero_init=cfg.rms_norm_zero_gamma,
        )

        # Pre-FFN LayerNorm
        self.ffn_norm = RMSNorm(
            cfg.hidden_size,
            eps=cfg.layernorm_epsilon,
            sequence_parallel=cfg.tp_cfg.sequence_parallel,
            use_zero_init=cfg.rms_norm_zero_gamma,
        )

        if self.layer_id in cfg.swa_layer_list:
            self.attention = cfg.swa_cfg.build_model(layer_id=layer_id)
        else:
            self.attention = cfg.attn_cfg.build_model(layer_id=layer_id)

        # FFN
        self.feed_forward = cfg.ffn_cfg.build_model(layer_id=layer_id)


class Step3p5Model(LlamaLikeModel):
    def build(self, layer_map: dict[int, dict[int, dict[int, dict]]]):
        # Build model components
        self.layers = nn.ModuleList()
        pp_rank, vp_rank = PM.rank_in("PP"), get_vpp_rank()

        for layer, kwargs in layer_map[pp_rank][vp_rank].items():
            self.layers.append(Step3p5Block(self.cfg, layer_id=layer, **kwargs))

        if len(self.layers) == 0:
            self.layers.append(NoopTransformerBlock())

        if self.is_pipeline_first_stage():
            self.tok_embeddings = self.cfg.tok_embed_cfg.build_model()

        if self.is_pipeline_last_stage():
            if self.cfg.tie_embedding:
                self.out_embeddings = self.cfg.out_embed_cfg.build_model(self.tok_embeddings.word_embeddings.weight)
            else:
                self.out_embeddings = self.cfg.out_embed_cfg.build_model()
