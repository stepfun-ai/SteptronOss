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

        if cfg.swa_layer_list[self.layer_id]:
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

    @property
    def reshaper(self):
        return self.build_reshaper()

    def build_reshaper(self):
        from steptronoss.checkpointing.reshape_ops import (
            ColumnParallel,
            Duplicate,
            FFNMergeGateUp,
            GQAMergeQKVG,
            KeepThisEP,
            KeepThisTP,
            OnlineReshaper,
            Rename,
            RowParallel,
            Script,
            UnbindMoE,
            VocabPad,
        )

        scripts = []
        if hasattr(self, "tok_embeddings"):
            scripts.append(
                Script(
                    src="model.embed_tokens.weight",
                    op=VocabPad(
                        target_vocab_size=self.cfg.tok_embed_cfg.vocab_size,
                        dim=0,
                        pad_type="last",
                    )
                    + ColumnParallel()
                    + KeepThisTP()
                    + Rename("tok_embeddings.word_embeddings.weight: model.embed_tokens.weight"),
                    dst="tok_embeddings.word_embeddings.weight",
                )
            )

        if hasattr(self, "out_embeddings"):
            scripts.append(
                Script(
                    src="lm_head.weight",
                    op=VocabPad(
                        target_vocab_size=self.cfg.out_embed_cfg.vocab_size,
                        dim=0,
                        pad_type="last",
                    )
                    + ColumnParallel()
                    + KeepThisTP()
                    + Rename("out_embeddings.output.weight: lm_head.weight"),
                    dst="out_embeddings.output.weight",
                )
            )
            scripts.append(
                Script(
                    src="model.norm.weight",
                    op=Duplicate() + KeepThisTP() + Rename("out_embeddings.norm.weight: model.norm.weight"),
                    dst="out_embeddings.norm.weight",
                )
            )

        def generate_block_scripts(layer: Step3p5Block, prefix_src, prefix_dst):
            block_scripts = []

            target_block = layer

            # Attention: WQKV
            block_scripts.append(
                Script(
                    src=f"{prefix_src}.self_attn.[qkvg]_proj.weight",
                    op=GQAMergeQKVG(
                        group_num=target_block.attention.num_kv_heads,
                        head_dim=target_block.attention.head_dim,
                        gate_dims=target_block.attention.wqkv_extra_dims,
                    )
                    + KeepThisTP()
                    + Rename(f"{prefix_dst}.attention.wqkv.weight: {prefix_src}.self_attn.qkv_proj.weight"),
                    dst=f"{prefix_dst}.attention.wqkv.weight",
                )
            )

            # Attention: WO
            block_scripts.append(
                Script(
                    src=f"{prefix_src}.self_attn.o_proj.weight",
                    op=RowParallel()
                    + KeepThisTP()
                    + Rename(f"{prefix_dst}.attention.wo.weight: {prefix_src}.self_attn.o_proj.weight"),
                    dst=f"{prefix_dst}.attention.wo.weight",
                )
            )

            # Norm: Attention
            block_scripts.append(
                Script(
                    src=f"{prefix_src}.input_layernorm.weight",
                    op=Duplicate()
                    + KeepThisTP()
                    + Rename(f"{prefix_dst}.attention_norm.weight: {prefix_src}.input_layernorm.weight"),
                    dst=f"{prefix_dst}.attention_norm.weight",
                )
            )

            # Norm: FFN
            block_scripts.append(
                Script(
                    src=f"{prefix_src}.post_attention_layernorm.weight",
                    op=Duplicate()
                    + KeepThisTP()
                    + Rename(f"{prefix_dst}.ffn_norm.weight: {prefix_src}.post_attention_layernorm.weight"),
                    dst=f"{prefix_dst}.ffn_norm.weight",
                )
            )

            # Q/K Norms
            for norm_type in ["q_norm", "k_norm"]:
                block_scripts.append(
                    Script(
                        src=f"{prefix_src}.self_attn.{norm_type}.weight",
                        op=Duplicate()
                        + KeepThisTP()
                        + Rename(
                            f"{prefix_dst}.attention.{norm_type}.weight: {prefix_src}.self_attn.{norm_type}.weight"
                        ),
                        dst=f"{prefix_dst}.attention.{norm_type}.weight",
                    )
                )

            # Feed Forward (Includes EP/MoE support)
            if hasattr(target_block.feed_forward, "moe"):
                # MOE Gate
                block_scripts.append(
                    Script(
                        src=f"{prefix_src}.moe.gate.weight",
                        op=Duplicate()
                        + KeepThisTP()
                        + Rename(f"{prefix_dst}.feed_forward.moe.gate.weight: {prefix_src}.moe.gate.weight"),
                        dst=f"{prefix_dst}.feed_forward.moe.gate.weight",
                    )
                )
                # Router Bias
                block_scripts.append(
                    Script(
                        src=f"{prefix_src}.moe.router_bias",
                        op=Duplicate()
                        + KeepThisTP()
                        + Rename(f"{prefix_dst}.feed_forward.moe.router_balance_bias: {prefix_src}.moe.router_bias"),
                        dst=f"{prefix_dst}.feed_forward.moe.router_balance_bias",
                    )
                )
                # MOE Experts (UnbindMoE for EP)
                block_scripts.append(
                    Script(
                        src=f"{prefix_src}.moe.[gu]*_proj.weight",
                        op=FFNMergeGateUp(group="ETP")
                        + KeepThisTP(group="ETP")
                        + KeepThisEP()
                        + Rename(f"{prefix_dst}.feed_forward.moe.experts.w1: {prefix_src}.moe.gate_up_proj.weight"),
                        dst=f"{prefix_dst}.feed_forward.moe.experts.w1",
                    )
                )
                block_scripts.append(
                    Script(
                        src=f"{prefix_src}.moe.down_proj.weight",
                        op=RowParallel(group="ETP")
                        + KeepThisTP(group="ETP")
                        + KeepThisEP()
                        + Rename(f"{prefix_dst}.feed_forward.moe.experts.w2: {prefix_src}.moe.down_proj.weight"),
                        dst=f"{prefix_dst}.feed_forward.moe.experts.w2",
                    )
                )
                # Shared Expert
                block_scripts.append(
                    Script(
                        src=f"{prefix_src}.share_expert.[gu]*_proj.weight",
                        op=FFNMergeGateUp()
                        + KeepThisTP()
                        + Rename(
                            f"{prefix_dst}.feed_forward.share_expert.w1.weight: {prefix_src}.share_expert.gate_up_proj.weight"
                        ),
                        dst=f"{prefix_dst}.feed_forward.share_expert.w1.weight",
                    )
                )
                block_scripts.append(
                    Script(
                        src=f"{prefix_src}.share_expert.down_proj.weight",
                        op=RowParallel()
                        + KeepThisTP()
                        + Rename(
                            f"{prefix_dst}.feed_forward.share_expert.w2.weight: {prefix_src}.share_expert.down_proj.weight"
                        ),
                        dst=f"{prefix_dst}.feed_forward.share_expert.w2.weight",
                    )
                )
            else:
                # Standard FFN
                block_scripts.append(
                    Script(
                        src=f"{prefix_src}.mlp.[gu]*_proj.weight",
                        op=FFNMergeGateUp()
                        + KeepThisTP()
                        + Rename(f"{prefix_dst}.feed_forward.w1.weight: {prefix_src}.mlp.gate_up_proj.weight"),
                        dst=f"{prefix_dst}.feed_forward.w1.weight",
                    )
                )
                block_scripts.append(
                    Script(
                        src=f"{prefix_src}.mlp.down_proj.weight",
                        op=RowParallel()
                        + KeepThisTP()
                        + Rename(f"{prefix_dst}.feed_forward.w2.weight: {prefix_src}.mlp.down_proj.weight"),
                        dst=f"{prefix_dst}.feed_forward.w2.weight",
                    )
                )
            return block_scripts

        # Main Layers
        for local_id, layer in enumerate(self.layers):
            if isinstance(layer, NoopTransformerBlock):
                continue
            scripts.extend(
                generate_block_scripts(
                    layer,
                    prefix_src=f"model.layers.{layer.layer_id}",
                    prefix_dst=f"layers.{local_id}",
                )
            )

        return OnlineReshaper(scripts)
