from functools import cached_property

from steptronoss.model.decoder_model import LlamaLikeModel, NoopTransformerBlock, TransformerBlock


class QwenModel(LlamaLikeModel):
    @cached_property
    def reshaper(self):
        return self.build_reshaper()

    def build_reshaper(self):
        from steptronoss.checkpointing.reshape_ops import (
            ColumnParallel,
            Duplicate,
            FFNMergeGateUp,
            GQAMergeQKV,
            GQAMergeQKVBias,
            Inverse,
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
        if self.is_pipeline_first_stage():
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
        if self.is_pipeline_last_stage():
            if not self.cfg.tie_embedding:
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

        def generate_block_scripts(layer: TransformerBlock, prefix_src, prefix_dst):
            block_scripts = []
            # Attention: WQKV
            block_scripts.append(
                Script(
                    src=f"{prefix_src}.self_attn.[qkv]_proj.weight",
                    op=GQAMergeQKV(
                        group_num=layer.attention.num_kv_heads,
                        head_dim=layer.attention.head_dim,
                    )
                    + KeepThisTP()
                    + Rename(f"{prefix_dst}.attention.wqkv.weight: {prefix_src}.self_attn.qkv_proj.weight"),
                    dst=f"{prefix_dst}.attention.wqkv.weight",
                )
            )
            if layer.attention.wqkv.bias is not None:
                block_scripts.append(
                    Script(
                        src=f"{prefix_src}.self_attn.[qkv]_proj.bias",
                        op=GQAMergeQKVBias(
                            group_num=layer.attention.num_kv_heads,
                            head_dim=layer.attention.head_dim,
                        )
                        + KeepThisTP()
                        + Rename(f"{prefix_dst}.attention.wqkv.bias: {prefix_src}.self_attn.qkv_proj.bias"),
                        dst=f"{prefix_dst}.attention.wqkv.bias",
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

            # Feed Forward
            if hasattr(layer.feed_forward, "moe"):
                # MOE Gate
                block_scripts.append(
                    Script(
                        src=f"{prefix_src}.mlp.gate.weight",
                        op=Duplicate()
                        + KeepThisTP()
                        + Rename(f"{prefix_dst}.feed_forward.moe.gate.weight: {prefix_src}.mlp.gate.weight"),
                        dst=f"{prefix_dst}.feed_forward.moe.gate.weight",
                    )
                )

                # MOE Experts
                block_scripts.append(
                    Script(
                        src=f"{prefix_src}.mlp.experts.*.[gu]*_proj.weight",
                        op=Inverse(UnbindMoE(moe_key_prefix="experts."))
                        + KeepThisEP()
                        + FFNMergeGateUp(group="ETP")
                        + KeepThisTP(group="ETP")
                        + Rename(
                            f"{prefix_dst}.feed_forward.moe.experts.w1: {prefix_src}.mlp.experts.gate_up_proj.weight"
                        ),
                        dst=f"{prefix_dst}.feed_forward.moe.experts.w1",
                    )
                )
                scripts.append(
                    Script(
                        src=f"{prefix_src}.mlp.experts.*.down_proj.weight",
                        op=Inverse(UnbindMoE(moe_key_prefix="experts."))
                        + KeepThisEP()
                        + RowParallel(group="ETP")
                        + KeepThisTP(group="ETP")
                        + Rename(
                            f"{prefix_dst}.feed_forward.moe.experts.w2: {prefix_src}.mlp.experts.down_proj.weight"
                        ),
                        dst=f"{prefix_dst}.feed_forward.moe.experts.w2",
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

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if self.cfg.tie_embedding:
            m, u = super().load_state_dict(state_dict, strict=False, assign=assign)
            if m or u:
                if len(m) == 1 and m[0] == "out_embeddings.output.weight":
                    pass
                else:
                    raise RuntimeError(f"Missing: {m}; Unexpected: {u}")
        else:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
