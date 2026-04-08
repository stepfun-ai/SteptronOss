from __future__ import annotations

import math
from functools import cached_property

import torch
import torch.nn.functional as F

from steptronoss.core import tensor_parallel
from steptronoss.core.parallel_state import PM
from steptronoss.model.common.rms_norm import RMSNorm
from steptronoss.model.common.rope import YARNRoPE, ropen_linspace
from steptronoss.model.decoder_model import LlamaLikeModel, NoopTransformerBlock, TransformerBlock


class Glm5YARNRoPE(YARNRoPE):
    """GLM-5 rotary embedding with reference-style apply semantics.

    Keep the cache in fp32 for stability, but apply RoPE in the input dtype to
    match the GLM/HF runtime behavior more closely.
    """

    def _get_frequencies(self, device="cpu"):
        S, C = self.max_position_embeddings, self.dim
        theta = self.theta
        ntk_ratio = self.ntk_interp_ratio

        Cv = C // 2
        base_freqs = 1 / (theta ** ropen_linspace(0, 1, Cv, device=device, dtype=torch.float32))
        # Reproduce the GLM bf16 inv_freq quantization path before cache generation.
        base_freqs = base_freqs.to(torch.bfloat16).to(torch.float32)
        if ntk_ratio == 1:
            return base_freqs

        ntk_freqs = base_freqs / ntk_ratio
        ntk_weight = (
            (S / (2 * math.pi / base_freqs) - self.yarn_beta_slow) / (self.yarn_beta_fast - self.yarn_beta_slow)
        ).clamp(0, 1)
        return base_freqs * ntk_weight + ntk_freqs * (1 - ntk_weight)

    def forward(
        self,
        feature: torch.Tensor,
        position_id: torch.IntTensor | None = None,
    ):
        cache_device = feature.device
        if position_id is not None:
            max_seqlen = self.max_position_embeddings
            if max_seqlen is None:
                max_seqlen = int(position_id.detach().amax().cpu()) + 1
            cos_cache, sin_cache = self._check_set_cos_sin_cache(max_seqlen, cache_device)
            cos = cos_cache[None, position_id, None, :].to(feature.dtype)
            sin = sin_cache[None, position_id, None, :].to(feature.dtype)
        else:
            max_seqlen = feature.shape[1] * PM.size_of("CP")
            cos_cache, sin_cache = self._check_set_cos_sin_cache(max_seqlen, cache_device)
            cos = cos_cache[None, :max_seqlen, None, :].to(feature.dtype)
            sin = sin_cache[None, :max_seqlen, None, :].to(feature.dtype)

        if self.use_adjacent_pair:
            b, s, h, c = feature.shape
            feature = feature.view(b, s, h, c // 2, 2).transpose(4, 3).reshape(b, s, h, c)

        return feature * cos + self.rotate_half(feature) * sin


class Glm5Indexer(torch.nn.Module):
    """Training-time DSA indexer used by GLM-5 attention.

    This first pass keeps the indexer duplicated across TP ranks so the rest of
    the MLA path can continue using existing TP linears without extra gathers.
    """

    def __init__(self, cfg, layer_id=None):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id

        self.index_n_heads = cfg.index_n_heads
        self.index_head_dim = cfg.index_head_dim
        self.q_lora_rank = cfg.q_lora_rank
        self.qk_rope_head_dim = cfg.qk_rope_head_dim
        self.softmax_scale = self.index_head_dim**-0.5

        self.wq_b = torch.nn.Linear(
            self.q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            bias=False,
            dtype=cfg.tp_cfg.params_dtype,
        )
        self.wk = torch.nn.Linear(
            cfg.hidden_size,
            self.index_head_dim,
            bias=False,
            dtype=cfg.tp_cfg.params_dtype,
        )
        self.k_norm = torch.nn.LayerNorm(
            self.index_head_dim,
            eps=cfg.index_layernorm_epsilon,
            elementwise_affine=True,
        )
        self.weights_proj = torch.nn.Linear(
            cfg.hidden_size,
            self.index_n_heads,
            bias=False,
            dtype=cfg.tp_cfg.params_dtype,
        )
        self.rope = Glm5YARNRoPE(
            dim=cfg.qk_rope_head_dim,
            theta=cfg.rope_theta,
            yarn_beta_fast=cfg.yarn_beta_fast,
            yarn_beta_slow=cfg.yarn_beta_slow,
            ntk_interp_ratio=cfg.ntk_interp_ratio,
            max_position_embeddings=cfg.max_position_embeddings,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_id: torch.IntTensor | None,
    ) -> torch.LongTensor:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.wq_b(q_resid)
        q = q.view(batch_size, seq_len, self.index_n_heads, self.index_head_dim)
        q_pe, q_nope = torch.split(q, [self.qk_rope_head_dim, self.index_head_dim - self.qk_rope_head_dim], dim=-1)
        q_pe = self.rope(q_pe, position_id=position_id)
        q = torch.cat([q_pe, q_nope], dim=-1)

        k = self.k_norm(self.wk(hidden_states))
        k_pe, k_nope = torch.split(k, [self.qk_rope_head_dim, self.index_head_dim - self.qk_rope_head_dim], dim=-1)
        k_pe = self.rope(k_pe.unsqueeze(2), position_id=position_id).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)

        weights = self.weights_proj(hidden_states).float() * (self.index_n_heads**-0.5)
        scores = torch.einsum("bshd,btd->bsht", q.float(), k.float()) * self.softmax_scale
        index_scores = torch.einsum("bsht,bsh->bst", scores, weights)
        if attention_mask is not None:
            index_scores = index_scores + attention_mask

        topk = min(self.cfg.index_topk, index_scores.shape[-1])
        return index_scores.topk(topk, dim=-1).indices


class Glm5Attention(torch.nn.Module):
    """GLM-5 MLA attention with a simple training-side DSA path.

    The model family uses Dynamic Sparse Attention at long context. To minimize
    the amount of new infrastructure for the initial SFT integration, this
    implementation keeps the official top-k indexer but applies the sparse mask
    through SDPA instead of a dedicated sparse kernel.
    """

    def __init__(self, cfg, layer_id=None):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id

        self.tp_size = PM.size_of("TP")
        self.sequence_parallel = cfg.tp_cfg.sequence_parallel

        self.num_heads = cfg.num_attention_heads
        self.num_local_heads = self.num_heads // self.tp_size
        if self.num_heads % self.tp_size != 0:
            raise ValueError(f"num_attention_heads={self.num_heads} must divide TP={self.tp_size}")

        self.q_lora_rank = cfg.q_lora_rank
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_rope_head_dim = cfg.qk_rope_head_dim
        self.qk_nope_head_dim = cfg.qk_nope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = cfg.v_head_dim
        if self.qk_head_dim != self.v_head_dim:
            raise NotImplementedError(
                "Initial GLM-5 SFT support expects qk_head_dim == v_head_dim so it can reuse SDPA directly."
            )

        self.q_a_proj = torch.nn.Linear(
            cfg.hidden_size,
            self.q_lora_rank,
            bias=cfg.use_qkv_bias,
            dtype=cfg.tp_cfg.params_dtype,
        )
        mla_layernorm_epsilon = getattr(cfg, "mla_layernorm_epsilon", cfg.layernorm_epsilon)
        self.q_a_layernorm = RMSNorm(
            self.q_lora_rank,
            eps=mla_layernorm_epsilon,
            sequence_parallel=False,
            use_fp32=True,
            use_zero_init=False,
        )
        self.q_b_proj = tensor_parallel.ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            gather_output=False,
            async_tensor_model_parallel_allreduce=cfg.tp_cfg.async_tensor_model_parallel_allreduce,
            **cfg.tp_cfg.get_tp_kwargs(),
        )

        self.kv_a_proj_with_mqa = torch.nn.Linear(
            cfg.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=cfg.use_qkv_bias,
            dtype=cfg.tp_cfg.params_dtype,
        )
        self.kv_a_layernorm = RMSNorm(
            self.kv_lora_rank,
            eps=mla_layernorm_epsilon,
            sequence_parallel=False,
            use_fp32=True,
            use_zero_init=False,
        )
        self.kv_b_proj = tensor_parallel.ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            gather_output=False,
            async_tensor_model_parallel_allreduce=cfg.tp_cfg.async_tensor_model_parallel_allreduce,
            **cfg.tp_cfg.get_tp_kwargs(),
        )
        self.o_proj = tensor_parallel.RowParallelLinear(
            self.num_heads * self.v_head_dim,
            cfg.hidden_size,
            bias=False,
            input_is_parallel=True,
            **cfg.tp_cfg.get_tp_kwargs(),
        )

        self.attention_dropout = cfg.attention_dropout
        self.rope = Glm5YARNRoPE(
            dim=cfg.qk_rope_head_dim,
            theta=cfg.rope_theta,
            yarn_beta_fast=cfg.yarn_beta_fast,
            yarn_beta_slow=cfg.yarn_beta_slow,
            ntk_interp_ratio=cfg.ntk_interp_ratio,
            max_position_embeddings=cfg.max_position_embeddings,
        )
        self.indexer = Glm5Indexer(cfg=cfg, layer_id=layer_id)

    @staticmethod
    def _build_final_attention_mask(
        batch_size: int,
        seq_len: int,
        cu_seqlens: torch.IntTensor | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        if cu_seqlens is None:
            neg_inf = float("-inf")
            causal = torch.triu(
                torch.full((seq_len, seq_len), neg_inf, device=device, dtype=torch.float32),
                diagonal=1,
            )
            return causal.unsqueeze(0).expand(batch_size, -1, -1)

        if batch_size != 1:
            raise NotImplementedError("Packed GLM-5 attention currently expects batch_size == 1")

        neg_inf = float("-inf")
        mask = torch.full((1, seq_len, seq_len), neg_inf, device=device, dtype=torch.float32)
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
            length = end - start
            local = torch.triu(
                torch.full((length, length), neg_inf, device=device, dtype=torch.float32),
                diagonal=1,
            )
            mask[:, start:end, start:end] = local
        return mask

    @staticmethod
    def _build_indexer_mask(
        batch_size: int,
        seq_len: int,
        cu_seqlens: torch.IntTensor | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        return Glm5Attention._build_final_attention_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            cu_seqlens=cu_seqlens,
            device=device,
        )

    @staticmethod
    def _default_position_id(seq_len: int, device: torch.device) -> torch.IntTensor:
        return torch.arange(seq_len, device=device, dtype=torch.int32)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seq_len: torch.Tensor | None = None,
        position_id: torch.Tensor | None = None,
        **kwargs,
    ):
        del max_seq_len, kwargs

        seq_len, batch_size, _ = x.shape
        x_bs = x.transpose(0, 1).contiguous()
        if position_id is None:
            position_id = self._default_position_id(seq_len, x.device)
        else:
            position_id = position_id.to(x.device)

        q_resid = self.q_a_layernorm(self.q_a_proj(x_bs))
        query_states, _ = self.q_b_proj(q_resid)
        query_states = query_states.view(batch_size, seq_len, self.num_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = self.rope(q_pe, position_id=position_id)
        query_states = torch.cat([q_nope, q_pe], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x_bs)
        k_compressed, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_compressed = self.kv_a_layernorm(k_compressed)
        kv_expanded, _ = self.kv_b_proj(k_compressed)
        kv_expanded = kv_expanded.view(
            batch_size,
            seq_len,
            self.num_local_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k_nope, value_states = torch.split(kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe = self.rope(k_pe.unsqueeze(2), position_id=position_id).expand(-1, -1, self.num_local_heads, -1)
        key_states = torch.cat([k_nope, k_pe], dim=-1)

        final_mask = self._build_final_attention_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            cu_seqlens=cu_seqlens,
            device=x.device,
        )
        indexer_mask = self._build_indexer_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            cu_seqlens=cu_seqlens,
            device=x.device,
        )
        topk_indices = self.indexer(
            hidden_states=x_bs,
            q_resid=q_resid,
            attention_mask=indexer_mask,
            position_id=position_id,
        )

        attn_mask = torch.full(
            (batch_size, seq_len, key_states.shape[1]),
            float("-inf"),
            device=x.device,
            dtype=query_states.dtype,
        )
        attn_mask.scatter_(-1, topk_indices, 0.0)
        if final_mask is not None:
            attn_mask = attn_mask + final_mask.to(attn_mask.dtype)
        attn_mask = attn_mask.unsqueeze(1)
        attn_output = F.scaled_dot_product_attention(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output, _ = self.o_proj(attn_output)
        return attn_output.transpose(0, 1).contiguous()


class Glm5Model(LlamaLikeModel):
    @cached_property
    def reshaper(self):
        return self.build_reshaper()

    def build_reshaper(self):
        from steptronoss.checkpointing.reshape_ops import (
            ColumnParallel,
            Duplicate,
            FFNMergeGateUp,
            KeepThisEP,
            KeepThisTP,
            OnlineReshaper,
            Rename,
            RowParallel,
            Script,
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
            block_scripts = [
                Script(
                    src=f"{prefix_src}.self_attn.q_a_proj.weight",
                    op=Duplicate()
                    + KeepThisTP()
                    + Rename(f"{prefix_dst}.attention.q_a_proj.weight: {prefix_src}.self_attn.q_a_proj.weight"),
                    dst=f"{prefix_dst}.attention.q_a_proj.weight",
                ),
                Script(
                    src=f"{prefix_src}.self_attn.q_a_layernorm.weight",
                    op=Duplicate()
                    + KeepThisTP()
                    + Rename(
                        f"{prefix_dst}.attention.q_a_layernorm.weight: {prefix_src}.self_attn.q_a_layernorm.weight"
                    ),
                    dst=f"{prefix_dst}.attention.q_a_layernorm.weight",
                ),
                Script(
                    src=f"{prefix_src}.self_attn.q_b_proj.weight",
                    op=ColumnParallel()
                    + KeepThisTP()
                    + Rename(f"{prefix_dst}.attention.q_b_proj.weight: {prefix_src}.self_attn.q_b_proj.weight"),
                    dst=f"{prefix_dst}.attention.q_b_proj.weight",
                ),
                Script(
                    src=f"{prefix_src}.self_attn.kv_a_proj_with_mqa.weight",
                    op=Duplicate()
                    + KeepThisTP()
                    + Rename(
                        f"{prefix_dst}.attention.kv_a_proj_with_mqa.weight: {prefix_src}.self_attn.kv_a_proj_with_mqa.weight"
                    ),
                    dst=f"{prefix_dst}.attention.kv_a_proj_with_mqa.weight",
                ),
                Script(
                    src=f"{prefix_src}.self_attn.kv_a_layernorm.weight",
                    op=Duplicate()
                    + KeepThisTP()
                    + Rename(
                        f"{prefix_dst}.attention.kv_a_layernorm.weight: {prefix_src}.self_attn.kv_a_layernorm.weight"
                    ),
                    dst=f"{prefix_dst}.attention.kv_a_layernorm.weight",
                ),
                Script(
                    src=f"{prefix_src}.self_attn.kv_b_proj.weight",
                    op=ColumnParallel()
                    + KeepThisTP()
                    + Rename(f"{prefix_dst}.attention.kv_b_proj.weight: {prefix_src}.self_attn.kv_b_proj.weight"),
                    dst=f"{prefix_dst}.attention.kv_b_proj.weight",
                ),
                Script(
                    src=f"{prefix_src}.self_attn.o_proj.weight",
                    op=RowParallel()
                    + KeepThisTP()
                    + Rename(f"{prefix_dst}.attention.o_proj.weight: {prefix_src}.self_attn.o_proj.weight"),
                    dst=f"{prefix_dst}.attention.o_proj.weight",
                ),
                Script(
                    src=f"{prefix_src}.self_attn.indexer.wq_b.weight",
                    op=Duplicate()
                    + KeepThisTP()
                    + Rename(f"{prefix_dst}.attention.indexer.wq_b.weight: {prefix_src}.self_attn.indexer.wq_b.weight"),
                    dst=f"{prefix_dst}.attention.indexer.wq_b.weight",
                ),
                Script(
                    src=f"{prefix_src}.self_attn.indexer.wk.weight",
                    op=Duplicate()
                    + KeepThisTP()
                    + Rename(f"{prefix_dst}.attention.indexer.wk.weight: {prefix_src}.self_attn.indexer.wk.weight"),
                    dst=f"{prefix_dst}.attention.indexer.wk.weight",
                ),
                Script(
                    src=f"{prefix_src}.self_attn.indexer.k_norm.weight",
                    op=Duplicate()
                    + KeepThisTP()
                    + Rename(
                        f"{prefix_dst}.attention.indexer.k_norm.weight: {prefix_src}.self_attn.indexer.k_norm.weight"
                    ),
                    dst=f"{prefix_dst}.attention.indexer.k_norm.weight",
                ),
                Script(
                    src=f"{prefix_src}.self_attn.indexer.k_norm.bias",
                    op=Duplicate()
                    + KeepThisTP()
                    + Rename(f"{prefix_dst}.attention.indexer.k_norm.bias: {prefix_src}.self_attn.indexer.k_norm.bias"),
                    dst=f"{prefix_dst}.attention.indexer.k_norm.bias",
                ),
                Script(
                    src=f"{prefix_src}.self_attn.indexer.weights_proj.weight",
                    op=Duplicate()
                    + KeepThisTP()
                    + Rename(
                        f"{prefix_dst}.attention.indexer.weights_proj.weight: {prefix_src}.self_attn.indexer.weights_proj.weight"
                    ),
                    dst=f"{prefix_dst}.attention.indexer.weights_proj.weight",
                ),
                Script(
                    src=f"{prefix_src}.input_layernorm.weight",
                    op=Duplicate()
                    + KeepThisTP()
                    + Rename(f"{prefix_dst}.attention_norm.weight: {prefix_src}.input_layernorm.weight"),
                    dst=f"{prefix_dst}.attention_norm.weight",
                ),
                Script(
                    src=f"{prefix_src}.post_attention_layernorm.weight",
                    op=Duplicate()
                    + KeepThisTP()
                    + Rename(f"{prefix_dst}.ffn_norm.weight: {prefix_src}.post_attention_layernorm.weight"),
                    dst=f"{prefix_dst}.ffn_norm.weight",
                ),
            ]

            if hasattr(layer.feed_forward, "moe"):
                block_scripts.extend([
                    Script(
                        src=f"{prefix_src}.mlp.gate.weight",
                        op=Duplicate()
                        + KeepThisTP()
                        + Rename(f"{prefix_dst}.feed_forward.moe.gate.weight: {prefix_src}.mlp.gate.weight"),
                        dst=f"{prefix_dst}.feed_forward.moe.gate.weight",
                    ),
                    Script(
                        src=f"{prefix_src}.mlp.gate.e_score_correction_bias",
                        op=Duplicate()
                        + KeepThisTP()
                        + Rename(
                            f"{prefix_dst}.feed_forward.moe.router_balance_bias: {prefix_src}.mlp.gate.e_score_correction_bias"
                        ),
                        dst=f"{prefix_dst}.feed_forward.moe.router_balance_bias",
                    ),
                    Script(
                        src=f"{prefix_src}.mlp.experts.gate_up_proj",
                        op=KeepThisEP()
                        + Rename(f"{prefix_dst}.feed_forward.moe.experts.w1: {prefix_src}.mlp.experts.gate_up_proj"),
                        dst=f"{prefix_dst}.feed_forward.moe.experts.w1",
                    ),
                    Script(
                        src=f"{prefix_src}.mlp.experts.down_proj",
                        op=KeepThisEP()
                        + Rename(f"{prefix_dst}.feed_forward.moe.experts.w2: {prefix_src}.mlp.experts.down_proj"),
                        dst=f"{prefix_dst}.feed_forward.moe.experts.w2",
                    ),
                    Script(
                        src=f"{prefix_src}.mlp.shared_experts.[gu]*_proj.weight",
                        op=FFNMergeGateUp()
                        + KeepThisTP()
                        + Rename(
                            f"{prefix_dst}.feed_forward.share_expert.w1.weight: {prefix_src}.mlp.shared_experts.gate_up_proj.weight"
                        ),
                        dst=f"{prefix_dst}.feed_forward.share_expert.w1.weight",
                    ),
                    Script(
                        src=f"{prefix_src}.mlp.shared_experts.down_proj.weight",
                        op=RowParallel()
                        + KeepThisTP()
                        + Rename(
                            f"{prefix_dst}.feed_forward.share_expert.w2.weight: {prefix_src}.mlp.shared_experts.down_proj.weight"
                        ),
                        dst=f"{prefix_dst}.feed_forward.share_expert.w2.weight",
                    ),
                ])
            else:
                block_scripts.extend([
                    Script(
                        src=f"{prefix_src}.mlp.[gu]*_proj.weight",
                        op=FFNMergeGateUp()
                        + KeepThisTP()
                        + Rename(f"{prefix_dst}.feed_forward.w1.weight: {prefix_src}.mlp.gate_up_proj.weight"),
                        dst=f"{prefix_dst}.feed_forward.w1.weight",
                    ),
                    Script(
                        src=f"{prefix_src}.mlp.down_proj.weight",
                        op=RowParallel()
                        + KeepThisTP()
                        + Rename(f"{prefix_dst}.feed_forward.w2.weight: {prefix_src}.mlp.down_proj.weight"),
                        dst=f"{prefix_dst}.feed_forward.w2.weight",
                    ),
                ])

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
