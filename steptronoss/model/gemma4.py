from __future__ import annotations

from functools import cached_property

import torch
import torch.nn as nn
import torch.nn.functional as F

from steptronoss.core import tensor_parallel
from steptronoss.core.parallel_state import PM, get_vpp_rank
from steptronoss.core.tensor_parallel.random import checkpoint
from steptronoss.model.common.attention_core import parse_cu_seqlens
from steptronoss.model.common.parallel_embedding import WordEmbedding
from steptronoss.model.common.rms_norm import RMSNorm
from steptronoss.model.decoder_model import LlamaLikeModel, NoopTransformerBlock


class GemmaScaledWordEmbedding(WordEmbedding):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.scalar_embed_scale = float(cfg.hidden_size**0.5)
        self.register_buffer(
            "embed_scale", torch.tensor(self.scalar_embed_scale, dtype=torch.float32), persistent=False
        )

    def forward(self, input_ids, **kwargs):
        embeddings = super().forward(input_ids=input_ids, **kwargs)
        return embeddings * self.embed_scale.to(device=embeddings.device, dtype=embeddings.dtype)


class GemmaRoPE(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        theta: float,
        max_position_embeddings: int,
        partial_rotary_factor: float = 1.0,
        factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.max_position_embeddings = max_position_embeddings
        self.partial_rotary_factor = partial_rotary_factor
        self.factor = factor

        self._cached_seqlen = 0
        self.register_buffer("_cos_cache", torch.empty(0, dim, dtype=torch.float32), persistent=False)
        self.register_buffer("_sin_cache", torch.empty(0, dim, dtype=torch.float32), persistent=False)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _build_inv_freq(self, device: torch.device) -> torch.Tensor:
        rope_angles = int(self.partial_rotary_factor * self.dim // 2)
        if rope_angles > 0:
            rotated = 1.0 / (
                self.theta ** (torch.arange(0, 2 * rope_angles, 2, device=device, dtype=torch.float32) / self.dim)
            )
        else:
            rotated = torch.empty(0, device=device, dtype=torch.float32)

        noop_angles = self.dim // 2 - rope_angles
        if noop_angles > 0:
            rotated = torch.cat((rotated, torch.zeros(noop_angles, device=device, dtype=torch.float32)), dim=0)

        if self.factor != 1.0:
            rotated = rotated / self.factor
        return rotated

    def _check_set_cos_sin_cache(self, cur_seqlen: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if cur_seqlen > self.max_position_embeddings:
            raise ValueError(
                f"Requested position {cur_seqlen} beyond max_position_embeddings={self.max_position_embeddings}"
            )

        need_refresh = cur_seqlen > self._cached_seqlen or self._cos_cache.device != device
        if need_refresh:
            inv_freq = self._build_inv_freq(device=device)
            positions = torch.arange(cur_seqlen, device=device, dtype=torch.float32)
            angles = torch.outer(positions, inv_freq)
            repeated_angles = angles.repeat([1, 2])
            self._cos_cache = repeated_angles.cos().to(torch.float32)
            self._sin_cache = repeated_angles.sin().to(torch.float32)
            self._cached_seqlen = cur_seqlen

        return self._cos_cache, self._sin_cache

    def forward(self, feature: torch.Tensor, position_id: torch.Tensor | None = None) -> torch.Tensor:
        if position_id is None:
            position_id = torch.arange(feature.shape[1], device=feature.device, dtype=torch.long)
            position_id = position_id.unsqueeze(0).expand(feature.shape[0], -1)
        else:
            position_id = position_id.to(device=feature.device, dtype=torch.long)
            if position_id.ndim == 1:
                position_id = position_id.unsqueeze(0).expand(feature.shape[0], -1)

        if position_id.ndim != 2:
            raise ValueError(f"Unsupported position_id shape: {tuple(position_id.shape)}")

        inv_freq = self._build_inv_freq(device=feature.device)
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_id.shape[0], -1, 1)
        position_ids_expanded = position_id[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=feature.dtype)[:, :, None, :]
        sin = emb.sin().to(dtype=feature.dtype)[:, :, None, :]
        return feature * cos + self.rotate_half(feature) * sin


class GemmaAttentionCore(nn.Module):
    """Gemma4-specific SDPA wrapper kept next to the Gemma model definition.

    Unlike the generic `AttentionCore`, this class intentionally bakes in the
    Hugging Face Gemma4 behavior we need for alignment:
    - sliding causal masks use the strict lower bound `kv_idx > q_idx - window`
    - short sliding sequences fall back to plain causal SDPA instead of an
      explicitly materialized local mask
    - SDPA uses `enable_gqa=True` when q-heads and kv-heads differ
    - Gemma keeps an explicit `scale=1.0`

    The semantics are model-specific rather than reusable across families, so
    they live in `gemma4.py` instead of `model/common`.
    """

    def __init__(
        self,
        causal: bool = True,
        attention_dropout: float = 0.0,
        sliding_window: int = -1,
        scale: float = 1.0,
    ):
        super().__init__()
        self.causal = causal
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.scale = scale

    @staticmethod
    def _build_local_mask(q_len: int, k_len: int, window: int, causal: bool, device: torch.device) -> torch.Tensor:
        q_idx = torch.arange(q_len, device=device).unsqueeze(1)
        k_idx = torch.arange(k_len, device=device).unsqueeze(0)
        if causal:
            allowed = (k_idx <= q_idx) & (k_idx > (q_idx - window))
        else:
            allowed = (k_idx - q_idx).abs() <= window
        return allowed

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool,
        attn_mask: torch.Tensor | None,
        enable_gqa: bool,
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
            scale=self.scale,
            enable_gqa=enable_gqa,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seq_len: int | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, num_heads, head_dim = q.shape
        window = self.sliding_window
        use_local_mask = window is not None and window >= 0

        if cu_seqlens is not None:
            cu_seqlens_q, cu_seqlens_k, _max_q_len, _max_k_len = parse_cu_seqlens(cu_seqlens, max_seq_len)
            q_flat = q.reshape(-1, num_heads, head_dim)
            k_flat = k.reshape(-1, k.shape[2], head_dim)
            v_flat = v.reshape(-1, v.shape[2], head_dim)

            outputs = []
            q_cu = cu_seqlens_q.tolist()
            k_cu = cu_seqlens_k.tolist()
            for batch_idx in range(len(q_cu) - 1):
                q_start, q_end = q_cu[batch_idx], q_cu[batch_idx + 1]
                k_start, k_end = k_cu[batch_idx], k_cu[batch_idx + 1]

                q_seq = q_flat[q_start:q_end].transpose(0, 1).unsqueeze(0)
                k_seq = k_flat[k_start:k_end].transpose(0, 1).unsqueeze(0)
                v_seq = v_flat[k_start:k_end].transpose(0, 1).unsqueeze(0)

                enable_gqa = False
                local_mask_for_seq = use_local_mask and k_seq.shape[-2] >= window
                attn_mask = None
                is_causal = self.causal and not local_mask_for_seq
                if local_mask_for_seq:
                    kv_repeat = num_heads // k_seq.shape[1]
                    k_seq = k_seq.repeat_interleave(kv_repeat, dim=1)
                    v_seq = v_seq.repeat_interleave(kv_repeat, dim=1)
                    attn_mask = (
                        self
                        ._build_local_mask(
                            q_seq.shape[-2],
                            k_seq.shape[-2],
                            window,
                            self.causal,
                            device=q_seq.device,
                        )
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )
                elif q_seq.shape[1] != k_seq.shape[1]:
                    enable_gqa = True

                out = self._sdpa(
                    q_seq,
                    k_seq,
                    v_seq,
                    is_causal=is_causal,
                    attn_mask=attn_mask,
                    enable_gqa=enable_gqa,
                )
                outputs.append(out.squeeze(0).transpose(0, 1))

            output = torch.cat(outputs, dim=0)
            return output.reshape(batch_size, seq_len, num_heads, head_dim)

        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        use_local_mask = use_local_mask and k_t.shape[-2] >= window
        attn_mask = None
        is_causal = self.causal and not use_local_mask
        enable_gqa = False
        if use_local_mask:
            kv_repeat = num_heads // k_t.shape[1]
            k_t = k_t.repeat_interleave(kv_repeat, dim=1)
            v_t = v_t.repeat_interleave(kv_repeat, dim=1)
            attn_mask = self._build_local_mask(seq_len, k_t.shape[-2], window, self.causal, q_t.device)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        elif q_t.shape[1] != k_t.shape[1]:
            enable_gqa = True

        output = self._sdpa(q_t, k_t, v_t, is_causal=is_causal, attn_mask=attn_mask, enable_gqa=enable_gqa)
        return output.transpose(1, 2)


class Gemma4Attention(nn.Module):
    """Full Gemma4 attention block: projections + norm + RoPE + attention core.

    This is intentionally a higher-level module than
    `steptronoss.model.common.attention_core.AttentionCore`.

    `AttentionCore` only describes the reusable kernel boundary for already-
    prepared q/k/v tensors. The local `GemmaAttentionCore` narrows that to the
    Hugging Face Gemma4 kernel semantics. `Gemma4Attention` additionally owns
    all Gemma4 architecture semantics before and after that kernel boundary:
    - sliding vs full-attention layer selection
    - head-dim and KV-head topology (`head_dim` vs `global_head_dim`)
    - optional `k == v` full-attention path (`attention_k_eq_v`)
    - separate q/k/v/o projections with Gemma TP sharding rules
    - Gemma q/k/v RMSNorm configuration
    - Gemma RoPE parameterization and application

    In other words, the core classes are reusable attention-kernel adapters,
    while this class is the Gemma4 architecture wrapper around them. If we add
    a future optimized Gemma attention implementation, it should be treated as a
    drop-in alternative for this whole module rather than for the generic
    `AttentionCore` interface alone.
    """

    def __init__(self, cfg, layer_id: int):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        self.sequence_parallel = cfg.tp_cfg.sequence_parallel
        self.tp_size = PM.size_of("TP")
        self.cp = PM.size_of("CP") > 1
        if self.cp:
            raise NotImplementedError("Gemma4Attention does not support context parallelism yet")

        self.layer_type = cfg.layer_types[layer_id]
        self.is_sliding = self.layer_type == "sliding_attention"
        self.head_dim = cfg.head_dim if self.is_sliding else cfg.global_head_dim
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_attention_groups if self.is_sliding else cfg.num_global_key_value_heads
        if self.num_heads % self.tp_size != 0:
            raise ValueError(
                f"Layer {layer_id} uses {self.num_heads} attention heads, but tensor parallel size is {self.tp_size}."
            )
        if self.num_kv_heads % self.tp_size != 0:
            raise ValueError(
                f"Layer {layer_id} uses {self.num_kv_heads} KV heads, but tensor parallel size is {self.tp_size}. "
                "Current Gemma4 support requires the KV-head count to divide TP evenly."
            )
        if self.num_kv_heads < self.tp_size:
            raise ValueError(
                f"Layer {layer_id} uses {self.num_kv_heads} KV heads, but tensor parallel size is {self.tp_size}. "
                "Current Gemma4 support requires TP to be no larger than the smallest KV-head count."
            )

        self.num_local_heads = self.num_heads // self.tp_size
        self.num_local_kv_heads = self.num_kv_heads // self.tp_size
        self.use_k_eq_v = (not self.is_sliding) and cfg.attention_k_eq_v
        self.sliding_window = cfg.sliding_window_size if self.is_sliding else -1

        self.q_proj = tensor_parallel.ColumnParallelLinear(
            cfg.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
            gather_output=False,
            async_tensor_model_parallel_allreduce=cfg.tp_cfg.async_tensor_model_parallel_allreduce,
            **cfg.tp_cfg.get_tp_kwargs(),
        )
        self.k_proj = tensor_parallel.ColumnParallelLinear(
            cfg.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            async_tensor_model_parallel_allreduce=cfg.tp_cfg.async_tensor_model_parallel_allreduce,
            **cfg.tp_cfg.get_tp_kwargs(),
        )
        self.v_proj = None
        if not self.use_k_eq_v:
            self.v_proj = tensor_parallel.ColumnParallelLinear(
                cfg.hidden_size,
                self.num_kv_heads * self.head_dim,
                bias=False,
                gather_output=False,
                async_tensor_model_parallel_allreduce=cfg.tp_cfg.async_tensor_model_parallel_allreduce,
                **cfg.tp_cfg.get_tp_kwargs(),
            )
        self.o_proj = tensor_parallel.RowParallelLinear(
            self.num_heads * self.head_dim,
            cfg.hidden_size,
            bias=False,
            input_is_parallel=True,
            fp32_output=self.tp_size > 1,
            **cfg.tp_cfg.get_tp_kwargs(),
        )

        self.q_norm = RMSNorm(
            self.head_dim,
            eps=cfg.layernorm_epsilon,
            sequence_parallel=self.sequence_parallel,
            use_fp32=True,
            use_zero_init=False,
            with_scale=True,
            math_mode="pow",
            cast_output_to_input_after_mul=True,
        )
        self.k_norm = RMSNorm(
            self.head_dim,
            eps=cfg.layernorm_epsilon,
            sequence_parallel=self.sequence_parallel,
            use_fp32=True,
            use_zero_init=False,
            with_scale=True,
            math_mode="pow",
            cast_output_to_input_after_mul=True,
        )
        self.v_norm = RMSNorm(
            self.head_dim,
            eps=cfg.layernorm_epsilon,
            sequence_parallel=self.sequence_parallel,
            use_fp32=True,
            use_zero_init=False,
            with_scale=False,
            math_mode="pow",
            cast_output_to_input_after_mul=True,
        )
        self.rope = GemmaRoPE(
            dim=self.head_dim,
            theta=cfg.rope_theta if self.is_sliding else cfg.full_rope_theta,
            max_position_embeddings=cfg.max_position_embeddings,
            partial_rotary_factor=1.0 if self.is_sliding else cfg.full_rotary_factor,
            factor=1.0 if self.is_sliding else cfg.full_rope_factor,
        )
        self.core_attention = GemmaAttentionCore(
            causal=cfg.causal,
            attention_dropout=cfg.attention_dropout,
            sliding_window=self.sliding_window,
            scale=1.0,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seq_len: torch.Tensor | None = None,
        position_id: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_len, batch_size, _hidden_size = x.shape
        if self.sequence_parallel:
            seq_len *= self.tp_size

        q_states = self.q_proj(x)[0].view(seq_len, batch_size, self.num_local_heads, self.head_dim)
        k_states = self.k_proj(x)[0].view(seq_len, batch_size, self.num_local_kv_heads, self.head_dim)
        if self.v_proj is None:
            v_states = k_states
        else:
            v_states = self.v_proj(x)[0].view(seq_len, batch_size, self.num_local_kv_heads, self.head_dim)

        q_states = self.q_norm(q_states.contiguous())
        k_states = self.k_norm(k_states.contiguous())
        v_states = self.v_norm(v_states.contiguous())

        q_states = q_states.transpose(0, 1).contiguous()
        k_states = k_states.transpose(0, 1).contiguous()
        v_states = v_states.transpose(0, 1).contiguous()

        if position_id is None:
            if cu_seqlens is not None:
                from steptronoss.utils.general import get_position_id_from_cu_seqlens

                position_id = get_position_id_from_cu_seqlens(cu_seqlens)
            else:
                position_id = torch.arange(seq_len, device=x.device, dtype=torch.long)
                if batch_size > 1:
                    position_id = position_id.unsqueeze(0).expand(batch_size, -1)

        q_states = self.rope(q_states, position_id)
        k_states = self.rope(k_states, position_id)

        attn_out = self.core_attention(
            q_states,
            k_states,
            v_states,
            cu_seqlens=cu_seqlens,
            max_seq_len=max_seq_len,
        )
        attn_out = attn_out.transpose(0, 1).contiguous().view(seq_len, batch_size, self.num_local_heads * self.head_dim)
        return self.o_proj(attn_out)[0].to(dtype=x.dtype)


class Gemma4Block(nn.Module):
    def __init__(self, cfg, layer_id: int, recompute: bool = False):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        self.recompute = recompute
        if not isinstance(self.recompute, list):
            if self.recompute is True:
                self.recompute = ["attention", "attn_norm", "feed_forward", "ffn_norm"]
            else:
                self.recompute = []
        self.distribute_saved_activations = cfg.tp_cfg.distribute_saved_activations

        self.input_layernorm = RMSNorm(
            cfg.hidden_size,
            eps=cfg.layernorm_epsilon,
            sequence_parallel=cfg.tp_cfg.sequence_parallel,
            use_fp32=True,
            use_zero_init=False,
            with_scale=True,
            math_mode="pow",
            cast_output_to_input_after_mul=True,
        )
        self.post_attention_layernorm = RMSNorm(
            cfg.hidden_size,
            eps=cfg.layernorm_epsilon,
            sequence_parallel=cfg.tp_cfg.sequence_parallel,
            use_fp32=True,
            use_zero_init=False,
            with_scale=True,
            math_mode="pow",
            cast_output_to_input_after_mul=True,
        )
        self.pre_feedforward_layernorm = RMSNorm(
            cfg.hidden_size,
            eps=cfg.layernorm_epsilon,
            sequence_parallel=cfg.tp_cfg.sequence_parallel,
            use_fp32=True,
            use_zero_init=False,
            with_scale=True,
            math_mode="pow",
            cast_output_to_input_after_mul=True,
        )
        self.post_feedforward_layernorm = RMSNorm(
            cfg.hidden_size,
            eps=cfg.layernorm_epsilon,
            sequence_parallel=cfg.tp_cfg.sequence_parallel,
            use_fp32=True,
            use_zero_init=False,
            with_scale=True,
            math_mode="pow",
            cast_output_to_input_after_mul=True,
        )

        self.attention = cfg.attn_cfg.build_model(layer_id=layer_id)
        self.feed_forward = cfg.ffn_cfg.build_model(layer_id=layer_id)
        self.register_buffer("layer_scalar", torch.ones(1, dtype=cfg.tp_cfg.params_dtype))

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seq_len: int | None = None,
        position_id: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = x
        if self.training and "attn_norm" in self.recompute:
            attn_in = checkpoint(self.input_layernorm, self.distribute_saved_activations, x)
        else:
            attn_in = self.input_layernorm(x)

        if self.training and "attention" in self.recompute:
            attn_out = checkpoint(
                self.attention,
                self.distribute_saved_activations,
                attn_in,
                cu_seqlens=cu_seqlens,
                max_seq_len=max_seq_len,
                position_id=position_id,
                **kwargs,
            )
        else:
            attn_out = self.attention(
                attn_in,
                cu_seqlens=cu_seqlens,
                max_seq_len=max_seq_len,
                position_id=position_id,
                **kwargs,
            )
        attn_out = self.post_attention_layernorm(attn_out)
        hidden_states = residual + attn_out

        residual = hidden_states
        if self.training and "ffn_norm" in self.recompute:
            ffn_in = checkpoint(self.pre_feedforward_layernorm, self.distribute_saved_activations, hidden_states)
        else:
            ffn_in = self.pre_feedforward_layernorm(hidden_states)
        ffn_out = self.feed_forward(ffn_in, recompute="feed_forward" in self.recompute)
        ffn_out = self.post_feedforward_layernorm(ffn_out)
        hidden_states = residual + ffn_out

        return hidden_states * self.layer_scalar.to(device=hidden_states.device, dtype=hidden_states.dtype)


class Gemma4Model(LlamaLikeModel):
    def build(self, layer_map: dict[int, dict[int, dict[int, dict]]]):
        self.layers = nn.ModuleList()
        pp_rank, vp_rank = PM.rank_in("PP"), get_vpp_rank()

        for layer, kwargs in layer_map[pp_rank][vp_rank].items():
            self.layers.append(Gemma4Block(self.cfg, layer_id=layer, **kwargs))

        if len(self.layers) == 0:
            self.layers.append(NoopTransformerBlock())

        if self.is_pipeline_first_stage():
            self.tok_embeddings = self.cfg.tok_embed_cfg.build_model()

        if self.is_pipeline_last_stage():
            tied_weight = self.tok_embeddings.word_embeddings.weight if self.cfg.tie_embedding else None
            self.out_embeddings = self.cfg.out_embed_cfg.build_model(tied_weight)

    @cached_property
    def reshaper(self):
        return self.build_reshaper()

    def build_reshaper(self):
        from steptronoss.checkpointing.reshape_ops import (
            ColumnParallel,
            Duplicate,
            FFNMergeGateUp,
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
                    src="model.language_model.embed_tokens.weight",
                    op=VocabPad(
                        target_vocab_size=self.cfg.tok_embed_cfg.vocab_size,
                        dim=0,
                        pad_type="last",
                    )
                    + ColumnParallel()
                    + KeepThisTP()
                    + Rename("tok_embeddings.word_embeddings.weight: model.language_model.embed_tokens.weight"),
                    dst="tok_embeddings.word_embeddings.weight",
                )
            )

        if self.is_pipeline_last_stage():
            scripts.append(
                Script(
                    src="model.language_model.norm.weight",
                    op=Duplicate()
                    + KeepThisTP()
                    + Rename("out_embeddings.norm.weight: model.language_model.norm.weight"),
                    dst="out_embeddings.norm.weight",
                )
            )

        def generate_block_scripts(layer: Gemma4Block, prefix_src: str, prefix_dst: str) -> list[Script]:
            block_scripts = [
                Script(
                    src=f"{prefix_src}.input_layernorm.weight",
                    op=Duplicate()
                    + KeepThisTP()
                    + Rename(f"{prefix_dst}.input_layernorm.weight: {prefix_src}.input_layernorm.weight"),
                    dst=f"{prefix_dst}.input_layernorm.weight",
                ),
                Script(
                    src=f"{prefix_src}.post_attention_layernorm.weight",
                    op=Duplicate()
                    + KeepThisTP()
                    + Rename(
                        f"{prefix_dst}.post_attention_layernorm.weight: {prefix_src}.post_attention_layernorm.weight"
                    ),
                    dst=f"{prefix_dst}.post_attention_layernorm.weight",
                ),
                Script(
                    src=f"{prefix_src}.pre_feedforward_layernorm.weight",
                    op=Duplicate()
                    + KeepThisTP()
                    + Rename(
                        f"{prefix_dst}.pre_feedforward_layernorm.weight: {prefix_src}.pre_feedforward_layernorm.weight"
                    ),
                    dst=f"{prefix_dst}.pre_feedforward_layernorm.weight",
                ),
                Script(
                    src=f"{prefix_src}.post_feedforward_layernorm.weight",
                    op=Duplicate()
                    + KeepThisTP()
                    + Rename(
                        f"{prefix_dst}.post_feedforward_layernorm.weight: {prefix_src}.post_feedforward_layernorm.weight"
                    ),
                    dst=f"{prefix_dst}.post_feedforward_layernorm.weight",
                ),
                Script(
                    src=f"{prefix_src}.layer_scalar",
                    op=Duplicate() + KeepThisTP() + Rename(f"{prefix_dst}.layer_scalar: {prefix_src}.layer_scalar"),
                    dst=f"{prefix_dst}.layer_scalar",
                ),
                Script(
                    src=f"{prefix_src}.self_attn.q_proj.weight",
                    op=ColumnParallel()
                    + KeepThisTP()
                    + Rename(f"{prefix_dst}.attention.q_proj.weight: {prefix_src}.self_attn.q_proj.weight"),
                    dst=f"{prefix_dst}.attention.q_proj.weight",
                ),
                Script(
                    src=f"{prefix_src}.self_attn.k_proj.weight",
                    op=ColumnParallel()
                    + KeepThisTP()
                    + Rename(f"{prefix_dst}.attention.k_proj.weight: {prefix_src}.self_attn.k_proj.weight"),
                    dst=f"{prefix_dst}.attention.k_proj.weight",
                ),
                Script(
                    src=f"{prefix_src}.self_attn.o_proj.weight",
                    op=RowParallel()
                    + KeepThisTP()
                    + Rename(f"{prefix_dst}.attention.o_proj.weight: {prefix_src}.self_attn.o_proj.weight"),
                    dst=f"{prefix_dst}.attention.o_proj.weight",
                ),
                Script(
                    src=f"{prefix_src}.self_attn.q_norm.weight",
                    op=Duplicate()
                    + KeepThisTP()
                    + Rename(f"{prefix_dst}.attention.q_norm.weight: {prefix_src}.self_attn.q_norm.weight"),
                    dst=f"{prefix_dst}.attention.q_norm.weight",
                ),
                Script(
                    src=f"{prefix_src}.self_attn.k_norm.weight",
                    op=Duplicate()
                    + KeepThisTP()
                    + Rename(f"{prefix_dst}.attention.k_norm.weight: {prefix_src}.self_attn.k_norm.weight"),
                    dst=f"{prefix_dst}.attention.k_norm.weight",
                ),
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
            ]
            if layer.attention.v_proj is not None:
                block_scripts.append(
                    Script(
                        src=f"{prefix_src}.self_attn.v_proj.weight",
                        op=ColumnParallel()
                        + KeepThisTP()
                        + Rename(f"{prefix_dst}.attention.v_proj.weight: {prefix_src}.self_attn.v_proj.weight"),
                        dst=f"{prefix_dst}.attention.v_proj.weight",
                    )
                )
            return block_scripts

        for local_id, layer in enumerate(self.layers):
            if isinstance(layer, NoopTransformerBlock):
                continue
            scripts.extend(
                generate_block_scripts(
                    layer,
                    prefix_src=f"model.language_model.layers.{layer.layer_id}",
                    prefix_dst=f"layers.{local_id}",
                )
            )

        return OnlineReshaper(scripts)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if self.cfg.tie_embedding:
            result = super().load_state_dict(state_dict, strict=False, assign=assign)
            missing = result.missing_keys
            unexpected = result.unexpected_keys
            if missing or unexpected:
                if len(missing) == 1 and missing[0] == "out_embeddings.output.weight" and not unexpected:
                    return result
                raise RuntimeError(f"Missing: {missing}; Unexpected: {unexpected}")
            return result
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def name_parameters(self):
        for parameter in self.parameters():
            parameter._log_name = "other"
        for block in self.layers:
            if isinstance(block, Gemma4Block):
                for _name, parameter in block.named_parameters():
                    parameter._log_name = f"layer{block.layer_id}"
        if self.is_pipeline_first_stage():
            for parameter in self.tok_embeddings.parameters():
                parameter._log_name = "tok_embeddings"
        if self.is_pipeline_last_stage():
            for parameter in self.out_embeddings.parameters():
                parameter._log_name = "output"
