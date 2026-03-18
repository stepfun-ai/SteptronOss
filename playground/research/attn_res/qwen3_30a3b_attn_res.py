"""Qwen3-30A3B Attention Residual experiment.

Notation from the paper
=======================

The paper replaces additive residuals with a depth-wise routing operator.
For sub-layer ``l``, let ``S_l = {s_0, ..., s_{N_l-1}}`` be the source stack,
``w_l`` be the learned pseudo-query for this sub-layer, and
``phi(s) = RMSNorm(s)``. The routing equations are:

``alpha_{l,n}(t) = exp(w_l^T phi(s_n(t))) / sum_m exp(w_l^T phi(s_m(t)))``
``h_l(t) = sum_n alpha_{l,n}(t) s_n(t)``
``v_l = f_l(LN(h_l))``

Here ``t`` indexes token position, ``h_l`` is the routed input to sub-layer
``l``, and ``v_l`` is the output after the actual transformer operation
``f_l`` (attention or FFN).

Full Attention Residuals
========================

The Full variant attends over the embedding output and all previous
sub-layer outputs:

```
v_0 = embedding(x)
for l in 1..L:
    S_l = [v_0, v_1, ..., v_{l-1}]
    h_l = AttnRes(S_l; w_l)
    v_l = f_l(LN(h_l))
```

Block Attention Residuals
=========================

The Block variant partitions the ``L`` sub-layers into ``N`` blocks, each
with size ``K = L / N``. It keeps completed block summaries plus the running
partial sum inside the current block:

```
blocks = [v_0]
partial = 0
for l in 1..L:
    S_l = blocks + [partial]
    h_l = AttnRes(S_l; w_l)
    v_l = f_l(LN(h_l))
    partial = partial + v_l
    if l % K == 0:
        blocks.append(partial)
        partial = 0
```

This implementation follows the paper at sub-layer granularity. A transformer
layer contributes two sub-layers (attention, then FFN), so the effective AttnRes
depth here is ``2 * num_layers``.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn

from playground.pretrain.qwen3.qwen3_30a3b import Qwen3_30A3BConfig
from playground.sft.qwen3.qwen3_30a3b_sft_step3_data import Exp as BaseSFTExp
from steptronoss.core.parallel_state import (
    PM,
    get_vpp_rank,
    get_vpp_size,
)
from steptronoss.core.tensor_parallel import get_cuda_rng_tracker
from steptronoss.core.tensor_parallel.random import checkpoint
from steptronoss.model.decoder_model import NoopTransformerBlock, TransformerBlock
from steptronoss.model.qwen_dense import QwenModel
from steptronoss.utils.general import get_position_id_from_cu_seqlens
from steptronoss.utils.memory_tracker import CMT

ATTN_RES_NONE = "none"
ATTN_RES_FULL = "full"
ATTN_RES_BLOCK = "block"
ATTN_RES_MODES = {ATTN_RES_NONE, ATTN_RES_FULL, ATTN_RES_BLOCK}


class RMSNormNoWeight(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x.float() * rms


class AttnResOperator(nn.Module):
    """Depth-wise attention over residual sources."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, sequence_parallel: bool = False):
        super().__init__()
        self.pseudo_query = nn.Parameter(torch.zeros(hidden_size))
        # When TP sequence parallel is enabled, each rank only sees a shard of tokens,
        # so this replicated parameter needs a TP all-reduce on its gradients.
        self.pseudo_query.sequence_parallel = sequence_parallel
        self.key_norm = RMSNormNoWeight(eps=eps)

    def forward(self, sources: torch.Tensor) -> torch.Tensor:
        normalized_sources = self.key_norm(sources)
        logits = torch.einsum("d,nsbd->nsb", self.pseudo_query.float(), normalized_sources)
        weights = torch.softmax(logits, dim=0)
        return torch.einsum("nsb,nsbd->sbd", weights, sources.float()).to(dtype=sources.dtype)


class Qwen3_30A3BAttnResConfig(Qwen3_30A3BConfig):
    attn_res_mode: str
    """Residual routing mode: `none`, `full`, or `block`."""

    attn_res_num_blocks: int
    """Number of AttnRes blocks when `attn_res_mode == "block"`."""

    def __init__(self):
        super().__init__()
        self.attn_res_mode = ATTN_RES_BLOCK
        self.attn_res_num_blocks = 8

    @property
    def attn_res_total_sublayers(self) -> int:
        return self.num_layers * 2

    @property
    def attn_res_block_size(self) -> int:
        mode = self.attn_res_mode.lower()
        if mode not in ATTN_RES_MODES:
            raise ValueError(
                f"Unsupported attn_res_mode={self.attn_res_mode!r}, expected one of {sorted(ATTN_RES_MODES)}"
            )
        if mode != ATTN_RES_BLOCK:
            return self.attn_res_total_sublayers
        return self.attn_res_total_sublayers // self.attn_res_num_blocks

    def sanity_check(self):
        super().sanity_check()

        mode = self.attn_res_mode.lower()
        if mode not in ATTN_RES_MODES:
            raise ValueError(
                f"Unsupported attn_res_mode={self.attn_res_mode!r}, expected one of {sorted(ATTN_RES_MODES)}"
            )
        if mode != ATTN_RES_NONE:
            if self.parallel_cfg.pipeline_model_parallel_size != 1:
                raise NotImplementedError("AttnRes research model currently requires PP=1")
            if self.parallel_cfg.virtual_pipeline_model_parallel_size != 1:
                raise NotImplementedError("AttnRes research model currently requires VPP=1")

        if mode == ATTN_RES_BLOCK:
            if self.attn_res_num_blocks < 1:
                raise ValueError("attn_res_num_blocks must be >= 1")
            if self.attn_res_total_sublayers % self.attn_res_num_blocks != 0:
                raise ValueError(
                    "2 * num_layers must be divisible by attn_res_num_blocks, "
                    f"got {self.attn_res_total_sublayers} and {self.attn_res_num_blocks}"
                )

    def build_model(self):
        model = AttnResQwenModel(cfg=self, layer_map=self.build_layer_map())

        base_std = 0.02
        output_std = base_std / (2 * self.num_layers) ** 0.5

        for name, param in model.named_parameters():
            with torch.no_grad():
                if name.endswith("pseudo_query") or name.endswith(".bias"):
                    param.zero_()
                elif param.ndim == 1:
                    param.fill_(1.0)
                else:
                    std = (
                        output_std
                        if name.endswith("attention.wo.weight") or name.endswith("feed_forward.moe.experts.w2")
                        else base_std
                    )
                    torch.nn.init.normal_(param, mean=0.0, std=std)

        return model


class AttnResTransformerBlock(TransformerBlock):
    def __init__(self, cfg: Qwen3_30A3BAttnResConfig, layer_id: int, recompute: bool = False):
        super().__init__(cfg=cfg, layer_id=layer_id, recompute=recompute)

        mode = cfg.attn_res_mode.lower()
        if mode not in ATTN_RES_MODES:
            raise ValueError(
                f"Unsupported attn_res_mode={cfg.attn_res_mode!r}, expected one of {sorted(ATTN_RES_MODES)}"
            )
        if mode == ATTN_RES_NONE:
            self.attn_res_attention = None
            self.attn_res_ffn = None
        else:
            self.attn_res_attention = AttnResOperator(
                cfg.hidden_size,
                eps=cfg.layernorm_epsilon,
                sequence_parallel=cfg.tp_cfg.sequence_parallel,
            )
            self.attn_res_ffn = AttnResOperator(
                cfg.hidden_size,
                eps=cfg.layernorm_epsilon,
                sequence_parallel=cfg.tp_cfg.sequence_parallel,
            )

    def forward_attention_sublayer(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seq_len: int | None = None,
        position_id: torch.IntTensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        CMT.mark(f"layer{self.layer_id}_attn_norm_in")
        if self.training and "attn_norm" in self.recompute:
            attn_in = checkpoint(self.attention_norm, self.distribute_saved_activations, x)
        else:
            attn_in = self.attention_norm(x)

        CMT.mark(f"layer{self.layer_id}_attn_in")
        if self.training and "attention" in self.recompute:
            return checkpoint(
                self.attention,
                self.distribute_saved_activations,
                attn_in,
                cu_seqlens=cu_seqlens,
                max_seq_len=max_seq_len,
                position_id=position_id,
                **kwargs,
            )
        return self.attention(
            attn_in,
            cu_seqlens=cu_seqlens,
            max_seq_len=max_seq_len,
            position_id=position_id,
            **kwargs,
        )

    def forward_ffn_sublayer(self, x: torch.Tensor) -> torch.Tensor:
        CMT.mark(f"layer{self.layer_id}_ffn_norm_in")
        if self.training and "ffn_norm" in self.recompute:
            ffn_in = checkpoint(self.ffn_norm, self.distribute_saved_activations, x)
        else:
            ffn_in = self.ffn_norm(x)

        CMT.mark(f"layer{self.layer_id}_ffn_in")
        out = self.feed_forward(ffn_in, recompute="feed_forward" in self.recompute)
        CMT.mark(f"layer{self.layer_id}_ffn_out")
        return out

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seq_len: int | None = None,
        position_id: torch.IntTensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_out = self.forward_attention_sublayer(
            x,
            cu_seqlens=cu_seqlens,
            max_seq_len=max_seq_len,
            position_id=position_id,
            **kwargs,
        )
        h = x + attn_out
        ffn_out = self.forward_ffn_sublayer(h)
        return h + ffn_out


class AttnResQwenModel(QwenModel):
    cfg: Qwen3_30A3BAttnResConfig

    def build(self, layer_map: dict[int, dict[int, dict[int, dict]]]):
        self.layers = nn.ModuleList()
        pp_rank, vp_rank = PM.rank_in("PP"), get_vpp_rank()

        for layer, kwargs in layer_map[pp_rank][vp_rank].items():
            self.layers.append(AttnResTransformerBlock(self.cfg, layer_id=layer, **kwargs))

        if len(self.layers) == 0:
            self.layers.append(NoopTransformerBlock())

        if self.is_pipeline_first_stage():
            self.tok_embeddings = self.cfg.tok_embed_cfg.build_model()

        if self.is_pipeline_last_stage():
            if self.cfg.tie_embedding:
                self.out_embeddings = self.cfg.out_embed_cfg.build_model(self.tok_embeddings.word_embeddings.weight)
            else:
                self.out_embeddings = self.cfg.out_embed_cfg.build_model()

    def _forward_full_attn_res(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.IntTensor | None = None,
        position_id: torch.IntTensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        sources = [hidden_states]

        for layer in self.layers:
            if isinstance(layer, NoopTransformerBlock):
                continue

            stacked_sources = torch.stack(sources, dim=0)
            if layer.attn_res_attention is None:
                attn_in = stacked_sources.sum(dim=0)
            else:
                attn_in = layer.attn_res_attention(stacked_sources)
            attn_out = layer.forward_attention_sublayer(
                attn_in,
                cu_seqlens=cu_seqlens,
                position_id=position_id,
                **kwargs,
            )
            sources.append(attn_out)

            stacked_sources = torch.stack(sources, dim=0)
            if layer.attn_res_ffn is None:
                ffn_in = stacked_sources.sum(dim=0)
            else:
                ffn_in = layer.attn_res_ffn(stacked_sources)
            ffn_out = layer.forward_ffn_sublayer(ffn_in)
            sources.append(ffn_out)

        total = sources[0].float()
        for source in sources[1:]:
            total = total + source.float()
        return total.to(dtype=sources[0].dtype)

    def _forward_block_attn_res(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.IntTensor | None = None,
        position_id: torch.IntTensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        blocks = [hidden_states]
        partial_block = None
        sublayer_idx = 0

        for layer in self.layers:
            if isinstance(layer, NoopTransformerBlock):
                continue

            block_sources = blocks if partial_block is None else blocks + [partial_block]
            stacked_sources = torch.stack(block_sources, dim=0)
            if layer.attn_res_attention is None:
                attn_in = stacked_sources.sum(dim=0)
            else:
                attn_in = layer.attn_res_attention(stacked_sources)
            attn_out = layer.forward_attention_sublayer(
                attn_in,
                cu_seqlens=cu_seqlens,
                position_id=position_id,
                **kwargs,
            )
            if sublayer_idx > 0 and sublayer_idx % self.cfg.attn_res_block_size == 0 and partial_block is not None:
                blocks.append(partial_block)
                partial_block = None
            partial_block = attn_out if partial_block is None else partial_block + attn_out
            sublayer_idx += 1

            block_sources = blocks if partial_block is None else blocks + [partial_block]
            stacked_sources = torch.stack(block_sources, dim=0)
            if layer.attn_res_ffn is None:
                ffn_in = stacked_sources.sum(dim=0)
            else:
                ffn_in = layer.attn_res_ffn(stacked_sources)
            ffn_out = layer.forward_ffn_sublayer(ffn_in)
            if sublayer_idx > 0 and sublayer_idx % self.cfg.attn_res_block_size == 0 and partial_block is not None:
                blocks.append(partial_block)
                partial_block = None
            partial_block = ffn_out if partial_block is None else partial_block + ffn_out
            sublayer_idx += 1

        final_sources = list(blocks)
        if partial_block is not None:
            final_sources.append(partial_block)
        total = final_sources[0].float()
        for source in final_sources[1:]:
            total = total + source.float()
        return total.to(dtype=final_sources[0].dtype)

    def forward_chunk(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.IntTensor = None,
        position_id: torch.IntTensor = None,
        **kwargs,
    ) -> torch.Tensor:
        mode = self.cfg.attn_res_mode.lower()
        if mode not in ATTN_RES_MODES:
            raise ValueError(
                f"Unsupported attn_res_mode={self.cfg.attn_res_mode!r}, expected one of {sorted(ATTN_RES_MODES)}"
            )

        if mode == ATTN_RES_NONE:
            return super().forward_chunk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_id=position_id,
                **kwargs,
            )

        if PM.size_of("CP") > 1:
            assert cu_seqlens is not None, "cu_seqlens required for CP, use [0, len(input_ids)] if you dont have one."
        if cu_seqlens is not None and position_id is None:
            position_id = get_position_id_from_cu_seqlens(cu_seqlens)

        if PM.size_of("PP") > 1 or get_vpp_size() > 1:
            raise NotImplementedError("AttnRes research model currently only supports PP=1 and VPP=1")

        if self.cfg.tp_cfg.sequence_parallel:
            rng_context = get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        with rng_context:
            if mode == ATTN_RES_FULL:
                return self._forward_full_attn_res(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_id=position_id,
                    **kwargs,
                )
            return self._forward_block_attn_res(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_id=position_id,
                **kwargs,
            )


class Exp(BaseSFTExp):
    model_cfg: Qwen3_30A3BAttnResConfig = Qwen3_30A3BAttnResConfig

    def __init__(self):
        super().__init__()
        self.model_cfg.attn_res_mode = ATTN_RES_BLOCK
        self.model_cfg.attn_res_num_blocks = 8
        # Base Qwen3 checkpoints do not contain the new pseudo-query parameters.
        self.checkpoint_cfg.strict_load_model = False


if __name__ == "__main__":
    Exp().train()
