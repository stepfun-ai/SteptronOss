"""Base LLM model components for SteptronOss.

This module provides the base transformer building blocks that can be
used to implement various LLM architectures (LLaMA, Qwen, etc.)
"""

from contextlib import nullcontext

import torch
import torch.nn as nn
from loguru import logger

from steptronoss.core.context_parallel import (
    scatter_to_balanced_cp_region,
)
from steptronoss.core.parallel_state import (
    PM,
    get_vpp_rank,
    get_vpp_size,
)
from steptronoss.core.tensor_parallel import get_cuda_rng_tracker
from steptronoss.core.tensor_parallel.random import (
    checkpoint,
)
from steptronoss.exp.base_exp import Megatron3DParallelModelConfig, MegatronTPConfig
from steptronoss.model.common.feed_forward import FeedForwardConfig
from steptronoss.model.common.grouped_query_attention import AttentionConfig
from steptronoss.model.common.parallel_embedding import (
    InputEmbeddingConfig,
    OutputEmbeddingConfig,
)
from steptronoss.model.common.rms_norm import RMSNorm
from steptronoss.model.module import MegatronModule
from steptronoss.utils.general import get_position_id_from_cu_seqlens
from steptronoss.utils.utils import format_layermap


class DecoderLLMConfig(Megatron3DParallelModelConfig):
    num_layers: int
    hidden_size: int
    layernorm_epsilon: float
    rms_norm_zero_gamma: bool
    recompute: list[str] | bool = []
    """Recompute level for each block.
    Values may include: 'attention', 'attn_norm', 'feed_forward', 'ffn_norm'.
    Set True to recompute all components. Manual pp_vp_allocation overrides this attr.
    """
    tie_embedding: bool

    tp_cfg: MegatronTPConfig = MegatronTPConfig
    ffn_cfg: FeedForwardConfig = FeedForwardConfig
    attn_cfg: AttentionConfig = AttentionConfig
    tok_embed_cfg: InputEmbeddingConfig = InputEmbeddingConfig
    out_embed_cfg: OutputEmbeddingConfig = OutputEmbeddingConfig

    def pp_vp_allocation(self, abs_pp_rank: int) -> list[dict]:
        from steptronoss.utils.general import list_split

        # use list_split: [0, 1, 2], split=2 -> [0, 1], [2]
        chunk_layers = len(list_split(range(self.num_layers), get_vpp_size() * PM.size_of("PP"))[abs_pp_rank])

        return [{}] * chunk_layers

    def build_layer_map(self):
        layer_map, layer_id = dict(), 0

        for vp in range(get_vpp_size()):
            for pp in range(PM.size_of("PP")):
                layer_map.setdefault(pp, dict())
                layer_map[pp].setdefault(vp, dict())
                per_vp_layers = self.pp_vp_allocation(vp * PM.size_of("PP") + pp)
                for i in range(len(per_vp_layers)):
                    layer_map[pp][vp][layer_id] = per_vp_layers[i]
                    layer_map[pp][vp][layer_id].setdefault("recompute", self.recompute)
                    layer_id += 1
        return layer_map

    def build_model(self):
        return LlamaLikeModel(cfg=self, layer_map=self.build_layer_map())

    def sanity_check(self):
        super().sanity_check()
        if self.tie_embedding:
            if self.parallel_cfg.pipeline_model_parallel_size != 1:
                raise NotImplementedError("Tie embedding for PP > 1")


class TransformerBlock(nn.Module):
    """Base transformer block with pre-normalization.

    This is the basic building block for transformer models.
    Subclasses should override the attention and feed_forward modules.
    """

    def __init__(self, cfg: DecoderLLMConfig, layer_id: int, recompute: bool = False):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        self.recompute = recompute
        if not isinstance(self.recompute, list):
            if self.recompute is True:
                self.recompute = ["attention", "attn_norm", "feed_forward", "ffn_norm"]
            else:
                self.recompute = []
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

        self.attention = cfg.attn_cfg.build_model(layer_id=layer_id)

        # FFN
        self.feed_forward = cfg.ffn_cfg.build_model(layer_id=layer_id)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seq_len: int | None = None,
        position_id: torch.IntTensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through the transformer block."""

        if self.training and "attn_norm" in self.recompute:
            attn_in = checkpoint(self.attention_norm, self.distribute_saved_activations, x)
        else:
            attn_in = self.attention_norm(x)

        if self.training and "attention" in self.recompute:
            h = x + checkpoint(
                self.attention,
                self.distribute_saved_activations,
                attn_in,
                cu_seqlens=cu_seqlens,
                max_seq_len=max_seq_len,
                position_id=position_id,
                **kwargs,
            )
        else:
            h = x + self.attention(
                attn_in,
                cu_seqlens=cu_seqlens,
                max_seq_len=max_seq_len,
                position_id=position_id,
                **kwargs,
            )

        if self.training and "ffn_norm" in self.recompute:
            ffn_in = checkpoint(self.ffn_norm, self.distribute_saved_activations, h)
        else:
            ffn_in = self.ffn_norm(h)

        # moe should not use recompute for router, so let it handle recompute inside.
        out = h + self.feed_forward(ffn_in, recompute="feed_forward" in self.recompute)

        return out


class NoopTransformerBlock(nn.Module):
    """A no-op transformer layer used as a placeholder."""

    is_noop = True

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x.clone()


class LlamaLikeModel(MegatronModule):
    """Base LLaMA-style model with tensor and pipeline parallelism support.

    This serves as a base class for LLaMA-family models including Qwen.
    """

    def __init__(self, cfg: DecoderLLMConfig, layer_map=None):
        super().__init__()
        self.cfg = cfg

        # Build layer map if not provided
        self.layer_map = layer_map or self._build_default_layer_map()

        logger.info(format_layermap(self.layer_map), at=0)

        self.build(self.layer_map)

        self.name_parameters()

    def build(self, layer_map: dict[int, dict[int, dict[int, dict]]]):
        # Build model components
        self.layers = nn.ModuleList()
        pp_rank, vp_rank = PM.rank_in("PP"), get_vpp_rank()

        for layer, kwargs in layer_map[pp_rank][vp_rank].items():
            self.layers.append(TransformerBlock(self.cfg, layer_id=layer, **kwargs))

        if len(self.layers) == 0:
            self.layers.append(NoopTransformerBlock())

        if self.is_pipeline_first_stage():
            self.tok_embeddings = self.cfg.tok_embed_cfg.build_model()

        if self.is_pipeline_last_stage():
            if self.cfg.tie_embedding:
                self.out_embeddings = self.cfg.out_embed_cfg.build_model(self.tok_embeddings.word_embeddings.weight)
            else:
                self.out_embeddings = self.cfg.out_embed_cfg.build_model()

    def _build_default_layer_map(self) -> dict:
        """Build default layer mapping for pipeline parallelism."""
        layer_map = dict()
        pp_size = PM.size_of("PP")
        vp_size = get_vpp_size()
        local_layers = self.cfg.num_layers // pp_size // vp_size

        layer_id = 0
        for vp in range(vp_size):
            for pp in range(pp_size):
                layer_map.setdefault(pp, dict())
                layer_map[pp].setdefault(vp, dict())
                for _i in range(local_layers):
                    layer_map[pp][vp][layer_id] = dict(recompute=self.cfg.recompute)
                    layer_id += 1
        return layer_map

    def forward_head(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Process input through embedding layer."""
        if PM.size_of("CP") > 1:
            input_ids = scatter_to_balanced_cp_region(input_ids, dim=1)
        return self.tok_embeddings(input_ids=input_ids, **kwargs)

    def forward_chunk(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.IntTensor = None,
        position_id: torch.IntTensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Process through transformer layers.
        input_ids: [B, S] (B should be 1 for most case) torch.long cuda
        cu_seqlens: [N, ] torch.int32 cuda
        position_id: [S, ] torch.int32 cuda
        """
        if PM.size_of("CP") > 1:
            assert cu_seqlens is not None, "cu_seqlens required for CP, use [0, len(input_ids)] if you dont have one."
        if cu_seqlens is not None and position_id is None:
            position_id = get_position_id_from_cu_seqlens(cu_seqlens)

        if self.cfg.tp_cfg.sequence_parallel:
            rng_context = get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        with rng_context:
            for layer in self.layers:
                hidden_states = layer(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_id=position_id,
                    **kwargs,
                )
        return hidden_states

    def forward_tail(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Process through output layer."""
        logits = self.out_embeddings(hidden_states, **kwargs)
        return logits

    def name_parameters(self):
        """Assign names to parameters for logging."""
        for p in self.parameters():
            p._log_name = "other"
        for block in self.layers:
            if isinstance(block, TransformerBlock):
                for _n, p in block.named_parameters():
                    p._log_name = f"layer{block.layer_id}"
        if self.is_pipeline_first_stage():
            for p in self.tok_embeddings.parameters():
                p._log_name = "tok_embeddings"
        if self.is_pipeline_last_stage():
            for p in self.out_embeddings.parameters():
                p._log_name = "output"
