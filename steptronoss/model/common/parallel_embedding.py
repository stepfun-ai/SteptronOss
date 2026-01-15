from __future__ import annotations

from typing import Optional

import torch
from configurize import Config, Ref

from steptronoss.core import tensor_parallel
from steptronoss.exp.base_exp import MegatronTPConfig
from steptronoss.model.common.rms_norm import RMSNorm


class InputEmbeddingConfig(Config):
    tp_cfg: MegatronTPConfig = Ref("..tp_cfg")
    vocab_size: int
    hidden_size: int
    embedding_weights_in_fp32: bool
    fp32_residual_connection: bool

    def build_model(self):
        return WordEmbedding(cfg=self)


class OutputEmbeddingConfig(Config):
    tp_cfg: MegatronTPConfig = Ref("..tp_cfg")
    vocab_size: int
    hidden_size: int
    fp32_rms_norm: bool

    rms_norm_zero_gamma: bool
    layernorm_epsilon: float

    gather_output: bool

    def build_model(self, tied_embedding_weight: Optional[torch.Tensor] = None):
        return OutputEmbedding(cfg=self, tied_word_embedding_weight=tied_embedding_weight)


class WordEmbedding(torch.nn.Module):
    """Language model embeddings."""

    def __init__(self, cfg: InputEmbeddingConfig):
        super().__init__()

        self.hidden_size = cfg.hidden_size

        # Word embeddings (parallel).
        self.embedding_weights_in_fp32 = cfg.embedding_weights_in_fp32
        self.params_dtype = cfg.tp_cfg.params_dtype

        self.word_embeddings = tensor_parallel.SimpleVocabParallelEmbedding(
            cfg.vocab_size, self.hidden_size, params_dtype=cfg.tp_cfg.params_dtype
        )

        self.fp32_residual_connection = cfg.fp32_residual_connection
        self.sequence_parallel = cfg.tp_cfg.sequence_parallel

    def forward(self, input_ids, **kwargs):
        # Embeddings.
        input_ids = input_ids.contiguous()
        if self.embedding_weights_in_fp32:
            self.word_embeddings = self.word_embeddings.to(torch.float32)
        embeddings = self.word_embeddings(input_ids)
        if self.embedding_weights_in_fp32:
            embeddings = embeddings.to(self.params_dtype)
            self.word_embeddings = self.word_embeddings.to(self.params_dtype)

        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        embeddings = embeddings.transpose(0, 1).contiguous()

        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()

        # Dropout.
        if self.sequence_parallel:
            embeddings = tensor_parallel.scatter_to_sequence_parallel_region(embeddings)

        return embeddings


class OutputEmbedding(torch.nn.Module):
    def __init__(
        self,
        cfg: OutputEmbeddingConfig,
        tied_word_embedding_weight: Optional[torch.Tensor],
    ):
        super().__init__()
        self.norm = RMSNorm(
            cfg.hidden_size,
            eps=cfg.layernorm_epsilon,
            sequence_parallel=cfg.tp_cfg.sequence_parallel,
            use_fp32=cfg.fp32_rms_norm,
            use_zero_init=cfg.rms_norm_zero_gamma,
        )
        self.output = tensor_parallel.ColumnParallelLinear(
            cfg.hidden_size,
            cfg.vocab_size,
            bias=False,
            gather_output=cfg.gather_output,
            async_tensor_model_parallel_allreduce=cfg.tp_cfg.async_tensor_model_parallel_allreduce,
            params_dtype=cfg.tp_cfg.params_dtype,
            gradient_accumulation_fusion=cfg.tp_cfg.gradient_accumulation_fusion,
            sequence_parallel_enabled=cfg.tp_cfg.sequence_parallel,
            tie_word_embeddings_weight=tied_word_embedding_weight,
        )

    def forward(self, hidden_states, **kwargs):
        hidden_states = self.norm(hidden_states)
        output = self.output(hidden_states)[0]
        return output
