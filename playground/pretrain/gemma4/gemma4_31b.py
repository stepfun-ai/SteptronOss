import torch
from configurize import Ref
from torch.nn import functional as F

from steptronoss.exp.base_exp import ParallelConfig
from steptronoss.model.common.feed_forward import FeedForwardConfig
from steptronoss.model.common.grouped_query_attention import AttentionConfig
from steptronoss.model.common.parallel_embedding import InputEmbeddingConfig, OutputEmbeddingConfig
from steptronoss.model.decoder_model import DecoderLLMConfig


class Gemma4AttentionConfig(AttentionConfig):
    """Gemma4 31B attention configuration."""

    num_global_key_value_heads: int
    global_head_dim: int
    full_rope_theta: float
    full_rope_factor: float
    full_rotary_factor: float
    attention_k_eq_v: bool
    layer_types: list[str]

    def __init__(self):
        super().__init__()
        self.causal = True
        self.attention_dropout = 0.0

        self.use_sliding_window = True
        self.sliding_window_size = 1024

        self.num_attention_heads = 32
        self.num_attention_groups = 16
        self.num_global_key_value_heads = 4

        self.head_dim = 256
        self.global_head_dim = 512
        self.hidden_size = Ref("..hidden_size")

        self.use_headwise_attn_gate = False
        self.use_qkv_bias = False
        self.use_qk_norm = False
        self.layernorm_epsilon = 1e-6
        self.rms_norm_zero_gamma = False
        self.recompute_qknorm_rope = False

        self.qk_rope_head_dim = None
        self.rope_theta = 10_000.0
        self.full_rope_theta = 1_000_000.0
        self.full_rope_factor = 1.0
        self.full_rotary_factor = 0.25
        self.yarn_beta_slow = 1.0
        self.yarn_beta_fast = 32.0
        self.ntk_interp_ratio = 1.0
        self.max_position_embeddings = 262144

        self.attention_k_eq_v = True
        self.layer_types = ["sliding_attention" if (layer_id + 1) % 6 else "full_attention" for layer_id in range(60)]

    def build_model(self, layer_id: int):
        from steptronoss.model.gemma4 import Gemma4Attention

        return Gemma4Attention(cfg=self, layer_id=layer_id)


class Gemma4FeedForwardConfig(FeedForwardConfig):
    """Gemma4 31B feed-forward configuration."""

    def __init__(self):
        super().__init__()
        self.hidden_size = Ref("..hidden_size")
        self.ffn_hidden_size = 21504

        self.layernorm_epsilon = 1e-6
        self.rms_norm_zero_gamma = False
        self.swiglu_recompute_silu_out_proj = False
        self.row_parallel_fp32_output_when_tp = True
        self.cast_output_to_input_dtype = True

    def activation(self, x, swiglu_limit=None):
        del swiglu_limit
        gate, up = torch.chunk(x, 2, dim=-1)
        return F.gelu(gate, approximate="tanh") * up


class Gemma4InputEmbeddingConfig(InputEmbeddingConfig):
    """Gemma4 31B input embedding configuration."""

    def __init__(self):
        super().__init__()
        self.vocab_size = 262144
        self.hidden_size = Ref("..hidden_size")
        self.embedding_weights_in_fp32 = False
        self.fp32_residual_connection = False

    def build_model(self):
        from steptronoss.model.gemma4 import GemmaScaledWordEmbedding

        return GemmaScaledWordEmbedding(cfg=self)


class Gemma4OutputEmbeddingConfig(OutputEmbeddingConfig):
    """Gemma4 31B output embedding configuration."""

    final_logit_softcapping: float | None

    def __init__(self):
        super().__init__()
        self.vocab_size = 262144
        self.hidden_size = Ref("..hidden_size")
        self.fp32_rms_norm = True

        self.rms_norm_zero_gamma = False
        self.layernorm_epsilon = 1e-6
        self.gather_output = False
        self.rms_norm_math_mode = "pow"
        self.rms_norm_cast_output_to_input_after_mul = True
        self.final_logit_softcapping = 30.0


class Gemma4ParallelConfig(ParallelConfig):
    """Gemma4 31B parallelism configuration."""

    def __init__(self):
        super().__init__()
        self.tensor_model_parallel_size = 4
        self.pipeline_model_parallel_size = 1
        self.virtual_pipeline_model_parallel_size = 1
        self.context_parallel_size = 1
        self.expert_model_parallel_size = 1
        self.expert_tensor_parallel_size = 1


class Gemma4_31BConfig(DecoderLLMConfig):
    """Gemma4 31B text-backbone training configuration."""

    ffn_cfg = Gemma4FeedForwardConfig
    attn_cfg = Gemma4AttentionConfig
    tok_embed_cfg = Gemma4InputEmbeddingConfig
    out_embed_cfg = Gemma4OutputEmbeddingConfig
    parallel_cfg = Gemma4ParallelConfig

    def __init__(self):
        super().__init__()
        self.num_layers = 60
        self.hidden_size = 5376
        self.layernorm_epsilon = 1e-6
        self.rms_norm_zero_gamma = False
        self.recompute = False
        self.tie_embedding = True

        self.params_dtype = torch.bfloat16

        self.variable_seq_lengths = True
        self.tp_cfg.sequence_parallel = True
        self.tp_cfg.async_tensor_model_parallel_allreduce = False

    def validate_attention_topology(self):
        if len(self.attn_cfg.layer_types) != self.num_layers:
            raise ValueError(
                f"Gemma4 layer_types has {len(self.attn_cfg.layer_types)} entries, expected {self.num_layers}."
            )
        if self.parallel_cfg.tensor_model_parallel_size > self.attn_cfg.num_global_key_value_heads:
            raise ValueError(
                "Gemma4 31B full-attention layers expose only 4 KV heads, so the current TP path requires "
                "tensor_model_parallel_size <= 4."
            )

    def sanity_check(self):
        super().sanity_check()
        self.validate_attention_topology()

    def build_model(self):
        from steptronoss.model.gemma4 import Gemma4Model

        self.validate_attention_topology()
        return Gemma4Model(cfg=self, layer_map=self.build_layer_map())
