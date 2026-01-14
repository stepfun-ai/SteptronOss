import torch

from steptronoss.model.decoder_model import (
    DecoderLLMConfig,
)
from steptronoss.model.common.feed_forward import FeedForwardConfig
from steptronoss.model.common.grouped_query_attention import AttentionConfig
from steptronoss.model.common.parallel_embedding import (
    InputEmbeddingConfig,
    OutputEmbeddingConfig,
    OutputEmbedding,
)
from steptronoss.exp.base_exp import ParallelConfig, BaseExp


class Qwen3AttentionConfig(AttentionConfig):
    """Qwen3 attention configuration."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.causal = True
        self.attention_dropout = 0.0

        self.use_sliding_window = False
        self.num_sliding_attention_heads = None

        self.num_attention_heads = 16
        self.num_attention_groups = 8  # KV heads for GQA

        self.head_dim = 128
        self.hidden_size = 2048

        self.use_headwise_attn_gate = False
        self.use_qkv_bias = False

        self.sliding_window_size = -1

        self.use_qk_norm = False  # Inverted in module: False here enables QK norm
        self.layernorm_epsilon = 1e-6
        self.rms_norm_zero_gamma = False

        self.recompute_qknorm_rope = False

        self.qk_rope_head_dim = None
        self.rope_theta = 1_000_000.0
        self.yarn_beta_slow = 1.0
        self.yarn_beta_fast = 32.0
        self.ntk_interp_ratio = 1.0
        self.max_position_embeddings = None


class Qwen3FeedForwardConfig(FeedForwardConfig):
    """Qwen3 feed-forward configuration."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.recompute_granularity = None

        self.hidden_size = 2048
        self.ffn_hidden_size = 6144

        self.layernorm_epsilon = 1e-6
        self.rms_norm_zero_gamma = False

        self.swiglu_limit = None
        self.swiglu_recompute_silu_out_proj = True


class Qwen3InputEmbeddingConfig(InputEmbeddingConfig):
    """Qwen3 input embedding configuration."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = 151936
        self.hidden_size = 2048
        self.embedding_weights_in_fp32 = False
        self.fp32_residual_connection = False


class Qwen3OutputEmbeddingConfig(OutputEmbeddingConfig):
    """Qwen3 output embedding configuration."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = 151936
        self.hidden_size = 2048
        self.fp32_rms_norm = True

        self.rms_norm_zero_gamma = False
        self.layernorm_epsilon = 1e-6

        self.gather_output = False


class Qwen3ParallelConfig(ParallelConfig):
    """Qwen3 parallelism configuration."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tensor_model_parallel_size = 2
        self.pipeline_model_parallel_size = 1
        self.virtual_pipeline_model_parallel_size = 1
        self.context_parallel_size = 1
        self.expert_model_parallel_size = 1
        self.expert_tensor_parallel_size = 1


class Qwen3_1p7BConfig(DecoderLLMConfig):
    """Qwen3 1.7B model configuration.

    Model architecture parameters based on Qwen3-1.7B:
    - 28 layers
    - 2048 hidden size
    - 16 attention heads with 8 KV heads (GQA)
    - 6144 FFN hidden size
    - 128 head dimension
    - 151936 vocab size
    """

    ffn_cfg = Qwen3FeedForwardConfig
    attn_cfg = Qwen3AttentionConfig
    tok_embed_cfg = Qwen3InputEmbeddingConfig
    out_embed_cfg = Qwen3OutputEmbeddingConfig
    parallel_cfg = Qwen3ParallelConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Model architecture
        self.num_layers = 28
        self.hidden_size = 2048
        self.layernorm_epsilon = 1e-6
        self.rms_norm_zero_gamma = False
        self.recompute_full = False

        # Precision
        self.params_dtype = torch.bfloat16

        # Other settings
        self.sequence_parallel = False
        self.variable_seq_lengths = True


from steptronoss.exp.base_exp import GradientManagerConfig, TrainerConfig

class Exp(BaseExp):
    model_cfg = Qwen3_1p7BConfig
    grad_manager_cfg = GradientManagerConfig
    trainer_cfg = TrainerConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grad_manager_cfg.optimizer_cfg.lr = 1
        self.grad_manager_cfg.optimizer_cfg.weight_decay = 0.9
        self.grad_manager_cfg.clip_grad = 0.0

        self.grad_manager_cfg.use_distributed_optimizer = False
        self.grad_manager_cfg.params_dtype = torch.bfloat16

    

# import sys, time
# sys.excepthook = lambda a,b,c:time.sleep(3600)

if __name__ == "__main__":
    from steptronoss.core.parallel_state import PM
    from steptronoss.initialize import set_mpu_random_seed
    from steptronoss.utils.logger import setup_logger

    logger = setup_logger('./tensorboard_dir/')
    exp = Exp()
    logger.info(exp)

    PM.initialize()
    PM.set_mesh(exp.model_cfg.parallel_cfg)
    set_mpu_random_seed(1234)

    model = exp.model_cfg.build_model()
    for p in model.parameters():
        from torch.nn.init import trunc_normal_
        trunc_normal_(p, mean=0.0, std=1.0)
    # from steptron import debug;debug()
    from steptronoss.model.module import Float16Module

    model = Float16Module(model, dtype=exp.model_cfg.params_dtype).cuda()
    gm = exp.grad_manager_cfg.build_gradient_manager(model)



    x = torch.arange(1024, dtype=torch.long, device="cuda").reshape(1, -1)
    cu_seqlens = torch.tensor([0, 1024], dtype=torch.int32, device='cuda')

    out = model(input_ids=x, cu_seqlens=cu_seqlens)
    loss = out.sum()
    loss.backward()
    
    success, grad_norm, grad_zeros = gm.step()
    gm.zero_grad()
    logger.warning(out.shape)
    logger.warning(model.module.tok_embeddings.word_embeddings.weight)
