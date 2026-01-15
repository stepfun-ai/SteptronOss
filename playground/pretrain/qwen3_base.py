import torch

from steptronoss.exp.base_exp import BaseExp, ParallelConfig
from steptronoss.model.common.feed_forward import FeedForwardConfig
from steptronoss.model.common.grouped_query_attention import AttentionConfig
from steptronoss.model.common.parallel_embedding import (
    InputEmbeddingConfig,
    OutputEmbeddingConfig,
)
from steptronoss.model.decoder_model import (
    DecoderLLMConfig,
)


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
        self.tie_embedding = True

        # Precision
        self.params_dtype = torch.bfloat16

        # Other settings
        self.variable_seq_lengths = True

    def build_model(self):
        from steptronoss.model.qwen_dense import QwenModel

        return QwenModel(cfg=self, layer_map=self.build_layermap())


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

        self.grad_manager_cfg.use_distributed_optimizer = True
        self.grad_manager_cfg.params_dtype = torch.bfloat16
        self.model_cfg.overlap_p2p_comm = True


# import sys, time
# sys.excepthook = lambda a,b,c:time.sleep(3600)

if __name__ == "__main__":
    from steptronoss.core.parallel_state import PM, get_vpp_size, set_vpp_rank
    from steptronoss.initialize import set_mpu_random_seed
    from steptronoss.utils import print_n_params, profile_allreduce
    from steptronoss.utils.logger import setup_logger
    from steptronoss.utils.weight_loader import HFWeights

    logger = setup_logger("./tensorboard_dir/")
    exp = Exp()
    logger.info(exp)

    PM.initialize()
    PM.set_mesh(exp.model_cfg.parallel_cfg)
    set_mpu_random_seed(1234)

    models = []
    for i in range(get_vpp_size()):
        set_vpp_rank(i)
        model = exp.model_cfg.build_model()
        model.load_hf_state_dict(
            HFWeights("/mnt/step2-alignment-jfs/zane/opensources_model/Qwen3-1.7B-Base/"), strict=False
        )
        # from steptron import debug;debug()
        from steptronoss.model.module import Float16Module

        model = Float16Module(model, dtype=exp.model_cfg.params_dtype).cuda()
        models.append(model)
    print_n_params(models)

    gms = [exp.grad_manager_cfg.build_gradient_manager(model) for model in models]

    x = torch.arange(1024, dtype=torch.long, device="cuda").reshape(1, -1)
    cu_seqlens = torch.tensor([0, 1024], dtype=torch.int32, device="cuda")
    data = dict(input_ids=x, cu_seqlens=cu_seqlens)

    profile_allreduce()

    pp_scheduler = exp.model_cfg.get_pp_scheduler()
    pp_scheduler.configure(
        models=models,
        data_iterators=[iter([data] * 100)] * get_vpp_size(),
        data_sync_fn=exp.trainer_cfg.sync_get_data,
        loss_fn=lambda data, logits: logits.sum(),
        data_proc_fn=lambda x: x,
        training=True,
        collect_output=False,
    )

    out = pp_scheduler.run(forward_num=16)
    if PM.i_am("PP", 0):
        logger.warning(models[0].module.tok_embeddings.word_embeddings.weight)

    for gm in gms:
        success, grad_norm, grad_zeros = gm.step()
        gm.zero_grad()
    if PM.i_am("PP", 0):
        logger.warning(models[0].module.tok_embeddings.word_embeddings.weight)
