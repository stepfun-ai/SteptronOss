"""
Toy setting for step3.5 flash, same feature, toy size.
"""

from playground.pretrain.step3p5.step3p5_flash import Step3p5FlashModelConfig

"""
MiniModelConfig(
    DDP_impl = 'local',
    accumulate_allreduce_grads_in_fp32 = True,
    actual_vocab_size = 128815 🈯 Ref(..tokenizer_cfg.vocab_size),
    async_tensor_model_parallel_allreduce = False,
    attention_dropout = 0.0,
    attention_type = 'gqa',
    build_layer_map = 'Func()',
    build_model = 'Func()',
    check_nan = True,
    context_parallel_load_balanced = False,
    context_parallel_size = 1 🈯 Ref(.parallel_cfg.context_parallel_size),
    continuous_memory_ffn = False,
    critical_keys = ['hidden_size', 'ffn_hidden_size', 'rope_theta'],
    data_parallel_size = 0,
    detect_abnormal_data = False,
    disable_qk_norm = False,
    distribute_saved_activations = False,
    embedding_weights_in_fp32 = False,
    expert_model_parallel_size = 1 🈯 Ref(.parallel_cfg.expert_model_parallel_size),
    expert_tensor_parallel_size = 1,
    ffn_hidden_size = 7168,
    fp32_lm_head_out = True,
    fp32_residual_connection = False,
    fp32_rms_norm = True,
    fp8_attention = False,
    fp8_qkv_log_scale_QEM_lambda = None,
    fp8_qkv_log_scale_step_clip = None,
    fp8_qkv_scale_grad_clip = None,
    fp8_qkv_scale_grad_factor = 32.0,
    fp8_qkv_scale_init = 1.0,
    gather_output = False,
    get_tp_kwargs = 'Func()',
    global_seq_length = 8192 🈯 Ref(..trainer_cfg.global_seq_length),
    gradient_accumulation_fusion = True,
    head_dim = 128,
    hidden_size = 2048,
    init_cfg = playground.pretrain.step4mini_base.Step4ModelInitConfig(
        do_not_rescale_params_by_depth = True,
        init_method_std = 0.06,
    ),
    layer_types = ['sliding_attention', 'sliding_attention', 'sliding_attention',
     'full_attention', 'sliding_attention', 'sliding_attention',
     'sliding_attention', 'full_attention', 'sliding_attention',
     'sliding_attention', 'sliding_attention', 'full_attention',
     'sliding_attention', 'sliding_attention', 'sliding_attention',
     'full_attention', 'sliding_attention', 'sliding_attention',
     'sliding_attention', 'full_attention', 'sliding_attention',
     'sliding_attention', 'sliding_attention', 'full_attention',
     'sliding_attention', 'sliding_attention', 'sliding_attention',
     'full_attention', 'sliding_attention', 'sliding_attention',
     'sliding_attention', 'full_attention', 'sliding_attention',
     'sliding_attention', 'sliding_attention', 'full_attention',
     'sliding_attention', 'sliding_attention', 'sliding_attention',
     'full_attention', 'sliding_attention', 'sliding_attention',
     'sliding_attention', 'full_attention', 'sliding_attention',
     'sliding_attention', 'sliding_attention', 'full_attention', 'full_attention'],
    layernorm_epsilon = 1e-05,
    log_attn_gate = False,
    log_attn_maximum_logits = False,
    loss_func = 'Func(data: dict, logits: torch.Tensor)',
    max_position_embeddings = 8192,
    mfa_kv_channels = None,
    mfa_q_channels = None 🈯 Ref(.mfa_kv_channels),
    mfa_use_inter_norm = False,
    micro_batch_size = 1 🈯 Ref(..trainer_cfg.micro_batch_size),
    model_type = 1,
    moe = playground.pretrain.step4mini_base.Step4MiniMoEConfig(
        moe_hidden_size = 768,
        moe_layer_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
         23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
         43, 44, 45, 46, 47, 48],
        moe_num_experts = 128,
        moe_top_k = 8,
        share_expert_dim = 768,
    ),
    mtp_cfg = playground.pretrain.step4mini_base.Step4MiniMTPConfig(
        mtp_loss_weight = 0.3,
        num_mtp_layers = 1,
        use_mtpv2 = False,
        use_optimus_cross_entropy = True,
    ),
    ntk_interp_ratio = 1.0,
    num_attention_groups = 8,
    num_attention_heads = 32,
    num_layers = 48,
    sliding_window_size = 512,
"""


class Step3p5ToyModelConfig(Step3p5FlashModelConfig):
    def __init__(self):
        super().__init__()

        self.hidden_size = 2048
        self.num_layers = 48
        self.swa_layer_list = [True, True, True, False] * 12

        self.attn_cfg.num_attention_heads = 32
        self.swa_cfg.num_attention_heads = 48

        self.ffn_cfg.moe_cfg.moe_num_experts = 128
        self.ffn_cfg.moe_cfg.moe_hidden_size = 768
        self.ffn_cfg.moe_cfg.share_expert_dim = 768
        self.ffn_cfg.moe_cfg.moe_layer_list = list(range(1, 48))
        self.ffn_cfg.moe_cfg.enable_auxiliary_loss_free_load_balance = True
        self.ffn_cfg.moe_cfg.router_bias_update_rate = 0.1

        self.ffn_cfg.ffn_hidden_size = 7168

        self.parallel_cfg.tensor_model_parallel_size = 8
        self.parallel_cfg.pipeline_model_parallel_size = 1
        self.parallel_cfg.virtual_pipeline_model_parallel_size = 1
        self.parallel_cfg.context_parallel_size = 1
        self.parallel_cfg.expert_model_parallel_size = 8
        self.parallel_cfg.expert_tensor_parallel_size = 1

        self.tp_cfg.sequence_parallel = True

    def build_model(self):
        model = super().build_model()
        for p in model.parameters():
            from torch.nn.init import trunc_normal_

            trunc_normal_(p)
            p.has_initialized = True
        return model
