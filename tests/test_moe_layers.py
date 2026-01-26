import pytest
import torch
import torch.distributed as dist

pytestmark = [
    pytest.mark.xdist_group("torchrun"),
    pytest.mark.gpu,
    pytest.mark.node2,
]
ops = pytest.importorskip("grouped_gemm.ops")


@pytest.fixture(scope="session", autouse=True)
def init_dist():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for distributed MoE tests")
        return
    if not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 not supported")
        return

    try:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    except Exception as e:
        print(f"Failed to initialize torch.distributed: {e}")
        pytest.skip("Failed to initialize torch.distributed")
        return

    if dist.get_world_size() != 2:
        pytest.skip(
            "Need 2 processes in dist group. "
            "You can run with `torchrun --nproc-per-node=2 -m pytest tests/test_moe_layers.py`."
        )

    from steptronoss.core.parallel_state import PM

    PM.initialize(backend="nccl")
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def _set_parallel(tp: int, ep: int):
    from steptronoss.core.parallel_state import PM
    from steptronoss.exp.base_exp import ParallelConfig

    parallel_cfg = ParallelConfig()
    parallel_cfg.tensor_model_parallel_size = tp
    parallel_cfg.pipeline_model_parallel_size = 1
    parallel_cfg.context_parallel_size = 1
    parallel_cfg.expert_model_parallel_size = ep
    parallel_cfg.expert_tensor_parallel_size = 1
    parallel_cfg.virtual_pipeline_model_parallel_size = 1
    PM.set_mesh(parallel_cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="moe_layers requires CUDA")
@pytest.mark.skipif(not torch.cuda.is_bf16_supported(), reason="bf16 not supported")
@pytest.mark.parametrize("tp,ep", [(1, 1), (1, 2), (2, 1)])
def test_moe_share_expert_ffn_backward_reference(monkeypatch, tp, ep):
    import steptronoss.model.common.moe_block as moe_block
    from playground.pretrain.qwen3.qwen3_1p7b import Qwen3FeedForwardConfig
    from steptronoss.core.parallel_state import PM
    from steptronoss.exp.base_exp import GradientManagerConfig, MegatronTPConfig
    from steptronoss.initialize import set_mpu_random_seed
    from steptronoss.model.common.moe_share_expert_ffn import MoeShareExpertFFN

    _set_parallel(tp=tp, ep=ep)

    class _DummyMetric:
        def add(self, *args, **kwargs):
            del args
            del kwargs

    class _DummyGlobalMetrics:
        def __getattr__(self, name):
            del name
            return _DummyMetric()

    monkeypatch.setattr(moe_block, "GlobalMetrics", _DummyGlobalMetrics())
    import steptronoss.model.common.moe_share_expert_ffn as moe_share_expert_ffn

    monkeypatch.setattr(moe_share_expert_ffn, "GlobalMetrics", _DummyGlobalMetrics())

    ffn_cfg = Qwen3FeedForwardConfig()
    tp_cfg = MegatronTPConfig()
    tp_cfg.params_dtype = torch.bfloat16
    tp_cfg.sequence_parallel = True
    tp_cfg.async_tensor_model_parallel_allreduce = False
    ffn_cfg.tp_cfg = tp_cfg

    moe_cfg = moe_block.MoEConfig()
    moe_cfg.tp_cfg = ffn_cfg.tp_cfg
    moe_cfg.hidden_size = ffn_cfg.hidden_size
    moe_cfg.activation = ffn_cfg.activation
    moe_cfg.moe_num_experts = 8
    moe_cfg.moe_top_k = 2
    moe_cfg.moe_aux_loss_coef = 0.1
    moe_cfg.moe_hidden_size = 1024
    moe_cfg.routed_scaling_factor = 2.0
    moe_cfg.enable_sigmoid_router = False
    moe_cfg.router_bias_update_rate = 0.01
    moe_cfg.moe_enable_deepep = True
    moe_cfg.moe_deepep_num_sms = 8
    moe_cfg.fuse_moescatter_and_moecolumn = False
    moe_cfg.enable_auxiliary_loss_free_load_balance = True
    moe_cfg.norm_expert_weight = False
    moe_cfg.moe_layer_list = [0]
    moe_cfg.share_expert_dim = 512

    set_mpu_random_seed(1234)
    module = MoeShareExpertFFN(
        moe_cfg=moe_cfg,
        ffn_cfg=ffn_cfg,
        layer_id=0,
    )
    with torch.no_grad():
        torch.nn.init.trunc_normal_(module.moe.gate.weight)
        module.share_expert.w1.weight.fill_(0.1)
        module.share_expert.w2.weight.fill_(0.2)
        for eid in range(len(module.moe.w1)):
            value = (eid + 1 + PM.rank_in("EP") * 4) / 10
            module.moe.w1[eid].fill_(value)
            module.moe.w2[eid].fill_(value)

    module = module.cuda().bfloat16()

    grad_cfg = GradientManagerConfig()
    grad_cfg.optimizer_cfg.lr = 1
    grad_cfg.optimizer_cfg.weight_decay = 0.9
    grad_cfg.clip_grad = 0.0
    grad_cfg.params_dtype = torch.bfloat16
    grad_cfg.use_distributed_optimizer = True
    grad_manager = grad_cfg.build_gradient_manager(model=module)

    x = torch.randn(
        (16, 1, ffn_cfg.hidden_size),
        dtype=torch.bfloat16,
        device="cuda",
    )
    x = torch.chunk(x, PM.size_of("TP"))[PM.rank_in("TP")]
    x = torch.chunk(x, PM.size_of("EP"))[PM.rank_in("EP")]

    y = module(x)
    y.sum().backward()
    assert y.shape == x.shape
    if PM.world_rank == 0:
        assert y[0, 0, 0] == 1664
    if PM.world_rank == 1:
        assert y[-1, 0, -1] == 9.3125
    if PM.world_rank == 0:
        assert module.moe.w1.main_grad[0, 0, 0].item() == 1888
    if PM.size_of("EP") == 2:
        if PM.world_rank == 1:
            assert module.moe.w1.main_grad[0, 0, 0].item() == 2768
    else:
        if PM.world_rank == 1:
            assert module.moe.w1.main_grad[0, 0, 0].item() == 1888

    y = module(x)
    y.sum().backward()
    if PM.world_rank == 0:
        assert module.moe.w1.main_grad[0, 0, 0].item() == 3776
    if PM.size_of("EP") == 2:
        if PM.world_rank == 1:
            assert module.moe.w1.main_grad[0, 0, 0].item() == 5536
    else:
        if PM.world_rank == 1:
            assert module.moe.w1.main_grad[0, 0, 0].item() == 3776
