import pytest
import torch
import torch.distributed as dist

pytestmark = [
    pytest.mark.xdist_group("torchrun"),
    pytest.mark.gpu,
    pytest.mark.node2,
]


@pytest.fixture(scope="session", autouse=True)
def init_dist():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for qwen3_8b distributed tests")
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
            "You can run with `torchrun --nproc-per-node=2 -m pytest tests/test_qwen3_8b.py`."
        )

    from steptronoss.core.parallel_state import PM

    PM.initialize(backend="nccl")
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def _set_parallel(tp: int):
    from steptronoss.core.parallel_state import PM
    from steptronoss.exp.base_exp import ParallelConfig

    parallel_cfg = ParallelConfig()
    parallel_cfg.tensor_model_parallel_size = tp
    parallel_cfg.pipeline_model_parallel_size = 1
    parallel_cfg.context_parallel_size = 1
    parallel_cfg.expert_model_parallel_size = 1
    parallel_cfg.expert_tensor_parallel_size = 1
    parallel_cfg.virtual_pipeline_model_parallel_size = 1
    PM.set_mesh(parallel_cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="qwen3_8b test requires CUDA")
@pytest.mark.skipif(not torch.cuda.is_bf16_supported(), reason="bf16 not supported")
@pytest.mark.parametrize("tp", [1, 2])
def test_qwen3_8b_embed_1block_hardcoded_backward_reference(tp):
    from playground.pretrain.qwen3.qwen3_8 import Qwen3_8BConfig
    from steptronoss.core.parallel_state import PM
    from steptronoss.exp.base_exp import MegatronTPConfig
    from steptronoss.initialize import set_mpu_random_seed

    _set_parallel(tp=tp)
    set_mpu_random_seed(1234)

    model_cfg = Qwen3_8BConfig()
    model_cfg.num_layers = 1
    model_cfg.parallel_cfg.tensor_model_parallel_size = tp
    tp_cfg = MegatronTPConfig()
    tp_cfg.params_dtype = torch.bfloat16
    tp_cfg.sequence_parallel = False
    tp_cfg.async_tensor_model_parallel_allreduce = False
    tp_cfg.gradient_accumulation_fusion = False
    model_cfg.tp_cfg = tp_cfg

    module = model_cfg.build_model().cuda().bfloat16()

    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.ndim == 1:
                parameter.fill_(1.0)
            else:
                parameter.fill_(1e-2)
            parameter.requires_grad_(False)

        module.tok_embeddings.word_embeddings.weight[1].fill_(0.75)
        module.tok_embeddings.word_embeddings.weight[2].fill_(-0.5)
        module.tok_embeddings.word_embeddings.weight[1, 0] = 1.25
        module.tok_embeddings.word_embeddings.weight[2, 0] = -1.5

        module.layers[0].attention.wqkv.weight.fill_(5e-2)
        module.layers[0].attention.wo.weight.fill_(-4e-2)
        module.layers[0].feed_forward.w1.weight.fill_(3e-2)
        module.layers[0].feed_forward.w2.weight.fill_(-2e-2)

        module.out_embeddings.output.weight.fill_(1e-1)
        module.out_embeddings.output.weight[0].fill_(2e-1)
        module.out_embeddings.output.weight[1].fill_(-2e-1)
    module.out_embeddings.output.weight.requires_grad_(True)

    input_ids = torch.tensor(
        [[1, 2]],
        dtype=torch.long,
        device="cuda",
    )

    y = module.abstract_forward(input_ids=input_ids)
    y.sum().backward()

    # assert len(module.layers) == 1
    assert y.shape == (2, 1, model_cfg.out_embed_cfg.vocab_size // PM.size_of("TP"))
    assert torch.isfinite(y).all().item()
    assert y[0, 0, 0].item() == -820.0
    assert y[0, 0, 1].item() == 820.0
    assert y[1, 0, 0].item() == -820.0
    assert y[1, 0, 1].item() == 820.0

    first_grad_00 = module.out_embeddings.output.weight.grad[0, 0].item()
    first_grad_10 = module.out_embeddings.output.weight.grad[1, 0].item()
    assert first_grad_00 == -2.0
    assert first_grad_10 == -2.0

    y = module.abstract_forward(input_ids=input_ids)
    y.sum().backward()

    second_grad_00 = module.out_embeddings.output.weight.grad[0, 0].item()
    second_grad_10 = module.out_embeddings.output.weight.grad[1, 0].item()
    assert second_grad_00 == -4.0
    assert second_grad_10 == -4.0
