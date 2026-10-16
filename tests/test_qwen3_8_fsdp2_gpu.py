"""Real two-GPU FSDP2 trainer, rank-zero/meta initialization and exact resume.

Run with torchrun --standalone --nproc-per-node=2 -m pytest -q
tests/test_qwen3_8_fsdp2_gpu.py -m node2. Not executed in CPU-only CI.
"""

import gc
import hashlib
import json
import tempfile
from pathlib import Path

import pytest
import torch
from torch.distributed.tensor import DTensor

from playground.sft.qwen3_8.qwen3_8_27b_fsdp2_lora import Exp
from steptronoss.core.parallel_state import PM
from steptronoss.core.trainers.qwen3_8_fsdp2_trainer import Qwen38FSDP2Trainer
from tests.test_qwen3_8_sft import tiny_base


def full_adapters(model):
    return {
        name: (p.full_tensor() if isinstance(p, DTensor) else p.detach()).cpu()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


@pytest.mark.gpu
@pytest.mark.node2
@pytest.mark.xdist_group("torchrun")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPUs required")
def test_two_gpu_fsdp2_matches_uninterrupted_and_exports_adapter():
    PM.initialize(backend="nccl")
    shared = [tempfile.mkdtemp(prefix="qwen38-fsdp2-smoke-") if PM.world_rank == 0 else None]
    torch.distributed.broadcast_object_list(shared, src=0)
    root = Path(shared[0])
    if PM.world_rank == 0:
        torch.manual_seed(1234)
        tiny_base().save_pretrained(root / "base")
        records = [{"input_ids": [3, 4, 5, i % 20 + 6], "labels": [-100, -100, 5, i % 20 + 6]} for i in range(32)]
        (root / "tokens.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
        (root / "tokens.jsonl.meta.json").write_text(
            json.dumps({
                "format": "steptron-qwen38-text-v1",
                "model": str(root / "base"),
                "revision": None,
                "max_length": 16,
                "pad_token_id": 0,
                "sha256": hashlib.sha256((root / "tokens.jsonl").read_bytes()).hexdigest(),
            })
        )
    torch.distributed.barrier()

    def experiment(iters, output):
        exp = Exp()
        exp.resource_cfg.gpu = 2
        exp.model_cfg.model_path = str(root / "base")
        exp.model_cfg.hidden_size = 32
        exp.model_cfg.lora_rank = 2
        exp.model_cfg.lora_alpha = 4
        exp.model_cfg.attn_implementation = "sdpa"  # Tiny GPU test needs no optional kernels.
        exp.model_cfg.loss_chunk_size = 2
        exp.data_cfg.tokenized_path = str(root / "tokens.jsonl")
        exp.data_cfg.pad_token_id = 0
        exp.trainer_cfg.global_seq_length = 16
        exp.trainer_cfg.global_batch_size = 4
        exp.trainer_cfg.train_iters = iters
        exp.scheduler_cfg.total_schedule = 10
        exp.scheduler_cfg.warmup_schedule = 0
        exp.checkpoint_cfg.save_dir = str(root / output)
        exp.log_dir = str(root / "logs" / output)
        exp.sanity_check()
        return exp

    reference = Qwen38FSDP2Trainer(experiment(4, "reference"))
    reference.train()
    expected = full_adapters(reference.models[0])
    del reference
    gc.collect()
    torch.cuda.empty_cache()

    first = Qwen38FSDP2Trainer(experiment(2, "resume"))
    first.train()
    after_two = full_adapters(first.models[0])
    for param in first.models.parameters():
        assert isinstance(param, DTensor)
    del first
    gc.collect()
    torch.cuda.empty_cache()

    resumed = Qwen38FSDP2Trainer(experiment(4, "resume"))
    resumed.before_train()
    assert resumed.start_iteration == 2
    assert resumed.train_data_iterators[0].cursor == 4
    assert resumed.grad_manager.optimizer.state
    for name, value in full_adapters(resumed.models[0]).items():
        torch.testing.assert_close(value, after_two[name], rtol=0, atol=0)
    resumed.train_loop()
    resumed.after_train()
    assert resumed.iteration == 4
    for name, value in full_adapters(resumed.models[0]).items():
        torch.testing.assert_close(value, expected[name], rtol=1e-5, atol=1e-6)
    checkpoint = root / "resume" / "qwen3_8_27b_fsdp2_lora" / "it3"
    assert (checkpoint / f"rank{PM.world_rank}.pt").exists()
    assert (checkpoint / "hf_export" / "adapter_model.safetensors").exists()
    local = torch.load(checkpoint / f"rank{PM.world_rank}.pt", map_location="cpu", weights_only=False)
    assert all("lora_" in name for name in local["model"][0] if name != "_qwen38_identity")
