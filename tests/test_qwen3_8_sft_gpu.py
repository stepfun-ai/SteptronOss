"""Real native trainer / gradient-manager / checkpoint test on CUDA.

Run: torchrun --standalone --nproc-per-node=2 -m pytest -q
     tests/test_qwen3_8_sft_gpu.py -m node2
"""

import hashlib
import json
import tempfile
from pathlib import Path

import pytest
import torch

from playground.sft.qwen3_8.qwen3_8_27b_lora import Exp
from steptronoss.core.parallel_state import PM
from steptronoss.core.trainers.qwen3_8_sft_trainer import Qwen38SFTTrainer
from tests.test_qwen3_8_sft import tiny_base


@pytest.mark.gpu
@pytest.mark.node2
@pytest.mark.xdist_group("torchrun")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPUs required")
def test_native_two_gpu_train_and_resume():
    PM.initialize(backend="nccl")
    shared = [tempfile.mkdtemp(prefix="qwen38-native-smoke-") if PM.world_rank == 0 else None]
    torch.distributed.broadcast_object_list(shared, src=0)
    root = Path(shared[0])
    if PM.world_rank == 0:
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

    def experiment(iters):
        exp = Exp()
        exp.resource_cfg.gpu = 2
        exp.model_cfg.model_path = str(root / "base")
        exp.model_cfg.hidden_size = 32
        exp.model_cfg.lora_rank = 2
        exp.model_cfg.lora_alpha = 4
        exp.data_cfg.tokenized_path = str(root / "tokens.jsonl")
        exp.data_cfg.pad_token_id = 0
        exp.trainer_cfg.global_seq_length = 16
        exp.trainer_cfg.global_batch_size = 4
        exp.trainer_cfg.train_iters = iters
        exp.scheduler_cfg.total_schedule = 10
        exp.scheduler_cfg.warmup_schedule = 0
        exp.checkpoint_cfg.save_dir = str(root / "checkpoints")
        exp.log_dir = str(root / "logs")
        exp.sanity_check()
        return exp

    first = Qwen38SFTTrainer(experiment(2))
    first.train()
    assert first.iteration == 2
    params = [p for p in first.models.parameters() if p.requires_grad]
    flat = torch.cat([p.detach().float().flatten() for p in params])
    replicas = [torch.empty_like(flat) for _ in range(2)]
    torch.distributed.all_gather(replicas, flat)
    torch.testing.assert_close(replicas[0], replicas[1])
    assert first.grad_manager.optimizer.state
    first.grad_manager._release_grad_acc_hooks()
    del first

    resumed = Qwen38SFTTrainer(experiment(4))
    resumed.before_train()
    assert resumed.start_iteration == 2
    assert resumed.train_data_iterators[0].cursor == 4
    assert resumed.grad_manager.optimizer.state
    resumed.train_loop()
    resumed.after_train()
    assert resumed.iteration == 4
    assert (root / "checkpoints" / "qwen3_8_27b_lora" / "it3" / "hf_export" / "adapter_config.json").exists()
    resumed.grad_manager._release_grad_acc_hooks()
