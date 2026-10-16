"""Standalone torchrun validation for CP=2/4 and multiple data groups.

Unlike node2 pytest markers, this entrypoint deliberately accepts 4/8 workers.
"""

import argparse
import gc
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from steptronoss.core.parallel_state import PM
from steptronoss.exp.base_exp import ParallelConfig
from steptronoss.model.optimizations.chunked_causal_loss import chunked_causal_loss, cp_chunked_causal_loss
from steptronoss.model.optimizations.context_parallel_qwen38 import install_context_parallel
from tests.test_qwen3_8_context_parallel import _gdn_only_model


def mesh(cp):
    cfg = ParallelConfig()
    cfg.context_parallel_size = cp
    PM.set_mesh(cfg)


def grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def compare(reference, control, actual, label, fp32):
    assert reference.keys() == actual.keys(), (label, reference.keys() - actual.keys())
    worst = noise = 0.0
    for name, ref in reference.items():
        assert torch.isfinite(actual[name]).all(), name
        scale = max(float(ref.abs().max()), 1e-8)
        err = float((actual[name] - ref).abs().max()) / scale
        n = float((control[name] - ref).abs().max()) / scale
        if err > 2e-3:
            print(json.dumps(dict(parameter=name, relative_error=err, scale=scale)), flush=True)
        worst, noise = max(worst, err), max(noise, n)
        if fp32:
            torch.testing.assert_close(actual[name], ref, rtol=1e-4, atol=1e-5, msg=name)
    assert noise <= 2e-3, ("unstable control", label, noise)
    assert worst <= max(10 * noise, 2e-3), (label, worst, noise)
    print(json.dumps(dict(check=label, gradient_relative_error=worst, noise=noise)), flush=True)


def components(args):
    cp = args.cp
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    device = args.device
    if device == "cpu":
        import inspect

        from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

        hf.torch_chunk_gated_delta_rule = inspect.unwrap(hf.torch_chunk_gated_delta_rule)
        hf.causal_conv1d_fn = inspect.unwrap(hf.causal_conv1d_fn)
    seq = cp * 128
    group_id = PM.world_rank // cp
    # Different data in each group catches mistaken global-rank point-to-point peers.
    torch.manual_seed(31 + group_id)
    ids = torch.randint(3, 80, (2, seq), device=device)
    cases = {
        "gdn": ["linear_attention"] * 4,
        "attention": ["full_attention"] * 4,
        "mixed": ["linear_attention", "full_attention"] * 2,
        "mixed_frozen_prefix": ["linear_attention", "full_attention"] * 2,
        "attention_empty_query_tail": ["full_attention"] * 4,
        "mixed_empty_query_tail": ["linear_attention", "full_attention"] * 2,
    }
    for name, layers in cases.items():
        # Unlike a zero-supervision rank, these trailing ranks have no valid
        # queries in ANY batch row. They must still gather and reduce gradients.
        mask = torch.ones_like(ids)
        labels = ids.clone()
        if name.endswith("empty_query_tail"):
            labels.fill_(-100)
            for row, valid_length in enumerate((77, 101)):
                mask[row, valid_length:] = 0
                labels[row, valid_length - 17 : valid_length] = ids[row, valid_length - 17 : valid_length]
        else:
            mask[1, seq // 2 + 7 :] = 0
            labels[:, : seq - 17] = -100
            labels[mask == 0] = -100
        print(f"rank={PM.world_rank} case={name} reference", flush=True)

        def build(layers=layers, name=name):
            m = _gdn_only_model(layer_types=layers).to(device=device, dtype=dtype)
            m.config._attn_implementation = args.attention
            m.config.text_config._attn_implementation = args.attention
            m.lm_head.requires_grad_(False)
            if dtype == torch.bfloat16:
                from peft import LoraConfig, get_peft_model

                targets = [
                    n
                    for n, v in m.named_modules()
                    if n.startswith("model.language_model.layers.") and isinstance(v, torch.nn.Linear)
                ]
                m = get_peft_model(m, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=targets))
            if name == "mixed_frozen_prefix":
                raw = m.get_base_model() if hasattr(m, "get_base_model") else m
                raw.model.language_model.layers[0].requires_grad_(False)
            if args.recompute:
                m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            return m

        def reference(mask=mask, labels=labels):
            m = build()
            raw = m.get_base_model() if hasattr(m, "get_base_model") else m
            h = raw.model.language_model(
                input_ids=ids, attention_mask=mask, use_cache=False, deterministic=True
            ).last_hidden_state
            loss = chunked_causal_loss(h, m.lm_head.weight, labels, 8)
            loss.backward()
            return h.detach(), loss.detach(), grads(m)

        expected_h, expected_loss, expected = reference()
        print(f"rank={PM.world_rank} case={name} control", flush=True)
        _, _, control = reference()
        m = build()
        print(f"rank={PM.world_rank} case={name} CP forward", flush=True)
        install_context_parallel(m)
        ctx = (
            torch.autograd.graph.save_on_cpu(pin_memory=False)
            if args.offload
            else __import__("contextlib").nullcontext()
        )
        with ctx:
            raw = m.get_base_model() if hasattr(m, "get_base_model") else m
            h = raw.model.language_model(input_ids=ids, attention_mask=mask, deterministic=True).last_hidden_state
            loss = cp_chunked_causal_loss(h, m.lm_head.weight, labels, 8) / cp
        print(f"rank={PM.world_rank} case={name} CP backward", flush=True)
        loss.backward()
        print(f"rank={PM.world_rank} case={name} CP backward done", flush=True)
        actual = grads(m)
        for value in actual.values():
            dist.all_reduce(value, group=PM.group_of("CP"))
        parts = [torch.empty_like(h) for _ in range(cp)]
        dist.all_gather(parts, h.detach(), group=PM.group_of("CP"))
        complete = torch.cat(parts, 1)
        dist.all_reduce(loss, group=PM.group_of("CP"))
        torch.testing.assert_close(
            complete[mask.bool()],
            expected_h[mask.bool()],
            rtol=3e-3 if dtype == torch.bfloat16 else 1e-4,
            atol=3e-3 if dtype == torch.bfloat16 else 1e-5,
        )
        torch.testing.assert_close(loss, expected_loss, rtol=3e-3 if dtype == torch.bfloat16 else 1e-4, atol=1e-5)
        compare(expected, control, actual, name, dtype == torch.float32)
        del m, h, parts
        gc.collect()
        torch.cuda.empty_cache()


def full_adapters(model):
    return {
        n: (p.full_tensor() if isinstance(p, DTensor) else p.detach()).cpu().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }


def trainer(args):
    from playground.sft.qwen3_8.qwen3_8_27b_fsdp2_lora import Exp
    from steptronoss.core.trainers.qwen3_8_fsdp2_trainer import Qwen38FSDP2Trainer

    shared = [tempfile.mkdtemp(prefix="qwen38-cp-") if PM.world_rank == 0 else None]
    dist.broadcast_object_list(shared, src=0)
    root = Path(shared[0])
    maxlen = args.cp * 128
    if PM.world_rank == 0:
        _gdn_only_model(layer_types=["linear_attention", "full_attention"] * 2).save_pretrained(root / "base")
        records = []
        for i in range(32):
            length = 48 + i % 13 if args.short_padding and i % 3 == 0 else maxlen - i % 17
            tokens = [(j + i) % 70 + 3 for j in range(length)]
            targets = [-100] * (length - 5) + tokens[-5:]
            records.append(dict(input_ids=tokens, labels=targets))
        raw = "\n".join(json.dumps(x) for x in records) + "\n"
        (root / "tokens.jsonl").write_text(raw)
        (root / "tokens.jsonl.meta.json").write_text(
            json.dumps(
                dict(
                    format="steptron-qwen38-text-v1",
                    model=str(root / "base"),
                    revision=None,
                    max_length=maxlen,
                    pad_token_id=0,
                    sha256=hashlib.sha256(raw.encode()).hexdigest(),
                )
            )
        )
    dist.barrier()

    def experiment(iters, out, cp=args.cp):
        e = Exp()
        e.resource_cfg.gpu = PM.world_size
        e.model_cfg.model_path = str(root / "base")
        e.model_cfg.hidden_size = 32
        e.model_cfg.lora_rank = 2
        e.model_cfg.lora_alpha = 4
        e.model_cfg.attn_implementation = args.attention
        e.model_cfg.gradient_checkpointing = args.recompute
        e.model_cfg.fsdp2_activation_cpu_offload = args.offload
        e.model_cfg.fsdp2_activation_cpu_offload_pin_memory = False
        e.model_cfg.parallel_cfg.context_parallel_size = cp
        e.data_cfg.tokenized_path = str(root / "tokens.jsonl")
        e.data_cfg.pad_token_id = 0
        # Keep padding identical for the CP=1 control; isolate parallelism from
        # different low-precision kernels selected for different padded shapes.
        e.data_cfg.cp_size = args.cp
        e.trainer_cfg.global_seq_length = maxlen
        e.trainer_cfg.global_batch_size = PM.world_size * 2  # accumulation >1
        e.trainer_cfg.train_iters = iters
        e.scheduler_cfg.total_schedule = 4
        e.scheduler_cfg.warmup_schedule = 0
        e.checkpoint_cfg.save_dir = str(root / out)
        e.log_dir = str(root / "logs" / out)
        e.sanity_check()
        return e

    def run(iters, out, cp=args.cp, capture_gradient=False):
        from unittest.mock import patch

        from steptronoss.optimizer.fsdp2_gradient_manager import FSDP2GradientManager

        t = Qwen38FSDP2Trainer(experiment(iters, out, cp))
        first_gradient = {}
        original_sync = FSDP2GradientManager._sync_cp_gradients

        def capture(manager):
            original_sync(manager)
            if capture_gradient and not first_gradient:
                # Read AFTER native FSDP reduction, BEFORE clipping/Adam: adaptive
                # updates alone could hide an incorrect constant gradient scale.
                for name, param in manager.model[0].named_parameters():
                    if param.requires_grad and param.grad is not None:
                        grad = param.grad
                        grad = grad.full_tensor() if isinstance(grad, DTensor) else grad
                        first_gradient[name] = grad.detach().cpu().clone()

        with patch.object(FSDP2GradientManager, "_sync_cp_gradients", capture):
            t.train()
        state = full_adapters(t.models[0])
        del t
        gc.collect()
        torch.cuda.empty_cache()
        return (state, first_gradient) if capture_gradient else state

    if args.short_padding:
        reference, cp_gradient = run(4, "continuous", capture_gradient=True)
    else:
        reference = run(4, "continuous")
    run(2, "resume")
    resumed = run(4, "resume")
    for n in reference:
        torch.testing.assert_close(resumed[n], reference[n], rtol=1e-5, atol=1e-6)
    ckpt = root / "resume/qwen3_8_27b_fsdp2_lora/it3"
    manifest = json.loads((ckpt / "manifest.json").read_text())
    assert manifest["context_parallel_size"] == args.cp
    assert (ckpt / "hf_export/adapter_model.safetensors").is_file()
    # Compare full CP training against CP=1 with the same global batch.
    if args.short_padding:
        baseline, baseline_gradient = run(4, "cp1", 1, capture_gradient=True)
        _, control_gradient = run(1, "cp1_gradient_control", 1, capture_gradient=True)
        compare(baseline_gradient, control_gradient, cp_gradient, "fsdp2_first_step", False)
    else:
        baseline = run(4, "cp1", 1)
    for n in reference:
        delta = float((reference[n] - baseline[n]).abs().max())
        if delta > 3e-5:
            print(f"CP1_UPDATE_DELTA {n} {delta:.3e}", flush=True)
        torch.testing.assert_close(
            reference[n], baseline[n], rtol=3e-3, atol=3e-5, msg=lambda msg, name=n: name + " " + msg
        )
    from steptronoss.checkpointing.fsdp2_checkpoint import FSDP2Checkpointer

    other = experiment(1, "reject", 1)
    mesh(1)
    try:
        FSDP2Checkpointer().load_ckpt(ckpt, other.checkpoint_cfg)
    except ValueError as exc:
        assert "same context_parallel_size" in str(exc)
    else:
        raise AssertionError("accepted a changed CP topology for full resume")
    mesh(args.cp)
    # Model-only loading across CP topologies starts a fresh optimizer/data cursor.
    e = experiment(1, "import_cp1")
    e.checkpoint_cfg.auto_resume = False
    e.checkpoint_cfg.load_path = str(root / "cp1/qwen3_8_27b_fsdp2_lora/it3")
    for flag in ("optimizer", "scheduler", "data", "rng_state", "iter", "exp"):
        setattr(e.checkpoint_cfg.load_option, flag, False)
    imported = Qwen38FSDP2Trainer(e)
    imported.before_train()
    assert imported.start_iteration == 0
    if imported.train_data_iterators[0] is not None:
        assert imported.train_data_iterators[0].cursor == 0
    for name, value in full_adapters(imported.models[0]).items():
        torch.testing.assert_close(value, baseline[name], rtol=0, atol=0)
    del imported
    gc.collect()
    torch.cuda.empty_cache()
    # A standard PEFT consumer can reload the exported adapter without CP/FSDP.
    from peft import PeftModel
    from transformers import Qwen3_5ForConditionalGeneration

    loaded = PeftModel.from_pretrained(
        Qwen3_5ForConditionalGeneration.from_pretrained(root / "base"), ckpt / "hf_export", is_trainable=False
    )
    loaded_params = dict(loaded.named_parameters())
    for name, value in resumed.items():
        key = name.removeprefix("hf_model.")
        torch.testing.assert_close(loaded_params[key].detach().cpu(), value, rtol=0, atol=0)
    print("CP_TRAINER_RESUME_AND_CP1_EQUIVALENCE_PASSED", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cp", type=int, default=2)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--mode", choices=["components", "trainer"], default="components")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--attention", choices=["sdpa", "flash_attention_2"], default="sdpa")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--short-padding", action="store_true", help="trainer fixture includes entirely padded CP ranks")
    args = ap.parse_args()
    if args.device == "cpu" and args.mode == "trainer":
        ap.error("The native FSDP2 trainer lifecycle requires CUDA; use CPU component validation instead")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    import faulthandler

    faulthandler.dump_traceback_later(90, repeat=True)
    print("rank=" + os.getenv("RANK", "?") + " init start", flush=True)
    if args.device == "cpu":
        dist.init_process_group("gloo")
        PM.world_size, PM.world_rank = dist.get_world_size(), dist.get_rank()
    else:
        PM.initialize(backend="nccl")
    print("rank=" + os.getenv("RANK", "?") + " init done", flush=True)
    print(f"rank={PM.world_rank} mesh start", flush=True)
    mesh(args.cp)
    print(f"rank={PM.world_rank} mesh done", flush=True)
    if args.device == "cuda":
        torch.cuda.set_device(PM.world_rank % torch.cuda.device_count())
    try:
        (components if args.mode == "components" else trainer)(args)
        dist.barrier()
        print("CP_VALIDATION_DONE", flush=True)
    finally:
        faulthandler.cancel_dump_traceback_later()
        if args.device == "cpu":
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
