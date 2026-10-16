import copy
import hashlib
import json

import pytest
import torch
import transformers
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from steptronoss.data.qwen3_8_sft import ResumableTextBatches, encode_chat
from steptronoss.model.qwen3_8_hf import Qwen38HFConfig, Qwen38HFModel

pytest.importorskip("transformers", minversion="5.19", reason="Install requirements/qwen38-fsdp2-runtime.txt")
pytest.importorskip("peft", minversion="0.21.2", reason="Install requirements/qwen38-fsdp2-runtime.txt")


def tiny_base():
    config = transformers.Qwen3_5Config(
        text_config=dict(
            vocab_size=96,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            layer_types=["linear_attention"] * 3 + ["full_attention"],
            rope_parameters=dict(
                rope_type="default", rope_theta=10000.0, partial_rotary_factor=1.0, mrope_section=[1, 1, 2]
            ),
        ),
        vision_config=dict(
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            out_hidden_size=32,
            patch_size=2,
            num_position_embeddings=16,
        ),
        image_token_id=90,
        video_token_id=91,
        vision_start_token_id=92,
        vision_end_token_id=93,
    )
    return transformers.Qwen3_5ForConditionalGeneration(config)


def tiny_wrapper(checkpointing=True):
    cfg = Qwen38HFConfig()
    cfg.model_path = "local-tiny-qwen38-test"
    cfg.params_dtype = torch.float32
    cfg.lora_rank = 2
    cfg.lora_alpha = 4
    cfg.gradient_checkpointing = checkpointing
    return Qwen38HFModel(tiny_base(), cfg)


@pytest.mark.cpu
def test_actual_hybrid_qwen_loss_lora_update_and_native_checkpoint(tmp_path):
    torch.manual_seed(42)
    model = tiny_wrapper()
    model.train()
    ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
    labels = torch.tensor([[-100, -100, -100, 6, 7, 8]])
    mask = torch.ones_like(ids)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n and "language_model.layers" in n for n in trainable)
    assert any("linear_attn" in n for n in trainable)
    assert any("self_attn" in n for n in trainable)
    frozen_name, frozen = next((n, p) for n, p in model.named_parameters() if not p.requires_grad)
    frozen_before = frozen.detach().clone()
    before = {n: p.detach().clone() for n, p in trainable.items()}
    # Compare the loss against an independently shifted causal CE calculation.
    with torch.no_grad():
        logits = model.hf_model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        expected = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
    loss = model(input_ids=ids, labels=labels, attention_mask=mask)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable.values())
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    optimizer = torch.optim.Adam(trainable.values(), lr=0.01)
    optimizer.step()
    assert any(not torch.equal(before[n], p) for n, p in trainable.items())
    torch.testing.assert_close(frozen, frozen_before)
    saved = copy.deepcopy(model.state_dict())
    assert frozen_name not in saved
    assert set(saved) == set(trainable) | {"_qwen38_identity"}
    with torch.no_grad():
        for p in trainable.values():
            p.zero_()
    model.load_state_dict(saved)
    for n, p in trainable.items():
        torch.testing.assert_close(p, saved[n])
    bad = copy.deepcopy(saved)
    bad.pop(next(iter(trainable)))
    with pytest.raises(ValueError, match="Incomplete adapter"):
        model.load_state_dict(bad)
    bad = copy.deepcopy(saved)
    bad["_qwen38_identity"]["model_path"] = "different-base"
    with pytest.raises(ValueError, match="does not match"):
        model.load_state_dict(bad)
    model.save_pretrained(tmp_path / "adapter")
    assert (tmp_path / "adapter" / "adapter_config.json").exists()
    assert (tmp_path / "adapter" / "adapter_model.safetensors").exists()


def local_tokenizer():
    backend = Tokenizer(models.BPE(unk_token="[UNK]"))  # noqa: S106
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.train_from_iterator(
        ["system user assistant alpha beta gamma delta <think> </think>\n"],
        trainers.BpeTrainer(vocab_size=300, special_tokens=["[UNK]", "<|im_start|>", "<|im_end|>"]),
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")  # noqa: S106
    tokenizer.chat_template = (
        "{% for m in messages %}{{ '<|im_start|>' + m.role + '\n' }}"
        "{% if m.role == 'assistant' %}{{ '<think>\n\n</think>\n\n' }}{% endif %}"
        "{{ m.content + '<|im_end|>\n' }}{% endfor %}"
    )
    return tokenizer


@pytest.mark.cpu
def test_assistant_mask_multiturn_eos_and_truncation():
    tokenizer = local_tokenizer()
    messages = [
        {"role": "user", "content": "alpha"},
        {"role": "assistant", "content": "beta"},
        {"role": "user", "content": "gamma"},
        {"role": "assistant", "content": "delta"},
    ]
    record = encode_chat(tokenizer, {"messages": messages}, 1024)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False)
    offsets = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    for marker, expected in [("alpha", False), ("beta", True), ("gamma", False), ("delta", True)]:
        position = rendered.index(marker)
        i = next(i for i, (a, b) in enumerate(offsets) if a <= position < b)
        assert (record["labels"][i] != -100) == expected
    assert sum(y == tokenizer.convert_tokens_to_ids("<|im_end|>") for y in record["labels"]) == 2
    with pytest.raises(ValueError, match="truncation is disabled"):
        encode_chat(tokenizer, {"messages": messages}, 2)
    with pytest.raises(ValueError, match="Image/video"):
        encode_chat(tokenizer, {"messages": messages, "images": ["x.png"]}, 1024)


def token_file(tmp_path, count=18):
    path = tmp_path / "train.tokens.jsonl"
    path.write_text(
        "\n".join(json.dumps({"input_ids": [i + 1, 2, 3], "labels": [-100, 2, 3]}) for i in range(count)) + "\n"
    )
    return path


@pytest.mark.cpu
def test_dp_disjoint_batches_and_dp0_resume_state(tmp_path):
    path = token_file(tmp_path)
    loaders = [ResumableTextBatches(path, r, 2, 2, 8, 16, 0) for r in range(2)]
    assert len(loaders[0]) == 16  # Drop only the incomplete global batch.
    assert set(loaders[0].indices).isdisjoint(loaders[1].indices)
    for loader in loaders:
        next(loader)
        next(loader)  # One global optimizer step (two micro batches per DP rank).
    state = loaders[0].state_dict()
    for r, loader in enumerate(loaders):
        expected = next(loader)
        restored = ResumableTextBatches(path, r, 2, 2, 8, 16, 0)
        restored.load_state_dict(state)
        actual = next(restored)
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key])
    changed = ResumableTextBatches(path, 0, 1, 2, 8, 16, 0)
    with pytest.raises(ValueError, match="dp_size"):
        changed.load_state_dict(state)


@pytest.mark.cpu
def test_experiment_guard_and_resume_configuration(monkeypatch):
    from playground.sft.qwen3_8.qwen3_8_27b_lora import Exp

    monkeypatch.setenv("WORLD_SIZE", "8")
    exp = Exp()
    exp.sanity_check()
    previous = {"exp": exp.to_dict()}
    exp.assert_critical_attrs_expected(previous)
    exp.model_cfg.lora_rank = 64
    with pytest.raises(ValueError, match="lora_rank"):
        exp.assert_critical_attrs_expected(previous)
    exp = Exp()
    exp.model_cfg.parallel_cfg.tensor_model_parallel_size = 2
    with pytest.raises(ValueError, match="tensor_model_parallel_size must be 1"):
        exp.sanity_check()


@pytest.mark.cpu
def test_full_text_sft_is_outside_lora_backend_scope(monkeypatch):
    from playground.sft.qwen3_8.qwen3_8_27b_lora import Exp

    monkeypatch.setenv("WORLD_SIZE", "8")
    exp = Exp()
    cfg = exp.model_cfg
    cfg.tuner = "full"
    with pytest.raises(ValueError, match="LoRA only"):
        cfg.sanity_check()
    with pytest.raises(ValueError, match="LoRA only"):
        Qwen38HFModel(tiny_base(), cfg)


@pytest.mark.cpu
def test_original_steptron_forward_backward_scheduler(monkeypatch):
    from steptronoss.core.pipeline_parallel import schedules

    monkeypatch.setattr(schedules, "is_pipeline_last_stage", lambda: True)
    model = tiny_wrapper(checkpointing=False)
    reference = tiny_wrapper(checkpointing=False)
    reference.hf_model.load_state_dict(model.hf_model.state_dict())
    batches = [
        {"input_ids": torch.tensor([[3, 4, 5, n]]), "labels": torch.tensor([[-100, -100, 5, n]])} for n in [6, 7]
    ]
    for batch in batches:
        (reference(**batch) / 2).backward()
    cfg = Qwen38HFConfig()
    cfg.check_nan = False  # No distributed process group is needed in this CPU test.
    scheduler = schedules.FWBWScheduler(cfg)
    scheduler.configure(
        models=[model],
        data_iterators=[iter(batches)],
        data_sync_fn=next,
        data_proc_fn=lambda data: data,
        loss_fn=lambda data, loss: loss,
    )
    scheduler.run(2)
    actual = dict(model.named_parameters())
    for name, p in reference.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(actual[name].grad, p.grad)


@pytest.mark.cpu
def test_original_checkpointer_collects_only_adapter_state(tmp_path, monkeypatch):
    from steptronoss.checkpointing.local_checkpoint import Checkpointer
    from steptronoss.core.parallel_state import PM
    from steptronoss.exp.checkpointing import CheckpointConfig

    monkeypatch.setattr(PM, "i_am", lambda group, rank: True)
    monkeypatch.setattr(PM, "rank_in", lambda group: 0)
    model = tiny_wrapper(checkpointing=False)
    data = ResumableTextBatches(token_file(tmp_path), 0, 1, 1, 2, 16, 0)
    next(data)
    cfg = CheckpointConfig()
    cfg.save_option.none(but=["model", "data"])
    cfg.use_distributed_optimizer = False
    files = Checkpointer().make_ckpt(cfg, model=[model], dataloader=data)
    state = next(value["model"] for key, value in files.items() if key.endswith(".pt") and "model" in value)
    assert all("lora_" in name for name in state if name != "_qwen38_identity")
    assert files["data.pkl"]["cursor"] == 1
    model.load_state_dict(state)


@pytest.mark.cpu
def test_from_pretrained_builder_on_actual_qwen_architecture(tmp_path):
    original = tiny_base()
    original.save_pretrained(tmp_path / "base")
    cfg = Qwen38HFConfig()
    cfg.model_path = str(tmp_path / "base")
    cfg.params_dtype = torch.float32
    cfg.gradient_checkpointing = False
    cfg.lora_rank = 2
    model = cfg.build_model()
    torch.testing.assert_close(
        model.hf_model.base_model.model.model.language_model.embed_tokens.weight,
        original.model.language_model.embed_tokens.weight,
    )
    assert torch.isfinite(model(input_ids=torch.tensor([[3, 4, 5]]), labels=torch.tensor([[-100, 4, 5]])))


@pytest.mark.cpu
def test_training_rejects_wrong_tokenizer_or_modified_data(tmp_path):
    from playground.sft.qwen3_8.qwen3_8_27b_lora import Exp

    path = token_file(tmp_path)
    exp = Exp()
    exp.data_cfg.tokenized_path = str(path)
    exp.trainer_cfg.global_batch_size = 8
    exp.trainer_cfg.global_seq_length = 16
    metadata = {
        "format": "steptron-qwen38-text-v1",
        "model": exp.model_cfg.model_path,
        "revision": None,
        "max_length": 16,
        "pad_token_id": exp.data_cfg.pad_token_id,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    meta_path = tmp_path / "train.tokens.jsonl.meta.json"
    meta_path.write_text(json.dumps(metadata))
    exp.data_cfg.build_dataloader(dp_rank=0, dp_size=2)
    metadata["model"] = "wrong-tokenizer"
    meta_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata mismatch: model"):
        exp.data_cfg.build_dataloader(dp_rank=0, dp_size=2)
    metadata["model"] = exp.model_cfg.model_path
    meta_path.write_text(json.dumps(metadata))
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after preparation"):
        exp.data_cfg.build_dataloader(dp_rank=0, dp_size=2)


@pytest.mark.cpu
def test_turn_mask_preserves_input_but_removes_supervision():
    tokenizer = local_tokenizer()
    messages = [
        {"role": "user", "content": "alpha"},
        {"role": "assistant", "content": "beta"},
        {"role": "user", "content": "gamma"},
        {"role": "assistant", "content": "delta"},
    ]
    reference = encode_chat(tokenizer, {"messages": messages}, 1024)
    messages[1]["loss_mask"] = 0
    masked = encode_chat(tokenizer, {"messages": messages}, 1024)
    assert masked["input_ids"] == reference["input_ids"]
    assert sum(x != -100 for x in masked["labels"]) < sum(x != -100 for x in reference["labels"])
    messages[-1]["loss_mask"] = 0
    with pytest.raises(ValueError, match="no assistant next-token target"):
        encode_chat(tokenizer, {"messages": messages}, 1024)
    messages[-1]["loss_mask"] = 2
    with pytest.raises(ValueError, match="loss_mask must be 0 or 1"):
        encode_chat(tokenizer, {"messages": messages}, 1024)


@pytest.mark.cpu
@pytest.mark.parametrize("offload", [False, True])
def test_activation_offload_is_optional_and_preserves_gradients(monkeypatch, offload):
    torch.manual_seed(67)
    reference = tiny_wrapper()
    actual = tiny_wrapper()
    actual.hf_model.load_state_dict(reference.hf_model.state_dict())
    assert actual.cfg.fsdp2_activation_cpu_offload is False
    actual.cfg.fsdp2_activation_cpu_offload = offload
    actual.cfg.fsdp2_activation_cpu_offload_pin_memory = False
    calls = []
    original = torch.autograd.graph.save_on_cpu

    def track(*args, **kwargs):
        calls.append(kwargs["pin_memory"])
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.autograd.graph, "save_on_cpu", track)
    batch = {"input_ids": torch.tensor([[3, 4, 5, 6]]), "labels": torch.tensor([[-100, -100, 5, 6]])}
    expected, result = reference(**batch), actual(**batch)
    torch.testing.assert_close(result, expected)
    expected.backward()
    result.backward()
    parameters = dict(reference.named_parameters())
    for name, param in actual.named_parameters():
        if param.requires_grad:
            torch.testing.assert_close(param.grad, parameters[name].grad)
    assert calls == ([False] if offload else [])


@pytest.mark.cpu
def test_recipe_uses_native_overrides_and_cosine_scheduler(tmp_path, monkeypatch):
    import sys

    from playground.sft.qwen3_8.qwen3_8_27b_fsdp2_lora import Exp
    from steptronoss.exp.lr_schedulers import CosineSchedulerConfig

    monkeypatch.setenv("WORLD_SIZE", "8")
    exp = Exp()
    exp.log_dir = str(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "experiment",
            "model_cfg.lora_rank=8",
            "model_cfg.lora_alpha=16",
            "model_cfg.parallel_cfg.context_parallel_size=4",
            "trainer_cfg.global_batch_size=4",
            "trainer_cfg.train_iters=100",
            "scheduler_cfg.total_schedule=100",
            "scheduler_cfg.lr=5e-5",
        ],
    )
    exp.update_from_args()
    exp.sanity_check()
    assert exp.model_cfg.lora_rank == 8 and exp.model_cfg.lora_alpha == 16
    assert exp.model_cfg.parallel_cfg.context_parallel_size == 4
    assert exp.trainer_cfg.global_batch_size == 4
    assert exp.scheduler_cfg.lr == 5e-5
    assert isinstance(exp.scheduler_cfg, CosineSchedulerConfig)
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=exp.scheduler_cfg.lr)
    scheduler = exp.scheduler_cfg.build_scheduler(optimizer)
    scheduler.step(exp.scheduler_cfg.total_schedule // 2)
    assert exp.scheduler_cfg.min_lr < optimizer.param_groups[0]["lr"] < exp.scheduler_cfg.lr
    scheduler.step(exp.scheduler_cfg.total_schedule // 2)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(exp.scheduler_cfg.min_lr)
