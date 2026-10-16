"""Qwen3.8 text SFT inside the SteptronOSS trainer (DP or FSDP2)."""

from collections import OrderedDict
from contextlib import nullcontext
from typing import Any, cast

import torch
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys

from steptronoss.exp.base_exp import Megatron3DParallelModelConfig
from steptronoss.model.module import MegatronModule


class Qwen38HFConfig(Megatron3DParallelModelConfig):
    model_path: str = "Qwen/Qwen3.8-27B"
    """HF model ID or local directory containing the complete original model."""
    revision: str | None = None
    """Pin a Hub commit for reproducible base weights."""
    tuner: str = "lora"
    """This backend supports text LoRA only."""
    lora_rank: int = 16
    """Low-rank adapter dimension."""
    lora_alpha: int = 32
    """Adapter scale numerator."""
    lora_dropout: float = 0.0
    """Adapter dropout; zero makes the initial DP replicas deterministic."""
    lora_target_suffixes: list[str] = []
    """Empty selects every Linear in text decoder layers, excluding vision/head."""
    gradient_checkpointing: bool = True
    """Use non-reentrant HF activation recomputation."""
    fsdp2_activation_cpu_offload: bool = False
    """Keep recomputation's decoder-block boundary tensors on CPU, not in HBM.

    Graph-preserving, like the native PP scheduler's pipeline_activation_cpu_offload,
    but owned by this path because FSDP2 rejects the native pipeline flag. Trades
    PCIe bandwidth for the term that dominates long-context memory.
    """
    fsdp2_activation_cpu_offload_pin_memory: bool = True
    """Pinned staging is faster but reserves host memory per rank; disable if it runs out."""
    attn_implementation: str = "sdpa"
    """Full-attention implementation; Gated DeltaNet uses its own kernels."""
    hidden_size: int = 5120
    """Qwen3.8-27B hidden size; no native PP partition is constructed."""
    variable_seq_lengths: bool = True
    """Allow padded batches with different sequence lengths."""
    loss_backend: str = "hf"
    """hf for the original loss, chunked for bounded logits memory with a frozen head."""
    loss_chunk_size: int = 128
    """Maximum projected token rows per chunk, in both forward and backward."""

    def sanity_check(self):
        super().sanity_check()
        for name in (
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "virtual_pipeline_model_parallel_size",
            "expert_model_parallel_size",
            "expert_tensor_parallel_size",
        ):
            if getattr(self.parallel_cfg, name) != 1:
                raise ValueError(f"Qwen38HFConfig uses DP or FSDP2 without native model parallelism: {name} must be 1")
        cp_size = int(self.parallel_cfg.context_parallel_size)
        if cp_size not in (1, 2, 4):
            raise ValueError("context_parallel_size must be 1, 2 or 4")
        if cp_size > 1:
            if self.tuner != "lora" or self.loss_backend != "chunked":
                raise ValueError("CP requires LoRA with chunked loss")
            # Context parallelism is the one model-parallel axis this path supports,
            # because it is the only one that needs no KV cache: the Gated DeltaNet
            # state crosses ranks as a plain argument, and GradientCheckpointingLayer
            # only strips use_cache / past_key_values / layer_past. That is what lets
            # per-layer recomputation and activation offload stay on.
            # 64 is the Gated DeltaNet kernel's internal chunk. Splitting off a
            # multiple of it is what makes resuming from a carried state exact
            # rather than approximate -- measured bit-exact at 2-way and 4-way.
        if self.tuner != "lora":
            raise ValueError("This Qwen backend supports LoRA only")
        if self.params_dtype not in {torch.bfloat16, torch.float32}:
            raise ValueError("This path supports bf16/fp32, not fp16 or quantized training")
        if self.lora_rank <= 0 or self.lora_alpha <= 0 or not 0 <= self.lora_dropout < 1:
            raise ValueError("Invalid LoRA rank, alpha, or dropout")
        if self.tp_cfg.sequence_parallel or self.tp_cfg.gradient_accumulation_fusion:
            raise ValueError("Native TP sequence parallel and gradient fusion must be disabled")
        if self.loss_backend not in {"hf", "chunked"} or self.loss_chunk_size <= 0:
            raise ValueError("Invalid loss backend or chunk size")
        if self.loss_backend == "chunked" and self.tuner != "lora":
            raise ValueError("Chunked loss requires LoRA with a frozen LM head")
        if self.fsdp2_activation_cpu_offload and not self.gradient_checkpointing:
            # Without recomputation every saved intermediate crosses PCIe, not just
            # the block boundaries. Same precondition the native pipeline flag has.
            raise ValueError("fsdp2_activation_cpu_offload requires gradient_checkpointing")

    def build_model(self, meta_init=False):
        from transformers import AutoConfig, Qwen3_5ForConditionalGeneration

        config = AutoConfig.from_pretrained(self.model_path, revision=self.revision)
        if config.model_type != "qwen3_5":
            raise ValueError(f"Expected qwen3_5 architecture, got {config.model_type}")
        config.use_cache = False
        config.text_config.use_cache = False
        if meta_init:
            config._attn_implementation = self.attn_implementation
            config.text_config._attn_implementation = self.attn_implementation
            with torch.device("meta"):
                model = Qwen3_5ForConditionalGeneration(config).to(dtype=self.params_dtype)
            # Match HF's from_pretrained keep-in-fp32 modules on rank zero.
            for name, param in model.named_parameters():
                if any(key in name for key in model._keep_in_fp32_modules or []):
                    param.data = param.data.float()
        else:
            model = Qwen3_5ForConditionalGeneration.from_pretrained(
                self.model_path,
                config=config,
                revision=self.revision,
                dtype=self.params_dtype,
                attn_implementation=self.attn_implementation,
                device_map="cpu",
            )
        return Qwen38HFModel(model, self)


def _cp_activation_offload(pin_memory=False):
    """Pageable CPU storage with asynchronous unpack, preserving autograd edges.

    save_on_cpu(pin_memory=False) synchronizes H2D unpack. Under CP/FSDP this
    may wait on a collective whose peer cannot be scheduled until unpack returns.
    Nonblocking unpack avoids that CPU/device circular wait without pinning all
    activations. PyTorch owns the copy/staging lifetime; stored CPU tensors are
    immutable until their consumers finish.
    """

    def pack(tensor):
        if tensor.device.type != "cuda":
            return tensor.device, tensor
        if pin_memory:
            host = torch.empty_like(tensor, device="cpu", pin_memory=True)
            host.copy_(tensor)
        else:
            host = tensor.to("cpu")
        return tensor.device, host

    def unpack(saved):
        device, tensor = saved
        return tensor.to(device, non_blocking=True)

    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)


class Qwen38HFModel(MegatronModule):
    """Return scalar HF causal loss; SteptronOSS owns backward/DP/optimizer."""

    def __init__(self, model: Any, cfg: Qwen38HFConfig):
        super().__init__()
        self.cfg = cfg
        if not hasattr(model.model, "language_model"):
            raise ValueError("Expected Qwen3.5 conditional-generation text backbone")
        # Text-only training: never add adapters to the vision tower or MTP.
        model.requires_grad_(False)
        if cfg.loss_backend == "chunked":
            from steptronoss.model.optimizations.chunked_causal_loss import install_chunked_causal_loss

            install_chunked_causal_loss(model, cfg.loss_chunk_size)
        if cfg.tuner != "lora":
            raise ValueError("This Qwen backend supports LoRA only")
        from peft import LoraConfig, TaskType, get_peft_model

        targets = [
            name
            for name, module in model.named_modules()
            if name.startswith("model.language_model.layers.")
            and isinstance(module, nn.Linear)
            and (not cfg.lora_target_suffixes or name.rsplit(".", 1)[-1] in cfg.lora_target_suffixes)
        ]
        if not targets:
            raise ValueError("No text-layer Linear modules matched the LoRA targets")
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=targets,
                bias="none",
            ),
        )
        model.config.use_cache = False
        model.config.text_config.use_cache = False
        if cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if int(cfg.parallel_cfg.context_parallel_size) > 1:
            if cfg.loss_backend != "chunked" or cfg.tuner != "lora":
                raise ValueError("CP requires LoRA and chunked loss")
            from steptronoss.model.optimizations.context_parallel_qwen38 import install_context_parallel

            install_context_parallel(model)
        self.hf_model = model
        self.checkpoint_identity = {
            "format": "steptron-qwen38-hf-v1",
            "model_path": cfg.model_path,
            "revision": cfg.revision,
            "tuner": cfg.tuner,
            "lora_rank": cfg.lora_rank,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
            "trainable_names": [name for name, p in self.named_parameters() if p.requires_grad],
        }
        for param in self.parameters():
            cast(Any, param).has_initialized = True

    def forward(self, input_ids, labels, attention_mask=None, **kwargs):
        # Our labels are UNshifted. HF shifts them exactly once internally.
        if not torch.any(labels[:, 1:] != -100):
            raise ValueError("Batch has no supervised next-token targets")
        saved_tensor_ctx = (
            (
                _cp_activation_offload(self.cfg.fsdp2_activation_cpu_offload_pin_memory)
                if int(self.cfg.parallel_cfg.context_parallel_size) > 1
                else torch.autograd.graph.save_on_cpu(pin_memory=self.cfg.fsdp2_activation_cpu_offload_pin_memory)
            )
            if self.cfg.fsdp2_activation_cpu_offload and self.training
            else nullcontext()
        )
        with saved_tensor_ctx:
            return self.hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
                return_dict=True,
            ).loss

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        self._reshard_for_checkpoint()
        if destination is None:
            destination = OrderedDict()
        # Keep frozen base weights out of adapter-only resume checkpoints.
        for name, param in self.named_parameters():
            if param.requires_grad:
                destination[prefix + name] = param if keep_vars else param.detach()
        destination[prefix + "_qwen38_identity"] = self.checkpoint_identity
        return destination

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self._reshard_for_checkpoint()
        if assign:
            raise ValueError("assign=True would invalidate the native gradient manager")
        state_dict = dict(state_dict)
        identity = state_dict.pop("_qwen38_identity", None)
        if identity != self.checkpoint_identity:
            raise ValueError("Checkpoint base model / LoRA configuration does not match this experiment")
        params = dict(self.named_parameters())
        expected = {name for name, p in params.items() if p.requires_grad}
        missing, unexpected = sorted(expected - state_dict.keys()), sorted(state_dict.keys() - expected)
        if missing or unexpected:
            raise ValueError(f"Incomplete adapter checkpoint: missing={missing}, unexpected={unexpected}")
        if hasattr(self, "fsdp_mesh"):
            from steptronoss.optimizer.fsdp2_gradient_manager import unpack_local_state

            state_dict = unpack_local_state(state_dict, self.fsdp_mesh, next(self.parameters()).device)
        for name in expected:
            if state_dict[name].shape != params[name].shape:
                raise ValueError(f"Adapter shape mismatch: {name}")
        with torch.no_grad():
            for name in expected:
                params[name].copy_(state_dict[name])
        return _IncompatibleKeys([], [])

    def save_pretrained(self, output_dir):
        """Export a standard PEFT adapter without gathering frozen base weights."""
        self._reshard_for_checkpoint()
        if hasattr(self, "fsdp_mesh"):
            from torch.distributed.tensor import DTensor

            # Every rank participates, even though only rank zero writes files.
            state = {}
            for name, param in self.hf_model.named_parameters():
                if param.requires_grad:
                    value = param.full_tensor() if isinstance(param, DTensor) else param.detach()
                    if torch.distributed.get_rank() == 0:
                        state[name] = value.detach().cpu()
            if torch.distributed.get_rank() == 0:
                self.hf_model.save_pretrained(output_dir, state_dict=state, safe_serialization=True)
        else:
            self.hf_model.save_pretrained(output_dir, safe_serialization=True)

    def _reshard_for_checkpoint(self):
        if hasattr(self, "fsdp_mesh"):
            from steptronoss.model.qwen3_8_fsdp2 import reshard_for_checkpoint

            reshard_for_checkpoint(self.hf_model)
