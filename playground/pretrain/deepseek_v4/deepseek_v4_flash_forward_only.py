"""
Forward-only smoke for the downloaded DeepSeek V4 Flash base checkpoint.

This entrypoint intentionally bypasses the trainer stack: it builds the model,
loads the published safetensors through the model's OnlineReshaper, encodes one
Step3 SFT sample with the DeepSeek V4 tokenizer, and prints the forward loss.
Launch with 8 GPUs:

    torchrun --standalone --nproc-per-node=8 playground/pretrain/deepseek_v4/deepseek_v4_flash_forward_only.py
"""

from __future__ import annotations

import json
import os
from os.path import exists, join

import torch
import torch.distributed as dist
from configurize import Config

from steptronoss.core import tensor_parallel
from steptronoss.core.parallel_state import PM
from steptronoss.core.pipeline_parallel import p2p_communication as p2p
from steptronoss.exp.base_exp import BaseExp
from steptronoss.model.deepseek_v4 import DeepseekV4ModelConfig
from steptronoss.utils import print_n_params, setup_logger
from steptronoss.utils.weight_loader import HFWeights

CHECKPOINT_DIR = os.environ.get(
    "DEEPSEEK_V4_FLASH_CHECKPOINT_DIR",
    "/oss/opensources_model/DeepSeek-V4-Flash-Base",
)
STEP3_DATA_PATH = os.environ.get(
    "DEEPSEEK_V4_FLASH_STEP3_DATA_PATH",
    "/oss/data/step_sft_data/0312_rtu/general/chunk_0.json",
)


class DeepseekV4FlashForwardOnlyModelConfig(DeepseekV4ModelConfig):
    checkpoint_dir: str
    """Published DeepSeek V4 Flash safetensors directory."""

    def __init__(self):
        super().__init__()
        self.checkpoint_dir = CHECKPOINT_DIR
        self._apply_release_config()
        self.moe_cfg.vocab_size = self.vocab_size
        self.global_seq_length = 128
        self.micro_batch_size = 1

        self.parallel_cfg.pipeline_model_parallel_size = 2
        self.parallel_cfg.tensor_model_parallel_size = 4
        self.parallel_cfg.expert_model_parallel_size = 4
        self.parallel_cfg.expert_tensor_parallel_size = 1
        self.parallel_cfg.context_parallel_size = 1
        self.parallel_cfg.virtual_pipeline_model_parallel_size = 1

        self.tp_cfg.sequence_parallel = False
        self.tp_cfg.async_tensor_model_parallel_allreduce = False
        self.tp_cfg.gradient_accumulation_fusion = False
        self.recompute = False

    def _apply_release_config(self):
        config_path = join(self.checkpoint_dir, "config.json")
        if not exists(config_path):
            return

        with open(config_path) as f:
            cfg = json.load(f)

        self.vocab_size = cfg["vocab_size"]
        self.hidden_size = cfg["hidden_size"]
        self.num_layers = cfg["num_hidden_layers"]
        self.layernorm_epsilon = cfg["rms_norm_eps"]
        self.params_dtype = torch.bfloat16

        self.attn_cfg.num_attention_heads = cfg["num_attention_heads"]
        self.attn_cfg.num_key_value_heads = cfg["num_key_value_heads"]
        self.attn_cfg.head_dim = cfg["head_dim"]
        self.attn_cfg.q_lora_rank = cfg["q_lora_rank"]
        self.attn_cfg.qk_rope_head_dim = cfg["qk_rope_head_dim"]
        self.attn_cfg.rope_theta = float(cfg["rope_theta"])
        self.attn_cfg.compress_rope_theta = float(cfg["compress_rope_theta"])
        self.attn_cfg.attention_dropout = float(cfg["attention_dropout"])
        self.attn_cfg.sliding_window = cfg["sliding_window"]
        self.attn_cfg.o_groups = cfg["o_groups"]
        self.attn_cfg.o_lora_rank = cfg["o_lora_rank"]
        self.attn_cfg.index_n_heads = cfg["index_n_heads"]
        self.attn_cfg.index_head_dim = cfg["index_head_dim"]
        self.attn_cfg.index_topk = cfg["index_topk"]
        self.attn_cfg.layernorm_epsilon = self.layernorm_epsilon
        self.attn_cfg.compress_rates = {
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 128,
        }

        ratios = cfg["compress_ratios"][: self.num_layers]
        self.attn_cfg.layer_types = [_attention_type_from_ratio(ratio) for ratio in ratios]

        self.moe_cfg.moe_intermediate_size = cfg["moe_intermediate_size"]
        self.moe_cfg.num_experts_per_tok = cfg["num_experts_per_tok"]
        self.moe_cfg.n_routed_experts = cfg["n_routed_experts"]
        self.moe_cfg.n_shared_experts = cfg["n_shared_experts"]
        self.moe_cfg.scoring_func = cfg["scoring_func"]
        self.moe_cfg.routed_scaling_factor = float(cfg["routed_scaling_factor"])
        self.moe_cfg.swiglu_limit = float(cfg["swiglu_limit"])
        self.moe_cfg.mlp_layer_types = ["hash_moe"] * cfg["num_hash_layers"] + ["moe"] * (
            self.num_layers - cfg["num_hash_layers"]
        )
        self.moe_cfg.vocab_size = self.vocab_size
        self.moe_cfg.layernorm_epsilon = self.layernorm_epsilon

        self.hc_cfg.hc_mult = cfg["hc_mult"]
        self.hc_cfg.hc_sinkhorn_iters = cfg["hc_sinkhorn_iters"]
        self.hc_cfg.hc_eps = float(cfg["hc_eps"])
        self.hc_cfg.layernorm_epsilon = self.layernorm_epsilon

        self.tok_embed_cfg.vocab_size = self.vocab_size
        self.out_embed_cfg.vocab_size = self.vocab_size
        self.out_embed_cfg.layernorm_epsilon = self.layernorm_epsilon


class ForwardOnlyConfig(Config):
    checkpoint_dir: str
    """Published DeepSeek V4 Flash safetensors directory."""
    tokenizer_dir: str
    """HF tokenizer directory used for Step3-data encoding."""
    step3_data_path: str
    """Raw Step3 SFT json file to sample from."""
    sample_index: int
    """Raw json array index to start from when sampling Step3 data."""
    seq_len: int
    """Maximum next-token positions to score."""
    preview_tokens: int
    """Number of encoded tokens to decode for a short provenance preview."""

    def __init__(self):
        super().__init__()
        self.checkpoint_dir = CHECKPOINT_DIR
        self.tokenizer_dir = CHECKPOINT_DIR
        self.step3_data_path = STEP3_DATA_PATH
        self.sample_index = 0
        self.seq_len = 512
        self.preview_tokens = 64


class Exp(BaseExp):
    log_dir = "./tensorboard_logs/"
    model_cfg = DeepseekV4FlashForwardOnlyModelConfig
    forward_cfg = ForwardOnlyConfig

    def __init__(self):
        super().__init__()
        self.exp_name = "deepseek_v4_flash_forward_only"

    @property
    def log_path(self) -> str:
        return join(self.log_dir, self.exp_name)

    def train(self):
        self.update_from_args()
        self.model_cfg.checkpoint_dir = self.forward_cfg.checkpoint_dir
        self.model_cfg._apply_release_config()
        self.model_cfg.global_seq_length = self.forward_cfg.seq_len
        self.model_cfg.micro_batch_size = 1
        self.sanity_check()

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        PM.initialize(backend="nccl")
        PM.set_mesh(self.model_cfg.parallel_cfg)

        setup_logger(self.log_path, filename="forward_only", mode="a")

        batch = _make_step3_batch(self.forward_cfg)
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        loss_masks = batch["loss_masks"]

        model = self.model_cfg.build_model()
        print_n_params([model])
        model.to(dtype=torch.bfloat16, device=torch.cuda.current_device())
        model.eval()

        model.load_hf_state_dict(HFWeights(self.forward_cfg.checkpoint_dir), strict=True)
        torch.cuda.empty_cache()
        dist.barrier()

        with torch.no_grad():
            logits = _run_pipeline_forward(model, self.model_cfg, input_ids)
            if PM.rank_in("PP") == PM.size_of("PP") - 1:
                labels_t = labels.transpose(0, 1).contiguous()
                loss_masks_t = loss_masks.transpose(0, 1).contiguous()
                losses, acc = tensor_parallel.vocab_parallel_cross_entropy(
                    logits.float(),
                    labels_t,
                )
                masked_token_count = loss_masks_t.sum()
                masked_loss = (losses * loss_masks_t).sum() / masked_token_count.clamp_min(1)
                all_token_loss = losses.mean()
                all_token_acc, masked_acc = _distributed_accuracy(logits, labels_t, loss_masks_t)
                if PM.rank_in("TP") == 0:
                    snapshot = {
                        **batch["metadata"],
                        "loss": masked_loss.detach().cpu().item(),
                        "all_token_loss": all_token_loss.detach().cpu().item(),
                        "masked_acc": masked_acc.detach().cpu().item(),
                        "all_token_acc": all_token_acc.detach().cpu().item(),
                        "tp_local_acc": acc.detach().cpu().item() if torch.is_tensor(acc) else float(acc),
                        "masked_token_count": masked_token_count.detach().cpu().item(),
                        "logits_slice": logits[0, 0, :6].detach().float().cpu().tolist(),
                        "input_ids_head": input_ids[0, :12].detach().cpu().tolist(),
                        "labels_head": labels[0, :12].detach().cpu().tolist(),
                        "loss_masks_head": loss_masks[0, :12].detach().float().cpu().tolist(),
                    }
                    print(f"DEEPSEEK_V4_FLASH_FORWARD_ONLY={json.dumps(snapshot, sort_keys=True)}", flush=True)

        dist.barrier()


def _attention_type_from_ratio(ratio: int) -> str:
    if ratio == 0:
        return "sliding_attention"
    if ratio == 4:
        return "compressed_sparse_attention"
    if ratio == 128:
        return "heavily_compressed_attention"
    raise ValueError(f"Unsupported DeepSeek V4 compress ratio {ratio}")


def _make_step3_batch(forward_cfg: ForwardOnlyConfig) -> dict:
    import ijson
    from megfile import smart_open

    from steptronoss.data.datasets.stepchat_dataset import StepChatJsonDataset
    from steptronoss.tokenizer.hf_compat_tokenizer import load_hf_tokenizer

    tokenizer = load_hf_tokenizer(forward_cfg.tokenizer_dir, trust_remote_code=True)

    raw_sample_index = None
    dialog = None
    tokens = None
    loss_mask = None
    with smart_open(forward_cfg.step3_data_path, "rb") as f:
        for idx, item in enumerate(ijson.items(f, "item", use_float=True)):
            if idx < forward_cfg.sample_index:
                continue
            candidate = StepChatJsonDataset.convert_dialog(item, forward_cfg.step3_data_path)
            if StepChatJsonDataset.check_error(candidate) is not None:
                continue
            encoded = _encode_step3_plain_roles(candidate, tokenizer)
            candidate_tokens = torch.as_tensor(encoded["tokens"], dtype=torch.long)
            candidate_loss_mask = torch.as_tensor(encoded["loss_mask"], dtype=torch.float32)
            scored_tokens = min(int(candidate_tokens.numel()) - 1, forward_cfg.seq_len)
            if scored_tokens <= 0:
                continue
            if candidate_loss_mask[1 : scored_tokens + 1].sum().item() == 0:
                continue
            raw_sample_index = idx
            dialog = candidate
            tokens = candidate_tokens
            loss_mask = candidate_loss_mask
            break
    if dialog is None or raw_sample_index is None:
        raise ValueError(f"No valid Step3 sample with scored loss tokens found from index {forward_cfg.sample_index}")
    assert tokens is not None
    assert loss_mask is not None
    if tokens.numel() < 2:
        raise ValueError(f"Step3 sample {raw_sample_index} has fewer than 2 tokens")

    original_num_tokens = int(tokens.numel())
    scored_tokens = min(original_num_tokens - 1, forward_cfg.seq_len)
    tokens = tokens[: scored_tokens + 1].contiguous()
    loss_mask = loss_mask[: scored_tokens + 1].contiguous()

    input_ids = tokens[:-1][None].contiguous()
    labels = tokens[1:][None].contiguous()
    loss_masks = loss_mask[1:][None].contiguous()
    decoded_preview = tokenizer.decode(tokens[: forward_cfg.preview_tokens].tolist(), skip_special_tokens=False)
    decoded_preview = " ".join(decoded_preview.split())
    roles = [message["role"] for message in dialog["conversations"]]
    metadata = {
        "step3_data_path": forward_cfg.step3_data_path,
        "raw_sample_index": raw_sample_index,
        "dialog_roles": roles,
        "original_num_tokens": original_num_tokens,
        "sample_seq_len": scored_tokens,
        "truncated": original_num_tokens - 1 > scored_tokens,
        "decoded_preview": decoded_preview,
        "tokenizer_dir": forward_cfg.tokenizer_dir,
    }
    dist.barrier()
    return _batch_to_cuda(input_ids, labels, loss_masks, metadata)


def _encode_step3_plain_roles(dialog: dict, tokenizer) -> dict:
    token_ids = []
    loss_mask = []
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if isinstance(bos_token_id, int):
        token_ids.append(bos_token_id)
        loss_mask.append(0.0)

    for message in dialog["conversations"]:
        prefix_ids = tokenizer.encode(f"{message['role']}: ", add_special_tokens=False)
        content_ids = tokenizer.encode(_message_text(message), add_special_tokens=False)
        suffix_ids = tokenizer.encode("\n", add_special_tokens=False)
        token_ids.extend(prefix_ids)
        loss_mask.extend([0.0] * len(prefix_ids))
        should_score = message["role"] == "assistant" and message.get("loss_mask", 1) == 1
        token_ids.extend(content_ids)
        loss_mask.extend([1.0 if should_score else 0.0] * len(content_ids))
        token_ids.extend(suffix_ids)
        loss_mask.extend([0.0] * len(suffix_ids))

    return {"tokens": token_ids, "loss_mask": loss_mask}


def _message_text(message: dict) -> str:
    parts = []
    for part in message["content"]:
        value = part.get("value", "")
        if value:
            parts.append(value)
    return "\n".join(parts)


def _batch_to_cuda(input_ids: torch.Tensor, labels: torch.Tensor, loss_masks: torch.Tensor, metadata: dict) -> dict:
    device = torch.cuda.current_device()
    return {
        "input_ids": input_ids.cuda(device, non_blocking=True),
        "labels": labels.cuda(device, non_blocking=True),
        "loss_masks": loss_masks.cuda(device, non_blocking=True),
        "metadata": metadata,
    }


def _distributed_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tp_size = PM.size_of("TP")
    tp_rank = PM.rank_in("TP")
    partition_vocab_size = logits.size(-1)
    local_max_values, local_indices = torch.max(logits.float(), dim=-1)
    local_indices = local_indices + tp_rank * partition_vocab_size
    gathered_values = [torch.empty_like(local_max_values) for _ in range(tp_size)]
    gathered_indices = [torch.empty_like(local_indices) for _ in range(tp_size)]
    dist.all_gather(gathered_values, local_max_values, group=PM.group_of("TP"))
    dist.all_gather(gathered_indices, local_indices, group=PM.group_of("TP"))
    values = torch.stack(gathered_values, dim=0)
    indices = torch.stack(gathered_indices, dim=0)
    best_rank = values.argmax(dim=0, keepdim=True)
    preds = indices.gather(0, best_rank).squeeze(0)
    correct = (preds == labels).float()
    all_token_acc = correct.mean()
    masked_acc = (correct * loss_masks).sum() / loss_masks.sum().clamp_min(1)
    return all_token_acc, masked_acc


def _run_pipeline_forward(model, cfg, input_ids: torch.Tensor) -> torch.Tensor | None:
    pp_rank = PM.rank_in("PP")
    pp_size = PM.size_of("PP")
    if pp_size == 1:
        return model.abstract_forward(input_ids=input_ids)

    if pp_rank == 0:
        hidden_states = model.forward_head(input_ids=input_ids)
    else:
        hidden_states = p2p.recv_forward(cfg)

    hidden_states = model.forward_chunk(hidden_states, input_ids=input_ids)

    if pp_rank == pp_size - 1:
        return model.forward_tail(hidden_states)

    p2p.send_forward(cfg, hidden_states)
    return None


if __name__ == "__main__":
    Exp().train()
