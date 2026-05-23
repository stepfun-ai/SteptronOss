from __future__ import annotations

import re
from collections.abc import Callable
from functools import cached_property

import torch
import torch.nn.functional as F
from configurize import Config, Ref
from torch import nn

from steptronoss.checkpointing.reshape_ops import ReshapeOp
from steptronoss.core import tensor_parallel
from steptronoss.core.parallel_state import PM
from steptronoss.exp.base_exp import MegatronTPConfig
from steptronoss.model.common.parallel_embedding import (
    InputEmbeddingConfig,
    OutputEmbeddingConfig,
)
from steptronoss.model.decoder_model import DecoderLLMConfig, NoopTransformerBlock
from steptronoss.model.ep_dispatcher.token_dispatcher import TokenDispatcher
from steptronoss.model.module import MegatronModule
from steptronoss.utils.general import get_position_id_from_cu_seqlens, safediv
from steptronoss.utils.utils import format_layermap

_FP8_BLOCK_SIZE = (128, 128)


def _sqrt_softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(F.softplus(x))


def _get_score_fn(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if name == "sqrtsoftplus":
        return _sqrt_softplus
    if name == "sigmoid":
        return torch.sigmoid
    if name == "softmax":
        return lambda x: F.softmax(x, dim=-1)
    raise ValueError(f"Unsupported DeepSeek V4 router scoring_func={name!r}")


def _dequant_fp8_blockwise(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    weight_f32 = weight.float()
    expanded_scale = scale.repeat_interleave(_FP8_BLOCK_SIZE[0], dim=0).repeat_interleave(_FP8_BLOCK_SIZE[1], dim=1)
    expanded_scale = expanded_scale[: weight_f32.shape[0], : weight_f32.shape[1]]
    return (weight_f32 * expanded_scale).to(torch.bfloat16)


class DeepseekV4FlashFP8Dequant(ReshapeOp):
    """Decode DeepSeek V4 Flash block-FP8 `*.weight` tensors using sibling `*.scale` tensors."""

    def forward(self, piece: dict) -> dict:
        output = {}
        for key, tensor in piece.items():
            if key.endswith(".scale"):
                continue
            if tensor.dtype == torch.float8_e4m3fn:
                scale_key = f"{key.removesuffix('.weight')}.scale"
                if scale_key not in piece:
                    raise KeyError(f"Missing FP8 scale tensor for {key}")
                tensor = _dequant_fp8_blockwise(tensor, piece[scale_key])
            output[key] = tensor
        return output

    def backward(self, piece: dict) -> dict:
        raise NotImplementedError(f"{self} is safetensors->model only")


class DeepseekV4FlashExperts(ReshapeOp):
    def __init__(self, dst_key: str, gate_up: bool):
        self.dst_key = dst_key
        self.gate_up = gate_up

    def forward(self, piece: dict) -> dict:
        expert_ids = sorted({int(match.group(1)) for key in piece if (match := re.search(r"\.experts\.(\d+)\.", key))})
        tensors = []
        for expert_id in expert_ids:
            prefix = self._expert_prefix(piece, expert_id)
            if self.gate_up:
                tensors.append(torch.cat([piece[f"{prefix}.w1.weight"], piece[f"{prefix}.w3.weight"]], dim=0))
            else:
                tensors.append(piece[f"{prefix}.w2.weight"])
        return {self.dst_key: torch.stack(tensors, dim=0)}

    @staticmethod
    def _expert_prefix(piece: dict, expert_id: int) -> str:
        suffix = f".experts.{expert_id}."
        for key in piece:
            if suffix in key and key.endswith(".weight"):
                return key.split(suffix)[0] + suffix[:-1]
        raise KeyError(f"Expert {expert_id} not found")

    def backward(self, piece: dict) -> dict:
        raise NotImplementedError(f"{self} is safetensors->model only")


def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_deepseek_v4_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """Apply DeepSeek V4 interleaved partial RoPE to the trailing rope slice."""

    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    rope_dim = cos.shape[-1]
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    rotated = rope.float() * cos + _rotate_half_interleaved(rope).float() * sin
    return torch.cat([nope, rotated.to(x.dtype)], dim=-1)


class DeepseekV4RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, qk_rope_head_dim: int, theta: float):
        super().__init__()
        self.head_dim = head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, qk_rope_head_dim, 2, dtype=torch.float32) / qk_rope_head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids = position_ids[:, None, :].float()
        freqs = (inv_freq @ position_ids).transpose(1, 2)
        cos = freqs.cos()
        sin = freqs.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class DeepseekV4UnweightedRMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)


class DeepseekV4RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class DeepseekV4GroupedLinear(nn.Linear):
    """Block-diagonal grouped linear used by DeepSeek V4 grouped output projection."""

    def __init__(self, in_features_per_group: int, out_features: int, n_groups: int, bias: bool = False):
        super().__init__(in_features_per_group, out_features, bias=bias)
        self.n_groups = n_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[:-2]
        hidden_dim = x.shape[-1]
        w = self.weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2)
        x = x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1)
        y = torch.bmm(x, w).transpose(0, 1)
        return y.reshape(*input_shape, self.n_groups, -1)


class DeepseekV4AttentionConfig(Config):
    tp_cfg: MegatronTPConfig = Ref("..tp_cfg")
    """Tensor-parallel runtime options."""
    hidden_size: int = Ref("..hidden_size")
    """Token hidden size."""
    num_attention_heads: int
    """Number of query heads."""
    num_key_value_heads: int
    """Number of KV heads. DeepSeek V4 uses single-KV MQA."""
    head_dim: int
    """Per-head attention dimension."""
    q_lora_rank: int
    """Low-rank query projection width."""
    qk_rope_head_dim: int
    """Number of trailing head channels that receive RoPE."""
    rope_theta: float
    """RoPE base for the main sliding branch."""
    compress_rope_theta: float
    """RoPE base for compressed branches."""
    attention_dropout: float
    """Attention dropout probability."""
    sliding_window: int
    """Local causal window for the normal KV branch."""
    layer_types: list[str]
    """Per-layer attention type: sliding_attention, compressed_sparse_attention, or heavily_compressed_attention."""
    compress_rates: dict[str, int]
    """Compression rates keyed by attention layer type."""
    o_groups: int
    """Number of grouped output projection groups."""
    o_lora_rank: int
    """Per-group output projection intermediate dim."""
    index_n_heads: int
    """Number of CSA indexer query heads."""
    index_head_dim: int
    """CSA indexer head dimension."""
    index_topk: int
    """Number of compressed entries selected per query by CSA."""
    layernorm_epsilon: float
    """RMSNorm epsilon."""

    def build_model(self, layer_id: int):
        return DeepseekV4Attention(self, layer_id)


class DeepseekV4HCACompressor(nn.Module):
    rope_layer_type = "compress"

    def __init__(self, cfg: DeepseekV4AttentionConfig):
        super().__init__()
        self.compress_rate = cfg.compress_rates["heavily_compressed_attention"]
        self.head_dim = cfg.head_dim
        self.kv_proj = nn.Linear(cfg.hidden_size, self.head_dim, bias=False)
        self.gate_proj = nn.Linear(cfg.hidden_size, self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.empty(self.compress_rate, self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=cfg.layernorm_epsilon)
        self.rotary_emb = DeepseekV4RotaryEmbedding(cfg.head_dim, cfg.qk_rope_head_dim, cfg.compress_rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        del q_residual, position_ids
        batch, _, _ = hidden_states.shape
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
        chunk_kv = kv[:, :usable]
        chunk_gate = gate[:, :usable]

        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, self.compress_rate, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, self.compress_rate, -1)
            chunk_gate = chunk_gate + self.position_bias.to(chunk_gate.dtype)
            compressed = self.kv_norm(
                (chunk_kv * chunk_gate.softmax(dim=2, dtype=torch.float32).to(chunk_kv.dtype)).sum(dim=2)
            )
            positions = torch.arange(n_windows, device=compressed.device)
            positions = (positions * self.compress_rate).unsqueeze(0).expand(batch, -1)
            cos, sin = self.rotary_emb(compressed, positions)
            compressed = apply_deepseek_v4_rotary(compressed.unsqueeze(1), cos, sin).squeeze(1)
        else:
            compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))
        return compressed.unsqueeze(1)


class DeepseekV4Indexer(nn.Module):
    def __init__(self, cfg: DeepseekV4AttentionConfig):
        super().__init__()
        self.compress_rate = cfg.compress_rates["compressed_sparse_attention"]
        self.num_heads = cfg.index_n_heads
        self.head_dim = cfg.index_head_dim
        self.index_topk = cfg.index_topk
        self.softmax_scale = self.head_dim**-0.5
        self.weights_scaling = self.num_heads**-0.5
        self.kv_proj = nn.Linear(cfg.hidden_size, 2 * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(cfg.hidden_size, 2 * self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.empty(self.compress_rate, 2 * self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=cfg.layernorm_epsilon)
        self.q_b_proj = nn.Linear(cfg.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(cfg.hidden_size, self.num_heads, bias=False)
        self.rotary_emb = DeepseekV4RotaryEmbedding(cfg.head_dim, cfg.qk_rope_head_dim, cfg.compress_rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        batch, seq_len, _ = hidden_states.shape
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
        chunk_kv = kv[:, :usable]
        chunk_gate = gate[:, :usable]

        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            ratio = self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + self.position_bias.to(chunk_gate.dtype)
            new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
            new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, self.head_dim), float("-inf"))
            new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim :]
            new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim :]
            if n_windows > 1:
                new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
                new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : self.head_dim]
            compressed = self.kv_norm(
                (new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2)
            )
            positions = torch.arange(n_windows, device=compressed.device)
            positions = (positions * self.compress_rate).unsqueeze(0).expand(batch, -1)
            cos, sin = self.rotary_emb(compressed, positions)
            compressed = apply_deepseek_v4_rotary(compressed.unsqueeze(1), cos, sin).squeeze(1)
        else:
            compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

        cos_q, sin_q = self.rotary_emb(hidden_states, position_ids)
        q = self.q_b_proj(q_residual).view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
        q = apply_deepseek_v4_rotary(q, cos_q, sin_q).transpose(1, 2)
        scores = torch.matmul(q.float(), compressed.transpose(-1, -2).float().unsqueeze(1))
        scores = F.relu(scores) * self.softmax_scale
        weights = self.weights_proj(hidden_states).float() * self.weights_scaling
        index_scores = (scores * weights.unsqueeze(-1)).sum(dim=2)
        topk = min(self.index_topk, compressed.shape[1])
        return index_scores.topk(topk, dim=-1).indices


class DeepseekV4CSACompressor(nn.Module):
    def __init__(self, cfg: DeepseekV4AttentionConfig):
        super().__init__()
        self.compress_rate = cfg.compress_rates["compressed_sparse_attention"]
        self.head_dim = cfg.head_dim
        self.kv_proj = nn.Linear(cfg.hidden_size, 2 * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(cfg.hidden_size, 2 * self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.empty(self.compress_rate, 2 * self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=cfg.layernorm_epsilon)
        self.rotary_emb = DeepseekV4RotaryEmbedding(cfg.head_dim, cfg.qk_rope_head_dim, cfg.compress_rope_theta)
        self.indexer = DeepseekV4Indexer(cfg)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
        chunk_kv = kv[:, :usable]
        chunk_gate = gate[:, :usable]

        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            ratio = self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + self.position_bias.to(chunk_gate.dtype)
            new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
            new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, self.head_dim), float("-inf"))
            new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim :]
            new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim :]
            if n_windows > 1:
                new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
                new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : self.head_dim]
            compressed = self.kv_norm(
                (new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2)
            )
            positions = torch.arange(n_windows, device=compressed.device)
            positions = (positions * self.compress_rate).unsqueeze(0).expand(batch, -1)
            cos, sin = self.rotary_emb(compressed, positions)
            compressed = apply_deepseek_v4_rotary(compressed.unsqueeze(1), cos, sin).squeeze(1)
        else:
            compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

        compressed_kv = compressed.unsqueeze(1)
        topk = self.indexer(hidden_states, q_residual, position_ids)
        if topk.shape[-1] == 0:
            return compressed_kv.new_zeros((batch, 1, 0, self.head_dim))
        expanded = compressed_kv.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
        idx = topk.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, -1, self.head_dim)
        return torch.gather(expanded, 3, idx).reshape(batch, 1, -1, self.head_dim)


class DeepseekV4Attention(nn.Module):
    def __init__(self, cfg: DeepseekV4AttentionConfig, layer_id: int):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        self.layer_type = cfg.layer_types[layer_id]
        self.tp_size = PM.size_of("TP")
        self.tp_rank = PM.rank_in("TP")
        self.num_heads = cfg.num_attention_heads
        self.num_local_heads = safediv(self.num_heads, self.tp_size)
        self.num_key_value_heads = cfg.num_key_value_heads
        if self.tp_size > 1 and self.num_key_value_heads != 1:
            raise NotImplementedError("DeepSeek V4 TP attention currently supports single-KV MQA only")
        self.num_local_key_value_heads = self.num_key_value_heads
        self.num_key_value_groups = self.num_local_heads // self.num_local_key_value_heads
        self.head_dim = cfg.head_dim
        self.sliding_window = cfg.sliding_window
        self.scaling = self.head_dim**-0.5
        self.local_o_groups = safediv(cfg.o_groups, self.tp_size)

        self.q_a_proj = nn.Linear(cfg.hidden_size, cfg.q_lora_rank, bias=False)
        self.q_a_norm = DeepseekV4RMSNorm(cfg.q_lora_rank, eps=cfg.layernorm_epsilon)
        if self.tp_size == 1:
            self.q_b_proj = nn.Linear(cfg.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        else:
            self.q_b_proj = tensor_parallel.ColumnParallelLinear(
                cfg.q_lora_rank,
                self.num_heads * self.head_dim,
                bias=False,
                gather_output=False,
                async_tensor_model_parallel_allreduce=cfg.tp_cfg.async_tensor_model_parallel_allreduce,
                **cfg.tp_cfg.get_tp_kwargs(),
            )
        self.q_b_norm = DeepseekV4UnweightedRMSNorm(eps=cfg.layernorm_epsilon)
        self.kv_proj = nn.Linear(cfg.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.kv_norm = DeepseekV4RMSNorm(self.num_key_value_heads * self.head_dim, eps=cfg.layernorm_epsilon)
        self.o_a_proj = DeepseekV4GroupedLinear(
            self.num_local_heads * self.head_dim // self.local_o_groups,
            self.local_o_groups * cfg.o_lora_rank,
            self.local_o_groups,
        )
        if self.tp_size == 1:
            self.o_b_proj = nn.Linear(cfg.o_groups * cfg.o_lora_rank, cfg.hidden_size, bias=False)
        else:
            self.o_b_proj = tensor_parallel.RowParallelLinear(
                cfg.o_groups * cfg.o_lora_rank,
                cfg.hidden_size,
                bias=False,
                input_is_parallel=True,
                **cfg.tp_cfg.get_tp_kwargs(),
            )
        self.sinks = nn.Parameter(torch.empty(self.num_local_heads))
        self.rotary_emb = DeepseekV4RotaryEmbedding(cfg.head_dim, cfg.qk_rope_head_dim, cfg.rope_theta)
        if self.layer_type == "compressed_sparse_attention":
            self.compressor = DeepseekV4CSACompressor(cfg)
        elif self.layer_type == "heavily_compressed_attention":
            self.compressor = DeepseekV4HCACompressor(cfg)
        elif self.layer_type == "sliding_attention":
            self.compressor = None
        else:
            raise ValueError(f"Unsupported DeepSeek V4 attention layer_type={self.layer_type!r}")

    @staticmethod
    def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, kv_heads, seq_len, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, kv_heads, n_rep, seq_len, head_dim)
        return hidden_states.reshape(batch, kv_heads * n_rep, seq_len, head_dim)

    @staticmethod
    def _build_sliding_mask(q_len: int, k_len: int, window: int, device: torch.device) -> torch.Tensor:
        q_idx = torch.arange(q_len, device=device).unsqueeze(1)
        k_idx = torch.arange(k_len, device=device).unsqueeze(0)
        return (k_idx <= q_idx) & (k_idx >= (q_idx - window + 1))

    @staticmethod
    def _linear_output(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        output = module(x)
        return output[0] if isinstance(output, tuple) else output

    def _attention_with_sink(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        key_states = self._repeat_kv(kv, self.num_key_value_groups)
        value_states = self._repeat_kv(kv, self.num_key_value_groups)
        attn_weights = torch.matmul(q, key_states.transpose(2, 3)) * self.scaling

        normal_k_len = min(q.shape[-2], attn_weights.shape[-1])
        if normal_k_len > 0:
            mask = self._build_sliding_mask(q.shape[-2], normal_k_len, self.sliding_window, q.device)
            attn_weights[..., :normal_k_len] = attn_weights[..., :normal_k_len].masked_fill(~mask, float("-inf"))

        sinks = self.sinks.reshape(1, -1, 1, 1).expand(q.shape[0], -1, q.shape[-2], -1)
        combined_logits = torch.cat([attn_weights, sinks], dim=-1)
        combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
        scores = probs[..., :-1]
        scores = F.dropout(scores, p=self.cfg.attention_dropout, training=self.training).to(value_states.dtype)
        return torch.matmul(scores, value_states)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.LongTensor) -> torch.Tensor:
        # Steptron uses [S, B, D]; HF DeepSeek V4 uses [B, S, D].
        hidden_states = hidden_states.transpose(0, 1).contiguous()
        batch, seq_len, _ = hidden_states.shape
        position_ids = position_ids.unsqueeze(0).expand(batch, -1).to(torch.long)
        q_hidden_shape = (batch, seq_len, self.num_local_heads, self.head_dim)
        kv_hidden_shape = (batch, seq_len, self.num_local_key_value_heads, self.head_dim)
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self._linear_output(self.q_b_proj, q_residual).view(*q_hidden_shape).transpose(1, 2)
        q = self.q_b_norm(q)
        q = apply_deepseek_v4_rotary(q, cos, sin)

        kv = self.kv_norm(self.kv_proj(hidden_states)).view(*kv_hidden_shape).transpose(1, 2)
        kv = apply_deepseek_v4_rotary(kv, cos, sin)
        kv = tensor_parallel.copy_to_tensor_model_parallel_region(kv)

        if self.compressor is not None:
            compressed_kv = self.compressor(hidden_states, q_residual, position_ids)
            compressed_kv = tensor_parallel.copy_to_tensor_model_parallel_region(compressed_kv)
            kv = torch.cat([kv, compressed_kv], dim=2)

        attn_output = self._attention_with_sink(q, kv)
        attn_output = apply_deepseek_v4_rotary(attn_output, cos, -sin).transpose(1, 2)
        grouped = attn_output.reshape(batch, seq_len, self.local_o_groups, -1)
        grouped = self.o_a_proj(grouped).flatten(2)
        output = self._linear_output(self.o_b_proj, grouped)
        return output.transpose(0, 1).contiguous()


class DeepseekV4HyperConnectionConfig(Config):
    hidden_size: int = Ref("..hidden_size")
    """Token hidden size."""
    hc_mult: int
    """Number of residual streams."""
    hc_sinkhorn_iters: int
    """Sinkhorn projection iterations."""
    hc_eps: float
    """Numerical floor for hyper-connection mixing."""
    layernorm_epsilon: float
    """RMSNorm epsilon."""


class DeepseekV4HyperConnection(nn.Module):
    def __init__(self, cfg: DeepseekV4HyperConnectionConfig):
        super().__init__()
        self.hc_mult = cfg.hc_mult
        self.hc_sinkhorn_iters = cfg.hc_sinkhorn_iters
        self.hc_eps = cfg.hc_eps
        self.input_norm = DeepseekV4UnweightedRMSNorm(eps=cfg.layernorm_epsilon)
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * cfg.hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        self.scale = nn.Parameter(torch.empty(3))

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        mix = F.linear(flat, self.fn.float())
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)
        hc = self.hc_mult
        pre = torch.sigmoid(mix[..., :hc] * pre_scale + self.base[:hc]) + self.hc_eps
        post = torch.sigmoid(mix[..., hc : 2 * hc] * post_scale + self.base[hc : 2 * hc]) + self.hc_eps
        comb = (
            torch.sigmoid(
                mix[..., 2 * hc :].view(*mix.shape[:-1], hc, hc) * comb_scale + self.base[2 * hc :].view(hc, hc)
            )
            + self.hc_eps
        )
        for _ in range(self.hc_sinkhorn_iters):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        return post, comb, collapsed


class DeepseekV4HyperHead(nn.Module):
    def __init__(self, cfg: DeepseekV4HyperConnectionConfig):
        super().__init__()
        self.hc_mult = cfg.hc_mult
        self.input_norm = DeepseekV4UnweightedRMSNorm(eps=cfg.layernorm_epsilon)
        self.eps = cfg.hc_eps
        self.hc_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * cfg.hidden_size))
        self.hc_base = nn.Parameter(torch.empty(self.hc_mult))
        self.hc_scale = nn.Parameter(torch.empty(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = self.input_norm(x.flatten(2).float())
        mixes = F.linear(flat, self.hc_fn.float())
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps
        return (pre.unsqueeze(-1) * x).sum(dim=2).to(x.dtype)


class DeepseekV4MoEConfig(Config):
    tp_cfg: MegatronTPConfig = Ref("..tp_cfg")
    """Tensor-parallel runtime options."""
    hidden_size: int = Ref("..hidden_size")
    """Token hidden size."""
    moe_intermediate_size: int
    """Expert and shared-expert inner dimension."""
    num_experts_per_tok: int
    """Top-k experts selected per token."""
    n_routed_experts: int
    """Total routed experts."""
    n_shared_experts: int
    """Number of shared experts represented by the shared MLP width multiplier."""
    scoring_func: str
    """Router activation: sqrtsoftplus, sigmoid, or softmax."""
    routed_scaling_factor: float
    """Scaling applied to normalized routed weights."""
    swiglu_limit: float
    """Clamp applied to routed expert gate/up projections."""
    mlp_layer_types: list[str]
    """Per-layer MLP type: moe or hash_moe."""
    vocab_size: int
    """Vocabulary size for hash router token-id table."""
    layernorm_epsilon: float
    """RMSNorm epsilon."""


class DeepseekV4MLP(nn.Module):
    def __init__(self, cfg: DeepseekV4MoEConfig):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.intermediate_size = cfg.moe_intermediate_size * cfg.n_shared_experts
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DeepseekV4Experts(nn.Module):
    def __init__(self, cfg: DeepseekV4MoEConfig):
        super().__init__()
        self.num_experts = cfg.n_routed_experts
        self.tp_size = PM.size_of("TP")
        self.ep_size = PM.size_of("EP")
        self.ep_rank = PM.rank_in("EP")
        self.num_local_experts = safediv(self.num_experts, self.ep_size)
        self.local_expert_offset = self.ep_rank * self.num_local_experts
        self.hidden_dim = cfg.hidden_size
        self.intermediate_dim = cfg.moe_intermediate_size
        self.limit = cfg.swiglu_limit
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_local_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(torch.empty(self.num_local_experts, self.hidden_dim, self.intermediate_dim))
        if self.ep_size > 1:
            self.gate_up_proj.expert_model_parallel = True
            self.down_proj.expert_model_parallel = True
        if self.ep_size > 1 and self.tp_size > 1:
            self.gate_up_proj.register_hook(lambda grad: grad / self.tp_size)
            self.down_proj.register_hook(lambda grad: grad / self.tp_size)

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return F.silu(gate) * up

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final = torch.zeros_like(hidden_states)
        for expert_idx in range(self.num_local_experts):
            token_idx, top_k_pos = torch.where(top_k_index == expert_idx)
            if token_idx.numel() == 0:
                continue
            current = self._apply_gate(F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx]))
            current = F.linear(current, self.down_proj[expert_idx])
            current = current * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final


class DeepseekV4TopKRouter(nn.Module):
    def __init__(self, cfg: DeepseekV4MoEConfig):
        super().__init__()
        self.top_k = cfg.num_experts_per_tok
        self.num_experts = cfg.n_routed_experts
        self.hidden_dim = cfg.hidden_size
        self.score_fn = _get_score_fn(cfg.scoring_func)
        self.routed_scaling_factor = cfg.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts), persistent=True)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat.float(), self.weight.float())
        scores = self.score_fn(logits)
        indices = torch.topk(scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices


class DeepseekV4HashRouter(nn.Module):
    def __init__(self, cfg: DeepseekV4MoEConfig):
        super().__init__()
        self.top_k = cfg.num_experts_per_tok
        self.num_experts = cfg.n_routed_experts
        self.hidden_dim = cfg.hidden_size
        self.score_fn = _get_score_fn(cfg.scoring_func)
        self.routed_scaling_factor = cfg.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.register_buffer("tid2eid", torch.zeros(cfg.vocab_size, self.top_k, dtype=torch.long), persistent=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat.float(), self.weight.float())
        scores = self.score_fn(logits)
        indices = self.tid2eid[input_ids.reshape(-1)].long()
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices


class DeepseekV4SparseMoeBlock(nn.Module):
    def __init__(self, cfg: DeepseekV4MoEConfig, layer_id: int):
        super().__init__()
        self.is_hash = cfg.mlp_layer_types[layer_id] == "hash_moe"
        self.gate = DeepseekV4HashRouter(cfg) if self.is_hash else DeepseekV4TopKRouter(cfg)
        self.experts = DeepseekV4Experts(cfg)
        self.shared_experts = DeepseekV4MLP(cfg)
        self.dispatcher = TokenDispatcher("EP", cfg.n_routed_experts) if PM.size_of("EP") > 1 else None

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        seq_len, batch, hidden_dim = hidden_states.shape
        residual = hidden_states
        hidden_states_bsh = hidden_states.transpose(0, 1).contiguous()
        flat = hidden_states_bsh.view(-1, hidden_dim)
        if self.is_hash:
            if input_ids is None:
                raise ValueError("DeepSeek V4 hash_moe layers require input_ids")
            _, weights, indices = self.gate(hidden_states_bsh, input_ids)
        else:
            _, weights, indices = self.gate(hidden_states_bsh)
        if self.dispatcher is not None:
            dispatched_flat, indices, weights = self.dispatcher.dispatch(flat, indices, weights)
            routed_flat = self.experts(dispatched_flat, indices, weights)
            routed_flat = self.dispatcher.combine(routed_flat)
        else:
            routed_flat = self.experts(flat, indices, weights)
        routed = routed_flat.view(batch, seq_len, hidden_dim).transpose(0, 1).contiguous()
        return routed + self.shared_experts(residual)


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(self, cfg: DeepseekV4ModelConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = cfg.attn_cfg.build_model(layer_id)
        self.mlp = DeepseekV4SparseMoeBlock(cfg.moe_cfg, layer_id)
        self.input_layernorm = DeepseekV4RMSNorm(cfg.hidden_size, eps=cfg.layernorm_epsilon)
        self.post_attention_layernorm = DeepseekV4RMSNorm(cfg.hidden_size, eps=cfg.layernorm_epsilon)
        self.attn_hc = DeepseekV4HyperConnection(cfg.hc_cfg)
        self.ffn_hc = DeepseekV4HyperConnection(cfg.hc_cfg)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        dtype = hidden_states.dtype
        post, comb, collapsed = self.attn_hc(hidden_states)
        attn_output = self.self_attn(self.input_layernorm(collapsed), position_ids)
        hidden_states = post.to(dtype).unsqueeze(-1) * attn_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype), hidden_states
        )

        post, comb, collapsed = self.ffn_hc(hidden_states)
        mlp_output = self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)
        return post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2) + torch.matmul(comb.to(dtype), hidden_states)


class DeepseekV4InputEmbeddingConfig(InputEmbeddingConfig):
    def __init__(self):
        super().__init__()
        self.vocab_size = 129280
        self.hidden_size = Ref("..hidden_size")
        self.embedding_weights_in_fp32 = False
        self.fp32_residual_connection = False


class DeepseekV4OutputEmbeddingConfig(OutputEmbeddingConfig):
    def build_model(self, tied_embedding_weight: torch.Tensor | None = None):
        return DeepseekV4OutputEmbedding(cfg=self, tied_word_embedding_weight=tied_embedding_weight)

    def __init__(self):
        super().__init__()
        self.vocab_size = 129280
        self.hidden_size = Ref("..hidden_size")
        self.fp32_rms_norm = True
        self.fp32_lm_head_out = True
        self.rms_norm_zero_gamma = False
        self.layernorm_epsilon = 1e-6
        self.gather_output = False


class DeepseekV4OutputEmbedding(torch.nn.Module):
    def __init__(
        self,
        cfg: DeepseekV4OutputEmbeddingConfig,
        tied_word_embedding_weight: torch.Tensor | None,
    ):
        super().__init__()
        self.norm = DeepseekV4RMSNorm(cfg.hidden_size, eps=cfg.layernorm_epsilon)
        self.output = tensor_parallel.ColumnParallelLinear(
            cfg.hidden_size,
            cfg.vocab_size,
            bias=False,
            gather_output=cfg.gather_output,
            async_tensor_model_parallel_allreduce=cfg.tp_cfg.async_tensor_model_parallel_allreduce,
            params_dtype=cfg.tp_cfg.params_dtype,
            gradient_accumulation_fusion=cfg.tp_cfg.gradient_accumulation_fusion,
            sequence_parallel_enabled=cfg.tp_cfg.sequence_parallel,
            tie_word_embeddings_weight=tied_word_embedding_weight,
            fp32_output=cfg.fp32_lm_head_out,
        )

    def forward(self, hidden_states, **kwargs):
        del kwargs
        hidden_states = self.norm(hidden_states)
        return self.output(hidden_states)[0]


class DeepseekV4ModelConfig(DecoderLLMConfig):
    """DeepSeek V4 model config following the HuggingFace `deepseek_v4` layout."""

    attn_cfg = DeepseekV4AttentionConfig
    moe_cfg = DeepseekV4MoEConfig
    hc_cfg = DeepseekV4HyperConnectionConfig
    tok_embed_cfg = DeepseekV4InputEmbeddingConfig
    out_embed_cfg = DeepseekV4OutputEmbeddingConfig
    vocab_size: int
    """Vocabulary size shared by input embedding, hash router, and LM head."""

    def __init__(self):
        super().__init__()
        self.vocab_size = 129280
        self.hidden_size = 4096
        self.num_layers = 43
        self.layernorm_epsilon = 1e-6
        self.rms_norm_zero_gamma = False
        self.recompute = False
        self.tie_embedding = False
        self.params_dtype = torch.bfloat16
        self.variable_seq_lengths = True

        self.attn_cfg.hidden_size = Ref("..hidden_size")
        self.attn_cfg.tp_cfg = Ref("..tp_cfg")
        self.attn_cfg.num_attention_heads = 64
        self.attn_cfg.num_key_value_heads = 1
        self.attn_cfg.head_dim = 512
        self.attn_cfg.q_lora_rank = 1024
        self.attn_cfg.qk_rope_head_dim = 64
        self.attn_cfg.rope_theta = 10000.0
        self.attn_cfg.compress_rope_theta = 160000.0
        self.attn_cfg.attention_dropout = 0.0
        self.attn_cfg.sliding_window = 128
        self.attn_cfg.layer_types = ["heavily_compressed_attention"] * min(self.num_layers, 2) + [
            "compressed_sparse_attention" if i % 2 else "heavily_compressed_attention"
            for i in range(max(self.num_layers - 2, 0))
        ]
        self.attn_cfg.compress_rates = {
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 128,
        }
        self.attn_cfg.o_groups = 8
        self.attn_cfg.o_lora_rank = 1024
        self.attn_cfg.index_n_heads = 64
        self.attn_cfg.index_head_dim = 128
        self.attn_cfg.index_topk = 512
        self.attn_cfg.layernorm_epsilon = 1e-6

        self.moe_cfg.hidden_size = Ref("..hidden_size")
        self.moe_cfg.tp_cfg = Ref("..tp_cfg")
        self.moe_cfg.moe_intermediate_size = 2048
        self.moe_cfg.num_experts_per_tok = 6
        self.moe_cfg.n_routed_experts = 256
        self.moe_cfg.n_shared_experts = 1
        self.moe_cfg.scoring_func = "sqrtsoftplus"
        self.moe_cfg.routed_scaling_factor = 1.5
        self.moe_cfg.swiglu_limit = 10.0
        self.moe_cfg.mlp_layer_types = ["hash_moe"] * min(self.num_layers, 3) + ["moe"] * max(self.num_layers - 3, 0)
        self.moe_cfg.vocab_size = Ref("..tok_embed_cfg.vocab_size")
        self.moe_cfg.layernorm_epsilon = 1e-6

        # DecoderLLMConfig still owns this generic FFN sub-config. DeepSeek V4
        # does not build it, but configurize sanity_check recurses through it.
        self.ffn_cfg.hidden_size = Ref("..hidden_size")
        self.ffn_cfg.ffn_hidden_size = Ref("..moe_cfg.moe_intermediate_size")
        self.ffn_cfg.layernorm_epsilon = Ref("..layernorm_epsilon")
        self.ffn_cfg.rms_norm_zero_gamma = False
        self.ffn_cfg.swiglu_recompute_silu_out_proj = True

        self.hc_cfg.hidden_size = Ref("..hidden_size")
        self.hc_cfg.hc_mult = 4
        self.hc_cfg.hc_sinkhorn_iters = 20
        self.hc_cfg.hc_eps = 1e-6
        self.hc_cfg.layernorm_epsilon = 1e-6

        self.tok_embed_cfg.vocab_size = Ref("..vocab_size")
        self.out_embed_cfg.vocab_size = Ref("..vocab_size")
        self.out_embed_cfg.layernorm_epsilon = Ref("..layernorm_epsilon")

        self.parallel_cfg.tensor_model_parallel_size = 1
        self.parallel_cfg.pipeline_model_parallel_size = 1
        self.parallel_cfg.virtual_pipeline_model_parallel_size = 1
        self.parallel_cfg.context_parallel_size = 1
        self.parallel_cfg.expert_model_parallel_size = 1
        self.parallel_cfg.expert_tensor_parallel_size = 1
        self.tp_cfg.sequence_parallel = False
        self.tp_cfg.async_tensor_model_parallel_allreduce = False

    def sanity_check(self):
        super().sanity_check()
        if len(self.attn_cfg.layer_types) != self.num_layers:
            raise ValueError("DeepSeek V4 attention layer_types must match num_layers")
        if len(self.moe_cfg.mlp_layer_types) != self.num_layers:
            raise ValueError("DeepSeek V4 mlp_layer_types must match num_layers")
        tp_size = self.parallel_cfg.tensor_model_parallel_size
        ep_size = self.parallel_cfg.expert_model_parallel_size
        if self.parallel_cfg.context_parallel_size != 1:
            raise NotImplementedError("DeepSeek V4 initial training path currently supports CP=1 only")
        if self.parallel_cfg.expert_tensor_parallel_size != 1:
            raise NotImplementedError("DeepSeek V4 routed experts support EP only; keep ETP=1")
        if self.attn_cfg.num_attention_heads % tp_size != 0:
            raise ValueError("DeepSeek V4 num_attention_heads must be divisible by TP")
        if tp_size > 1 and self.attn_cfg.num_key_value_heads != 1:
            raise NotImplementedError("DeepSeek V4 TP attention currently supports single-KV MQA only")
        if self.attn_cfg.o_groups % tp_size != 0:
            raise ValueError("DeepSeek V4 o_groups must be divisible by TP")
        if self.moe_cfg.n_routed_experts % ep_size != 0:
            raise ValueError("DeepSeek V4 n_routed_experts must be divisible by EP")

    def build_model(self):
        return DeepseekV4Model(cfg=self, layer_map=self.build_layer_map())


class DeepseekV4Model(MegatronModule):
    def __init__(self, cfg: DeepseekV4ModelConfig, layer_map=None):
        super().__init__()
        self.cfg = cfg
        self.layer_map = layer_map or cfg.build_layer_map()
        from loguru import logger

        logger.bind(at=0).info(format_layermap(self.layer_map))
        self.build(self.layer_map)
        self.name_parameters()

    def build(self, layer_map: dict[int, dict[int, dict[int, dict]]]):
        from steptronoss.core.parallel_state import get_vpp_rank

        self.layers = nn.ModuleList()
        pp_rank, vp_rank = PM.rank_in("PP"), get_vpp_rank()
        for layer_id in layer_map[pp_rank][vp_rank]:
            self.layers.append(DeepseekV4DecoderLayer(self.cfg, layer_id=layer_id))
        if len(self.layers) == 0:
            self.layers.append(NoopTransformerBlock())

        if self.is_pipeline_first_stage():
            self.tok_embeddings = self.cfg.tok_embed_cfg.build_model()

        if self.is_pipeline_last_stage():
            self.hc_head = DeepseekV4HyperHead(self.cfg.hc_cfg)
            self.out_embeddings = self.cfg.out_embed_cfg.build_model()

    def forward_head(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.tok_embeddings(input_ids=input_ids, **kwargs)

    def forward_chunk(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.IntTensor | None = None,
        position_id: torch.IntTensor | None = None,
        input_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if cu_seqlens is not None and position_id is None:
            position_id = get_position_id_from_cu_seqlens(cu_seqlens)
        if position_id is None:
            position_id = torch.arange(hidden_states.shape[0], device=hidden_states.device, dtype=torch.long)
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.unsqueeze(2).expand(-1, -1, self.cfg.hc_cfg.hc_mult, -1).contiguous()
        elif hidden_states.dim() != 4:
            raise ValueError(f"DeepSeek V4 expected 3D embeddings or 4D hyper streams, got {hidden_states.shape}")
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids=position_id, input_ids=input_ids)
        return hidden_states

    def forward_tail(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = self.hc_head(hidden_states)
        return self.out_embeddings(hidden_states, **kwargs)

    @cached_property
    def _hf_key_map(self) -> dict[str, str]:
        mapping = {}
        if self.is_pipeline_first_stage():
            mapping["tok_embeddings.word_embeddings.weight"] = "model.embed_tokens.weight"
        if self.is_pipeline_last_stage():
            mapping.update({
                "hc_head.hc_fn": "model.hc_head.hc_fn",
                "hc_head.hc_base": "model.hc_head.hc_base",
                "hc_head.hc_scale": "model.hc_head.hc_scale",
                "out_embeddings.norm.weight": "model.norm.weight",
                "out_embeddings.output.weight": "lm_head.weight",
            })
        for local_id, layer in enumerate(self.layers):
            if isinstance(layer, NoopTransformerBlock):
                continue
            hf_id = layer.layer_id
            prefix_local = f"layers.{local_id}"
            prefix_hf = f"model.layers.{hf_id}"
            direct_suffixes = [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "attn_hc.fn",
                "attn_hc.base",
                "attn_hc.scale",
                "ffn_hc.fn",
                "ffn_hc.base",
                "ffn_hc.scale",
                "self_attn.q_a_proj.weight",
                "self_attn.q_a_norm.weight",
                "self_attn.q_b_proj.weight",
                "self_attn.kv_proj.weight",
                "self_attn.kv_norm.weight",
                "self_attn.o_a_proj.weight",
                "self_attn.o_b_proj.weight",
                "self_attn.sinks",
                "mlp.gate.weight",
                "mlp.experts.gate_up_proj",
                "mlp.experts.down_proj",
                "mlp.shared_experts.gate_proj.weight",
                "mlp.shared_experts.up_proj.weight",
                "mlp.shared_experts.down_proj.weight",
            ]
            if hasattr(layer.mlp.gate, "e_score_correction_bias"):
                direct_suffixes.append("mlp.gate.e_score_correction_bias")
            if hasattr(layer.mlp.gate, "tid2eid"):
                direct_suffixes.append("mlp.gate.tid2eid")
            if layer.self_attn.compressor is not None:
                direct_suffixes.extend([
                    "self_attn.compressor.kv_proj.weight",
                    "self_attn.compressor.gate_proj.weight",
                    "self_attn.compressor.position_bias",
                    "self_attn.compressor.kv_norm.weight",
                ])
            if isinstance(layer.self_attn.compressor, DeepseekV4CSACompressor):
                direct_suffixes.extend([
                    "self_attn.compressor.indexer.kv_proj.weight",
                    "self_attn.compressor.indexer.gate_proj.weight",
                    "self_attn.compressor.indexer.position_bias",
                    "self_attn.compressor.indexer.kv_norm.weight",
                    "self_attn.compressor.indexer.q_b_proj.weight",
                    "self_attn.compressor.indexer.weights_proj.weight",
                ])
            for suffix in direct_suffixes:
                mapping[f"{prefix_local}.{suffix}"] = f"{prefix_hf}.{suffix}"
        return mapping

    @cached_property
    def reshaper(self):
        return self.build_reshaper()

    def build_reshaper(self):
        from steptronoss.checkpointing.reshape_ops import (
            ColumnParallel,
            Duplicate,
            KeepThisTP,
            OnlineReshaper,
            Rename,
            RowParallel,
            Script,
            VocabPad,
        )

        def src_with_scale(weight_key: str) -> list[str]:
            if weight_key.endswith(".weight"):
                return [weight_key, f"{weight_key.removesuffix('.weight')}.scale"]
            return [weight_key]

        def replicated(weight_key: str, dst_key: str):
            return Script(
                src=src_with_scale(weight_key),
                op=DeepseekV4FlashFP8Dequant() + Duplicate() + KeepThisTP() + Rename(f"{dst_key}: {weight_key}"),
                dst=dst_key,
            )

        def column(weight_key: str, dst_key: str):
            return Script(
                src=src_with_scale(weight_key),
                op=DeepseekV4FlashFP8Dequant() + ColumnParallel() + KeepThisTP() + Rename(f"{dst_key}: {weight_key}"),
                dst=dst_key,
            )

        def row(weight_key: str, dst_key: str):
            return Script(
                src=src_with_scale(weight_key),
                op=DeepseekV4FlashFP8Dequant() + RowParallel() + KeepThisTP() + Rename(f"{dst_key}: {weight_key}"),
                dst=dst_key,
            )

        scripts = []
        if self.is_pipeline_first_stage():
            scripts.append(
                Script(
                    src="embed.weight",
                    op=VocabPad(target_vocab_size=self.cfg.tok_embed_cfg.vocab_size, dim=0, pad_type="last")
                    + ColumnParallel()
                    + KeepThisTP()
                    + Rename("tok_embeddings.word_embeddings.weight: embed.weight"),
                    dst="tok_embeddings.word_embeddings.weight",
                )
            )
        if self.is_pipeline_last_stage():
            scripts.extend([
                Script(
                    src="head.weight",
                    op=VocabPad(target_vocab_size=self.cfg.out_embed_cfg.vocab_size, dim=0, pad_type="last")
                    + ColumnParallel()
                    + KeepThisTP()
                    + Rename("out_embeddings.output.weight: head.weight"),
                    dst="out_embeddings.output.weight",
                ),
                replicated("norm.weight", "out_embeddings.norm.weight"),
                replicated("hc_head_fn", "hc_head.hc_fn"),
                replicated("hc_head_base", "hc_head.hc_base"),
                replicated("hc_head_scale", "hc_head.hc_scale"),
            ])

        for local_id, layer in enumerate(self.layers):
            if isinstance(layer, NoopTransformerBlock):
                continue
            local = f"layers.{local_id}"
            release = f"layers.{layer.layer_id}"
            attn = f"{local}.self_attn"
            rel_attn = f"{release}.attn"
            mlp = f"{local}.mlp"
            rel_ffn = f"{release}.ffn"

            scripts.extend([
                replicated(f"{release}.attn_norm.weight", f"{local}.input_layernorm.weight"),
                replicated(f"{release}.ffn_norm.weight", f"{local}.post_attention_layernorm.weight"),
                replicated(f"{release}.hc_attn_fn", f"{local}.attn_hc.fn"),
                replicated(f"{release}.hc_attn_base", f"{local}.attn_hc.base"),
                replicated(f"{release}.hc_attn_scale", f"{local}.attn_hc.scale"),
                replicated(f"{release}.hc_ffn_fn", f"{local}.ffn_hc.fn"),
                replicated(f"{release}.hc_ffn_base", f"{local}.ffn_hc.base"),
                replicated(f"{release}.hc_ffn_scale", f"{local}.ffn_hc.scale"),
                column(f"{rel_attn}.attn_sink", f"{attn}.sinks"),
                replicated(f"{rel_attn}.wq_a.weight", f"{attn}.q_a_proj.weight"),
                replicated(f"{rel_attn}.q_norm.weight", f"{attn}.q_a_norm.weight"),
                column(f"{rel_attn}.wq_b.weight", f"{attn}.q_b_proj.weight"),
                replicated(f"{rel_attn}.wkv.weight", f"{attn}.kv_proj.weight"),
                replicated(f"{rel_attn}.kv_norm.weight", f"{attn}.kv_norm.weight"),
                column(f"{rel_attn}.wo_a.weight", f"{attn}.o_a_proj.weight"),
                row(f"{rel_attn}.wo_b.weight", f"{attn}.o_b_proj.weight"),
            ])

            if layer.self_attn.compressor is not None:
                scripts.extend([
                    replicated(f"{rel_attn}.compressor.wkv.weight", f"{attn}.compressor.kv_proj.weight"),
                    replicated(f"{rel_attn}.compressor.wgate.weight", f"{attn}.compressor.gate_proj.weight"),
                    replicated(f"{rel_attn}.compressor.ape", f"{attn}.compressor.position_bias"),
                    replicated(f"{rel_attn}.compressor.norm.weight", f"{attn}.compressor.kv_norm.weight"),
                ])
            if hasattr(layer.self_attn.compressor, "indexer"):
                scripts.extend([
                    replicated(
                        f"{rel_attn}.indexer.compressor.wkv.weight",
                        f"{attn}.compressor.indexer.kv_proj.weight",
                    ),
                    replicated(
                        f"{rel_attn}.indexer.compressor.wgate.weight",
                        f"{attn}.compressor.indexer.gate_proj.weight",
                    ),
                    replicated(f"{rel_attn}.indexer.compressor.ape", f"{attn}.compressor.indexer.position_bias"),
                    replicated(
                        f"{rel_attn}.indexer.compressor.norm.weight",
                        f"{attn}.compressor.indexer.kv_norm.weight",
                    ),
                    replicated(f"{rel_attn}.indexer.wq_b.weight", f"{attn}.compressor.indexer.q_b_proj.weight"),
                    replicated(
                        f"{rel_attn}.indexer.weights_proj.weight",
                        f"{attn}.compressor.indexer.weights_proj.weight",
                    ),
                ])

            scripts.append(replicated(f"{rel_ffn}.gate.weight", f"{mlp}.gate.weight"))
            if hasattr(layer.mlp.gate, "tid2eid"):
                scripts.append(replicated(f"{rel_ffn}.gate.tid2eid", f"{mlp}.gate.tid2eid"))
            if hasattr(layer.mlp.gate, "e_score_correction_bias"):
                scripts.append(replicated(f"{rel_ffn}.gate.bias", f"{mlp}.gate.e_score_correction_bias"))
            scripts.extend([
                replicated(f"{rel_ffn}.shared_experts.w1.weight", f"{mlp}.shared_experts.gate_proj.weight"),
                replicated(f"{rel_ffn}.shared_experts.w3.weight", f"{mlp}.shared_experts.up_proj.weight"),
                replicated(f"{rel_ffn}.shared_experts.w2.weight", f"{mlp}.shared_experts.down_proj.weight"),
            ])

            experts = layer.mlp.experts
            expert_ids = range(experts.local_expert_offset, experts.local_expert_offset + experts.num_local_experts)
            gate_up_src = []
            down_src = []
            for expert_id in expert_ids:
                for proj in ("w1", "w3"):
                    key = f"{rel_ffn}.experts.{expert_id}.{proj}.weight"
                    gate_up_src.extend(src_with_scale(key))
                key = f"{rel_ffn}.experts.{expert_id}.w2.weight"
                down_src.extend(src_with_scale(key))
            scripts.extend([
                Script(
                    src=gate_up_src,
                    op=DeepseekV4FlashFP8Dequant()
                    + DeepseekV4FlashExperts(f"{mlp}.experts.gate_up_proj", gate_up=True),
                    dst=f"{mlp}.experts.gate_up_proj",
                ),
                Script(
                    src=down_src,
                    op=DeepseekV4FlashFP8Dequant() + DeepseekV4FlashExperts(f"{mlp}.experts.down_proj", gate_up=False),
                    dst=f"{mlp}.experts.down_proj",
                ),
            ])

        return OnlineReshaper(scripts)

    def load_hf_state_dict(self, state_dict, strict=True):
        if "embed.weight" in state_dict:
            return self.load_state_dict(self.reshaper.forward(state_dict), strict=strict)
        translated = {}
        for local_key, hf_key in self._hf_key_map.items():
            if hf_key in state_dict:
                translated[local_key] = state_dict[hf_key]
        return self.load_state_dict(translated, strict=strict)

    def hf_state_dict(self):
        state_dict = self.state_dict()
        return {hf_key: state_dict[local_key].detach().cpu() for local_key, hf_key in self._hf_key_map.items()}

    def name_parameters(self):
        for p in self.parameters():
            p._log_name = "other"
        for layer in self.layers:
            for _n, p in layer.named_parameters():
                p._log_name = f"layer{layer.layer_id}"
        if self.is_pipeline_first_stage():
            for p in self.tok_embeddings.parameters():
                p._log_name = "tok_embeddings"
        if self.is_pipeline_last_stage():
            for p in self.out_embeddings.parameters():
                p._log_name = "output"
