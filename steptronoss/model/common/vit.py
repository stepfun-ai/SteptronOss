from __future__ import annotations

import math
from functools import cached_property

import torch
from configurize import Config
from torch import nn
from torch.nn import functional as F

from steptronoss.core.parallel_state import PM
from steptronoss.core.tensor_parallel import set_tensor_model_parallel_attributes
from steptronoss.core.tensor_parallel.mappings import (
    reduce_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from steptronoss.exp.base_exp import ParallelConfig


def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(1.702 * x)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs in the last dimension for RoPE."""

    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_emb(
    freqs: torch.Tensor,
    tensor: torch.Tensor,
    *,
    start_index: int = 0,
    scale: float = 1.0,
) -> torch.Tensor:
    """Apply cached rotary frequencies to the last dimension of a q/k tensor."""

    dtype = tensor.dtype
    freqs = freqs.to(device=tensor.device, dtype=torch.float32)
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    if rot_dim > tensor.shape[-1]:
        raise ValueError(f"feature dimension {tensor.shape[-1]} is too small for rot_dim {rot_dim}")

    tensor_left = tensor[..., :start_index]
    tensor_rot = tensor[..., start_index:end_index].float()
    tensor_right = tensor[..., end_index:]
    tensor_rot = (tensor_rot * freqs.cos() * scale) + (rotate_half(tensor_rot) * freqs.sin() * scale)
    return torch.cat((tensor_left, tensor_rot.to(dtype), tensor_right), dim=-1)


class EncoderRope2D(nn.Module):
    """Cacheable 2D rotary positional embedding for vision attention."""

    def __init__(
        self,
        dim: int,
        max_grid_height: int,
        max_grid_width: int,
        *,
        use_cls_token: bool = False,
        theta: int | float = 10000,
        max_freq: int = 10,
        num_freqs: int = 1,
        theta_rescale_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_grid_height = max_grid_height
        self.max_grid_width = max_grid_width
        self.use_cls_token = use_cls_token
        self.theta = theta * theta_rescale_factor ** (dim / (dim - 2))
        self.max_freq = max_freq
        self.num_freqs = num_freqs
        self.freqs_cache: torch.Tensor
        self.register_buffer("freqs_cache", self._compute_2d_freqs(), persistent=False)

    @staticmethod
    def _compute_inv_freq(base: int | float, dim: int) -> torch.Tensor:
        return 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    @staticmethod
    def _compute_freqs(position: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
        freqs = torch.einsum("..., f -> ... f", position.to(inv_freq.dtype), inv_freq)
        return freqs.repeat_interleave(2, dim=-1)

    def _compute_2d_freqs(self) -> torch.Tensor:
        grid_h_range = torch.arange(self.max_grid_height, dtype=torch.float32)
        grid_w_range = torch.arange(self.max_grid_width, dtype=torch.float32)
        if self.use_cls_token:
            grid_h_range += 1
            grid_w_range += 1

        inv_freq = self._compute_inv_freq(self.theta, self.dim // 2)
        freqs_h = self._compute_freqs(grid_h_range, inv_freq)[:, None].expand(
            self.max_grid_height, self.max_grid_width, -1
        )
        freqs_w = self._compute_freqs(grid_w_range, inv_freq)[None, :].expand(
            self.max_grid_height, self.max_grid_width, -1
        )
        freqs = torch.cat([freqs_w, freqs_h], dim=-1).reshape(self.max_grid_height * self.max_grid_width, -1)
        if self.use_cls_token:
            freqs = torch.cat([torch.zeros(1, freqs.shape[-1]), freqs], dim=0)
        return freqs[None, None, ...]

    def forward(self, q: torch.Tensor, k: torch.Tensor, grid_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        freqs: torch.Tensor
        if grid_hw == (self.max_grid_height, self.max_grid_width):
            freqs = self.freqs_cache
        else:
            rows = torch.arange(grid_hw[0], device=q.device).view(-1, 1)
            cols = torch.arange(grid_hw[1], device=q.device).view(1, -1)
            positions = (rows * self.max_grid_width + cols).reshape(-1).to(torch.long)
            if self.use_cls_token:
                positions = torch.cat([torch.zeros(1, device=q.device, dtype=torch.long), positions + 1], dim=0)
            freqs = self.freqs_cache.index_select(2, positions.to(self.freqs_cache.device))

        return apply_rotary_emb(freqs, q), apply_rotary_emb(freqs, k)


def _interpolate_positional_embedding(
    position_embedding: torch.Tensor, target_tokens: int, *, use_cls_token: bool
) -> torch.Tensor:
    if position_embedding.dim() == 2:
        position_embedding = position_embedding.unsqueeze(0)

    if position_embedding.shape[1] == target_tokens:
        return position_embedding

    if use_cls_token:
        cls_token = position_embedding[:, :1]
        grid = position_embedding[:, 1:]
    else:
        cls_token = None
        grid = position_embedding

    source_tokens = grid.shape[1]
    source_size = math.isqrt(source_tokens)
    target_grid_tokens = target_tokens - (1 if use_cls_token else 0)
    target_size = math.isqrt(target_grid_tokens)

    if source_size * source_size != source_tokens:
        raise ValueError(f"Expected square positional grid, got {source_tokens} tokens")
    if target_size * target_size != target_grid_tokens:
        raise ValueError(f"Expected square target grid, got {target_grid_tokens} tokens")

    dtype = position_embedding.dtype
    grid = grid.reshape(1, source_size, source_size, -1).permute(0, 3, 1, 2).float()
    grid = F.interpolate(
        grid,
        size=(target_size, target_size),
        mode="bicubic",
        align_corners=False,
    ).to(dtype)
    grid = grid.permute(0, 2, 3, 1).reshape(1, target_grid_tokens, -1)
    if use_cls_token:
        assert cls_token is not None
        return torch.cat([cls_token, grid], dim=1)
    return grid


class VisionTransformerParallelConfig(ParallelConfig):
    """Default single-rank mesh for the vision encoder."""

    def __init__(self):
        super().__init__()
        self.tensor_model_parallel_size = 1
        self.pipeline_model_parallel_size = 1
        self.virtual_pipeline_model_parallel_size = 1
        self.context_parallel_size = 1
        self.expert_model_parallel_size = 1
        self.expert_tensor_parallel_size = 1


class VisionTransformerConfig(Config):  # type: ignore[no-any-unimported]
    parallel_cfg = VisionTransformerParallelConfig
    """Parallel mesh used while building/running the vision encoder."""

    in_channels: int
    """Input image channel count."""

    image_size: int
    """Reference input size used by the patch embedder."""

    patch_size: int
    """Square patch size for the initial convolution."""

    hidden_size: int
    """Vision backbone width."""

    ffn_hidden_size: int
    """MLP hidden size inside each vision block."""

    num_layers: int
    """Number of vision transformer blocks."""

    num_attention_heads: int
    """Attention head count."""

    vit_downsampler_hidden_dim: int
    """Intermediate channel count for the spatial downsampler."""

    output_dim: int
    """Channel count produced for the language projector."""

    layernorm_epsilon: float
    """Layer norm epsilon."""

    attention_dropout: float = 0.0
    """Attention dropout probability."""

    use_rope2d: bool = False
    """Whether to apply 2D RoPE in each vision self-attention layer."""

    rope_theta: int | float = 10000
    """Base theta for 2D RoPE."""

    rope_max_freq: int = 10
    """Maximum frequency parameter kept for HF Step3-VL config parity."""

    rope_num_freqs: int = 1
    """Number of frequency bands kept for HF Step3-VL config parity."""

    rope_theta_rescale_factor: float = 1.0
    """Theta rescale factor for 2D RoPE."""

    layer_scale_init_value: float | None = None
    """Optional residual layer-scale initialization."""

    use_cls_token: bool = True
    """Whether to prepend a learnable CLS token before the transformer."""

    use_ln_pre: bool = False
    """Whether to apply a pre-transformer layer norm to patch embeddings."""

    patch_embed_bias: bool = True
    """Whether the patch embedding convolution uses bias."""

    vit_downsampler1_kernel_size: int = 2
    """Kernel size for the first spatial downsampler."""

    vit_downsampler1_padding: int = 0
    """Padding for the first spatial downsampler."""

    vit_downsampler2_kernel_size: int = 3
    """Kernel size for the second spatial downsampler."""

    vit_downsampler2_padding: int = 1
    """Padding for the second spatial downsampler."""

    def build_model(self) -> VisionTransformer:
        return VisionTransformer(cfg=self)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class VisionMLP(nn.Module):
    def __init__(self, cfg: VisionTransformerConfig):
        super().__init__()
        self.fc1 = TPColumnLinear(cfg.hidden_size, cfg.ffn_hidden_size, bias=True)
        self.fc2 = TPRowLinear(cfg.ffn_hidden_size, cfg.hidden_size, bias=True, input_is_parallel=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(quick_gelu(self.fc1(x)))


class TPColumnLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, *, bias: bool):
        super().__init__()
        tp_world_size = PM.size_of("TP")
        if output_size % tp_world_size != 0:
            raise ValueError(f"output_size={output_size} must be divisible by TP={tp_world_size}")

        self.output_size_per_partition = output_size // tp_world_size
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else None
        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, input_size, device=device))
        self.bias = nn.Parameter(torch.empty(self.output_size_per_partition, device=device)) if bias else None
        set_tensor_model_parallel_attributes(self.weight, True, 0, 1)
        if self.bias is not None:
            set_tensor_model_parallel_attributes(self.bias, True, 0, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class TPRowLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, *, bias: bool, input_is_parallel: bool):
        super().__init__()
        tp_world_size = PM.size_of("TP")
        if input_size % tp_world_size != 0:
            raise ValueError(f"input_size={input_size} must be divisible by TP={tp_world_size}")

        self.input_is_parallel = input_is_parallel
        self.input_size_per_partition = input_size // tp_world_size
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else None
        self.weight = nn.Parameter(torch.empty(output_size, self.input_size_per_partition, device=device))
        self.bias = nn.Parameter(torch.empty(output_size, device=device)) if bias else None
        set_tensor_model_parallel_attributes(self.weight, True, 1, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_parallel = x if self.input_is_parallel else scatter_to_tensor_model_parallel_region(x)
        output_parallel = F.linear(input_parallel, self.weight, None)
        output = reduce_from_tensor_model_parallel_region(output_parallel)
        if self.bias is not None:
            output = output + self.bias
        return output


class VisionAttention(nn.Module):
    def __init__(self, cfg: VisionTransformerConfig):
        super().__init__()
        if cfg.hidden_size % cfg.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        tp_world_size = PM.size_of("TP")
        if cfg.num_attention_heads % tp_world_size != 0:
            raise ValueError(f"num_attention_heads={cfg.num_attention_heads} must be divisible by TP={tp_world_size}")

        self.num_heads = cfg.num_attention_heads
        self.num_local_heads = cfg.num_attention_heads // tp_world_size
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.attention_dropout = cfg.attention_dropout
        self.rope = (
            EncoderRope2D(
                dim=self.head_dim,
                max_grid_height=cfg.image_size // cfg.patch_size,
                max_grid_width=cfg.image_size // cfg.patch_size,
                use_cls_token=cfg.use_cls_token,
                theta=cfg.rope_theta,
                max_freq=cfg.rope_max_freq,
                num_freqs=cfg.rope_num_freqs,
                theta_rescale_factor=cfg.rope_theta_rescale_factor,
            )
            if cfg.use_rope2d
            else None
        )

        self.qkv_proj = TPColumnLinear(cfg.hidden_size, cfg.hidden_size * 3, bias=True)
        self.out_proj = TPRowLinear(cfg.hidden_size, cfg.hidden_size, bias=True, input_is_parallel=True)

    def forward(self, x: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        qkv = self.qkv_proj(x).reshape(batch_size, seq_len, 3, self.num_local_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope(q, k, grid_hw=grid_hw)
        v = v.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, hidden_size // PM.size_of("TP"))
        return self.out_proj(attn_output)


class VisionBlock(nn.Module):
    def __init__(self, cfg: VisionTransformerConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layernorm_epsilon)
        self.attn = VisionAttention(cfg)
        self.ls_1 = (
            LayerScale(cfg.hidden_size, cfg.layer_scale_init_value)
            if cfg.layer_scale_init_value is not None
            else nn.Identity()
        )

        self.ln_2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layernorm_epsilon)
        self.mlp = VisionMLP(cfg)
        self.ls_2 = (
            LayerScale(cfg.hidden_size, cfg.layer_scale_init_value)
            if cfg.layer_scale_init_value is not None
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        x = x + self.ls_1(self.attn(self.ln_1(x), grid_hw=grid_hw))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class VisionBackbone(nn.Module):
    def __init__(self, cfg: VisionTransformerConfig):
        super().__init__()
        self.resblocks = nn.ModuleList([VisionBlock(cfg) for _ in range(cfg.num_layers)])

    def forward(self, x: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        for block in self.resblocks:
            x = block(x, grid_hw=grid_hw)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, cfg: VisionTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(
            in_channels=cfg.in_channels,
            out_channels=cfg.hidden_size,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
            bias=cfg.patch_embed_bias,
        )
        if cfg.use_cls_token:
            self.class_embedding = nn.Parameter(torch.randn(cfg.hidden_size))

        self.num_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.positional_embedding = nn.Parameter(
            torch.randn(self.num_patches + (1 if cfg.use_cls_token else 0), cfg.hidden_size)
        )
        self.ln_pre = nn.LayerNorm(cfg.hidden_size, eps=cfg.layernorm_epsilon) if cfg.use_ln_pre else nn.Identity()

        self.transformer = VisionBackbone(cfg)

        self.vit_downsampler1 = nn.Conv2d(
            cfg.hidden_size,
            cfg.vit_downsampler_hidden_dim,
            kernel_size=cfg.vit_downsampler1_kernel_size,
            stride=2,
            padding=cfg.vit_downsampler1_padding,
            bias=True,
        )
        self.vit_downsampler2 = nn.Conv2d(
            cfg.vit_downsampler_hidden_dim,
            cfg.output_dim,
            kernel_size=cfg.vit_downsampler2_kernel_size,
            stride=2,
            padding=cfg.vit_downsampler2_padding,
            bias=True,
        )

    def _build_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeddings = self.conv1(pixel_values).flatten(2).transpose(1, 2)
        if self.cfg.use_cls_token:
            class_embedding = self.class_embedding.to(dtype=patch_embeddings.dtype, device=patch_embeddings.device)
            class_embedding = class_embedding.expand(batch_size, 1, -1)
            embeddings = torch.cat([class_embedding, patch_embeddings], dim=1)
        else:
            embeddings = patch_embeddings
        positional_embedding = _interpolate_positional_embedding(
            self.positional_embedding.to(dtype=embeddings.dtype, device=embeddings.device),
            embeddings.shape[1],
            use_cls_token=self.cfg.use_cls_token,
        )
        return self.ln_pre(embeddings + positional_embedding)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self._build_embeddings(pixel_values)
        grid_hw = (pixel_values.shape[-2] // self.cfg.patch_size, pixel_values.shape[-1] // self.cfg.patch_size)
        x = self.transformer(x, grid_hw=grid_hw)
        if self.cfg.use_cls_token:
            x = x[:, 1:, :]

        batch_size, seq_len, hidden_size = x.shape
        side = math.isqrt(seq_len)
        if side * side != seq_len:
            raise ValueError(f"Expected square patch grid, got {seq_len} tokens")

        x = x.transpose(1, 2).reshape(batch_size, hidden_size, side, side)
        x = self.vit_downsampler1(x)
        x = self.vit_downsampler2(x)
        return x.flatten(2).transpose(1, 2)

    @cached_property
    def reshaper(self):
        from steptronoss.checkpointing.reshape_ops import (
            MHA_TP_CHUNK,
            ColumnParallel,
            Duplicate,
            KeepThisTP,
            OnlineReshaper,
            Rename,
            RowParallel,
            Script,
        )

        scripts = [
            Script(src="positional_embedding", op=Duplicate() + KeepThisTP(), dst="positional_embedding"),
            Script(src="conv1.weight", op=Duplicate() + KeepThisTP(), dst="conv1.weight"),
            Script(src="vit_downsampler1.*", op=Duplicate() + KeepThisTP(), dst="vit_downsampler1.*"),
            Script(src="vit_downsampler2.*", op=Duplicate() + KeepThisTP(), dst="vit_downsampler2.*"),
        ]

        if hasattr(self, "class_embedding"):
            scripts.insert(0, Script(src="class_embedding", op=Duplicate() + KeepThisTP(), dst="class_embedding"))
        if self.conv1.bias is not None:
            scripts.append(Script(src="conv1.bias", op=Duplicate() + KeepThisTP(), dst="conv1.bias"))
        if not isinstance(self.ln_pre, nn.Identity):
            scripts.extend([
                Script(src="ln_pre.weight", op=Duplicate() + KeepThisTP(), dst="ln_pre.weight"),
                Script(src="ln_pre.bias", op=Duplicate() + KeepThisTP(), dst="ln_pre.bias"),
            ])

        for layer_id, _block in enumerate(self.transformer.resblocks):
            prefix = f"transformer.resblocks.{layer_id}"
            scripts.extend([
                Script(
                    src=f"{prefix}.attn.qkv_proj.*",
                    op=MHA_TP_CHUNK() + KeepThisTP(),
                    dst=f"{prefix}.attn.qkv_proj.*",
                ),
                Script(
                    src=f"{prefix}.attn.out_proj.weight",
                    op=RowParallel() + KeepThisTP(),
                    dst=f"{prefix}.attn.out_proj.weight",
                ),
                Script(
                    src=f"{prefix}.attn.out_proj.bias",
                    op=Duplicate() + KeepThisTP(),
                    dst=f"{prefix}.attn.out_proj.bias",
                ),
                Script(
                    src=f"{prefix}.ln_*.weight",
                    op=Duplicate() + KeepThisTP(),
                    dst=f"{prefix}.ln_*.weight",
                ),
                Script(
                    src=f"{prefix}.ln_*.bias",
                    op=Duplicate() + KeepThisTP(),
                    dst=f"{prefix}.ln_*.bias",
                ),
                Script(
                    src=f"{prefix}.ls_*.gamma",
                    op=Duplicate() + KeepThisTP(),
                    dst=f"{prefix}.ls_*.gamma",
                ),
                Script(
                    src=f"{prefix}.mlp.c_fc.*",
                    op=ColumnParallel() + KeepThisTP() + Rename(f"{prefix}.mlp.fc1.*: {prefix}.mlp.c_fc.*"),
                    dst=f"{prefix}.mlp.fc1.*",
                ),
                Script(
                    src=f"{prefix}.mlp.c_proj.weight",
                    op=RowParallel() + KeepThisTP() + Rename(f"{prefix}.mlp.fc2.weight: {prefix}.mlp.c_proj.weight"),
                    dst=f"{prefix}.mlp.fc2.weight",
                ),
                Script(
                    src=f"{prefix}.mlp.c_proj.bias",
                    op=Duplicate() + KeepThisTP() + Rename(f"{prefix}.mlp.fc2.bias: {prefix}.mlp.c_proj.bias"),
                    dst=f"{prefix}.mlp.fc2.bias",
                ),
            ])
        return OnlineReshaper(scripts)
