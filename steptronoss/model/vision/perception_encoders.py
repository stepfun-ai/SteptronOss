from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from functools import partial
from typing import Literal, Optional

import torch
from configurize import Config, Ref
from loguru import logger
from torch import nn
from torch.nn import functional as F

from steptronoss.core.tensor_parallel.random import checkpoint
from steptronoss.exp.abstract import ModelConfig
from steptronoss.exp.base_exp import MegatronTPConfig
from steptronoss.model.common.feed_forward import MLPConfig
from steptronoss.model.common.multi_head_attention import MultiHeadAttentionConfig
from steptronoss.model.common.rope import RoPEConfig


class PerceptionEmbedConfig(Config):
    """Vision patch embedding config."""

    hidden_size: int
    """Model hidden size."""
    in_channels: int
    """Input image channels."""
    patch_size: int
    """Patch size for patch embedding."""
    image_size: int | tuple[int, ...]
    """Input image size (square int or tuple)."""
    use_abs_posemb: bool
    """Use absolute position embeddings."""
    use_cls_token: bool
    """Use a class token."""
    rope_cfg: RoPEConfig | None
    """Optional RoPE config."""

    def __init__(self):
        super().__init__()
        self.hidden_size = Ref("..hidden_size")
        self.in_channels = 3
        self.image_size = (224,)
        self.use_abs_posemb = True
        self.use_cls_token = False
        self.rope_cfg = None

    def build_model(self):
        return PerceptionPatchEmbedding(cfg=self)


class PerceptionPoolConfig(Config):
    """Vision pooling config."""

    hidden_size: int
    """Model hidden size."""
    act_layer: Callable
    """Activation layer."""
    norm_layer: Callable
    """Normalization layer."""

    pool_type: Literal["attn", "tok", "avg", "none"]
    """Pooling type for output tokens."""
    attn_pooler_heads: int
    """Number of heads for attention pooling."""
    output_dim: int | None
    """Optional output projection dimension."""

    def __init__(self):
        super().__init__()
        self.hidden_size = Ref("..hidden_size")
        self.act_layer = nn.GELU
        self.norm_layer = Ref("..norm_layer")
        self.pool_type = "none"
        self.attn_pooler_heads = -1
        self.output_dim = None

    def build_model(self):
        return PerceptionPooler(cfg=self)


class VisionAttentionConfig(MultiHeadAttentionConfig):
    """Vision attention config."""


class VisionMLPConfig(MLPConfig):
    """Vision MLP config."""


class PerceptionEncoderConfig(ModelConfig):
    """Vision tower config."""

    tp_cfg: MegatronTPConfig = MegatronTPConfig
    """Tensor-parallel config."""
    attn_cfg: VisionAttentionConfig = VisionAttentionConfig
    """Attention block config."""
    ffn_cfg: VisionMLPConfig = VisionMLPConfig
    """FFN block config."""
    embed_cfg: PerceptionEmbedConfig = PerceptionEmbedConfig
    """Patch embedding config."""
    pool_cfg: PerceptionPoolConfig = PerceptionPoolConfig
    """Pooling config."""

    load_path: str | None
    """Optional path to a checkpoint to load."""
    params_dtype: torch.dtype
    """Model parameters dtype."""
    hidden_size: int
    """Model hidden size."""
    num_layers: int
    """Number of transformer blocks."""
    norm_layer: Callable
    """Normalization layer."""
    ls_init_value: float | None
    """LayerScale init value when enabled."""
    drop_path: float
    """Stochastic depth rate."""

    use_ln_pre: bool
    """Apply pre-attention layer norm."""
    use_ln_post: bool
    """Apply post-encoder layer norm."""

    recompute: bool
    """Enable full recompute for activation checkpointing."""
    recompute_list: list[int] | None
    """Selective recompute layers."""

    def __init__(self):
        super().__init__()
        self.load_path = None
        self.params_dtype = torch.bfloat16
        self.norm_layer = partial(nn.LayerNorm, eps=1e-5)
        self.ls_init_value = None
        self.drop_path = 0.0
        self.use_ln_pre = True
        self.use_ln_post = True
        self.recompute = False
        self.recompute_list = None

    def sanity_check(self):
        super().sanity_check()
        ffn_cfg = self.ffn_cfg
        assert ffn_cfg.ffn_hidden_size > 0, "ffn_hidden_size must be specified"
        attn_cfg = self.attn_cfg
        assert attn_cfg.num_attention_heads > 0, "num_attention_heads must be specified"
        assert not (self.recompute and (self.recompute_list is not None and len(self.recompute_list) > 0)), (
            "Cannot specify recompute_list when recompute is True"
        )

    def build_model(self):
        model = PerceptionEncoder(cfg=self)
        if self.load_path:
            _sd = torch.load(self.load_path, weights_only=True)
            m, u = model.load_state_dict(_sd, strict=False)
            if m:
                logger.info(f"Missing keys for loading vision encoder: {m}")
            if u:
                logger.info(f"Unexpected keys for loading vision encoder: {u}")
        return model


class PerceptionPatchEmbedding(nn.Module):
    def __init__(self, cfg: PerceptionEmbedConfig):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.patch_size = cfg.patch_size
        self.use_abs_posemb = cfg.use_abs_posemb
        self.use_cls_token = cfg.use_cls_token

        self.conv1 = nn.Conv2d(
            in_channels=cfg.in_channels,
            out_channels=cfg.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        if self.use_cls_token:
            self.class_embedding = nn.Parameter(torch.empty(1, 1, self.hidden_size))
        else:
            self.class_embedding = None

        if self.use_abs_posemb:
            image_size = cfg.image_size
            if isinstance(image_size, tuple):
                if len(image_size) == 1:
                    grid_h = grid_w = image_size[0]
                else:
                    grid_h, grid_w = image_size[:2]
            else:
                grid_h = grid_w = image_size
            self.posemb_grid_h = grid_h // self.patch_size
            self.posemb_grid_w = grid_w // self.patch_size
            num_tokens = self.posemb_grid_h * self.posemb_grid_w + int(self.use_cls_token)
            self.positional_embedding = nn.Parameter(torch.empty(num_tokens, self.hidden_size))
        else:
            self.positional_embedding = None

    def _sample_abs_posemb(self, grid_h: int, grid_w: int) -> torch.Tensor:
        if self.posemb_grid_h == grid_h and self.posemb_grid_w == grid_w:
            return self.positional_embedding

        pos_embed = self.positional_embedding
        if self.use_cls_token:
            cls_token_embed, pos_embed = pos_embed[:1], pos_embed[1:]

        pos_embed = pos_embed.reshape(self.posemb_grid_h, self.posemb_grid_w, -1).permute(2, 0, 1).unsqueeze(0)
        pos_embed = F.interpolate(pos_embed, size=(grid_h, grid_w), mode="bilinear", align_corners=False)
        pos_embed = pos_embed.squeeze(0).permute(1, 2, 0).reshape(-1, self.hidden_size).contiguous()

        if self.use_cls_token:
            pos_embed = torch.cat([cls_token_embed, pos_embed], dim=0)

        return pos_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, h, w = x.shape
        grid_h, grid_w = h // self.patch_size, w // self.patch_size

        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).reshape(batch, -1, self.hidden_size)

        if self.use_cls_token:
            x = torch.cat(
                [self.class_embedding.expand(batch, -1, -1).to(x.dtype), x],
                dim=1,
            )

        if self.use_abs_posemb:
            x = x + self._sample_abs_posemb(grid_h, grid_w).unsqueeze(0)

        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(torch.empty(dim))
        nn.init.constant_(self.gamma, init_values)

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


class AttentionPooling(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_probe: int = 1,
        mlp_ratio: int = 4,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.probe = nn.Parameter(torch.randn(1, num_probe, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.layernorm = norm_layer(embed_dim)
        self.mlp_width = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict([
                ("c_fc", nn.Linear(embed_dim, self.mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(self.mlp_width, embed_dim)),
            ])
        )

    def forward(self, x: torch.Tensor):
        batch, _, _ = x.shape

        q = self.probe.repeat((batch, 1, 1)).to(x.dtype)
        x = self.attn(q, x, x, need_weights=False)[0]
        x = x + self.mlp(self.layernorm(x))

        return x


class PerceptionPooler(nn.Module):
    def __init__(self, cfg: PerceptionPoolConfig):
        super().__init__()
        if cfg.pool_type is not None:
            assert cfg.pool_type in ("attn", "tok", "avg", "none")
        self.pool_type = cfg.pool_type

        if self.pool_type == "attn":
            self.attn_pool = AttentionPooling(
                embed_dim=cfg.hidden_size,
                num_heads=cfg.attn_pooler_heads,
                act_layer=cfg.act_layer,
                norm_layer=cfg.norm_layer,
            )
        else:
            self.attn_pool = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool_type == "tok":
            return x[:, 0]
        if self.pool_type == "avg":
            return x.mean(dim=1)
        if self.pool_type == "attn":
            return self.attn_pool(x).squeeze(1)
        if self.pool_type == "none":
            return x
        raise NotImplementedError


class ResidualAttentionBlock(nn.Module):
    def __init__(self, cfg: PerceptionEncoderConfig, layer_id: int):
        super().__init__()

        self.attn = cfg.attn_cfg.build_model(layer_id=layer_id)

        self.ls_1 = LayerScale(cfg.hidden_size, cfg.ls_init_value) if cfg.ls_init_value is not None else nn.Identity()
        self.ls_2 = LayerScale(cfg.hidden_size, cfg.ls_init_value) if cfg.ls_init_value is not None else nn.Identity()

        self.ln_1 = cfg.norm_layer(cfg.hidden_size)
        self.ln_2 = cfg.norm_layer(cfg.hidden_size)

        self.drop_path1 = DropPath(cfg.drop_path) if cfg.drop_path > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(cfg.drop_path) if cfg.drop_path > 0.0 else nn.Identity()

        self.mlp = cfg.ffn_cfg.build_model(layer_id=layer_id)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cu_seqlens = None
        max_seq_len = None
        if x.shape[1] > 1:
            seq_len = x.shape[0]
            cu_seqlens = torch.arange(
                0,
                (x.shape[1] + 1) * seq_len,
                step=seq_len,
                dtype=torch.int32,
                device=x.device,
            )
            max_seq_len = seq_len
        x = x + self.drop_path1(self.ls_1(self.attn(self.ln_1(x), cu_seqlens=cu_seqlens, max_seq_len=max_seq_len)))
        x = x + self.drop_path2(self.ls_2(self.mlp(self.ln_2(x))))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: PerceptionEncoderConfig):
        super().__init__()
        self.num_layers = cfg.num_layers
        self.grad_checkpointing = False
        self.recompute_list = cfg.recompute_list

        if self.grad_checkpointing:
            self.recompute_list = range(self.num_layers)
            logger.warning(
                f"grad_checkpointing is enabled, all layers will be recomputed. Ignore {self.recompute_list}!!!"
            )

        self.resblocks = nn.ModuleList([ResidualAttentionBlock(cfg, layer_id) for layer_id in range(cfg.num_layers)])

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def truncate(self, layer_idx: int):
        """Delete layers so the last layer is the given layer index."""
        self.num_layers = ((self.num_layers + layer_idx) % self.num_layers) + 1
        self.resblocks = nn.ModuleList(self.resblocks[: self.num_layers])

    def forward(self, x: torch.Tensor, layer_idx: int = -1):
        stop_idx = (self.num_layers + layer_idx) % self.num_layers
        for lid, layer in enumerate(self.resblocks):
            if self.recompute_list and lid in self.recompute_list and not torch.jit.is_scripting():
                x = checkpoint(layer, False, x)
            else:
                x = layer(x)

            if lid == stop_idx:
                break

        return x


class VisionTransformer(nn.Module):
    def __init__(self, cfg: PerceptionEncoderConfig):
        super().__init__()
        self.pool_type = cfg.pool_cfg.pool_type

        self.width = cfg.hidden_size

        self.use_cls_token = cfg.embed_cfg.use_cls_token
        self.patch_embed = cfg.embed_cfg.build_model()

        self.ln_pre = cfg.norm_layer(cfg.hidden_size) if cfg.use_ln_pre else nn.Identity()
        self.ln_post = cfg.norm_layer(self.width) if cfg.use_ln_post else nn.Identity()

        self.transformer = Transformer(cfg)
        self.pooler = cfg.pool_cfg.build_model()

    def truncate(self, layer_idx: int):
        """Delete layers so the last layer is the given layer index."""
        self.transformer.truncate(layer_idx)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.set_grad_checkpointing(enable=enable)

    def forward_features(
        self,
        x: torch.Tensor,
        norm: bool = False,
        layer_idx: int = -1,
        strip_cls_token: bool = False,
    ):
        x = self.patch_embed(x)
        x = self.ln_pre(x)
        x = x.transpose(0, 1).contiguous()
        x = self.transformer(x, layer_idx=layer_idx)

        if norm:
            x = self.ln_post(x)

        if strip_cls_token and self.use_cls_token:
            x = x[1:, :, :]

        return x.transpose(0, 1).contiguous()

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.forward_features(x, norm=True, strip_cls_token=True, **kwargs)
        x = self.pooler(x)
        return x

    def name_parameters(self, prefix: str = ""):
        prefix = f"{prefix}/" if len(prefix) > 0 else ""

        for p in self.ln_pre.parameters():
            p._log_name = prefix + "ln_pre"
        for p in self.ln_post.parameters():
            p._log_name = prefix + "ln_post"
        for layer_id, layer in enumerate(self.transformer.resblocks):
            for p in layer.parameters():
                p._log_name = prefix + f"resblocks.{layer_id}"
        if self.patch_embed.positional_embedding is not None:
            self.patch_embed.positional_embedding._log_name = prefix + "positional_embedding"
        if self.patch_embed.class_embedding is not None:
            self.patch_embed.class_embedding._log_name = prefix + "class_embedding"
        if self.pool_type == "attn":
            for p in self.pooler.attn_pool.parameters():
                p._log_name = prefix + "attn_pool"


class PerceptionEncoder(VisionTransformer):
    """Vision transformer with spatial downsampling."""

    def __init__(self, cfg: PerceptionEncoderConfig):
        super().__init__(cfg)
        self.recompute = cfg.recompute

        self.vit_downsampler1 = nn.Conv2d(
            cfg.hidden_size,
            cfg.hidden_size * 2,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.vit_downsampler2 = nn.Conv2d(
            cfg.hidden_size * 2,
            cfg.hidden_size * 4,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor, **kwargs):
        x = x.detach()
        x.requires_grad = True

        if self.recompute:
            x = checkpoint(self.forward_features, False, x, norm=True, strip_cls_token=True, **kwargs)
        else:
            x = self.forward_features(x, norm=True, strip_cls_token=True, **kwargs)

        x = self.pooler(x)
        if x.dim() == 2:
            return x

        batch, patches, channels = x.shape
        grid = int(patches**0.5)
        x = x.transpose(2, 1).contiguous().view(batch, channels, grid, grid)

        x = self.vit_downsampler1(x)
        x = self.vit_downsampler2(x)

        batch, channels, grid_h, grid_w = x.shape
        return x.view(batch, -1, grid_h * grid_w).transpose(1, 2)


class PE_CORE_G14_448_EmbedConfig(PerceptionEmbedConfig):
    def __init__(self):
        super().__init__()
        self.rope_cfg = RoPEConfig(
            dim=1536 // 16,
            theta=10_000.0,
            ntk_interp_ratio=1.0,
            yarn_beta_slow=1.0,
            yarn_beta_fast=32.0,
            max_position_embeddings=4096,
        )
        self.image_size = 448
        self.patch_size = 14
        self.use_cls_token = False


class PE_LANG_L14_448_EmbedConfig(PerceptionEmbedConfig):
    def __init__(self):
        super().__init__()
        self.rope_cfg = RoPEConfig(
            dim=1024 // 16,
            theta=10_000.0,
            ntk_interp_ratio=1.0,
            yarn_beta_slow=1.0,
            yarn_beta_fast=32.0,
            max_position_embeddings=4096,
        )
        self.image_size = 448
        self.patch_size = 14
        self.use_cls_token = True


class PE_LANG_L14_728_EmbedConfig(PE_LANG_L14_448_EmbedConfig):
    def __init__(self):
        super().__init__()
        self.image_size = 728


class PE_LANG_G14_728_EmbedConfig(PE_CORE_G14_448_EmbedConfig):
    def __init__(self):
        super().__init__()
        self.image_size = 728


class PE_CORE_G14_448_AttnConfig(VisionAttentionConfig):
    def __init__(self):
        super().__init__()
        self.num_attention_heads = 16
        self.head_dim = 1536 // 16


class PE_LANG_L14_448_AttnConfig(VisionAttentionConfig):
    def __init__(self):
        super().__init__()
        self.num_attention_heads = 16
        self.head_dim = 1024 // 16


class PE_LANG_G14_448_AttnConfig(VisionAttentionConfig):
    def __init__(self):
        super().__init__()
        self.num_attention_heads = 16
        self.head_dim = 1536 // 16


class PE_CORE_G14_448_MLPConfig(VisionMLPConfig):
    def __init__(self):
        super().__init__()
        self.ffn_hidden_size = 8960


class PE_LANG_L14_448_MLPConfig(VisionMLPConfig):
    def __init__(self):
        super().__init__()
        self.ffn_hidden_size = 4096


class PE_CORE_G14_448_TP(PerceptionEncoderConfig):
    embed_cfg = PE_CORE_G14_448_EmbedConfig
    attn_cfg = PE_CORE_G14_448_AttnConfig
    ffn_cfg = PE_CORE_G14_448_MLPConfig

    def __init__(self):
        super().__init__()
        self.hidden_size = 1536
        self.num_layers = 50
        self.recompute_list = list(range(1, 50))


class PE_LANG_L14_448_TP(PerceptionEncoderConfig):
    embed_cfg = PE_LANG_L14_448_EmbedConfig
    attn_cfg = PE_LANG_L14_448_AttnConfig
    ffn_cfg = PE_LANG_L14_448_MLPConfig

    def __init__(self):
        super().__init__()
        self.hidden_size = 1024
        self.num_layers = 23
        self.use_ln_post = False
        self.ls_init_value = 0.1
        self.recompute_list = list(range(1, 23))


class PE_LANG_L14_728_TP(PE_LANG_L14_448_TP):
    embed_cfg = PE_LANG_L14_728_EmbedConfig


class PE_LANG_G14_448_TP(PE_CORE_G14_448_TP):
    attn_cfg = PE_LANG_G14_448_AttnConfig

    def __init__(self):
        super().__init__()
        self.use_ln_post = False
        self.ls_init_value = 0.1
        self.num_layers = 47


class PE_LANG_G14_728_TP(PE_CORE_G14_448_TP):
    embed_cfg = PE_LANG_G14_728_EmbedConfig
    attn_cfg = PE_LANG_G14_448_AttnConfig

    def __init__(self):
        super().__init__()
        self.use_ln_post = False
        self.ls_init_value = 0.1
        self.num_layers = 47


class PE_LANG_G14_728_TP_WithLLRD(PE_LANG_G14_728_TP):
    llrd_rate: float
    """Layer-wise LR decay multiplier."""

    def __init__(self):
        super().__init__()
        self.llrd_rate = 0.4

    def build_model(self):
        model = super().build_model()
        for p in model.parameters():
            p._lr_scale = self.llrd_rate
        return model
