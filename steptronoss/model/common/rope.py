import math

import torch
from loguru import logger
from torch import Tensor

from steptronoss.core.parallel_state import PM
from steptronoss.exp.abstract import ModelConfig as AbstractModelConfig


def ropen_linspace(start, end, steps, **kwargs):
    """for start=0, end=1, steps=4
    linspace:       [0.0000, 0.3333, 0.6667, 1.0000]
    ropen_linspace: [0.0000, 0.2500, 0.5000, 0.7500]
    """
    return torch.linspace(start, end, steps + 1, **kwargs)[:-1]


class RoPEConfig(AbstractModelConfig):
    dim: int
    theta: float
    ntk_interp_ratio: float
    yarn_beta_slow: float
    yarn_beta_fast: float
    max_position_embeddings: int

    def build_model(self):
        return YARNRoPE(
            dim=self.dim,
            theta=self.theta,
            ntk_interp_ratio=self.ntk_interp_ratio,
            yarn_beta_slow=self.yarn_beta_slow,
            yarn_beta_fast=self.yarn_beta_fast,
            max_position_embeddings=self.max_position_embeddings,
        )


class YARNRoPE(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        theta: float = 10000.0,
        ntk_interp_ratio: float = 1.0,
        yarn_beta_slow: float = 1,
        yarn_beta_fast: float = 32,
        max_position_embeddings: int | None = None,
        use_adjacent_pair: bool = False,
    ):
        """RoPE implementation with YARN support, use ntk_interp_ratio=1 for plain RoPE
        NOTE: this implement split Channels with (real, virt) = X[:C/2], X[C/2:]

        Args:
            dim (int): Dim of one head (C).
            theta (float, optional): Theta in RoPE, note that period T = 2pi * theta.
                Defaults to 10000.0 (T ~ 62k)s
            ntk_interp_ratio (float, optional): interp ratio, should >= 1. Defaults to 1.0.
            yarn_beta_slow (float, optional): beta_slow in YARN. Defaults to 1.
            yarn_beta_fast (float, optional): beta_fast in YARN. Defaults to 32.
            max_position_embeddings (int, optional): max seqlen allowed, this param
                is only used when YARN is activated, otherwise its just for sanity and
                pre-allocation. Defaults to 8192.
        """
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.ntk_interp_ratio = ntk_interp_ratio
        self.yarn_beta_slow = yarn_beta_slow
        self.yarn_beta_fast = yarn_beta_fast

        self.max_position_embeddings = max_position_embeddings
        non_repetitive_seqlen = theta * 2 * math.pi * ntk_interp_ratio
        if max_position_embeddings:
            if max_position_embeddings > non_repetitive_seqlen:
                logger.warning(
                    f"max_position_embedding larger than one rotation period ({int(non_repetitive_seqlen)})!"
                )

        if ntk_interp_ratio != 1.0:
            assert max_position_embeddings is not None
            logger.warning(
                f"Use Python Rope with YARN support with ratio={ntk_interp_ratio}, max_pos_emb={max_position_embeddings},"
                f" yarn_beta_slow: {yarn_beta_slow}, yarn_beta_fast: {yarn_beta_fast}, use_adjacent_pair: {use_adjacent_pair}"
            )

        self.use_adjacent_pair = use_adjacent_pair

        self._cached_seqlen = -1
        self._cos_cache: Tensor
        self._sin_cache: Tensor
        self._check_set_cos_sin_cache(self.max_position_embeddings or 8192, torch.device("cpu"))

    @staticmethod
    def ref1(x: Tensor, theta: float):
        # Raw complex imple
        S, C = x.shape
        frequencies = 1 / (theta ** ropen_linspace(0, 1, C // 2))
        angles = torch.outer(torch.arange(S), frequencies)  # S, C / 2
        # NOTE: here different, this use [r, v, r, v], ref2 use [r, r, v, v]
        x_ = torch.view_as_complex(x.reshape(S, C // 2, 2))

        rotation_complex = torch.polar(torch.ones_like(angles), angles)

        x_ = torch.view_as_real(x_ * rotation_complex)
        return x_.type_as(x).reshape(S, C)

    @staticmethod
    def ref2(inputs: Tensor, theta: float):
        # Cos Sin imple, we use this, its faster.
        S, C = inputs.shape
        frequencies = 1 / (theta ** ropen_linspace(0, 1, C // 2))
        angles = torch.outer(torch.arange(S), frequencies)  # S, C / 2

        x, y = inputs.chunk(2, dim=-1)
        inputs_rotate_90 = torch.cat([-y, x], dim=-1)

        repeated_angles = angles.repeat([1, 2])
        cos = repeated_angles.cos()
        sin = repeated_angles.sin()
        output = inputs * cos + inputs_rotate_90 * sin

        return output

    @staticmethod
    def yarn_ref(inputs: Tensor, theta: float, beta_fast=32, beta_slow=1, ntk_ratio=1):
        # ntk_ratio >= 1
        S, C = inputs.shape
        Cv = C // 2  # complex dim
        base_freqs = 1 / (theta ** ropen_linspace(0, 1, Cv))  # C / 2
        ntk_freqs = base_freqs / ntk_ratio  # lower freqencies

        d_left = math.floor(Cv * (math.log((S / beta_fast) / (2 * math.pi)) / math.log(theta)))  # theta ~ S / (2 * pi)
        d_right = math.ceil(Cv * (math.log((S / beta_slow) / (2 * math.pi)) / math.log(theta)))  # theta ~ S / (2 * pi)
        d_left, d_right = max(0, d_left), min(Cv, d_right)
        # weight: [0, 0, 0, 0.3, 0.6, 0.9, 1, 1, 1]
        ntk_weight = ((torch.arange(Cv) - d_left) / (d_right - d_left)).clamp(0, 1)
        # use ntk only on low freq (right)
        yarn_freqs = base_freqs * (1 - ntk_weight) + ntk_freqs * ntk_weight

        angles = torch.outer(torch.arange(S), yarn_freqs)  # S, C / 2

        # Now, just like plain RoPE

        x, y = inputs.chunk(2, dim=-1)
        inputs_rotate_90 = torch.cat([-y, x], dim=-1)

        repeated_angles = angles.repeat([1, 2])
        cos = repeated_angles.cos()
        sin = repeated_angles.sin()
        output = inputs * cos + inputs_rotate_90 * sin

        return output

    @staticmethod
    def rotate_half(x: Tensor):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _get_frequencies(self, device="cpu"):
        S, C = self.max_position_embeddings, self.dim
        theta = self.theta
        ntk_ratio = self.ntk_interp_ratio

        Cv = C // 2  # complex dim
        base_freqs = 1 / (theta ** ropen_linspace(0, 1, Cv, device=device))  # C / 2
        if ntk_ratio == 1:
            return base_freqs

        ntk_freqs = base_freqs / ntk_ratio  # lower freqencies

        ntk_weight = (
            (S / (2 * math.pi / base_freqs) - self.yarn_beta_slow) / (self.yarn_beta_fast - self.yarn_beta_slow)
        ).clamp(0, 1)
        # use ntk only on low freq (right)
        yarn_freqs = base_freqs * ntk_weight + ntk_freqs * (1 - ntk_weight)

        return yarn_freqs

    def _check_set_cos_sin_cache(self, cur_seqlen=None, device="cuda"):
        if cur_seqlen and cur_seqlen > self._cached_seqlen:
            frequencies = self._get_frequencies(device=device)
            angles = torch.outer(torch.arange(cur_seqlen, device=device), frequencies)  # S, C / 2
            repeated_angles = angles.repeat([1, 2])
            cos = repeated_angles.cos()
            sin = repeated_angles.sin()
            self._cos_cache = cos
            self._sin_cache = sin
            self._cached_seqlen = cur_seqlen

        if self._cos_cache.device != device:
            self._cos_cache = self._cos_cache.to(device)
            self._sin_cache = self._sin_cache.to(device)

        return self._cos_cache, self._sin_cache

    def forward(
        self,
        feature: Tensor,
        position_id: torch.IntTensor = None,
    ):
        # feature: [B, S, H, C]
        # position_id: [S, ]

        if position_id is not None:  # packed sample
            max_seqlen = position_id.max() + 1
            cos_cache, sin_cache = self._check_set_cos_sin_cache(max_seqlen, position_id.device)
            cos = cos_cache[None, position_id, None, :].to(feature.device).to(feature.dtype)
            sin = sin_cache[None, position_id, None, :].to(feature.device).to(feature.dtype)
        else:  # no packing
            max_seqlen = feature.shape[1] * PM.size_of("CP")
            cos_cache, sin_cache = self._check_set_cos_sin_cache(max_seqlen, feature.device)
            cos = cos_cache[None, :max_seqlen, None, :].to(feature.device)
            sin = sin_cache[None, :max_seqlen, None, :].to(feature.device)

        if self.use_adjacent_pair:
            # rotate to adjacent
            b, s, h, c = feature.shape
            feature = feature.view(b, s, h, c // 2, 2).transpose(4, 3).reshape(b, s, h, c)

        dtype = feature.dtype
        feature = feature.to(torch.float32)
        feature = feature * cos + self.rotate_half(feature) * sin
        return feature.to(dtype)
