import torch
from configurize import Config, Ref
from einops import rearrange

from steptronoss.core import tensor_parallel
from steptronoss.core.context_parallel import (
    cu_seqlens_to_balanced_cp,
    gather_from_balanced_cp_region,
)
from steptronoss.core.parallel_state import PM
from steptronoss.exp.base_exp import MegatronTPConfig
from steptronoss.model.common.attention_core import AttentionCore
from steptronoss.model.common.rms_norm import RMSNorm
from steptronoss.model.common.rope import YARNRoPE
from steptronoss.utils import safediv


class AttentionConfig(Config):
    tp_cfg: MegatronTPConfig = Ref("..tp_cfg")
    causal: bool
    attention_dropout: float

    use_sliding_window: bool

    num_attention_heads: int
    num_attention_groups: int

    head_dim: int | None
    hidden_size: int

    use_headwise_attn_gate: bool
    use_qkv_bias: bool

    sliding_window_size: int | None

    use_qk_norm: bool
    layernorm_epsilon: float
    rms_norm_zero_gamma: bool

    recompute_qknorm_rope: bool

    qk_rope_head_dim: int | list[int] | None
    rope_theta: float | list[float]
    yarn_beta_slow: float
    yarn_beta_fast: float
    ntk_interp_ratio: float
    max_position_embeddings: int

    def build_model(self, layer_id: int):
        return GroupedQueryAttention(cfg=self, layer_id=layer_id)


class GroupedQueryAttention(torch.nn.Module):
    def __init__(self, cfg: AttentionConfig, layer_id=None):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id

        self.sequence_parallel = cfg.tp_cfg.sequence_parallel

        self.tp_size = PM.size_of("TP")

        self.cp = PM.size_of("CP") > 1

        self.sliding_window = self.cfg.use_sliding_window

        self.num_heads = cfg.num_attention_heads

        self.num_local_heads = safediv(self.num_heads, self.tp_size)
        self.n_local_heads = self.num_local_heads  # name adatped for

        assert self.num_heads % cfg.num_attention_groups == 0
        self.num_kv_heads = cfg.num_attention_groups

        if self.tp_size > self.num_kv_heads:
            kv_gather_group_size = safediv(self.tp_size, self.num_kv_heads)
            # subgroup is too cumbersome in torch. only support the case where kv gather group is tp group
            assert kv_gather_group_size == self.tp_size
            self.kv_gather_group = PM.group_of("TP")
            # after gather, each tp rank has one attn group
            self.num_local_kv_heads = 1
        else:
            self.num_local_kv_heads = safediv(self.num_kv_heads, self.tp_size)

        self.n_local_kv_heads = self.num_local_kv_heads  # name adatped for muon
        self.kv_per_head = safediv(self.num_heads, cfg.num_attention_groups)

        if cfg.head_dim is not None:
            self.head_dim = cfg.head_dim
        else:
            # 确保 hidden_size 能被 num_heads 整除
            assert cfg.hidden_size % self.num_heads == 0, (
                f"hidden_size ({cfg.hidden_size}) must be divisible by num_heads ({self.num_heads})"
            )
            self.head_dim = cfg.hidden_size // self.num_heads

        # gated-attn
        self.wqkv_extra_dims = 0
        if self.cfg.use_headwise_attn_gate:
            self.wqkv_extra_dims = self.num_heads
        self.local_wqkv_extra_dims = self.wqkv_extra_dims // self.tp_size

        self.wqkv = tensor_parallel.ColumnParallelLinear(
            cfg.hidden_size,
            self.head_dim * self.num_heads + self.head_dim * 2 * self.num_kv_heads + self.wqkv_extra_dims,
            bias=cfg.use_qkv_bias,
            gather_output=False,
            async_tensor_model_parallel_allreduce=self.cfg.tp_cfg.async_tensor_model_parallel_allreduce,
            **cfg.tp_cfg.get_tp_kwargs(),
        )
        self.wo = tensor_parallel.RowParallelLinear(
            self.head_dim * self.num_heads,
            cfg.hidden_size,
            bias=False,
            input_is_parallel=True,
            custom_pre_recompute_function=(
                self.head_wise_attn_gate_function if self.cfg.use_headwise_attn_gate else None
            ),
            **cfg.tp_cfg.get_tp_kwargs(),
        )

        self.core_attention = AttentionCore(
            causal=self.cfg.causal,
            attention_dropout=self.cfg.attention_dropout,
            sliding_window=(self.cfg.sliding_window_size if self.sliding_window else -1),
        )
        self.use_qk_norm = not cfg.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(
                self.head_dim,
                eps=self.cfg.layernorm_epsilon,
                sequence_parallel=self.sequence_parallel,
                use_zero_init=self.cfg.rms_norm_zero_gamma,
            )
            self.k_norm = RMSNorm(
                self.head_dim,
                eps=self.cfg.layernorm_epsilon,
                sequence_parallel=self.sequence_parallel,
                use_zero_init=self.cfg.rms_norm_zero_gamma,
            )

        self.recompute_qknorm_rope = self.cfg.recompute_qknorm_rope
        self.rope = self.build_rope()

    def build_rope(self):

        if isinstance(self.cfg.qk_rope_head_dim, list):
            self.qk_rope_head_dim = self.cfg.qk_rope_head_dim[self.layer_id]
        else:
            self.qk_rope_head_dim = self.cfg.qk_rope_head_dim
        if self.qk_rope_head_dim is None:
            self.qk_rope_head_dim = self.cfg.head_dim

        if isinstance(self.cfg.rope_theta, list):
            self.rope_theta = self.cfg.rope_theta[self.layer_id]
        else:
            self.rope_theta = self.cfg.rope_theta

        rope = YARNRoPE(
            dim=self.qk_rope_head_dim,
            theta=self.rope_theta,
            yarn_beta_fast=self.cfg.yarn_beta_fast,
            yarn_beta_slow=self.cfg.yarn_beta_slow,
            ntk_interp_ratio=self.cfg.ntk_interp_ratio,
            max_position_embeddings=self.cfg.max_position_embeddings,
        )
        return rope

    def head_wise_attn_gate_function(self, output: torch.FloatTensor, gate_weight: torch.FloatTensor):
        S, B = output.shape[:2]
        attn_out = output.view(S, B, self.num_local_heads, self.head_dim)

        attn_out = attn_out * gate_weight.unsqueeze(-1).sigmoid()

        return attn_out.view(S, B, -1)

    def forward_rope(self, xq, xk, pos_id_q, pos_id_k):
        # xq, xk: B, S, H, C
        if self.qk_rope_head_dim is not None and self.qk_rope_head_dim != self.head_dim:
            rot_dim = self.qk_rope_head_dim
            xq_rope, xq_nope = torch.split(xq, [rot_dim, xq.shape[-1] - rot_dim], dim=-1)
            xk_rope, xk_nope = torch.split(xk, [rot_dim, xk.shape[-1] - rot_dim], dim=-1)
            xq_rope = self.rope(xq_rope, pos_id_q)
            xk_rope = self.rope(xk_rope, pos_id_k)
            xq = torch.cat([xq_rope, xq_nope], dim=-1)
            xk = torch.cat([xk_rope, xk_nope], dim=-1)
        else:
            xq = self.rope(xq, pos_id_q)
            xk = self.rope(xk, pos_id_k)
        return xq, xk

    def forward_attention_core(
        self,
        xq: torch.FloatTensor,
        xk: torch.FloatTensor,
        xv: torch.FloatTensor,
        cu_seqlens: torch.IntTensor | dict[str, torch.IntTensor] | None = None,
        max_seqlen: int | dict[str, int] | None = None,
    ):

        if not self.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                output = self.core_attention(
                    xq,
                    xk,
                    xv,
                    cu_seqlens=cu_seqlens,
                    max_seq_len=max_seqlen,
                )
        else:
            output = self.core_attention(
                xq,
                xk,
                xv,
                cu_seqlens=cu_seqlens,
                max_seq_len=max_seqlen,
            )
        return output

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seq_len: torch.Tensor | None = None,
        position_id: torch.Tensor | None = None,
        **kwargs,
    ):
        # here, we need pass pos_ids, because calc it from cu_seqlens can be slow.
        if self.cp:
            assert cu_seqlens is not None, "Context parallel (CP) attention requires cu_seqlens to be provided"
            assert position_id is not None, "Context parallel (CP) attention requires position_id to be provided"

        S, B, C = x.shape  # cfg.hidden_size from x.shape

        xqkv, _ = self.wqkv(x)

        q_sharded_dim = self.head_dim * self.num_local_heads
        kv_sharded_dim = self.head_dim * 2 * self.num_local_kv_heads

        xq, xkv, gate_weight = torch.split(
            xqkv,
            (q_sharded_dim, kv_sharded_dim, self.local_wqkv_extra_dims),
            dim=-1,
        )

        if self.tp_size > self.num_kv_heads:
            raise NotImplementedError

        if self.sequence_parallel:
            S *= self.tp_size

        xq = xq.view(S, B, self.num_local_heads, self.head_dim)
        xkv = xkv.view(S, B, self.num_local_kv_heads, 2 * self.head_dim)

        xk, xv = xkv.chunk(2, dim=-1)
        xq = xq.contiguous()
        xk = xk.contiguous()

        if cu_seqlens is None:
            assert B == 1, "cu_seqlens is not supported for bsz > 1!"
            cu_seqlens = torch.arange(
                0,
                (xq.shape[0] + 1) * xq.shape[1],
                step=S,
                dtype=torch.int32,
                device=xq.device,
            )
            max_seq_len = torch.max(cu_seqlens[1:] - cu_seqlens[:-1]).cpu()
        elif max_seq_len is None:
            max_seq_len = torch.max(cu_seqlens[1:] - cu_seqlens[:-1]).cpu()

        if self.use_qk_norm:
            xq = self.q_norm(xq.contiguous())
            xk = self.k_norm(xk.contiguous())

        if self.cp:
            # Compute balanced CP splits for this rank (left and right halves)
            info0, info1 = cu_seqlens_to_balanced_cp(cu_seqlens, cp_rank=PM.rank_in("CP"), cp_size=PM.size_of("CP"))
            # Left half
            q_range0, q_cumlen0, max_q0, k_range0, k_cumlen0, max_k0 = info0
            # Right half
            q_range1, q_cumlen1, max_q1, k_range1, k_cumlen1, max_k1 = info1

            # Gather K and V back to original order across CP ranks
            xk_full = gather_from_balanced_cp_region(xk)
            xv_full = gather_from_balanced_cp_region(xv)

            # Slice KV for each half using precomputed k_range from above
            xk0 = xk_full[k_range0[0] : k_range0[1]]
            xv0 = xv_full[k_range0[0] : k_range0[1]]
            xk1 = xk_full[k_range1[0] : k_range1[1]]
            xv1 = xv_full[k_range1[0] : k_range1[1]]

            # Split Q into two balanced halves for local CP shard
            xq0, xq1 = xq.chunk(2, 0)

            # Half 0
            xq0, xk0, xv0 = [rearrange(i, "s b h d -> b s h d") for i in [xq0, xk0, xv0]]
            xq0, xk0 = self.forward_rope(
                xq0,
                xk0,
                pos_id_q=position_id[q_range0[0] : q_range0[1]],
                pos_id_k=position_id[k_range0[0] : k_range0[1]],
            )
            out0 = self.forward_attention_core(
                xq0,
                xk0,
                xv0,
                cu_seqlens={"q": q_cumlen0, "k": k_cumlen0},
                max_seqlen={"q": max_q0, "k": max_k0},
            )

            # Half 1
            xq1, xk1, xv1 = [rearrange(i, "s b h d -> b s h d") for i in [xq1, xk1, xv1]]
            xq1, xk1 = self.forward_rope(
                xq1,
                xk1,
                pos_id_q=position_id[q_range1[0] : q_range1[1]],
                pos_id_k=position_id[k_range1[0] : k_range1[1]],
            )
            out1 = self.forward_attention_core(
                xq1,
                xk1,
                xv1,
                cu_seqlens={"q": q_cumlen1, "k": k_cumlen1},
                max_seqlen={"q": max_q1, "k": max_k1},
            )
            output = torch.cat([out0, out1], dim=1)
            output = rearrange(output, "b s h d -> s b (h d)")
        else:
            # Non-CP path via shared _forward
            xq, xk, xv = [rearrange(i, "s b h d -> b s h d") for i in [xq, xk, xv]]

            xq, xk = self.forward_rope(xq, xk, pos_id_q=position_id, pos_id_k=position_id)
            output = self.forward_attention_core(
                xq,
                xk,
                xv,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seq_len,
            )
            output = rearrange(output, "b s h d -> s b (h d)")

        output = self.wo.forward(
            output,
            custom_pre_recompute_function_input=gate_weight,
        )[0]

        return output
