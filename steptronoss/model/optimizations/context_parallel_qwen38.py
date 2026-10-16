"""Contiguous CP blocks for hybrid Gated DeltaNet / full-attention models.

GDN boundaries align to its 64-token kernel chunks. State communication stays
outside checkpointed layer calls so backward recomputation cannot repeat P2P.
"""

import os
from types import MethodType

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

CHUNK_ALIGN = 64


def _linear_attention_layer_class():
    """Preserve autograd history instead of overwriting inference cache buffers."""
    from transformers.cache_utils import LinearAttentionLayer

    class GradCarryingLinearAttentionLayer(LinearAttentionLayer):
        def lazy_initialization(
            self, conv_states=None, recurrent_states=None, state_idx=0, conv_kernel_size=None
        ) -> None:
            if conv_states is not None:
                if self.device is None:
                    self.dtype, self.device = conv_states.dtype, conv_states.device
                self.conv_kernel_size[state_idx] = (
                    conv_states.shape[-1] if conv_kernel_size is None else conv_kernel_size
                )
                self.is_conv_states_initialized[state_idx] = True
            if recurrent_states is not None:
                self.is_recurrent_states_initialized[state_idx] = True

        def update_conv_state(self, conv_states, state_idx=0, conv_kernel_size=None, **kwargs):
            if not self.is_conv_states_initialized[state_idx]:
                self.lazy_initialization(
                    conv_states=conv_states, state_idx=state_idx, conv_kernel_size=conv_kernel_size
                )
            width = self.conv_kernel_size[state_idx]
            if not self.has_previous_state[state_idx]:
                full = conv_states
                self.has_previous_state[state_idx] = True
                if full.shape[-1] < width:
                    full = F.pad(full, (width - full.shape[-1], 0))
            else:
                full = torch.cat([self.conv_states[state_idx], conv_states], dim=-1)
            self.conv_states[state_idx] = full[..., -width:]
            return full

        def update_recurrent_state(self, recurrent_states, state_idx=0, **kwargs):
            if not self.is_recurrent_states_initialized[state_idx]:
                self.lazy_initialization(recurrent_states=recurrent_states, state_idx=state_idx)
            self.recurrent_states[state_idx] = recurrent_states
            return recurrent_states

    return GradCarryingLinearAttentionLayer


def _trace(message):
    if os.environ.get("QWEN38_CP_TRACE") == "1":
        print(f"CP_TRACE rank={dist.get_rank()} {message}", flush=True)


def cp_info():
    """(cp_size, cp_rank, group) for this rank, from PM -- the repo's CP authority."""
    from steptronoss.core.parallel_state import PM

    if not dist.is_initialized() or "CP" not in PM.parallels:
        return 1, 0, None
    size = PM.size_of("CP")
    if size <= 1:
        return 1, 0, None
    return size, PM.rank_in("CP"), PM.group_of("CP")


def split_contiguous(tensor, cp_size, cp_rank, dim=1):
    """This rank's contiguous block. Mirrors steptron's scatter_to_sequential_cp_region."""
    total = tensor.shape[dim]
    if total % (cp_size * CHUNK_ALIGN):
        raise ValueError(
            f"sequence length {total} must be a multiple of cp_size * {CHUNK_ALIGN} "
            f"= {cp_size * CHUNK_ALIGN}; off-multiple boundaries lose the GDN kernel's exactness"
        )
    block = total // cp_size
    return tensor.narrow(dim, cp_rank * block, block)


class _RecvStatePair(torch.autograd.Function):
    """Receive one GDN layer's (recurrent, conv) state from the previous CP rank.

    Both are received in one Function so the pair is atomic. NCCL point-to-point
    has no tags -- a receive matches whatever arrives next from that peer -- so if
    autograd happened to run the two gradients' sends in the opposite order, the
    peer would read the conv gradient into the recurrent slot. Wrong numbers, no
    error. Keeping the pair inside one node removes that freedom.
    """

    @staticmethod
    def forward(ctx, trigger, recurrent_meta, conv_meta, src, group):
        ctx.src, ctx.group = src, group
        ctx.meta = (recurrent_meta, conv_meta)
        ctx.device = trigger.device
        _trace(f"recv_state_begin src={src}")
        recurrent = torch.empty(recurrent_meta[0], dtype=recurrent_meta[1], device=trigger.device)
        conv = torch.empty(conv_meta[0], dtype=conv_meta[1], device=trigger.device)
        dist.recv(recurrent, src=dist.get_global_rank(group, src), group=group)
        dist.recv(conv, src=dist.get_global_rank(group, src), group=group)
        _trace(f"recv_state_done src={src}")
        return recurrent, conv

    @staticmethod
    def backward(ctx, grad_recurrent, grad_conv):
        # Same order the sender's backward expects to receive them in.
        grad_recurrent = (
            torch.zeros(ctx.meta[0][0], dtype=ctx.meta[0][1], device=ctx.device)
            if grad_recurrent is None
            else grad_recurrent
        )
        grad_conv = (
            torch.zeros(ctx.meta[1][0], dtype=ctx.meta[1][1], device=ctx.device) if grad_conv is None else grad_conv
        )
        _trace(f"send_state_grad_begin dst={ctx.src}")
        dist.send(grad_recurrent.contiguous(), dst=dist.get_global_rank(ctx.group, ctx.src), group=ctx.group)
        dist.send(grad_conv.contiguous(), dst=dist.get_global_rank(ctx.group, ctx.src), group=ctx.group)
        _trace(f"send_state_grad_done dst={ctx.src}")
        return torch.zeros((), device=ctx.device), None, None, None, None


class _SendStatePair(torch.autograd.Function):
    """Send one GDN layer's (recurrent, conv) state onward; take its gradient back.

    Takes and returns a `token` scalar so the per-layer sends CHAIN, which is
    load-bearing for the same tagless-NCCL reason: chaining makes each send depend
    on the previous one, so autograd's reverse-topological order is exactly the
    reverse of the forward order -- which is the order the peer receives in.
    Without it, all the sends hang off one node and fire in arbitrary order.
    """

    @staticmethod
    def forward(ctx, recurrent, conv, token, dst, group):
        ctx.dst, ctx.group = dst, group
        ctx.meta = [(t.shape, t.dtype, t.device) for t in (recurrent, conv)]
        _trace(f"send_state_begin dst={dst}")
        dist.send(recurrent.contiguous(), dst=dist.get_global_rank(group, dst), group=group)
        dist.send(conv.contiguous(), dst=dist.get_global_rank(group, dst), group=group)
        _trace(f"send_state_done dst={dst}")
        return recurrent.new_zeros(())

    @staticmethod
    def backward(ctx, grad_token):
        _trace(f"recv_state_grad_begin src={ctx.dst}")
        grads = []
        for shape, dtype, device in ctx.meta:
            buffer = torch.empty(shape, dtype=dtype, device=device)
            dist.recv(buffer, src=dist.get_global_rank(ctx.group, ctx.dst), group=ctx.group)
            grads.append(buffer)
        _trace(f"recv_state_grad_done src={ctx.dst}")
        return grads[0], grads[1], grad_token, None, None


class _OneLayerCache:
    """The smallest object the stock GDN forward will accept as `cache_params`.

    It calls exactly five things on it: `has_previous_state(idx, state_idx)`,
    `layers[idx].recurrent_states[0]`, `update_conv_state(...)`,
    `update_recurrent_state(...)`, and the truthiness check behind
    `output_final_state=cache_params is not None`. Supplying those lets the
    recurrence run unmodified while the state comes from, and returns to, here.
    """

    def __init__(self, layer_idx, recurrent=None, conv=None):
        self._idx = layer_idx
        self._layer = _linear_attention_layer_class()()
        self.layers = {layer_idx: self._layer}
        if recurrent is not None:
            self._layer.recurrent_states[0] = recurrent
            self._layer.conv_states[0] = conv
            self._layer.conv_kernel_size[0] = conv.shape[-1]
            self._layer.is_recurrent_states_initialized[0] = True
            self._layer.is_conv_states_initialized[0] = True
            # Tells the GDN forward to use initial_state rather than start from zero.
            self._layer.has_previous_state[0] = True

    def has_previous_state(self, layer_idx=None, state_idx=None):
        return bool(self._layer.has_previous_state[state_idx or 0])

    def update_conv_state(self, conv_states, layer_idx, conv_kernel_size=None, **kwargs):
        return self._layer.update_conv_state(conv_states, 0, conv_kernel_size=conv_kernel_size)

    def update_recurrent_state(self, recurrent_states, layer_idx, **kwargs):
        return self._layer.update_recurrent_state(recurrent_states, 0)

    def harvest(self):
        return self._layer.recurrent_states[0], self._layer.conv_states[0]


def _install_gated_deltanet(module):
    """Thread the recurrent and conv state through a kwarg, and return the new state."""
    original = module.forward

    def forward(self, hidden_states, cache_params=None, attention_mask=None, cp_state=None, **kwargs):
        recurrent, conv = cp_state if cp_state is not None else (None, None)
        cache = _OneLayerCache(self.layer_idx, recurrent, conv)
        output = original(hidden_states=hidden_states, cache_params=cache, attention_mask=attention_mask, **kwargs)
        return output, cache.harvest()

    module.forward = MethodType(forward, module)


class _GatherKV(torch.autograd.Function):
    """One ordered collective for K/V; backward sums every consumer's gradients."""

    @staticmethod
    def forward(ctx, key, value, group):
        _trace("gather_kv_forward")
        ctx.group = group
        ctx.size = dist.get_world_size(group)
        ctx.shape = key.shape
        pair = torch.stack((key, value)).contiguous()
        gathered = [torch.empty_like(pair) for _ in range(ctx.size)]
        dist.all_gather(gathered, pair, group=group)
        full = torch.cat(gathered, dim=3)  # pair, batch, heads, sequence, feature
        return full[0], full[1]

    @staticmethod
    def backward(ctx, grad_key, grad_value):
        _trace("gather_kv_backward")
        shape = list(ctx.shape)
        shape[2] *= ctx.size
        if grad_key is None:
            grad_key = torch.zeros(shape, device=grad_value.device, dtype=grad_value.dtype)
        if grad_value is None:
            grad_value = torch.zeros_like(grad_key)
        chunks = [
            torch.stack((k, v)).contiguous()
            for k, v in zip(grad_key.chunk(ctx.size, dim=2), grad_value.chunk(ctx.size, dim=2))
        ]
        result = torch.empty_like(chunks[0])
        if dist.get_backend(ctx.group) == "nccl":
            dist.reduce_scatter(result, chunks, op=dist.ReduceOp.SUM, group=ctx.group)
        else:
            # CPU reference uses gloo, which does not implement reduce_scatter.
            full = torch.stack(chunks)
            dist.all_reduce(full, group=ctx.group)
            result.copy_(full[dist.get_rank(ctx.group)])
        return result[0], result[1], None


class _PrefixFlash(torch.autograd.Function):
    """Local prefix forward, full-query replay backward before BF16 rounding.

    FA2 returns BF16 dK/dV. Adding independently rounded query-block gradients
    deviates from an unsplit backward even with FP32 collective accumulation.
    Replaying the full query objective once on each rank retains FA2's rounding
    point. K/V contributions are divided by CP before SUM reduce-scatter; Q's
    gradient is sliced to its owner. This prioritizes correctness over backward
    FLOPs and uses only linear-sized storage, never an explicit attention matrix.
    """

    @staticmethod
    def forward(ctx, query, key, value, mask, group, scaling, deterministic):
        from flash_attn import flash_attn_varlen_func

        size, rank = dist.get_world_size(group), dist.get_rank(group)
        batch, heads, length, dim = query.shape
        pieces = [torch.empty_like(query) for _ in range(size)]
        dist.all_gather(pieces, query.contiguous(), group=group)
        full_query = torch.cat(pieces, dim=2)
        key, value = key.to(query.dtype), value.to(query.dtype)
        ctx.save_for_backward(full_query, key, value, mask)
        ctx.group, ctx.rank, ctx.size = group, rank, size
        ctx.length, ctx.scaling, ctx.deterministic = length, scaling, deterministic
        end = (rank + 1) * length
        qm = mask[:, rank * length : end].bool()
        km = mask[:, :end].bool()
        out = query.new_zeros(batch, length, heads, dim)
        qlens = qm.sum(1, dtype=torch.int32)
        klens = km.sum(1, dtype=torch.int32)
        if int(qlens.sum()):
            values = flash_attn_varlen_func(
                query.transpose(1, 2)[qm],
                key[:, :, :end].transpose(1, 2)[km],
                value[:, :, :end].transpose(1, 2)[km],
                F.pad(qlens.cumsum(0, dtype=torch.int32), (1, 0)),
                F.pad(klens.cumsum(0, dtype=torch.int32), (1, 0)),
                int(qlens.max()),
                int(klens.max()),
                softmax_scale=scaling,
                causal=True,
                deterministic=deterministic,
            )
            out[qm] = values
        return out

    @staticmethod
    def backward(ctx, grad_output):
        from flash_attn import flash_attn_varlen_func

        _trace("flash_backward_unpack")
        query, key, value, mask = ctx.saved_tensors
        _trace("flash_backward_gather_dout")
        parts = [torch.empty_like(grad_output) for _ in range(ctx.size)]
        dist.all_gather(parts, grad_output.contiguous(), group=ctx.group)
        full_grad = torch.cat(parts, dim=1)
        _trace("flash_backward_compute")
        valid = mask.bool()
        lengths = valid.sum(1, dtype=torch.int32)
        cumulative = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
        with torch.enable_grad():
            q = query.transpose(1, 2)[valid].detach().requires_grad_(True)
            k = key.transpose(1, 2)[valid].detach().requires_grad_(True)
            v = value.transpose(1, 2)[valid].detach().requires_grad_(True)
            out = flash_attn_varlen_func(
                q,
                k,
                v,
                cumulative,
                cumulative,
                int(lengths.max()),
                int(lengths.max()),
                softmax_scale=ctx.scaling,
                causal=True,
                deterministic=ctx.deterministic,
            )
            dq, dk, dv = torch.autograd.grad(out, (q, k, v), grad_outputs=full_grad[valid])
        _trace("flash_backward_compute_done")
        qgrad = torch.zeros_like(query).transpose(1, 2)
        kgrad = torch.zeros_like(key, dtype=torch.float32).transpose(1, 2)
        vgrad = torch.zeros_like(value, dtype=torch.float32).transpose(1, 2)
        qgrad[valid] = dq
        kgrad[valid] = dk.float() / ctx.size
        vgrad[valid] = dv.float() / ctx.size
        offset = ctx.rank * ctx.length
        return (
            qgrad[:, offset : offset + ctx.length].transpose(1, 2).contiguous(),
            kgrad.transpose(1, 2),
            vgrad.transpose(1, 2),
            None,
            None,
            None,
            None,
        )


def _install_attention(module):
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    def forward(self, hidden_states, position_embeddings, attention_mask=None, cp_mask=None, **kwargs):
        size, rank, group = cp_info()
        batch, length, _ = hidden_states.shape
        shape = (batch, length, -1, self.head_dim)
        q, gate = self.q_proj(hidden_states).view(batch, length, -1, self.head_dim * 2).chunk(2, -1)
        gate = gate.reshape(batch, length, -1)
        q = self.q_norm(q.reshape(shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        # Keep the distributed K/V gradient accumulation in FP32. Cast to the
        # kernel dtype only at the attention call, and round once on the owner.
        k, v = _GatherKV.apply(k.float(), v.float(), group)
        full_key, full_value = k, v
        prefix = (rank + 1) * length
        k, v = k[:, :, :prefix], v[:, :, :prefix]
        full_mask = cp_mask
        if full_mask is None:
            full_mask = torch.ones(batch, size * length, dtype=torch.bool, device=q.device)
        key_mask = full_mask[:, :prefix].bool()
        if self.config._attn_implementation == "flash_attention_2":
            if self.training and self.attention_dropout:
                raise ValueError("CP full-query backward currently requires attention_dropout=0")
            # The backward needs all keys even though the forward uses a prefix.
            # Re-gather here is avoided: pass the untrimmed pair into the Function.
            out = _PrefixFlash.apply(
                q, full_key, full_value, full_mask, group, self.scaling, kwargs.get("deterministic", False)
            )
        else:
            if length * prefix > 4 * 1024 * 1024:
                raise ValueError("CP SDPA reference is limited to small sequences; use flash_attention_2")
            qpos = torch.arange(length, device=q.device) + rank * length
            kpos = torch.arange(prefix, device=q.device)
            mask = (kpos[None, :] <= qpos[:, None])[None, None] & key_mask[:, None, None, :]
            keys = k.to(q.dtype).repeat_interleave(self.num_key_value_groups, dim=1)
            values = v.to(q.dtype).repeat_interleave(self.num_key_value_groups, dim=1)
            out = F.scaled_dot_product_attention(
                q,
                keys,
                values,
                attn_mask=mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                scale=self.scaling,
            ).transpose(1, 2)
        out = out.reshape(batch, length, -1) * torch.sigmoid(gate)
        return self.o_proj(out), None

    module.forward = MethodType(forward, module)


def _install_decoder_layer(module):
    """FSDP hooks enclose communication; only the pure local block is recomputed.

    The stock HF __call__ checkpoints the entire forward. Disable that wrapper:
    replaying a recv would hang, and receiving before FSDP's unshard would hang
    its all-gather. Adapter groups are unsharded collectively before the recv too.
    FSDP's output pre-backward hook unshards before the state-gradient exchange.
    """
    module._cp_recompute = bool(module.gradient_checkpointing)
    module.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        cp_mask=None,
        **kwargs,
    ):
        size, rank, group = cp_info()
        _trace(f"block_forward idx={getattr(self, 'layer_idx', '?')} type={self.block_type}")
        for adapter_group in getattr(self, "_cp_adapter_groups", ()):
            adapter_group.unshard()
        state = None
        if self.block_type == "linear_attention" and rank > 0:
            state = _recv_state(self.linear_attn, hidden_states, rank, group)

        def compute(hidden, incoming_recurrent=None, incoming_conv=None):
            residual = hidden
            hidden = self.input_layernorm(hidden)
            if self.block_type == "linear_attention":
                incoming = None if incoming_recurrent is None else (incoming_recurrent, incoming_conv)
                hidden, outgoing = self.linear_attn(
                    hidden_states=hidden, attention_mask=attention_mask, cp_state=incoming, **kwargs
                )
            else:
                hidden, _ = self.self_attn(
                    hidden_states=hidden,
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                    cp_mask=cp_mask,
                    **kwargs,
                )
                outgoing = (None, None)
            hidden = residual + hidden
            hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
            return hidden, *outgoing

        args = (hidden_states,) if state is None else (hidden_states, *state)
        if self._cp_recompute and self.training and torch.is_grad_enabled():
            hidden, recurrent, conv = checkpoint(compute, *args, use_reentrant=False)
        else:
            hidden, recurrent, conv = compute(*args)
        if self.block_type == "linear_attention" and rank < size - 1:
            token = _SendStatePair.apply(
                recurrent, conv, recurrent.new_zeros((), requires_grad=torch.is_grad_enabled()), rank + 1, group
            )
            hidden = hidden + token * 0.0
        return hidden

    module.forward = MethodType(forward, module)


def _install_text_model(module, cp_size, cp_rank, group):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ModelOutputWithPast

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("CP needs exactly one of input_ids or inputs_embeds")
        source = input_ids if input_ids is not None else inputs_embeds
        total = source.shape[1]
        length = total // cp_size
        if attention_mask is not None:
            if attention_mask.ndim != 2 or bool((attention_mask[:, 1:] > attention_mask[:, :-1]).any()):
                raise ValueError("CP accepts only a 2D right-padding mask")
        local = split_contiguous(source, cp_size, cp_rank)
        hidden = self.embed_tokens(local) if input_ids is not None else local
        local_mask = None if attention_mask is None else split_contiguous(attention_mask, cp_size, cp_rank)
        absolute = torch.arange(length, device=hidden.device) + cp_rank * length
        positions = absolute.view(1, 1, -1).expand(3, hidden.shape[0], -1)
        embeddings = self.rotary_emb(hidden, positions)
        # The outer HF multimodal wrapper supplies full-sequence positions and
        # cache flags. CP owns local positions and never uses a shared KV cache.
        for name in ("position_ids", "cache_position", "past_key_values", "use_cache", "return_dict"):
            kwargs.pop(name, None)
        for layer in self.layers[: self.config.num_hidden_layers]:
            hidden = layer(
                hidden,
                position_embeddings=embeddings,
                attention_mask=local_mask,
                position_ids=absolute[None].expand(hidden.shape[0], -1),
                cp_mask=attention_mask,
                **kwargs,
            )
        return Qwen3_5ModelOutputWithPast(last_hidden_state=self.norm(hidden), past_key_values=None)

    module.forward = MethodType(forward, module)


def _recv_state(text_model, inputs_embeds, cp_rank, group):
    """Receive (recurrent, conv) for one GDN layer from the previous CP rank."""
    cfg = text_model
    batch = inputs_embeds.shape[0]
    recurrent_shape = (batch, cfg.num_v_heads, cfg.head_k_dim, cfg.head_v_dim)
    conv_shape = (batch, cfg.key_dim * 2 + cfg.value_dim, cfg.conv_kernel_size)
    # A scalar autograd trigger avoids retaining a full unused state template
    # for every layer. Communication buffers themselves hold the received state.
    trigger = inputs_embeds.reshape(-1)[0].float() * 0.0
    if torch.is_grad_enabled() and not trigger.requires_grad:
        trigger.requires_grad_(True)
    return _RecvStatePair.apply(
        trigger, (recurrent_shape, torch.float32), (conv_shape, inputs_embeds.dtype), cp_rank - 1, group
    )


def install_context_parallel(model):
    """Patch the text decoder to run context-parallel over the CP group.

    Patches instances rather than classes, the way install_chunked_causal_loss
    does, so a second model in the same process is unaffected -- the test suite
    builds several.
    """
    cp_size, cp_rank, group = cp_info()
    if cp_size <= 1:
        raise ValueError("install_context_parallel called with context_parallel_size <= 1")

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    text_model = base.model.language_model
    layer_types = list(text_model.config.layer_types)

    for idx, layer in enumerate(text_model.layers[: text_model.config.num_hidden_layers]):
        if layer_types[idx] == "linear_attention":
            _install_gated_deltanet(layer.linear_attn)
        else:
            _install_attention(layer.self_attn)
        _install_decoder_layer(layer)
    _install_text_model(text_model, cp_size, cp_rank, group)
