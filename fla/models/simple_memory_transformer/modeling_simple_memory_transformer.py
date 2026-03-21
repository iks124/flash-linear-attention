"""
SimpleMemoryTransformer: Transformer with GLA-based online compression memory.

Design overview
───────────────
Each block has three pre-norm sublayers.  The INPUT hidden states are stored
into the buffer BEFORE compression runs, so the oldest overflow tokens are
always from prior steps — never the current input.  This ensures:
  • Causality: current tokens never appear in the gla_state they query.
  • Train/inference consistency: both use the same chunked path for long inputs.
  • Compressor trainability: gla_state is live (in-graph) when the reader uses it.

    _append_to_buffer(x)    # store detached input FIRST
    _compress_overflow()    # compress oldest overflow → live gla_state (in-graph)
    residual = x
    h_sa  = residual + Self-Attention(attn_norm(residual))   # local window
    h_mem = h_sa + MemoryRead(h_sa, gla_state)               # read live gla_state
    out   = h_mem + MLP(mlp_norm(h_mem))

Compression pipeline
────────────────────
After each forward call, `out.detach()` is appended to the per-layer `raw_buffer`.
When the buffer exceeds `raw_buffer_max_size`, the oldest overflow tokens are
compressed into the persistent GLA recurrent state via `GLAMemoryCompressor`.
Compression is purely overflow-driven: triggered only when the buffer grows
beyond `raw_buffer_max_size`.

Memory storage: GLA recurrent state (B, H, K, V)
──────────────────────────────────────────────────
`GLAMemoryCompressor` runs `fused_recurrent_gla` on the overflow segment, using
the previous state as `initial_state`. This accumulates compressed representations
with natural gating/forgetting (older content decays as new content is absorbed).

Memory retrieval: linear attention read
────────────────────────────────────────
`GLAMemoryReader` projects current hidden states to queries Q, then retrieves:
    o_t = Q_t @ S          (B, T, H, K) × (B, H, K, V) → (B, T, H, V)
Per-head RMSNorm + sigmoid gate + output projection complete the retrieval.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from fla.layers.attn import Attention
from fla.models.simple_memory_transformer.configuration_simple_memory_transformer import SimpleMemoryTransformerConfig
from fla.models.utils import Cache, FLAGenerationMixin
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm
from fla.modules import GatedMLP as MemTransformerMLP
from fla.modules.l2warp import l2_warp
from fla.ops.gla import chunk_gla, fused_recurrent_gla

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer

logger = logging.get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Output dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BaseModelOutputWithPastAndMemory(BaseModelOutputWithPast):
    memory_state: Optional[Any] = None


@dataclass
class CausalLMOutputWithPastAndMemory(CausalLMOutputWithPast):
    memory_state: Optional[Any] = None


# ──────────────────────────────────────────────────────────────────────────────
# MemoryCache
# ──────────────────────────────────────────────────────────────────────────────

class MemoryCache(Cache):
    """
    Extends the standard KV Cache with per-layer GLA compression state.

    Per-layer fields
    ----------------
    raw_buffer  : (B, L_buf, C)         – detached hidden states not yet compressed.
    gla_state   : (B, H, K, V) | None  – accumulated GLA recurrent state (the memory).
    """

    def __init__(self, seen_tokens: int = 0, **kwargs: Any) -> None:
        super().__init__(seen_tokens=seen_tokens, **kwargs)
        self._raw_buffer: dict[int, torch.Tensor | None] = {}
        self._gla_state:  dict[int, torch.Tensor | None] = {}

    def _ensure_layer(self, layer_idx: int) -> None:
        if layer_idx not in self._raw_buffer:
            self._raw_buffer[layer_idx] = None
            self._gla_state[layer_idx]  = None

    def get_layer_memory(self, layer_idx: int) -> dict[str, Any]:
        self._ensure_layer(layer_idx)
        return {
            'raw_buffer': self._raw_buffer[layer_idx],
            'gla_state':  self._gla_state[layer_idx],
        }

    def update_layer_memory(
        self,
        layer_idx:  int,
        raw_buffer: torch.Tensor | None,
        gla_state:  torch.Tensor | None,
    ) -> None:
        self._ensure_layer(layer_idx)
        self._raw_buffer[layer_idx] = raw_buffer
        self._gla_state[layer_idx]  = gla_state

    def detach_(self) -> 'MemoryCache':
        """
        Detach all stored tensors in-place (truncated BPTT).

        Detaches both the KV cache (attn_state in each FLALayer) and the memory
        compression state (gla_state, raw_buffer).  Call this after each training
        chunk to break cross-chunk gradient chains.
        """
        for layer in self.layers:
            if layer.state is not None and layer.state.get('attn_state') is not None:
                layer.state['attn_state'] = tuple(t.detach() for t in layer.state['attn_state'])
        for d in (self._raw_buffer, self._gla_state):
            for k, t in d.items():
                if t is not None:
                    d[k] = t.detach()
        return self


# ──────────────────────────────────────────────────────────────────────────────
# GLA Memory Compressor
# ──────────────────────────────────────────────────────────────────────────────

class GLAMemoryCompressor(nn.Module):
    """
    Compresses a variable-length segment into a GLA recurrent state.

    Runs fused_recurrent_gla on the segment, continuing from `initial_state`
    if provided.  The GLA gating mechanism acts as a natural forgetting decay:
    older content is progressively weighted out as new segments arrive.

    Returns the updated recurrent state (B, H, K, V) which serves as the
    entire compressed memory — no explicit slot cap is needed.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        gate_low_rank_dim: int = 16,
        gate_logit_normalizer: int = 16,
        norm_eps: float = 1e-5,
        mode: str = 'auto',
    ):
        super().__init__()
        assert mode in ('auto', 'chunk', 'fused_recurrent'), \
            f"Unsupported mode `{mode}`. Choose from 'auto', 'chunk', 'fused_recurrent'."
        self.mode                 = mode
        self.num_heads            = num_heads
        self.key_dim              = int(hidden_size * expand_k)
        self.value_dim            = int(hidden_size * expand_v)
        self.head_k_dim           = self.key_dim  // num_heads
        self.head_v_dim           = self.value_dim // num_heads
        self.gate_logit_normalizer = gate_logit_normalizer

        assert self.key_dim   % num_heads == 0, "key_dim must be divisible by num_heads"
        assert self.value_dim % num_heads == 0, "value_dim must be divisible by num_heads"

        self.norm   = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.q_proj = nn.Linear(hidden_size, self.key_dim,   bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim,   bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.gk_proj = nn.Sequential(
            nn.Linear(hidden_size, gate_low_rank_dim, bias=False),
            nn.Linear(gate_low_rank_dim, self.key_dim, bias=True),
        )

    def forward(
        self,
        segment:       torch.Tensor,              # (B, T, C)
        initial_state: torch.Tensor | None = None, # (B, H, K, V)
    ) -> torch.Tensor:                             # new state (B, H, K, V)
        x  = self.norm(segment)
        q  = self.q_proj(x)   # (B, T, key_dim)
        k  = self.k_proj(x)
        v  = self.v_proj(x)   # (B, T, value_dim)
        gk = self.gk_proj(x)  # (B, T, key_dim)

        q  = rearrange(q,  'b t (h d) -> b t h d', d=self.head_k_dim)
        k  = rearrange(k,  'b t (h d) -> b t h d', d=self.head_k_dim)
        v  = rearrange(v,  'b t (h d) -> b t h d', d=self.head_v_dim)
        gk = rearrange(gk, 'b t (h d) -> b t h d', d=self.head_k_dim)
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer

        # Auto-selection (mirrors GatedLinearAttention's runtime override):
        #   T <= 64 : fused_recurrent — decode (compress_to=1) or very short segment.
        #             Purely sequential kernel, zero chunk-parallelism overhead.
        #   T >  64 : chunk — training or long-prompt prefill.
        #             Chunk-parallel, supports initial_state, gradient-compatible.
        # When mode is explicitly set ('chunk' or 'fused_recurrent'), the T<=64
        # guard still forces fused_recurrent to avoid degenerate chunk launches.
        T = segment.shape[1]
        if T <= 64 or self.mode == 'fused_recurrent':
            mode = 'fused_recurrent'
        else:
            mode = 'chunk'  # self.mode == 'auto' or 'chunk'

        if mode == 'fused_recurrent':
            _, new_state = fused_recurrent_gla(
                q=q, k=k, v=v, gk=gk,
                initial_state=initial_state,
                output_final_state=True,
            )
        else:  # chunk
            _, new_state = chunk_gla(
                q=q, k=k, v=v, g=gk,
                initial_state=initial_state,
                output_final_state=True,
            )

        return new_state  # (B, H, head_k_dim, head_v_dim)


# ──────────────────────────────────────────────────────────────────────────────
# GLA Memory Reader
# ──────────────────────────────────────────────────────────────────────────────

class GLAMemoryReader(nn.Module):
    """
    Reads from a GLA recurrent state (B, H, K, V) given current hidden states.

    Projects hidden_states → queries Q, then performs a linear attention read:
        o = Q @ S    (B, T, H, K) × (B, H, K, V) → (B, T, H, V)

    Queries are L2-normalised for stable retrieval from an accumulated state.
    Per-head RMSNorm + sigmoid gate + output projection map the result back to
    the model's hidden_size.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.num_heads  = num_heads
        self.key_dim    = int(hidden_size * expand_k)
        self.value_dim  = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim  // num_heads
        self.head_v_dim = self.value_dim // num_heads

        assert self.key_dim   % num_heads == 0
        assert self.value_dim % num_heads == 0

        self.norm      = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.q_proj    = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_norm    = nn.RMSNorm(self.head_v_dim, eps=norm_eps)  # per-head output norm
        self.o_proj    = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        hidden_states:   torch.Tensor,  # (B, T, C)
        recurrent_state: torch.Tensor,  # (B, H, K, V)
    ) -> torch.Tensor:                  # (B, T, C)
        x = self.norm(hidden_states)

        q = self.q_proj(x)                                                        # (B, T, key_dim)
        q = rearrange(q, 'b t (h d) -> b t h d', d=self.head_k_dim)              # (B, T, H, K)
        q = F.normalize(q, dim=-1)                                                # L2-norm for stability

        # Linear retrieval: (B, T, H, K) × (B, H, K, V) → (B, T, H, V)
        o = torch.einsum('bthk,bhkv->bthv', q, recurrent_state)
        o = self.v_norm(o)                                                        # (B, T, H, V)
        o = rearrange(o, 'b t h v -> b t (h v)')                                 # (B, T, value_dim)
        o = self.o_proj(o)                                                        # (B, T, C)

        gate = torch.sigmoid(self.gate_proj(x))
        return gate * o


# ──────────────────────────────────────────────────────────────────────────────
# Memory Transformer Block
# ──────────────────────────────────────────────────────────────────────────────

class SimpleMemoryTransformerBlock(GradientCheckpointingLayer):
    """
    Transformer block with GLA-based online compression memory.

    Forward pass (memory-enabled layer):

        _append_to_buffer(x)    # store detached INPUT first
        _compress_overflow()    # compress oldest overflow → live gla_state (in-graph)
        residual = x
        h_sa  = residual + Self-Attention(attn_norm(residual))   # local window
        h_mem = h_sa + MemoryRead(h_sa, gla_state)               # read live gla_state
        out   = h_mem + MLP(mlp_norm(h_mem))

    Key invariant: the tokens compressed into gla_state are ALWAYS from prior steps
    (oldest overflow), never from the current input.  This guarantees causality —
    current tokens can never query a state that contains themselves.

    Gradient flow: gla_state is live (in-graph) when GLAMemoryReader consumes it,
    so the compressor receives gradients via:
        loss → reader → gla_state_live → compressor params

    For layers without memory the MemoryRead step is skipped.
    """

    def __init__(self, config: SimpleMemoryTransformerConfig, layer_idx: int):
        super().__init__()
        self.config        = config
        self.layer_idx     = layer_idx
        self.enable_memory = (
            config.use_memory
            and layer_idx in set(config.memory_layers)
        )

        # ── Self-Attention ────────────────────────────────────────────────
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            qkv_bias=config.qkv_bias,
            qk_norm=config.qk_norm,
            window_size=config.local_window_size,   # bounded local window
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            layer_idx=layer_idx,
        )

        # ── MLP ───────────────────────────────────────────────────────────
        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = MemTransformerMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

        # ── Memory components (memory layers only) ────────────────────────
        if self.enable_memory:
            self.compressor = GLAMemoryCompressor(
                hidden_size=config.hidden_size,
                num_heads=config.gla_num_heads,
                expand_k=config.gla_expand_k,
                expand_v=config.gla_expand_v,
                gate_low_rank_dim=config.gla_gate_low_rank_dim,
                gate_logit_normalizer=config.gla_gate_logit_normalizer,
                norm_eps=config.norm_eps,
                mode=config.gla_mode,
            )
            self.memory_reader = GLAMemoryReader(
                hidden_size=config.hidden_size,
                num_heads=config.gla_num_heads,
                expand_k=config.gla_expand_k,
                expand_v=config.gla_expand_v,
                norm_eps=config.norm_eps,
            )

    # ── Buffer utilities ──────────────────────────────────────────────────────

    @staticmethod
    def _cat(old: torch.Tensor | None, new: torch.Tensor) -> torch.Tensor:
        """Concatenate along the sequence dimension."""
        return new if old is None else torch.cat([old, new], dim=1)

    def _find_compress_boundary(self, buffer: torch.Tensor) -> int | None:
        """
        Overflow-driven compression boundary.

        Returns the number of oldest tokens to compress (exclusive end index),
        or None if the buffer has not yet exceeded `raw_buffer_max_size`.

        The buffer is allowed to grow up to `raw_buffer_max_size` tokens.
        Once exceeded, all tokens older than the window are compressed at once,
        leaving exactly `raw_buffer_max_size` tokens in the raw buffer.
        """
        L_buf = buffer.shape[1]
        if L_buf <= self.config.raw_buffer_max_size:
            return None
        return L_buf - self.config.raw_buffer_max_size

    def _compress_one_segment(
        self,
        buffer:    torch.Tensor,              # (B, L_buf, C)
        gla_state: torch.Tensor | None,       # (B, H, K, V) or None
        compress_to: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run GLAMemoryCompressor on `buffer[:, :compress_to]` and return
        (remaining_buffer, updated_gla_state).
        """
        segment   = buffer[:, :compress_to]   # (B, compress_to, C)
        remaining = buffer[:, compress_to:]   # (B, L_buf - compress_to, C)
        new_state = self.compressor(segment, initial_state=gla_state)
        return remaining, new_state

    def _try_compress(
        self,
        buffer:    torch.Tensor,
        gla_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Compress any buffer overflow into the GLA state.
        Single-pass: after one compression the buffer is at most raw_buffer_max_size.
        """
        compress_to = self._find_compress_boundary(buffer)
        if compress_to is None:
            return buffer, gla_state
        return self._compress_one_segment(buffer, gla_state, compress_to)

    def _compress_overflow(self, memory_cache: MemoryCache) -> MemoryCache:
        """
        Called at the START of forward: compress any buffer overflow into a live
        GLA state (in the current backward graph).

        Because compression runs before the memory read, the freshly-produced
        gla_state is consumed by GLAMemoryReader in the same forward/backward
        pass, giving compressor parameters a gradient via:
            loss → reader → gla_state_live → compressor params
        """
        state = memory_cache.get_layer_memory(self.layer_idx)
        buffer = state['raw_buffer']
        if buffer is None:
            return memory_cache
        compress_to = self._find_compress_boundary(buffer)
        if compress_to is None:
            return memory_cache
        remaining, new_gla_state = self._compress_one_segment(
            buffer, state['gla_state'], compress_to
        )
        memory_cache.update_layer_memory(self.layer_idx, remaining, new_gla_state)
        return memory_cache

    def _append_to_buffer(
        self,
        hidden_states: torch.Tensor,  # post-MLP output (detached before storing)
        memory_cache:  MemoryCache,
    ) -> MemoryCache:
        """
        Called at the END of forward: append detached hidden_states to the raw
        buffer.  Compression is intentionally deferred to the next forward call
        (_compress_overflow) so it runs inside that call's backward graph.
        """
        state  = memory_cache.get_layer_memory(self.layer_idx)
        buffer = self._cat(state['raw_buffer'], hidden_states.detach())
        memory_cache.update_layer_memory(self.layer_idx, buffer, state['gla_state'])
        return memory_cache

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states:  torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        use_cache:       bool | None = False,
        output_attentions: bool | None = False,
        **kwargs,
    ) -> tuple:
        # ── 0. Append INPUT to buffer, then compress oldest overflow ───────
        # Input is stored BEFORE compression so that compressed tokens are
        # always from prior steps (oldest overflow), never the current input.
        # This preserves causality: no token ever queries a gla_state that
        # contains itself.  The live gla_state produced here is then consumed
        # by the reader below, giving compressor params a gradient via:
        #     loss → reader → gla_state_live → compressor params
        if self.enable_memory and isinstance(past_key_values, MemoryCache):
            past_key_values = self._append_to_buffer(hidden_states, past_key_values)
            past_key_values = self._compress_overflow(past_key_values)

        # ── 1. Self-Attention (local window) ──────────────────────────────
        residual      = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = residual + hidden_states  # h_sa

        # ── 2. Memory Read (from the live gla_state produced in step 0) ───
        if self.enable_memory and isinstance(past_key_values, MemoryCache):
            state     = past_key_values.get_layer_memory(self.layer_idx)
            gla_state = state['gla_state']
            if gla_state is not None:
                mem_delta     = self.memory_reader(hidden_states, gla_state)
                hidden_states = hidden_states + mem_delta  # h_mem

        # ── 3. MLP ────────────────────────────────────────────────────────
        residual      = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states  # out

        return hidden_states, attentions, past_key_values


# ──────────────────────────────────────────────────────────────────────────────
# PreTrainedModel base
# ──────────────────────────────────────────────────────────────────────────────

class SimpleMemoryTransformerPreTrainedModel(PreTrainedModel):
    config_class              = SimpleMemoryTransformerConfig
    base_model_prefix         = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules         = ['SimpleMemoryTransformerBlock']
    _supports_cache_class     = True

    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = False,
        num_residuals_per_layer:  int  = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

        if rescale_prenorm_residual:
            for attr in ('o_proj', 'down_proj'):
                p = getattr(getattr(module, attr, None), 'weight', None)
                if p is not None:
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)


# ──────────────────────────────────────────────────────────────────────────────
# SimpleMemoryTransformerModel
# ──────────────────────────────────────────────────────────────────────────────

class SimpleMemoryTransformerModel(SimpleMemoryTransformerPreTrainedModel):

    def __init__(self, config: SimpleMemoryTransformerConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size  = config.vocab_size

        self.embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList([
            SimpleMemoryTransformerBlock(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def _chunked_forward(
        self,
        inputs_embeds:  torch.FloatTensor,          # (B, L, C)
        attention_mask: Optional[torch.Tensor],      # (B, L) padding mask or None
        past_key_values: MemoryCache | None,
        return_dict:    bool,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPastAndMemory:
        """
        Chunked forward pass used for both training and long-context inference.

        The sequence is split into chunks of `training_chunk_size` tokens.  A
        single MemoryCache threads through all chunks, carrying:
          • KV cache (attn_state) – local sliding-window attention across chunks.
          • raw_buffer + gla_state – GLA memory compression pipeline.

        Each chunk's block.forward (new store-first design):
          0. _append_to_buffer(input): store current chunk's input (detached).
          1. _compress_overflow(): oldest overflow → LIVE gla_state (in-graph).
          2. Self-attention.
          3. Memory read from the live gla_state → compressor params get gradient.
          4. MLP.

        Training: cache.detach_() after each chunk for truncated BPTT and FSDP2
          compatibility.  gla_state becomes initial_state for the next chunk's
          compression; compressor params still receive gradient in every chunk
          because compression recomputes a fresh live gla_state from live params.

        Inference: no detach — gla_state chains naturally across chunks, giving
          the model full long-context memory without gradient overhead.
        """
        chunk_size   = self.config.training_chunk_size
        B, L, _      = inputs_embeds.shape
        position_ids = kwargs.get('position_ids', None)

        cache         = past_key_values if past_key_values is not None else MemoryCache()
        chunk_outputs: list[torch.Tensor] = []

        for chunk_start in range(0, L, chunk_size):
            chunk_end = min(chunk_start + chunk_size, L)
            h = inputs_embeds[:, chunk_start:chunk_end]

            if attention_mask is not None and attention_mask.dim() == 2:
                chunk_attn_mask = attention_mask[:, chunk_start:chunk_end]
            else:
                chunk_attn_mask = None

            kw = dict(kwargs)
            if position_ids is not None:
                kw['position_ids'] = position_ids[:, chunk_start:chunk_end]

            for layer in self.layers:
                h, _, cache = layer(
                    h,
                    attention_mask=chunk_attn_mask,
                    past_key_values=cache,
                    use_cache=True,
                    output_attentions=False,
                    **kw,
                )

            chunk_outputs.append(h)
            if self.training:
                # Truncated BPTT: detach cache between chunks.
                # raw_buffer tokens are already detached (_append_to_buffer).
                # Detaching gla_state truncates the gradient graph at this boundary;
                # the next chunk's _compress_overflow recomputes a fresh live
                # gla_state from (detached initial_state, detached buffer, live
                # compressor params), so compressor params get gradient every chunk.
                # Required for FSDP2: frees parameter shard storage after backward.
                cache.detach_()

        hidden_states = self.norm(torch.cat(chunk_outputs, dim=1))   # (B, L, C)

        if not return_dict:
            return tuple(v for v in [hidden_states, None, None, None, cache] if v is not None)

        return BaseModelOutputWithPastAndMemory(
            last_hidden_state=hidden_states,
            past_key_values=cache,
            hidden_states=None,
            attentions=None,
            memory_state=cache,
        )

    def forward(
        self,
        input_ids:       torch.LongTensor | None = None,
        attention_mask:  Optional[torch.Tensor]  = None,
        inputs_embeds:   torch.FloatTensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        use_cache:       bool | None = None,
        output_attentions:    bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict:          bool | None = None,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPastAndMemory:

        if output_attentions:
            warnings.warn(
                '`SimpleMemoryTransformerModel` does not support `output_attentions`; '
                'setting it to False.'
            )
            output_attentions = False

        output_attentions    = output_attentions    if output_attentions    is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache            = use_cache            if use_cache            is not None else (self.config.use_cache if not self.training else False)
        return_dict          = return_dict          if return_dict          is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('Specify either input_ids or inputs_embeds, not both.')
        if input_ids is None and inputs_embeds is None:
            raise ValueError('You must specify input_ids or inputs_embeds.')

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        hidden_states = inputs_embeds

        # ── Cache initialisation ──────────────────────────────────────────────
        if use_cache or self.config.use_memory:
            if past_key_values is not None and not isinstance(past_key_values, Cache):
                past_key_values = Cache.from_legacy_cache(past_key_values)
            elif past_key_values is None:
                past_key_values = MemoryCache() if self.config.use_memory else Cache()

        # ── Chunked forward (training + long-context inference) ──────────
        # Triggered whenever input exceeds the local window and chunking is
        # configured.  Applies to both training and inference so that prefill
        # of long prompts uses the same memory-read pathway as training.
        if (
            self.config.use_memory
            and self.config.training_chunk_size is not None
            and kwargs.get('cu_seqlens') is None   # varlen not supported in chunked mode
            and hidden_states.shape[1] > self.config.raw_buffer_max_size
        ):
            if output_hidden_states:
                warnings.warn(
                    '`output_hidden_states` is not supported in chunked '
                    'mode; setting it to False.'
                )
            return self._chunked_forward(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                return_dict=return_dict,
                **kwargs,
            )

        all_hidden_states = () if output_hidden_states else None
        all_attns         = () if output_attentions    else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states, attentions, past_key_values = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )

            if output_attentions:
                all_attns += (attentions,)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [
                    hidden_states, past_key_values,
                    all_hidden_states, all_attns,
                    past_key_values if isinstance(past_key_values, MemoryCache) else None,
                ] if v is not None
            )

        return BaseModelOutputWithPastAndMemory(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attns,
            memory_state=past_key_values if isinstance(past_key_values, MemoryCache) else None,
        )


# ──────────────────────────────────────────────────────────────────────────────
# SimpleMemoryTransformerForCausalLM
# ──────────────────────────────────────────────────────────────────────────────

class SimpleMemoryTransformerForCausalLM(SimpleMemoryTransformerPreTrainedModel, FLAGenerationMixin):
    _tied_weights_keys = ['lm_head.weight']

    def __init__(self, config: SimpleMemoryTransformerConfig):
        super().__init__(config)
        self.model      = SimpleMemoryTransformerModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head    = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion  = None
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @deprecate_kwarg('num_logits_to_keep', version='4.50', new_name='logits_to_keep')
    def forward(
        self,
        input_ids:       torch.LongTensor | None = None,
        attention_mask:  torch.Tensor | None = None,
        inputs_embeds:   torch.FloatTensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        labels:          torch.LongTensor | None = None,
        use_cache:       bool | None = None,
        output_attentions:    bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict:          bool | None = None,
        logits_to_keep:  int = 0,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPastAndMemory:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]
        loss, logits  = None, None

        if not self.config.fuse_linear_cross_entropy or labels is None:
            logits = self.lm_head(
                hidden_states if logits_to_keep is None
                else hidden_states[:, -logits_to_keep:]
            )

        if labels is not None:
            if getattr(self, 'criterion', None) is None:
                if self.config.fuse_linear_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss(use_l2warp=self.config.use_l2warp)
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion

            labels = labels.to(hidden_states.device)
            labels = torch.cat(
                (labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)),
                dim=1,
            )

            if self.config.fuse_linear_cross_entropy:
                loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))
                loss = l2_warp(loss, logits) if self.config.use_l2warp else loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastAndMemory(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            memory_state=outputs.memory_state,
        )
