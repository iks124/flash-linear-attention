"""
SimpleMemoryTransformer: Transformer with GLA-based async-eviction memory.

Design overview
───────────────
Each block has three pre-norm sublayers plus an async write step:

    residual = x
    h_sa     = residual + Self-Attention(attn_norm(residual))   # local window
    h_mem    = h_sa + Reader(h_sa, prev_write_buffer)           # read evicted tokens
    out      = h_mem + MLP(mlp_norm(h_mem))
    new_wb   = Compressor(h_sa, initial=prev_wb.detach())       # write for next chunk

Two-level memory hierarchy
──────────────────────────
  L1 (precise):    KV cache — local SA window covers recent tokens exactly.
  L2 (compressed): GLA state — GLAMemoryReader retrieves from tokens evicted one chunk ago.

No overlap: prev_write_buffer contains tokens from chunk N-1, which are outside the SA
window when chunk N processes.  SA and GLA cover disjoint token sets.

Joint optimization via one-chunk delay
───────────────────────────────────────
prev_write_buffer is NOT detached between chunks.  It stays live so that:

    loss_{N+1} → reader → prev_write_buffer_N (live) → compressor.{k,v,gk}_proj  ✓

All projections (q_proj in reader, k/v/gk_proj in compressor) receive direct gradients
every chunk.  The compressor is evaluated on "was what you stored useful for the next
window's queries?" — semantically the correct signal for long-range memory.

initial_state is prev_write_buffer.detach(), truncating the gradient chain at the write
boundary so gradients do not span more than one chunk.

Training via truncated BPTT
────────────────────────────
cache.detach_() after each chunk detaches attn_state (KV cache) but intentionally
leaves write_buffer live.  write_buffer is lazily detached when consumed as initial_state
by the compressor in the next chunk's forward.

Note: two consecutive chunks share a gradient graph through write_buffer.  FSDP2 users
should ensure parameter shards remain available across consecutive chunk boundaries.
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
    Extends the standard KV Cache with per-layer GLA write buffers.

    Per-layer fields
    ----------------
    write_buffer : (B, H, K, V) | None
        Live GLA state produced by the current chunk's GLAMemoryCompressor.
        Intentionally NOT detached between chunks — kept live so the next
        chunk's GLAMemoryReader can propagate gradients through it:

            loss_{N+1} → reader → write_buffer_N (live) → compressor.{k,v,gk}_proj ✓

        Lazily detached in GLAMemoryCompressor.forward() when it becomes the
        initial_state for the next write, truncating the gradient chain at one chunk.
    """

    def __init__(self, seen_tokens: int = 0, **kwargs: Any) -> None:
        super().__init__(seen_tokens=seen_tokens, **kwargs)
        self._write_buffer: dict[int, torch.Tensor | None] = {}

    def _ensure_layer(self, layer_idx: int) -> None:
        if layer_idx not in self._write_buffer:
            self._write_buffer[layer_idx] = None

    def get_write_buffer(self, layer_idx: int) -> torch.Tensor | None:
        self._ensure_layer(layer_idx)
        return self._write_buffer[layer_idx]

    def set_write_buffer(self, layer_idx: int, state: torch.Tensor | None) -> None:
        self._ensure_layer(layer_idx)
        self._write_buffer[layer_idx] = state

    def detach_(self) -> 'MemoryCache':
        """
        Detach KV cache (attn_state) in-place for truncated BPTT / FSDP2.

        write_buffer is intentionally NOT detached here — it must stay live so
        that the next chunk's reader can propagate gradients back to the
        compressor.  The lazy detach happens in GLAMemoryCompressor.forward()
        when write_buffer is passed as initial_state for the next write.
        """
        for layer in self.layers:
            if layer.state is not None and layer.state.get('attn_state') is not None:
                layer.state['attn_state'] = tuple(t.detach() for t in layer.state['attn_state'])
        # write_buffer: NOT detached — kept live for next chunk's reader.
        return self


# ──────────────────────────────────────────────────────────────────────────────
# GLAMemoryCompressor  (write-only, no q_proj)
# ──────────────────────────────────────────────────────────────────────────────

class GLAMemoryCompressor(nn.Module):
    """
    Write-only GLA compressor: accumulates current tokens into a recurrent state.

    Only k/v/gk projections are present — no q_proj.  The GLA state update

        S_t = diag(exp(gk_t)) ⊙ S_{t-1} + k_t^T v_t

    does not involve q, so q_proj would carry zero gradient and is omitted.

    The per-token read output of the GLA kernel is discarded; only new_state
    is returned and stored in the cache for the next chunk's reader.

    Gradient flow (one-chunk-delayed, via async write pattern):
        loss_{N+1} → GLAMemoryReader → write_buffer_N (live)
                   → ∂write_buffer_N / ∂{k,v,gk}_proj  ✓

    initial_state is always passed in as detached to prevent gradient chains
    from extending beyond one chunk boundary.
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
            f"mode must be 'auto', 'chunk', or 'fused_recurrent', got '{mode}'"
        self.mode                  = mode
        self.num_heads             = num_heads
        self.key_dim               = int(hidden_size * expand_k)
        self.value_dim             = int(hidden_size * expand_v)
        self.head_k_dim            = self.key_dim   // num_heads
        self.head_v_dim            = self.value_dim // num_heads
        self.gate_logit_normalizer = gate_logit_normalizer

        assert self.key_dim   % num_heads == 0, "key_dim must be divisible by num_heads"
        assert self.value_dim % num_heads == 0, "value_dim must be divisible by num_heads"

        self.norm    = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.k_proj  = nn.Linear(hidden_size, self.key_dim,   bias=False)
        self.v_proj  = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.gk_proj = nn.Sequential(
            nn.Linear(hidden_size, gate_low_rank_dim, bias=False),
            nn.Linear(gate_low_rank_dim, self.key_dim, bias=True),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,              # (B, T, C)
        state:         torch.Tensor | None = None, # (B, H, K, V) — must be detached
    ) -> torch.Tensor:                             # new_state (B, H, K, V)
        x  = self.norm(hidden_states)
        k  = rearrange(self.k_proj(x),  'b t (h d) -> b t h d', d=self.head_k_dim)
        v  = rearrange(self.v_proj(x),  'b t (h d) -> b t h d', d=self.head_v_dim)
        gk = rearrange(self.gk_proj(x), 'b t (h d) -> b t h d', d=self.head_k_dim)
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer

        # Dummy zero query: the GLA kernel API requires q, but the state update
        # S_t = g_t ⊙ S_{t-1} + k_t^T v_t does not involve q.  The per-token
        # output (o_t = q_t @ S_t) is discarded — only new_state is kept.
        q = torch.zeros(
            hidden_states.shape[0], hidden_states.shape[1],
            self.num_heads, self.head_k_dim,
            dtype=k.dtype, device=k.device,
        )

        T = hidden_states.shape[1]
        if T <= 64 or self.mode == 'fused_recurrent':
            _, new_state = fused_recurrent_gla(
                q=q, k=k, v=v, gk=gk,
                initial_state=state,
                output_final_state=True,
            )
        else:
            _, new_state = chunk_gla(
                q=q, k=k, v=v, g=gk,
                initial_state=state,
                output_final_state=True,
            )

        return new_state  # (B, H, head_k_dim, head_v_dim)


# ──────────────────────────────────────────────────────────────────────────────
# GLAMemoryReader
# ──────────────────────────────────────────────────────────────────────────────

class GLAMemoryReader(nn.Module):
    """
    Read-only GLA reader: retrieves from a recurrent state via linear attention.

    Projects hidden_states → Q, then:
        o = Q @ state    (B, T, H, K) × (B, H, K, V) → (B, T, H, V)

    Queries are L2-normalised for stable retrieval from an accumulated state.
    Per-head RMSNorm + sigmoid gate + output projection map back to hidden_size.

    Gradient flow:
        loss → mem_out → {q_proj, o_proj, gate_proj, v_norm, norm}   ✓  (direct)
        loss → mem_out → state (live write_buffer) → compressor.{k,v,gk}_proj  ✓
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
        self.head_k_dim = self.key_dim   // num_heads
        self.head_v_dim = self.value_dim // num_heads

        assert self.key_dim   % num_heads == 0
        assert self.value_dim % num_heads == 0

        self.norm      = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.q_proj    = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_norm    = nn.RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj    = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, hidden_size,   bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, T, C)
        state:         torch.Tensor,  # (B, H, K, V) — live write_buffer from prev chunk
    ) -> torch.Tensor:                # (B, T, C)
        x = self.norm(hidden_states)

        q = rearrange(self.q_proj(x), 'b t (h d) -> b t h d', d=self.head_k_dim)
        q = F.normalize(q, dim=-1)                                    # L2-norm for stability

        o = torch.einsum('bthk,bhkv->bthv', q, state.to(q.dtype))    # (B, T, H, V)
        o = self.v_norm(o)
        o = rearrange(o, 'b t h v -> b t (h v)')                     # (B, T, value_dim)
        o = self.o_proj(o)                                            # (B, T, C)

        gate = torch.sigmoid(self.gate_proj(x))
        return gate * o


# ──────────────────────────────────────────────────────────────────────────────
# Memory Transformer Block
# ──────────────────────────────────────────────────────────────────────────────

class SimpleMemoryTransformerBlock(GradientCheckpointingLayer):
    """
    Transformer block with async-eviction GLA memory.

    Forward pass (memory-enabled layer):

        h_sa   = x + Self-Attention(attn_norm(x))              # local window
        h_mem  = h_sa + Reader(h_sa, prev_write_buffer)        # read evicted tokens
        out    = h_mem + MLP(mlp_norm(h_mem))
        new_wb = Compressor(h_sa, initial=prev_wb.detach())    # write for next chunk

    Two-level memory:
      - SA covers the current window precisely (recent tokens).
      - Reader reads prev_write_buffer, built from the PREVIOUS chunk's tokens,
        which are now outside the SA window.  No overlap, no redundancy.

    Gradient path for the compressor (one-chunk delay):
        loss_{N+1} → reader → prev_write_buffer_N (live) → compressor.{k,v,gk}_proj

    prev_write_buffer is passed as detached initial_state to the compressor,
    truncating gradient chains at chunk boundaries without losing the read-side gradient.

    For layers without memory the Reader/Compressor steps are skipped.
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
            window_size=config.local_window_size,
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
            self.reader = GLAMemoryReader(
                hidden_size=config.hidden_size,
                num_heads=config.gla_num_heads,
                expand_k=config.gla_expand_k,
                expand_v=config.gla_expand_v,
                norm_eps=config.norm_eps,
            )

    def forward(
        self,
        hidden_states:   torch.Tensor,
        attention_mask:  torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        use_cache:       bool | None = False,
        output_attentions: bool | None = False,
        **kwargs,
    ) -> tuple:

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
        h_sa          = residual + hidden_states  # local-context representation
        hidden_states = h_sa

        # ── 2. Memory Read (from PREVIOUS chunk's write_buffer, kept live) ─
        # prev_write_buffer was produced by the last chunk and is intentionally
        # not yet detached.  Reading from it while live creates the gradient path:
        #   loss_{N+1} → reader → prev_write_buffer_N (live) → compressor.{k,v,gk}
        # Signal: "was what the compressor stored useful for this window's queries?"
        if self.enable_memory and isinstance(past_key_values, MemoryCache):
            prev_wb = past_key_values.get_write_buffer(self.layer_idx)
            if prev_wb is not None:
                mem_out       = self.reader(h_sa, prev_wb)
                hidden_states = h_sa + mem_out  # h_mem

        # ── 3. MLP ────────────────────────────────────────────────────────
        residual      = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        out           = residual + hidden_states

        # ── 4. Write current tokens into new write_buffer ─────────────────
        # h_sa is used as compressor input (same distribution as reader's q).
        #
        # prev_wb.detach() as initial_state: lazy truncation of the gradient
        # chain at the write boundary.  The live reference read in step 2 above
        # is unaffected — only the accumulated-state path (through initial_state)
        # is disconnected, preventing cross-chunk gradient chains beyond one step.
        #
        # In eval mode, new_wb is immediately detached to prevent unbounded
        # autograd graph growth across decode steps.
        if self.enable_memory and isinstance(past_key_values, MemoryCache):
            prev_wb = past_key_values.get_write_buffer(self.layer_idx)
            initial = prev_wb.detach() if prev_wb is not None else None
            new_wb  = self.compressor(h_sa, state=initial)
            if not self.training:
                new_wb = new_wb.detach()
            past_key_values.set_write_buffer(self.layer_idx, new_wb)

        return out, attentions, past_key_values


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
        inputs_embeds:   torch.FloatTensor,
        attention_mask:  Optional[torch.Tensor],
        past_key_values: MemoryCache | None,
        return_dict:     bool,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPastAndMemory:
        """
        Chunked forward pass for training and long-context inference.

        Splits the sequence into chunks of `training_chunk_size` tokens.
        A single MemoryCache threads through all chunks carrying:
          • attn_state    — local SA KV cache (sliding window across chunks).
          • write_buffer  — live GLA state from the previous chunk's compressor.

        Per-chunk block forward:
          1. SA on current tokens (local window).
          2. Read from prev_write_buffer (live) → gradient reaches compressor.k/v/gk.
          3. MLP.
          4. Compressor writes current tokens → new_write_buffer (live for next chunk).
             initial_state = prev_write_buffer.detach() (lazy truncation).

        cache.detach_() after each chunk:
          • Detaches attn_state for FSDP2 / truncated BPTT.
          • Does NOT detach write_buffer — kept live for the next chunk's reader.
          • write_buffer is lazily detached when consumed as initial_state (step 4).

        Inference: new_write_buffer is immediately detached in block forward
        (non-training mode) to avoid unbounded autograd graph growth.
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
                # Detach attn_state for FSDP2 / truncated BPTT.
                # write_buffer is intentionally NOT detached here — see MemoryCache.detach_().
                cache.detach_()

        hidden_states = self.norm(torch.cat(chunk_outputs, dim=1))

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

        # ── Chunked forward (training + long-context inference) ───────────────
        if (
            self.config.use_memory
            and self.config.training_chunk_size is not None
            and kwargs.get('cu_seqlens') is None
            and hidden_states.shape[1] > self.config.training_chunk_size
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
