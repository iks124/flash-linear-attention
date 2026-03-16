"""
MemoryTransformer: Transformer with online semantic-compression memory.

Design overview
───────────────
Each block has three pre-norm sublayers:

    residual = x
    h_sa  = residual + Self-Attention(attn_norm(residual))   # local window
    h_mem = h_sa + MemoryCrossAttention(h_sa, memory_tokens) # optional
    out   = h_mem + MLP(mlp_norm(h_mem))
    _update_memory(out)   # append to buffer, semantically segment, compress

The compression pipeline (_update_memory / _try_compress)
──────────────────────────────────────────────────────────
After each forward call, `out.detach()` is appended to a per-layer `raw_buffer`.
Compression is triggered by a two-layer decision:

  Layer 1 – Attention Pressure (when to compress)
    Reuses self.attn.q_proj / k_proj to compute how much each buffer position
    is attended to by the W most recent tokens of the current forward pass.
    The max over the W queries (union signal) averaged over batch and heads gives
    a per-buffer-position pressure score.  Positions whose pressure is below
    `attention_pressure_threshold` are safe to compress.  The leftmost
    high-pressure position defines a `pressure_frontier`: only tokens to its
    left are compression candidates.

  Layer 2 – Semantic Boundary (where to cut, within the pressure frontier)
    SemanticBoundaryDetector scores each inter-token gap in the candidate region.
    The leftmost gap whose score exceeds `boundary_threshold` (which softens
    as the buffer fills) defines the cut point.  If no semantic boundary is
    found within `max_segment_length` tokens a hard cut is forced.
    Buffer overflow acts as a final safety net.

The segment is compressed into K memory slots (Perceiver-style latent
cross-attention) and evicted from the buffer.  This repeats until the buffer
contains no more eligible segments (oldest-first / FIFO).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from fla.layers.attn import Attention
from fla.models.memory_transformer.configuration_memory_transformer import MemoryTransformerConfig
from fla.models.utils import Cache, FLAGenerationMixin
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm
from fla.modules import GatedMLP as MemTransformerMLP
from fla.modules.l2warp import l2_warp

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
    Extends the standard KV Cache with per-layer online compression state.

    Per-layer fields
    ----------------
    raw_buffer     : (B, L_buf, C)  –  detached hidden states not yet compressed.
    memory_tokens  : (B, M, C)      –  compressed memory slots (M ≤ max_memory_slots).
    segment_count  : int            –  monotonically increasing, used for pos embedding.
    """

    def __init__(self, seen_tokens: int = 0, **kwargs: Any) -> None:
        super().__init__(seen_tokens=seen_tokens, **kwargs)
        self._raw_buffer:    dict[int, torch.Tensor | None] = {}
        self._memory_tokens: dict[int, torch.Tensor | None] = {}
        self._segment_count: dict[int, int] = {}

    def _ensure_layer(self, layer_idx: int) -> None:
        if layer_idx not in self._raw_buffer:
            self._raw_buffer[layer_idx]    = None
            self._memory_tokens[layer_idx] = None
            self._segment_count[layer_idx] = 0

    def get_layer_memory(self, layer_idx: int) -> dict[str, Any]:
        self._ensure_layer(layer_idx)
        return {
            'raw_buffer':    self._raw_buffer[layer_idx],
            'memory_tokens': self._memory_tokens[layer_idx],
            'segment_count': self._segment_count[layer_idx],
        }

    def update_layer_memory(
        self,
        layer_idx:     int,
        raw_buffer:    torch.Tensor | None,
        memory_tokens: torch.Tensor | None,
        segment_count: int,
    ) -> None:
        self._ensure_layer(layer_idx)
        self._raw_buffer[layer_idx]    = raw_buffer
        self._memory_tokens[layer_idx] = memory_tokens
        self._segment_count[layer_idx] = segment_count


# ──────────────────────────────────────────────────────────────────────────────
# Semantic Boundary Detector
# ──────────────────────────────────────────────────────────────────────────────

class SemanticBoundaryDetector(nn.Module):
    """
    Produces boundary probabilities for each inter-token gap in a sequence.

    The final score is the sigmoid of:
        gate_logit  (learned MLP on the pair (h_i, h_{i+1}))
      + change_score  (1 – cosine-sim of projected representations)

    The cosine-change component is parameter-free and provides a reliable
    prior (topic shifts show up as sudden representation divergences).  The
    learned gate refines this with content-aware, trainable signal.

    Note on gradients
    -----------------
    At inference / during memory updates the input comes from a detached
    buffer, so the gate MLP will NOT receive gradients through the compression
    path.  The cosine-change component still works without training.
    If an auxiliary boundary-prediction loss is available (e.g. supervised by
    punctuation / sentence boundaries), the gate can be trained separately.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        sem_dim = max(32, hidden_size // 8)
        self.sem_proj = nn.Linear(hidden_size, sem_dim, bias=False)

        # Gate MLP: processes the concatenated pair (h_i ‖ h_{i+1})
        inner = max(32, hidden_size // 4)
        self.pair_norm = nn.RMSNorm(hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, inner, bias=False)
        self.fc2 = nn.Linear(inner, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, L, C)
        Returns:
            boundary_probs: (B, L-1)  probability of boundary AFTER position i
        """
        # ── Cosine-change score (parameter-free) ─────────────────────────
        sem    = F.normalize(self.sem_proj(hidden_states), dim=-1)   # (B, L, sem_dim)
        cos    = (sem[:, :-1] * sem[:, 1:]).sum(dim=-1)              # (B, L-1)
        change = (1.0 - cos) * 0.5                                   # ∈ [0, 1]

        # ── Learned gate ─────────────────────────────────────────────────
        pairs      = torch.cat([hidden_states[:, :-1],
                                 hidden_states[:, 1:]], dim=-1)       # (B, L-1, 2C)
        gate_logit = self.fc2(F.silu(self.fc1(self.pair_norm(pairs)))).squeeze(-1)  # (B, L-1)

        return torch.sigmoid(gate_logit + change)  # (B, L-1)


# ──────────────────────────────────────────────────────────────────────────────
# Segment Compressor
# ──────────────────────────────────────────────────────────────────────────────

class SegmentCompressor(nn.Module):
    """
    Compresses a variable-length segment into K fixed memory slots.

    Implements Perceiver-style cross-attention: K learnable latent queries
    attend over the segment tokens.  An optional `importance` bias (derived
    from boundary scores) steers attention toward semantically salient tokens.
    """

    def __init__(self, hidden_size: int, slots_per_segment: int):
        super().__init__()
        self.hidden_size       = hidden_size
        self.slots_per_segment = slots_per_segment
        # K learnable latent query vectors
        self.latent_queries = nn.Parameter(
            torch.randn(slots_per_segment, hidden_size) * 0.02
        )
        self.norm    = nn.RMSNorm(hidden_size)
        self.q_proj  = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj  = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj  = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        segment:    torch.Tensor,              # (B, Ls, C)
        importance: torch.Tensor | None = None, # (B, Ls) – additive attention bias
    ) -> torch.Tensor:                          # (B, K, C)
        B, Ls, C = segment.shape
        seg_norm = self.norm(segment)

        # Latent queries: (K, C) → broadcast to (B, K, C)
        q = self.q_proj(self.latent_queries).unsqueeze(0).expand(B, -1, -1)
        k = self.k_proj(seg_norm)   # (B, Ls, C)
        v = self.v_proj(seg_norm)   # (B, Ls, C)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(C)  # (B, K, Ls)
        if importance is not None:
            scores = scores + importance.unsqueeze(1)   # broadcast (B, 1, Ls)

        attn  = torch.softmax(scores, dim=-1)           # (B, K, Ls)
        slots = torch.matmul(attn, v)                   # (B, K, C)
        return self.out_proj(slots)


# ──────────────────────────────────────────────────────────────────────────────
# Memory Cross-Attention
# ──────────────────────────────────────────────────────────────────────────────

class MemoryCrossAttention(nn.Module):
    """
    Multi-head cross-attention from current hidden states to memory tokens.

    The output is element-wise gated before being returned, so the model can
    learn to suppress memory retrieval when it is not useful.

    The module applies its own pre-norm (RMSNorm) on the query input, so the
    block can pass post-residual representations directly.
    """

    def __init__(self, hidden_size: int, num_heads: int, norm_eps: float = 1e-6):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f'hidden_size ({hidden_size}) must be divisible by '
                f'memory_num_heads ({num_heads}).'
            )
        self.num_heads = num_heads
        self.head_dim  = hidden_size // num_heads
        self.scale     = self.head_dim ** -0.5

        self.norm     = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.q_proj   = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj   = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj   = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj   = nn.Linear(hidden_size, hidden_size, bias=False)
        # Element-wise gate: each dimension can independently gate memory access
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        hidden_states:  torch.Tensor,              # (B, L, C) – queries (post-SA)
        memory_tokens:  torch.Tensor,              # (B, M, C) – keys / values
        memory_mask:    torch.Tensor | None = None, # (B, M) bool – True = valid slot
    ) -> torch.Tensor:                              # (B, L, C) gated delta
        B, L, C = hidden_states.shape
        M = memory_tokens.shape[1]

        x = self.norm(hidden_states)   # pre-norm on queries

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory_tokens).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory_tokens).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, H, L, M)

        if memory_mask is not None:
            # Mask invalid / padding slots  →  shape (B, 1, 1, M)
            attn = attn.masked_fill(
                ~memory_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )

        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # guard against all-inf rows

        out = torch.matmul(attn, v)                         # (B, H, L, head_dim)
        out = out.transpose(1, 2).reshape(B, L, C)
        out = self.o_proj(out)

        # Element-wise gate (uses pre-normed query for consistency)
        gate = torch.sigmoid(self.gate_proj(x))
        return gate * out


# ──────────────────────────────────────────────────────────────────────────────
# Memory Transformer Block
# ──────────────────────────────────────────────────────────────────────────────

class MemoryTransformerBlock(GradientCheckpointingLayer):
    """
    Transformer block with optional online semantic-compression memory.

    Forward pass (memory-enabled layer):

        residual = x
        h_sa  = residual + Self-Attention(attn_norm(residual))
        h_mem = h_sa    + MemoryCrossAttention(h_sa, memory_tokens)
        out   = h_mem   + MLP(mlp_norm(h_mem))
        _update_memory(out)

    For layers without memory the Memory Cross-Attention step is skipped.
    """

    def __init__(self, config: MemoryTransformerConfig, layer_idx: int):
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
            self.boundary_detector = SemanticBoundaryDetector(config.hidden_size)
            self.compressor = SegmentCompressor(
                config.hidden_size, config.memory_slots_per_segment
            )
            self.memory_attn = MemoryCrossAttention(
                config.hidden_size, config.memory_num_heads, norm_eps=config.norm_eps
            )
            # Segment position embedding: maps segment_count % max_segments → (C,)
            self.segment_pos_embed = nn.Embedding(
                config.max_segments, config.hidden_size
            )

    # ── Buffer utilities ──────────────────────────────────────────────────────

    @staticmethod
    def _cat(old: torch.Tensor | None, new: torch.Tensor) -> torch.Tensor:
        """Concatenate along the sequence dimension."""
        return new if old is None else torch.cat([old, new], dim=1)

    def _compute_attention_pressure(
        self,
        hidden_states: torch.Tensor,  # (B, L_cur, C)  current forward's output
        buffer:        torch.Tensor,  # (B, L_buf, C)
    ) -> torch.Tensor:                # (L_buf,)  per-position pressure ∈ [0, 1]
        """
        Estimate how much each buffer position is referenced by the most
        recent W tokens of the current context.

        Reuses self.attn.q_proj / self.attn.k_proj; no extra parameters needed.
        Handles grouped-query attention (num_kv_heads ≠ num_heads) transparently.

        Returns a (L_buf,) tensor of attention probabilities averaged over
        batch and heads, max-pooled over the W query positions.
        A HIGH value means the buffer position is still actively attended to.
        """
        cfg = self.config
        B, L_buf, _ = buffer.shape
        W = min(cfg.attention_pressure_window, hidden_states.shape[1])

        H  = self.attn.num_heads
        Kh = self.attn.num_kv_heads
        Dh = self.attn.head_dim

        # Project the W most recent tokens as queries, all buffer tokens as keys
        q = self.attn.q_proj(hidden_states[:, -W:])  # (B, W,   H*Dh)
        k = self.attn.k_proj(buffer)                  # (B, L_buf, Kh*Dh)

        q = q.view(B, W,     H,  Dh).permute(0, 2, 1, 3)   # (B, H,  W,   Dh)
        k = k.view(B, L_buf, Kh, Dh).permute(0, 2, 1, 3)   # (B, Kh, L_buf, Dh)

        # Expand key heads for grouped-query attention
        if Kh != H:
            k = k.repeat_interleave(H // Kh, dim=1)         # (B, H, L_buf, Dh)

        scores   = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(Dh)  # (B, H, W, L_buf)
        pressure = torch.softmax(scores, dim=-1)                          # (B, H, W, L_buf)

        # Union across W queries (if ANY recent token attends here, keep it)
        # then average over batch and heads for a shared scalar per buffer position
        return pressure.max(dim=2).values.mean(dim=(0, 1))  # (L_buf,)

    def _find_compress_boundary(
        self,
        buffer:        torch.Tensor,  # (B, L_buf, C)
        hidden_states: torch.Tensor,  # (B, L_cur, C)  current context
    ) -> int | None:
        """
        Two-layer compression decision.

        Layer 1 – Attention Pressure:  establishes `pressure_frontier`, the
          leftmost high-pressure buffer position.  Only tokens before this
          index are eligible for compression.

        Layer 2 – Semantic Boundary:  within the eligible region, find the
          leftmost semantic boundary (softened threshold as buffer fills).
          Falls back to a hard cut at max_segment_length if needed.

        Returns the `compress_to` index (exclusive) or None.
        """
        B, L_buf, _ = buffer.shape
        cfg = self.config

        if L_buf < cfg.min_segment_length:
            return None

        # ── Layer 1: Attention pressure frontier ──────────────────────────
        with torch.no_grad():
            pressure = self._compute_attention_pressure(hidden_states, buffer)

        # pressure_frontier = leftmost position with HIGH pressure
        # = how many tokens from the left are safe to compress
        high_mask = pressure > cfg.attention_pressure_threshold
        if high_mask.any().item():
            pressure_frontier = int(high_mask.float().argmax().item())
        else:
            pressure_frontier = L_buf

        # If the frontier is too small to form even one minimal segment, bail out
        # early – the current context still "needs" those early tokens.
        if pressure_frontier < cfg.min_segment_length:
            # But if the buffer is critically overflowing, compress anyway as
            # a last resort (prevent unbounded buffer growth).
            if L_buf < cfg.raw_buffer_max_size:
                return None
            # Emergency override: ignore pressure and use hard cut
            pressure_frontier = L_buf

        # ── Layer 2: Semantic boundary within the eligible region ─────────
        # Soften the boundary threshold proportionally to how full the buffer is.
        buffer_fullness     = L_buf / cfg.raw_buffer_max_size          # [0, ∞)
        effective_threshold = cfg.boundary_threshold * max(0.0, 1.0 - 0.6 * buffer_fullness)
        effective_threshold = max(effective_threshold, 0.05)           # floor

        with torch.no_grad():
            bp_mean = self.boundary_detector(buffer).mean(dim=0)       # (L_buf-1,)

        lo = cfg.min_segment_length - 1
        hi = min(pressure_frontier - 1, cfg.max_segment_length - 1, L_buf - 2)

        if lo <= hi:
            region  = bp_mean[lo:hi + 1]
            boundary_mask = region > effective_threshold
            if boundary_mask.any().item():
                return lo + int(boundary_mask.float().argmax().item()) + 1  # leftmost semantic boundary

        # No semantic boundary in the eligible window – hard cut at the frontier
        # (capped by max_segment_length so we don't compress too much at once)
        if pressure_frontier >= cfg.min_segment_length:
            return min(pressure_frontier, cfg.max_segment_length)

        # Safety net: buffer overflow
        if L_buf >= cfg.raw_buffer_max_size:
            return min(cfg.max_segment_length, L_buf // 2)

        return None

    def _compress_one_segment(
        self,
        buffer:        torch.Tensor,
        memory_tokens: torch.Tensor | None,
        segment_count: int,
        compress_to:   int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Extract and compress `buffer[:, :compress_to]` into memory slots.
        Returns (remaining_buffer, updated_memory_tokens, updated_segment_count).
        """
        cfg = self.config
        B   = buffer.shape[0]

        segment   = buffer[:, :compress_to]    # (B, compress_to, C)
        remaining = buffer[:, compress_to:]    # (B, L_buf - compress_to, C)

        # ── Importance weights from boundary scores ───────────────────────
        # boundary_probs[i] = prob of boundary AFTER token i
        # Tokens closer to a boundary are semantically more "terminal" →
        # higher importance for the compressor's attention.
        if compress_to > 1:
            with torch.no_grad():
                bp = self.boundary_detector(segment)  # (B, compress_to-1)
            # Last token always gets full importance (it IS the boundary token)
            importance = torch.cat(
                [bp, bp.new_ones(B, 1)], dim=1
            )  # (B, compress_to)
        else:
            importance = None

        slots = self.compressor(segment, importance)  # (B, K, C)

        # ── Segment positional embedding ─────────────────────────────────
        # Encodes the temporal order of compressed segments in the memory bank.
        seg_idx = torch.tensor(
            segment_count % cfg.max_segments,
            device=buffer.device, dtype=torch.long,
        )
        pos_emb = self.segment_pos_embed(seg_idx)       # (C,)
        slots   = slots + pos_emb.unsqueeze(0).unsqueeze(0)  # broadcast (B, K)

        # ── Append to memory bank, evict oldest if over capacity ──────────
        memory_tokens = self._cat(memory_tokens, slots)  # (B, M+K, C)
        if memory_tokens.shape[1] > cfg.max_memory_slots:
            memory_tokens = memory_tokens[:, -cfg.max_memory_slots:]

        return remaining, memory_tokens, segment_count + 1

    def _try_compress(
        self,
        buffer:        torch.Tensor,
        memory_tokens: torch.Tensor | None,
        segment_count: int,
        hidden_states: torch.Tensor,   # current context; unchanged across recursion
    ) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        """
        Greedy left-to-right compression loop.
        Compresses the oldest segment first; recurses until the buffer contains
        no more complete segments.
        `hidden_states` is passed unchanged so every boundary decision uses
        the same current-context pressure signal.
        """
        compress_to = self._find_compress_boundary(buffer, hidden_states)
        if compress_to is None:
            return buffer, memory_tokens, segment_count

        buffer, memory_tokens, segment_count = self._compress_one_segment(
            buffer, memory_tokens, segment_count, compress_to
        )
        return self._try_compress(buffer, memory_tokens, segment_count, hidden_states)

    def _update_memory(
        self,
        hidden_states: torch.Tensor,  # post-MLP output of the current forward
        memory_cache:  MemoryCache,
    ) -> MemoryCache:
        """
        Append hidden_states to the raw buffer and compress any complete segments.

        hidden_states is used in two ways:
          • As a pressure probe (non-detached, wrapped in torch.no_grad inside
            _compute_attention_pressure) to decide compression boundaries.
          • As detached states appended to the raw buffer.
        """
        state         = memory_cache.get_layer_memory(self.layer_idx)
        buffer        = self._cat(state['raw_buffer'], hidden_states.detach())
        memory_tokens = state['memory_tokens']
        segment_count = state['segment_count']

        buffer, memory_tokens, segment_count = self._try_compress(
            buffer, memory_tokens, segment_count, hidden_states
        )

        memory_cache.update_layer_memory(
            self.layer_idx, buffer, memory_tokens, segment_count
        )
        return memory_cache

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states:  torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        use_cache:       bool | None = False,
        output_attentions: bool | None = False,
        memory_cache:    MemoryCache | None = None,
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
        hidden_states = residual + hidden_states  # h_sa

        # ── 2. Memory Cross-Attention (conditional on available memory) ───
        if self.enable_memory and memory_cache is not None:
            state = memory_cache.get_layer_memory(self.layer_idx)
            mem_tokens = state['memory_tokens']
            if mem_tokens is not None and mem_tokens.shape[1] > 0:
                B, M, _ = mem_tokens.shape
                # All current slots are valid (no padding in the bank)
                mem_mask = mem_tokens.new_ones(B, M, dtype=torch.bool)
                mem_delta = self.memory_attn(hidden_states, mem_tokens, mem_mask)
                hidden_states = hidden_states + mem_delta  # h_mem

        # ── 3. MLP ────────────────────────────────────────────────────────
        residual      = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states  # out

        # ── 4. Update memory buffer (after MLP for richer representations) ─
        if self.enable_memory and memory_cache is not None:
            memory_cache = self._update_memory(hidden_states, memory_cache)

        return hidden_states, attentions, past_key_values, memory_cache


# ──────────────────────────────────────────────────────────────────────────────
# PreTrainedModel base
# ──────────────────────────────────────────────────────────────────────────────

class MemoryTransformerPreTrainedModel(PreTrainedModel):
    config_class              = MemoryTransformerConfig
    base_model_prefix         = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules         = ['MemoryTransformerBlock']
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
# MemoryTransformerModel
# ──────────────────────────────────────────────────────────────────────────────

class MemoryTransformerModel(MemoryTransformerPreTrainedModel):

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size  = config.vocab_size

        self.embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList([
            MemoryTransformerBlock(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids:       torch.LongTensor | None = None,
        attention_mask:  Optional[torch.Tensor]  = None,
        inputs_embeds:   torch.FloatTensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        memory_cache:    MemoryCache | None = None,
        use_cache:       bool | None = None,
        output_attentions:    bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict:          bool | None = None,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPastAndMemory:

        if output_attentions:
            warnings.warn(
                '`MemoryTransformerModel` does not support `output_attentions`; '
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

        # ── KV cache initialisation ───────────────────────────────────────
        if use_cache:
            if past_key_values is not None and not isinstance(past_key_values, Cache):
                past_key_values = Cache.from_legacy_cache(past_key_values)
            elif past_key_values is None:
                past_key_values = Cache()

        # ── Memory cache initialisation ───────────────────────────────────
        if self.config.use_memory and memory_cache is None:
            memory_cache = MemoryCache()

        all_hidden_states = () if output_hidden_states else None
        all_attns         = () if output_attentions    else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states, attentions, past_key_values, memory_cache = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                memory_cache=memory_cache,
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
                    all_hidden_states, all_attns, memory_cache,
                ] if v is not None
            )

        return BaseModelOutputWithPastAndMemory(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attns,
            memory_state=memory_cache,
        )


# ──────────────────────────────────────────────────────────────────────────────
# MemoryTransformerForCausalLM
# ──────────────────────────────────────────────────────────────────────────────

class MemoryTransformerForCausalLM(MemoryTransformerPreTrainedModel, FLAGenerationMixin):
    _tied_weights_keys = ['lm_head.weight']

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__(config)
        self.model      = MemoryTransformerModel(config)
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
        memory_cache:    MemoryCache | None = None,
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
            memory_cache=memory_cache,
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
