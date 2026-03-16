from __future__ import annotations

import warnings

from transformers.configuration_utils import PretrainedConfig


class MemoryTransformerConfig(PretrainedConfig):
    """
    Configuration for MemoryTransformer: a standard Transformer augmented with
    an online semantic-compression memory.

    Architecture overview
    ---------------------
    Each MemoryTransformerBlock contains three pre-norm sublayers:

      1. **Local Self-Attention** (window_size = local_window_size)
         Only attends to the most recent `local_window_size` tokens,
         keeping the KV cache bounded.

      2. **Memory Cross-Attention** (memory layers only)
         Attends to compressed `memory_tokens` – the long-range history
         folded into a fixed-size bank of slots.

      3. **MLP**

    After each full block forward pass the hidden states are appended to a
    per-layer `raw_buffer`.  The buffer is scanned left-to-right for semantic
    segment boundaries.  Every complete segment is compressed (Perceiver-style
    cross-attention from K learnable latent queries) into `memory_slots_per_segment`
    memory slots and evicted from the buffer.  If no semantic boundary is found
    within `max_segment_length` tokens a hard boundary is forced.

    Parameters
    ----------
    Standard Transformer params (identical to TransformerConfig):
        hidden_size, num_hidden_layers, num_heads, num_kv_heads, qkv_bias,
        qk_norm, rope_theta, max_position_embeddings, hidden_ratio,
        intermediate_size, hidden_act, initializer_range, norm_eps, use_cache,
        fuse_norm, fuse_swiglu, fuse_cross_entropy, fuse_linear_cross_entropy,
        use_l2warp, vocab_size.

    Memory-specific params:
        use_memory (bool): Enable the memory compression mechanism.
        memory_layers (list[int] | None): Layer indices that receive memory
            augmentation. None means all layers.
        local_window_size (int): Sliding-window size for the main self-attention
            (KV cache is bounded to this many tokens). Set to None for full attention.
        raw_buffer_max_size (int): Maximum number of tokens allowed to accumulate
            in the raw buffer before a forced compression is triggered regardless
            of semantic boundaries.
        boundary_threshold (float): Sigmoid-probability threshold above which a
            position is classified as a segment boundary.
        min_segment_length (int): Minimum number of tokens a segment must contain
            before it is eligible for compression (avoids micro-segments).
        max_segment_length (int): Maximum segment length; a hard boundary is forced
            when no semantic boundary has been found within this window.
        memory_slots_per_segment (int): Number of compressed memory slots produced
            per segment (K in the Perceiver cross-attention).
        max_memory_slots (int): Maximum number of compressed slots retained in the
            memory bank; oldest slots are evicted when this limit is reached.
        max_segments (int): Size of the learnable segment-position embedding table.
            Segment indices are taken modulo this value.
        memory_num_heads (int | None): Number of attention heads used in the memory
            cross-attention layer. Defaults to num_heads.
    """

    model_type = 'memory_transformer'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        # ── Standard Transformer ──────────────────────────────────────────
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 2048,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = 'swish',
        initializer_range: float = 0.02,
        elementwise_affine: bool | None = True,
        norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = False,
        vocab_size: int = 32000,
        # ── Memory mechanism ──────────────────────────────────────────────
        use_memory: bool = True,
        memory_layers: list[int] | None = None,
        local_window_size: int | None = 512,
        raw_buffer_max_size: int = 256,
        boundary_threshold: float = 0.5,
        min_segment_length: int = 8,
        max_segment_length: int = 64,
        memory_slots_per_segment: int = 4,
        max_memory_slots: int = 512,
        max_segments: int = 256,
        memory_num_heads: int | None = None,
        # ── Attention-pressure compression trigger ────────────────────────
        attention_pressure_window: int = 8,
        attention_pressure_threshold: float | None = None,
        **kwargs,
    ):
        # ── Standard Transformer fields ───────────────────────────────────
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size

        # ── Memory fields ─────────────────────────────────────────────────
        self.use_memory = use_memory
        self.local_window_size = local_window_size
        self.raw_buffer_max_size = raw_buffer_max_size
        self.boundary_threshold = boundary_threshold
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.memory_slots_per_segment = memory_slots_per_segment
        self.max_memory_slots = max_memory_slots
        self.max_segments = max_segments
        self.memory_num_heads = memory_num_heads or num_heads

        # Attention-pressure params
        self.attention_pressure_window = attention_pressure_window
        # If not set, default to 1 / (2 * local_window_size) so that only buffer
        # positions receiving less than half of "uniform" attention are considered
        # low-pressure.  Falls back to 1/512 when local_window_size is None.
        if attention_pressure_threshold is None:
            ref = local_window_size if local_window_size is not None else 512
            attention_pressure_threshold = 1.0 / (2.0 * ref)
        self.attention_pressure_threshold = attention_pressure_threshold

        # Default memory layers: every layer
        if memory_layers is None:
            memory_layers = list(range(num_hidden_layers))
        self.memory_layers = sorted(set(memory_layers))

        # ── Validation ────────────────────────────────────────────────────
        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError(
                '`fuse_cross_entropy` and `fuse_linear_cross_entropy` '
                'cannot both be True.',
            )
        if fuse_linear_cross_entropy:
            warnings.warn(
                '`fuse_linear_cross_entropy` is enabled; this may reduce precision. '
                'If loss diverges, consider disabling it.',
            )
        if min_segment_length >= max_segment_length:
            raise ValueError(
                f'`min_segment_length` ({min_segment_length}) must be smaller '
                f'than `max_segment_length` ({max_segment_length}).',
            )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
