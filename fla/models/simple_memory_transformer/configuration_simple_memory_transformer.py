from __future__ import annotations

import warnings

from transformers.configuration_utils import PretrainedConfig


class SimpleMemoryTransformerConfig(PretrainedConfig):
    """
    Configuration for SimpleMemoryTransformer: a Transformer augmented with
    GLA-based online compression memory.

    Architecture overview
    ---------------------
    Each SimpleMemoryTransformerBlock contains three pre-norm sublayers:

      1. **Local Self-Attention** (window_size = local_window_size)
         Only attends to the most recent `local_window_size` tokens,
         keeping the KV cache bounded.

      2. **GLA Memory Read** (memory layers only)
         Reads from the accumulated GLA recurrent state (B, H, K, V) via a
         linear attention query: o = Q @ S.  The GLA state stores compressed
         representations of all context older than the current window.

      3. **MLP**

    After each full block forward pass the hidden states are appended to a
    per-layer `raw_buffer`.  When the buffer exceeds `raw_buffer_max_size`,
    the oldest overflow tokens are compressed into the GLA state via
    `GLAMemoryCompressor` (fused_recurrent_gla), which updates the state
    with natural gating/forgetting.

    Parameters
    ----------
    Standard Transformer params:
        hidden_size, num_hidden_layers, num_heads, num_kv_heads, qkv_bias,
        qk_norm, rope_theta, max_position_embeddings, hidden_ratio,
        intermediate_size, hidden_act, initializer_range, norm_eps, use_cache,
        fuse_norm, fuse_swiglu, fuse_cross_entropy, fuse_linear_cross_entropy,
        use_l2warp, vocab_size.

    Memory-specific params:
        use_memory (bool): Enable the memory compression mechanism.
        memory_layers (list[int] | None): Layer indices that receive memory
            augmentation. None means all layers.
        raw_buffer_max_size (int): Buffer overflow threshold. When the raw buffer
            exceeds this many tokens, the oldest overflow is compressed into the
            GLA state. Also determines the self-attention window size.
        gla_num_heads (int | None): Number of heads for GLAMemoryCompressor and
            GLAMemoryReader. Defaults to num_heads.
        gla_expand_k (float): Key expansion ratio for GLA compressor/reader.
        gla_expand_v (float): Value expansion ratio for GLA compressor/reader.
        gla_gate_low_rank_dim (int): Low-rank bottleneck for the GLA gate projection.
        gla_gate_logit_normalizer (int): Divisor applied after logsigmoid on gate logits.
    """

    model_type = 'simple_memory_transformer'
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
        raw_buffer_max_size: int = 512,
        # ── GLA compressor / reader dims ──────────────────────────────────
        gla_num_heads: int | None = None,
        gla_expand_k: float = 0.5,
        gla_expand_v: float = 1.0,
        gla_gate_low_rank_dim: int = 16,
        gla_gate_logit_normalizer: int = 16,
        # Kernel mode for GLAMemoryCompressor.
        # 'auto' (default): fused_recurrent for T<=64 (decode), chunk for T>64 (train/prefill).
        # 'chunk': always use chunk_gla (chunk-parallel, gradient-compatible).
        # 'fused_recurrent': always use fused_recurrent_gla (serial, minimal overhead).
        # Note: fused_chunk_gla is deprecated upstream and must not be used.
        gla_mode: str = 'auto',
        # ── Training ──────────────────────────────────────────────────────
        training_chunk_size: int | None = None,
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
        self.raw_buffer_max_size = raw_buffer_max_size
        # local_window_size is derived from raw_buffer_max_size to guarantee that
        # every uncompressed token (in raw_buffer) is always within the attention
        # window, eliminating any limbo zone between KV cache and GLA memory.
        self.local_window_size = raw_buffer_max_size

        # GLA compressor / reader
        self.gla_num_heads            = gla_num_heads or num_heads
        self.gla_expand_k             = gla_expand_k
        self.gla_expand_v             = gla_expand_v
        self.gla_gate_low_rank_dim    = gla_gate_low_rank_dim
        self.gla_gate_logit_normalizer = gla_gate_logit_normalizer
        self.gla_mode                 = gla_mode

        # Training
        self.training_chunk_size = training_chunk_size

        # Default memory layers: every layer
        if memory_layers is None:
            memory_layers = list(range(num_hidden_layers))
        self.memory_layers = sorted(set(memory_layers))

        # ── Validation ────────────────────────────────────────────────────
        if gla_mode not in ('auto', 'chunk', 'fused_recurrent'):
            raise ValueError(
                f'`gla_mode` must be one of "auto", "chunk", "fused_recurrent", '
                f'got "{gla_mode}". Note: "fused_chunk" is deprecated upstream.'
            )
        if training_chunk_size is not None and training_chunk_size <= 0:
            raise ValueError(
                f'`training_chunk_size` must be positive, got {training_chunk_size}.'
            )
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

        gla_key_dim = int(hidden_size * gla_expand_k)
        gla_val_dim = int(hidden_size * gla_expand_v)
        _gla_heads  = gla_num_heads or num_heads
        if gla_key_dim % _gla_heads != 0:
            raise ValueError(
                f'gla_expand_k ({gla_expand_k}) × hidden_size ({hidden_size}) = {gla_key_dim} '
                f'must be divisible by gla_num_heads ({_gla_heads}).'
            )
        if gla_val_dim % _gla_heads != 0:
            raise ValueError(
                f'gla_expand_v ({gla_expand_v}) × hidden_size ({hidden_size}) = {gla_val_dim} '
                f'must be divisible by gla_num_heads ({_gla_heads}).'
            )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
