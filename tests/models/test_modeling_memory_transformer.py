"""Smoke tests for MemoryTransformer.

Test tiers
──────────
CPU-0   Pure-PyTorch component tests – no flash_attn, no CUDA required.
CPU-1   Memory pipeline tests using _MemoryBlockCPU (stub attention) – CPU only.
GPU     Full model forward / save-load tests – requires CUDA + flash_attn.

Run CPU-only subset:
    pytest tests/models/test_modeling_memory_transformer.py -v -m "not gpu"
Run everything:
    pytest tests/models/test_modeling_memory_transformer.py -v
"""

from __future__ import annotations

import math
import tempfile

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.models.memory_transformer.configuration_memory_transformer import MemoryTransformerConfig
from fla.models.memory_transformer.modeling_memory_transformer import (
    MemoryCache,
    MemoryCrossAttention,
    MemoryTransformerBlock,
    MemoryTransformerForCausalLM,
    SemanticBoundaryDetector,
    SegmentCompressor,
)


# ── pytest marks ──────────────────────────────────────────────────────────────

def _has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


GPU_MARK = pytest.mark.skipif(
    not (torch.cuda.is_available() and _has_flash_attn()),
    reason='requires CUDA + flash_attn',
)
gpu = pytest.mark.gpu   # for -m "not gpu" filtering

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Config helper ─────────────────────────────────────────────────────────────

def _tiny_config(**overrides) -> MemoryTransformerConfig:
    base = dict(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_heads=4,
        num_kv_heads=4,
        hidden_ratio=2,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
        use_memory=True,
        memory_layers=[0],
        local_window_size=16,
        raw_buffer_max_size=32,
        min_segment_length=4,
        max_segment_length=16,
        memory_slots_per_segment=2,
        max_memory_slots=32,
        max_segments=16,
        attention_pressure_window=4,
    )
    base.update(overrides)
    return MemoryTransformerConfig(**base)


# ── CPU-testable block (stub attention, no flash_attn dependency) ─────────────

class _MemoryBlockCPU(nn.Module):
    """
    MemoryTransformerBlock with stub attention for CPU-only unit tests.

    fla.layers.attn.Attention requires flash_attn; this stub exposes the same
    q_proj / k_proj / num_heads / num_kv_heads / head_dim interface used by
    _compute_attention_pressure.  All memory pipeline methods are borrowed
    unchanged from MemoryTransformerBlock via unbound-method assignment.
    """

    def __init__(self, config: MemoryTransformerConfig):
        super().__init__()
        self.config        = config
        self.layer_idx     = 0
        self.enable_memory = True

        H  = config.num_heads
        Kh = config.num_kv_heads or H
        C  = config.hidden_size
        Dh = C // H

        class _AttnStub(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads    = H
                self.num_kv_heads = Kh
                self.head_dim     = Dh
                self.q_proj = nn.Linear(C, H  * Dh, bias=False)
                self.k_proj = nn.Linear(C, Kh * Dh, bias=False)

        self.attn              = _AttnStub()
        self.boundary_detector = SemanticBoundaryDetector(config.hidden_size)
        self.compressor        = SegmentCompressor(
            config.hidden_size, config.memory_slots_per_segment
        )
        self.memory_attn       = MemoryCrossAttention(
            config.hidden_size, config.memory_num_heads, norm_eps=config.norm_eps
        )
        self.segment_pos_embed = nn.Embedding(config.max_segments, config.hidden_size)

    # Borrow all memory pipeline methods from MemoryTransformerBlock.
    # _cat is a @staticmethod → re-wrap so it stays static in this class.
    _cat                        = staticmethod(MemoryTransformerBlock._cat)
    _compute_attention_pressure = MemoryTransformerBlock._compute_attention_pressure
    _find_compress_boundary     = MemoryTransformerBlock._find_compress_boundary
    _compress_one_segment       = MemoryTransformerBlock._compress_one_segment
    _try_compress               = MemoryTransformerBlock._try_compress
    _update_memory              = MemoryTransformerBlock._update_memory


# ═════════════════════════════════════════════════════════════════════════════
# TIER CPU-0 – Component unit tests
# ═════════════════════════════════════════════════════════════════════════════

class TestMemoryCache:

    def test_initialization_has_no_state(self):
        cache = MemoryCache()
        state = cache.get_layer_memory(0)
        assert state['raw_buffer']    is None
        assert state['memory_tokens'] is None
        assert state['segment_count'] == 0

    def test_update_get_roundtrip(self):
        cache = MemoryCache()
        buf   = torch.randn(2, 5, 32)
        mem   = torch.randn(2, 4, 32)
        cache.update_layer_memory(0, buf, mem, segment_count=3)
        state = cache.get_layer_memory(0)
        assert torch.equal(state['raw_buffer'],    buf)
        assert torch.equal(state['memory_tokens'], mem)
        assert state['segment_count'] == 3

    def test_layers_are_independent(self):
        cache = MemoryCache()
        buf0 = torch.randn(1, 3, 8)
        buf1 = torch.randn(1, 7, 8)
        cache.update_layer_memory(0, buf0, None, 0)
        cache.update_layer_memory(1, buf1, None, 2)
        s0, s1 = cache.get_layer_memory(0), cache.get_layer_memory(1)
        assert torch.equal(s0['raw_buffer'], buf0)
        assert torch.equal(s1['raw_buffer'], buf1)
        assert s0['segment_count'] == 0
        assert s1['segment_count'] == 2


class TestSemanticBoundaryDetector:

    def test_output_shape(self):
        B, L, C = 3, 10, 64
        det = SemanticBoundaryDetector(C).eval()
        x   = torch.randn(B, L, C)
        with torch.no_grad():
            out = det(x)
        assert out.shape == (B, L - 1), f"Expected ({B}, {L-1}), got {out.shape}"

    def test_output_in_zero_one(self):
        det = SemanticBoundaryDetector(64).eval()
        x   = torch.randn(2, 8, 64)
        with torch.no_grad():
            out = det(x)
        assert out.min() >= 0.0 and out.max() <= 1.0, (
            f"Boundary probs must be in [0,1]; got min={out.min():.4f} max={out.max():.4f}"
        )

    def test_cosine_change_fires_at_topic_transition(self):
        """
        With the learned gate zeroed out, the cosine-change component alone
        should give a higher score at a clear topic boundary than within topics.
        """
        torch.manual_seed(42)
        C, L_topic = 64, 6
        det = SemanticBoundaryDetector(C).eval()
        # Zero out learned gate → score = sigmoid(change_score)
        with torch.no_grad():
            det.fc1.weight.zero_()
            det.fc2.weight.zero_()

        # Two orthogonal topic centres
        v_a = F.normalize(torch.randn(C), dim=0)
        v_b = F.normalize(torch.randn(C), dim=0)
        v_b = F.normalize(v_b - (v_b @ v_a) * v_a, dim=0)

        noise = 0.05
        states = torch.zeros(1, L_topic * 2, C)
        states[0, :L_topic]  = v_a + noise * torch.randn(L_topic, C)
        states[0, L_topic:]  = v_b + noise * torch.randn(L_topic, C)

        with torch.no_grad():
            probs = det(states)[0]   # (2*L_topic - 1,)

        boundary_score  = probs[L_topic - 1].item()       # gap at the boundary
        within_a_mean   = probs[:L_topic - 1].mean().item()
        within_b_mean   = probs[L_topic:].mean().item()

        assert boundary_score > within_a_mean, (
            f"Boundary {boundary_score:.3f} should exceed within-A {within_a_mean:.3f}"
        )
        assert boundary_score > within_b_mean, (
            f"Boundary {boundary_score:.3f} should exceed within-B {within_b_mean:.3f}"
        )


class TestSegmentCompressor:

    def test_output_shape(self):
        B, Ls, C, K = 2, 12, 64, 3
        comp = SegmentCompressor(C, K).eval()
        seg  = torch.randn(B, Ls, C)
        with torch.no_grad():
            out = comp(seg)
        assert out.shape == (B, K, C), f"Expected ({B}, {K}, {C}), got {out.shape}"

    def test_importance_changes_output(self):
        torch.manual_seed(5)
        B, Ls, C, K = 1, 8, 32, 2
        comp = SegmentCompressor(C, K).eval()
        seg  = torch.randn(B, Ls, C)
        imp  = torch.rand(B, Ls)
        with torch.no_grad():
            out_plain = comp(seg)
            out_imp   = comp(seg, importance=imp)
        assert not torch.allclose(out_plain, out_imp), (
            "Importance bias should change compressor output"
        )

    def test_single_token_segment(self):
        comp = SegmentCompressor(32, 2).eval()
        seg  = torch.randn(1, 1, 32)
        with torch.no_grad():
            out = comp(seg)
        assert out.shape == (1, 2, 32)


class TestMemoryCrossAttention:

    def test_output_shape(self):
        B, L, M, C, H = 2, 6, 8, 64, 4
        mca = MemoryCrossAttention(C, H).eval()
        q   = torch.randn(B, L, C)
        mem = torch.randn(B, M, C)
        with torch.no_grad():
            out = mca(q, mem)
        assert out.shape == (B, L, C), f"Expected ({B}, {L}, {C}), got {out.shape}"

    def test_mask_changes_output(self):
        torch.manual_seed(9)
        B, L, M, C, H = 1, 4, 6, 32, 4
        mca  = MemoryCrossAttention(C, H).eval()
        q    = torch.randn(B, L, C)
        mem  = torch.randn(B, M, C)
        mask = torch.ones(B, M, dtype=torch.bool)
        mask[0, M // 2:] = False   # mask out second half
        full_mask = torch.ones(B, M, dtype=torch.bool)
        with torch.no_grad():
            out_partial = mca(q, mem, memory_mask=mask)
            out_full    = mca(q, mem, memory_mask=full_mask)
        assert not torch.allclose(out_partial, out_full), (
            "Masking memory slots should change the output"
        )

    def test_all_masked_produces_no_nan(self):
        """All-masked softmax → nan_to_num guard must prevent NaN output."""
        B, L, M, C, H = 1, 3, 4, 32, 4
        mca  = MemoryCrossAttention(C, H).eval()
        q    = torch.randn(B, L, C)
        mem  = torch.randn(B, M, C)
        mask = torch.zeros(B, M, dtype=torch.bool)   # all masked
        with torch.no_grad():
            out = mca(q, mem, memory_mask=mask)
        assert not out.isnan().any(), "NaN in output with all-masked memory"
        assert not out.isinf().any(), "Inf in output with all-masked memory"


# ═════════════════════════════════════════════════════════════════════════════
# TIER CPU-1 – Memory pipeline tests (stub attention)
# ═════════════════════════════════════════════════════════════════════════════

class TestAttentionPressure:

    def test_output_shape_and_range(self):
        torch.manual_seed(0)
        cfg   = _tiny_config()
        block = _MemoryBlockCPU(cfg).eval()
        B, L_cur, L_buf, C = 2, 6, 10, cfg.hidden_size
        hidden = torch.randn(B, L_cur, C)
        buffer = torch.randn(B, L_buf, C)
        with torch.no_grad():
            pressure = block._compute_attention_pressure(hidden, buffer)
        assert pressure.shape == (L_buf,), (
            f"Expected (L_buf={L_buf},), got {pressure.shape}"
        )
        assert pressure.min() >= 0.0, "Pressure must be non-negative"
        assert pressure.max() <= 1.0, "Pressure (softmax max) must be ≤ 1"

    def test_gqa_expand_is_handled(self):
        """num_kv_heads < num_heads (GQA) should not crash."""
        cfg   = _tiny_config(num_heads=4, num_kv_heads=2)
        block = _MemoryBlockCPU(cfg).eval()
        B, L_cur, L_buf, C = 1, 4, 8, cfg.hidden_size
        with torch.no_grad():
            p = block._compute_attention_pressure(
                torch.randn(B, L_cur, C), torch.randn(B, L_buf, C)
            )
        assert p.shape == (L_buf,)

    def test_window_clamped_to_seq_len(self):
        """W = min(attention_pressure_window, L_cur) – short sequences should work."""
        cfg   = _tiny_config(attention_pressure_window=8)
        block = _MemoryBlockCPU(cfg).eval()
        B, L_cur, L_buf, C = 1, 2, 6, cfg.hidden_size   # L_cur < window
        with torch.no_grad():
            p = block._compute_attention_pressure(
                torch.randn(B, L_cur, C), torch.randn(B, L_buf, C)
            )
        assert p.shape == (L_buf,)


class TestFindCompressBoundary:

    def test_returns_none_when_buffer_too_small(self):
        cfg   = _tiny_config()
        block = _MemoryBlockCPU(cfg).eval()
        B, C  = 1, cfg.hidden_size
        # Buffer strictly smaller than min_segment_length
        buf    = torch.randn(B, cfg.min_segment_length - 1, C)
        hidden = torch.randn(B, 4, C)
        with torch.no_grad():
            result = block._find_compress_boundary(buf, hidden)
        assert result is None, f"Expected None for tiny buffer, got {result}"

    def test_triggers_on_overflow(self):
        """A full buffer (≥ raw_buffer_max_size) must always return a cut point."""
        torch.manual_seed(3)
        cfg   = _tiny_config()
        block = _MemoryBlockCPU(cfg).eval()
        B, C  = 1, cfg.hidden_size
        buf    = torch.randn(B, cfg.raw_buffer_max_size, C)
        hidden = torch.randn(B, 4, C)
        with torch.no_grad():
            result = block._find_compress_boundary(buf, hidden)
        assert result is not None, "Full buffer must trigger compression"
        assert result >= cfg.min_segment_length, (
            f"cut ({result}) must be ≥ min_segment_length ({cfg.min_segment_length})"
        )

    def test_cut_does_not_exceed_max_segment_length(self):
        """Compression cut should never exceed max_segment_length tokens."""
        torch.manual_seed(4)
        cfg   = _tiny_config()
        block = _MemoryBlockCPU(cfg).eval()
        B, C  = 1, cfg.hidden_size
        buf    = torch.randn(B, cfg.raw_buffer_max_size, C)
        hidden = torch.randn(B, 4, C)
        with torch.no_grad():
            result = block._find_compress_boundary(buf, hidden)
        if result is not None:
            assert result <= cfg.max_segment_length, (
                f"cut ({result}) exceeds max_segment_length ({cfg.max_segment_length})"
            )


class TestCompressPipeline:

    def test_compress_one_segment_shapes(self):
        torch.manual_seed(6)
        cfg         = _tiny_config()
        block       = _MemoryBlockCPU(cfg).eval()
        B, L_buf, C = 1, 12, cfg.hidden_size
        K           = cfg.memory_slots_per_segment
        buffer      = torch.randn(B, L_buf, C)
        compress_to = 6

        with torch.no_grad():
            remaining, mem_tokens, seg_cnt = block._compress_one_segment(
                buffer, None, 0, compress_to
            )

        assert remaining.shape  == (B, L_buf - compress_to, C)
        assert mem_tokens.shape == (B, K, C)
        assert seg_cnt           == 1

    def test_try_compress_is_repeatable(self):
        """_try_compress should reduce a large buffer to ≤ max_segment_length tokens."""
        torch.manual_seed(7)
        cfg   = _tiny_config()
        block = _MemoryBlockCPU(cfg).eval()
        B, C  = 1, cfg.hidden_size
        # Create a very large buffer – forces multiple compression passes
        buf    = torch.randn(B, cfg.raw_buffer_max_size * 2, C)
        hidden = torch.randn(B, 4, C)

        with torch.no_grad():
            buf_out, mem_out, seg_cnt = block._try_compress(buf, None, 0, hidden)

        assert buf_out.shape[1] <= cfg.raw_buffer_max_size, (
            "After _try_compress, buffer must not exceed raw_buffer_max_size"
        )
        if mem_out is not None:
            assert mem_out.shape[0] == B
            assert mem_out.shape[2] == C
            assert seg_cnt > 0

    def test_update_memory_bounds_buffer_size(self):
        """Multi-step _update_memory: buffer must never exceed raw_buffer_max_size."""
        torch.manual_seed(8)
        cfg        = _tiny_config()
        block      = _MemoryBlockCPU(cfg).eval()
        cache      = MemoryCache()
        B, chunk   = 1, 8

        for _ in range((cfg.raw_buffer_max_size // chunk) + 3):
            chunk_h = torch.randn(B, chunk, cfg.hidden_size)
            with torch.no_grad():
                cache = block._update_memory(chunk_h, cache)

        state   = cache.get_layer_memory(0)
        buf_len = state['raw_buffer'].shape[1] if state['raw_buffer'] is not None else 0
        assert buf_len <= cfg.raw_buffer_max_size, (
            f"Buffer size {buf_len} exceeds max {cfg.raw_buffer_max_size}"
        )
        assert state['memory_tokens'] is not None, (
            "Expected memory tokens to have been created after overflow"
        )

    def test_memory_slots_evicted_when_over_max(self):
        """memory_tokens should never exceed max_memory_slots."""
        torch.manual_seed(9)
        cfg   = _tiny_config(
            max_memory_slots=8,
            memory_slots_per_segment=2,
            raw_buffer_max_size=16,
            min_segment_length=4,
            max_segment_length=8,
        )
        block = _MemoryBlockCPU(cfg).eval()
        cache = MemoryCache()
        B, chunk = 1, 6

        # Feed many chunks to accumulate memory
        for _ in range(10):
            with torch.no_grad():
                cache = block._update_memory(torch.randn(B, chunk, cfg.hidden_size), cache)

        state = cache.get_layer_memory(0)
        if state['memory_tokens'] is not None:
            assert state['memory_tokens'].shape[1] <= cfg.max_memory_slots, (
                "Memory bank exceeded max_memory_slots"
            )


# ═════════════════════════════════════════════════════════════════════════════
# TIER GPU – Full model tests (requires CUDA + flash_attn)
# ═════════════════════════════════════════════════════════════════════════════

@GPU_MARK
@gpu
class TestFullModelGPU:

    def test_forward_smoke(self):
        """Basic forward pass: correct logits shape, no crash."""
        torch.manual_seed(0)
        cfg   = _tiny_config()
        model = MemoryTransformerForCausalLM(cfg).to(DEVICE).eval()
        B, L  = 2, 8
        ids   = torch.randint(0, cfg.vocab_size, (B, L), device=DEVICE)

        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True, return_dict=True)

        assert out.logits.shape == (B, L, cfg.vocab_size)
        assert not out.logits.isnan().any(), "NaN in logits"
        assert isinstance(out.memory_state, MemoryCache)
        assert out.past_key_values is not None

    def test_memory_state_grows_with_steps(self):
        """
        After processing enough tokens, memory_tokens should become non-None
        and the buffer should remain bounded.
        """
        torch.manual_seed(1)
        cfg   = _tiny_config(local_window_size=8, raw_buffer_max_size=16)
        model = MemoryTransformerForCausalLM(cfg).to(DEVICE).eval()
        B     = 1
        cache = None

        for _ in range(6):   # 6 × 8 = 48 tokens total >> raw_buffer_max_size
            ids = torch.randint(0, cfg.vocab_size, (B, 8), device=DEVICE)
            with torch.no_grad():
                out   = model(input_ids=ids, memory_cache=cache,
                              use_cache=False, return_dict=True)
                cache = out.memory_state

        mem_layer0 = cache.get_layer_memory(0)
        buf_len    = (mem_layer0['raw_buffer'].shape[1]
                      if mem_layer0['raw_buffer'] is not None else 0)
        assert buf_len <= cfg.raw_buffer_max_size, (
            f"Buffer {buf_len} exceeded max {cfg.raw_buffer_max_size}"
        )
        assert mem_layer0['memory_tokens'] is not None, (
            "Expected memory_tokens after many forward steps"
        )

    def test_handles_sequence_longer_than_local_window(self):
        """Model should not crash on sequences much longer than local_window_size."""
        torch.manual_seed(2)
        cfg   = _tiny_config(local_window_size=8)
        model = MemoryTransformerForCausalLM(cfg).to(DEVICE).eval()
        B, L  = 1, 64   # 8× the local window
        ids   = torch.randint(0, cfg.vocab_size, (B, L), device=DEVICE)

        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True, return_dict=True)

        assert out.logits.shape == (B, L, cfg.vocab_size)
        assert not out.logits.isnan().any()

    def test_no_memory_vs_memory_output_differs(self):
        """
        With enough context built up, memory cross-attention should change the
        output compared to a model that has no memory tokens available.
        """
        torch.manual_seed(3)
        cfg   = _tiny_config(local_window_size=8, raw_buffer_max_size=16)
        model = MemoryTransformerForCausalLM(cfg).to(DEVICE).eval()

        # Build up memory by running a long sequence
        long_ids = torch.randint(0, cfg.vocab_size, (1, 48), device=DEVICE)
        with torch.no_grad():
            warm_out = model(input_ids=long_ids, use_cache=False, return_dict=True)
        warm_cache = warm_out.memory_state

        # Query tokens
        query_ids = torch.randint(0, cfg.vocab_size, (1, 4), device=DEVICE)
        with torch.no_grad():
            out_with_mem = model(input_ids=query_ids, memory_cache=warm_cache,
                                 use_cache=False, return_dict=True)
            out_no_mem   = model(input_ids=query_ids, memory_cache=None,
                                 use_cache=False, return_dict=True)

        mem_layer = warm_cache.get_layer_memory(0)
        if mem_layer['memory_tokens'] is not None:
            assert not torch.allclose(
                out_with_mem.logits, out_no_mem.logits, atol=1e-6
            ), "Memory should change logits when memory_tokens are available"

    def test_hf_config_save_load(self):
        """AutoConfig can round-trip a MemoryTransformerConfig via save/load."""
        from transformers import AutoConfig
        cfg = _tiny_config()
        with tempfile.TemporaryDirectory() as tmp:
            cfg.save_pretrained(tmp)
            loaded = AutoConfig.from_pretrained(tmp)
        assert loaded.model_type      == 'memory_transformer'
        assert loaded.hidden_size     == cfg.hidden_size
        assert loaded.use_memory      == cfg.use_memory
        assert loaded.memory_layers   == cfg.memory_layers
        assert loaded.max_memory_slots == cfg.max_memory_slots

    def test_model_save_load_weights_identical(self):
        """save_pretrained → from_pretrained must produce identical logits."""
        torch.manual_seed(4)
        cfg   = _tiny_config()
        model = MemoryTransformerForCausalLM(cfg).to(DEVICE).eval()
        ids   = torch.randint(0, cfg.vocab_size, (1, 6), device=DEVICE)

        with torch.no_grad():
            logits_before = model(input_ids=ids, return_dict=True).logits

        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            loaded = MemoryTransformerForCausalLM.from_pretrained(tmp).to(DEVICE).eval()
            with torch.no_grad():
                logits_after = loaded(input_ids=ids, return_dict=True).logits

        assert torch.allclose(logits_before, logits_after, atol=1e-5), (
            "Logits must be identical after save/load roundtrip"
        )

    def test_automodel_registration(self):
        """AutoModelForCausalLM.register check via config model_type."""
        from transformers import AutoConfig, AutoModelForCausalLM
        cfg = _tiny_config()
        assert cfg.model_type == 'memory_transformer'
        with tempfile.TemporaryDirectory() as tmp:
            cfg.save_pretrained(tmp)
            loaded_cfg = AutoConfig.from_pretrained(tmp)
            # AutoModelForCausalLM.from_config should resolve to MemoryTransformerForCausalLM
            m = AutoModelForCausalLM.from_config(loaded_cfg)
            assert isinstance(m, MemoryTransformerForCausalLM)
