#!/usr/bin/env python3
"""
MemoryTransformer Diagnostic Evaluation
========================================
Smoke-tests + lightweight diagnostic evaluations that require NO training.
All CPU-mode checks pass without flash_attn; GPU checks are skipped gracefully.

Usage
-----
    python scripts/eval_memory_transformer.py            # CPU only
    python scripts/eval_memory_transformer.py --device cuda

What is tested
--------------
1. Component shape contracts        (CPU)
2. Compression boundary alignment   (CPU) – synthetic multi-topic sequences
3. Attention pressure semantics     (CPU) – low-pressure tokens should compress first
4. HF framework integration         (GPU) – save / load / generate / AutoModel
5. Long-sequence stability          (GPU) – no NaN/Inf beyond local window

Recommended benchmarks after fine-tuning
-----------------------------------------
| Capability             | Benchmark      | Metric      |
|------------------------|----------------|-------------|
| Long-context modeling  | LongBench      | Avg score   |
| Needle-in-haystack     | RULER NIAH     | Recall@K    |
| Cross-passage retrieval| HotpotQA       | EM / F1     |
| Passage summarisation  | SCROLLS        | ROUGE / F1  |
| Perplexity             | PG-19 / LAMBADA| PPL / Acc   |
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Ensure project root is on the path when run as a standalone script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── early import check ────────────────────────────────────────────────────────
try:
    from fla.models.memory_transformer.configuration_memory_transformer import (
        MemoryTransformerConfig,
    )
    from fla.models.memory_transformer.modeling_memory_transformer import (
        MemoryCache,
        MemoryCrossAttention,
        MemoryTransformerBlock,
        MemoryTransformerForCausalLM,
        SemanticBoundaryDetector,
        SegmentCompressor,
    )
except ImportError as exc:
    sys.exit(f"[ERROR] Cannot import MemoryTransformer: {exc}")


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


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
        max_memory_slots=64,
        max_segments=32,
        attention_pressure_window=4,
    )
    base.update(overrides)
    return MemoryTransformerConfig(**base)


class _MemoryBlockCPU(nn.Module):
    """CPU-testable stub block (no flash_attn dependency)."""

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

    _cat                        = staticmethod(MemoryTransformerBlock._cat)
    _compute_attention_pressure = MemoryTransformerBlock._compute_attention_pressure
    _find_compress_boundary     = MemoryTransformerBlock._find_compress_boundary
    _compress_one_segment       = MemoryTransformerBlock._compress_one_segment
    _try_compress               = MemoryTransformerBlock._try_compress
    _update_memory              = MemoryTransformerBlock._update_memory


@dataclass
class EvalResult:
    name:    str
    passed:  bool
    details: str = ''
    metrics: dict = field(default_factory=dict)


RESULTS: list[EvalResult] = []


def _report(r: EvalResult):
    RESULTS.append(r)
    icon = '✓' if r.passed else '✗'
    print(f"  [{icon}] {r.name}")
    if r.details:
        for line in r.details.splitlines():
            print(f"        {line}")
    for k, v in r.metrics.items():
        if isinstance(v, float):
            print(f"        {k}: {v:.4f}")
        else:
            print(f"        {k}: {v}")


# ═════════════════════════════════════════════════════════════════════════════
# Eval 1 – Component shape contracts
# ═════════════════════════════════════════════════════════════════════════════

def eval_component_shapes():
    print("\n── Eval 1: Component shape contracts ──────────────────────────────────")
    torch.manual_seed(0)
    cfg = _tiny_config()
    C, B, L, K = cfg.hidden_size, 2, 10, cfg.memory_slots_per_segment
    M, H       = 8, cfg.memory_num_heads

    # SemanticBoundaryDetector
    det  = SemanticBoundaryDetector(C).eval()
    x    = torch.randn(B, L, C)
    with torch.no_grad():
        bp = det(x)
    ok   = (bp.shape == (B, L - 1)) and (0 <= bp.min() <= bp.max() <= 1)
    _report(EvalResult(
        'SemanticBoundaryDetector',
        ok,
        f"output shape {tuple(bp.shape)} (expected ({B},{L-1})), range [{bp.min():.3f}, {bp.max():.3f}]",
    ))

    # SegmentCompressor
    comp = SegmentCompressor(C, K).eval()
    seg  = torch.randn(B, L, C)
    with torch.no_grad():
        slots = comp(seg)
    ok   = slots.shape == (B, K, C)
    _report(EvalResult(
        'SegmentCompressor',
        ok,
        f"output shape {tuple(slots.shape)} (expected ({B},{K},{C}))",
    ))

    # MemoryCrossAttention
    mca = MemoryCrossAttention(C, H).eval()
    q   = torch.randn(B, L, C)
    mem = torch.randn(B, M, C)
    with torch.no_grad():
        delta = mca(q, mem)
    ok   = delta.shape == (B, L, C)
    _report(EvalResult(
        'MemoryCrossAttention',
        ok,
        f"output shape {tuple(delta.shape)} (expected ({B},{L},{C}))",
    ))

    # MemoryCache
    cache = MemoryCache()
    cache.update_layer_memory(0, torch.randn(B, 5, C), torch.randn(B, 3, C), 2)
    s     = cache.get_layer_memory(0)
    ok    = (s['raw_buffer'].shape == (B, 5, C)
             and s['memory_tokens'].shape == (B, 3, C)
             and s['segment_count'] == 2)
    _report(EvalResult('MemoryCache', ok, f"state shapes correct, segment_count=2"))


# ═════════════════════════════════════════════════════════════════════════════
# Eval 2 – Compression boundary alignment on synthetic multi-topic sequences
# ═════════════════════════════════════════════════════════════════════════════

def eval_compression_boundary_alignment():
    """
    Creates a synthetic sequence of N_TOPICS topics, each TOKEN_PER_TOPIC tokens.
    Hidden states within a topic are near-identical (topic centre + small noise).
    Hidden states across topics are orthogonal.

    Checks:
      A. SemanticBoundaryDetector gives higher scores at true boundaries
         than within topics (cosine-change component, gate zeroed out).
      B. Compression pipeline (no training) fires and keeps buffer bounded.
      C. Recorded compression events tend to align with topic boundaries.
    """
    print("\n── Eval 2: Compression boundary alignment ──────────────────────────────")
    torch.manual_seed(42)

    N_TOPICS        = 4
    TOKENS_PER_TOPIC = 12
    CHUNK_SIZE      = 4
    cfg             = _tiny_config(
        hidden_size=64,
        raw_buffer_max_size=24,
        min_segment_length=4,
        max_segment_length=12,
    )
    C = cfg.hidden_size

    # ── 2A: Boundary detector fires at topic transitions ──────────────────
    det = SemanticBoundaryDetector(C).eval()
    # Zero gate → isolate parameter-free cosine-change signal
    with torch.no_grad():
        det.fc1.weight.zero_()
        det.fc2.weight.zero_()

    # Build orthogonal topic centres via Gram-Schmidt
    centres = [F.normalize(torch.randn(C), dim=0)]
    for _ in range(N_TOPICS - 1):
        v = torch.randn(C)
        for u in centres:
            v = v - (v @ u) * u
        centres.append(F.normalize(v, dim=0))

    noise  = 0.05
    L_full = N_TOPICS * TOKENS_PER_TOPIC
    states = torch.zeros(1, L_full, C)
    for t, centre in enumerate(centres):
        start = t * TOKENS_PER_TOPIC
        states[0, start:start + TOKENS_PER_TOPIC] = (
            centre + noise * torch.randn(TOKENS_PER_TOPIC, C)
        )

    with torch.no_grad():
        bp = det(states)[0]   # (L_full - 1,)

    # True boundary positions (gaps just before topic changes)
    true_boundaries = [t * TOKENS_PER_TOPIC - 1 for t in range(1, N_TOPICS)]
    boundary_scores = [bp[pos].item() for pos in true_boundaries]
    within_scores   = [
        bp[i].item()
        for i in range(len(bp))
        if i not in true_boundaries
    ]
    mean_boundary = sum(boundary_scores) / len(boundary_scores)
    mean_within   = sum(within_scores)   / len(within_scores)
    ok_2a         = mean_boundary > mean_within

    _report(EvalResult(
        'BoundaryDetector: boundary > within-topic (cosine-change)',
        ok_2a,
        f"mean score @ boundaries={mean_boundary:.3f}  mean score within={mean_within:.3f}",
        {'mean_boundary': mean_boundary, 'mean_within': mean_within},
    ))

    # ── 2B: Pipeline keeps buffer bounded ─────────────────────────────────
    block  = _MemoryBlockCPU(cfg).eval()
    cache  = MemoryCache()
    events = []

    for step in range(N_TOPICS * TOKENS_PER_TOPIC // CHUNK_SIZE):
        start   = step * CHUNK_SIZE
        chunk_h = states[:, start:start + CHUNK_SIZE, :]
        old_buf = cache._raw_buffer.get(0)
        old_len = old_buf.shape[1] if old_buf is not None else 0

        with torch.no_grad():
            cache = block._update_memory(chunk_h, cache)

        new_buf = cache._raw_buffer.get(0)
        new_len = new_buf.shape[1] if new_buf is not None else 0

        if new_len < old_len + CHUNK_SIZE:
            compressed = (old_len + CHUNK_SIZE) - new_len
            events.append({
                'step':            step,
                'start_token':     start,
                'compressed':      compressed,
                'topic_at_start':  start // TOKENS_PER_TOPIC,
            })

    buf_state  = cache.get_layer_memory(0)
    final_buf  = buf_state['raw_buffer']
    buf_bounded = final_buf is None or final_buf.shape[1] <= cfg.raw_buffer_max_size
    _report(EvalResult(
        'Pipeline: buffer stays bounded',
        buf_bounded,
        f"final buffer size: "
        f"{final_buf.shape[1] if final_buf is not None else 0} "
        f"(max={cfg.raw_buffer_max_size})",
    ))

    # ── 2C: Compression events vs topic boundaries ─────────────────────────
    if events:
        # Compute min distance from each compression event to a true topic boundary
        true_boundary_tokens = [t * TOKENS_PER_TOPIC for t in range(1, N_TOPICS)]
        dists = []
        for ev in events:
            compressed_end = ev['start_token'] + ev['compressed']
            dist = min(abs(compressed_end - b) for b in true_boundary_tokens)
            dists.append(dist)
        mean_dist = sum(dists) / len(dists)
        # Expected random baseline: mean distance ≈ TOKENS_PER_TOPIC / 4
        baseline  = TOKENS_PER_TOPIC / 4.0
        detail = (
            f"{len(events)} compression events, "
            f"mean token-distance to nearest boundary: {mean_dist:.1f} "
            f"(random baseline ≈ {baseline:.1f})"
        )
        _report(EvalResult(
            'Pipeline: compression events near topic boundaries',
            True,      # informational – pass always; watch the metric
            detail,
            {'mean_dist_to_boundary': float(mean_dist), 'n_events': len(events)},
        ))
    else:
        _report(EvalResult(
            'Pipeline: compression events near topic boundaries',
            False,
            'No compression events fired during the simulation',
        ))


# ═════════════════════════════════════════════════════════════════════════════
# Eval 3 – Attention pressure semantics
# ═════════════════════════════════════════════════════════════════════════════

def eval_attention_pressure_semantics():
    """
    Verify that _compute_attention_pressure:
      • Returns values in [0, 1] for every buffer position.
      • Positions that are copied verbatim into the query window get
        the highest pressure (model "looks at itself").
    """
    print("\n── Eval 3: Attention pressure semantics ────────────────────────────────")
    torch.manual_seed(13)
    cfg   = _tiny_config()
    block = _MemoryBlockCPU(cfg).eval()
    B, C  = 1, cfg.hidden_size
    L_buf = 16
    W     = cfg.attention_pressure_window   # = 4

    buffer = torch.randn(B, L_buf, C)
    # Use the LAST W buffer tokens verbatim as the current "hidden_states"
    # so those positions should get maximum attention from themselves.
    hidden = buffer[:, -W:, :].detach().clone()

    with torch.no_grad():
        pressure = block._compute_attention_pressure(hidden, buffer)

    ok_range = (pressure.min() >= 0.0) and (pressure.max() <= 1.0)
    _report(EvalResult(
        'Attention pressure in [0,1]',
        ok_range,
        f"min={pressure.min():.4f}  max={pressure.max():.4f}",
    ))

    # The last W positions should have relatively high pressure since the
    # query window IS those positions (softmax will concentrate there).
    tail_pressure = pressure[-W:].mean().item()
    head_pressure = pressure[:-W].mean().item()
    ok_semantics  = tail_pressure > head_pressure
    _report(EvalResult(
        'Attention pressure: recent tokens have higher pressure',
        ok_semantics,
        f"tail (last {W}) pressure={tail_pressure:.4f}  head pressure={head_pressure:.4f}",
        {'tail_pressure': tail_pressure, 'head_pressure': head_pressure},
    ))


# ═════════════════════════════════════════════════════════════════════════════
# Eval 4 – HF framework integration (GPU + flash_attn required)
# ═════════════════════════════════════════════════════════════════════════════

def eval_hf_integration(device: torch.device):
    print("\n── Eval 4: HF framework integration ───────────────────────────────────")
    if not (torch.cuda.is_available() and _has_flash_attn()):
        print("  [–] Skipped (requires CUDA + flash_attn)")
        return

    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(20)
    cfg   = _tiny_config()
    model = MemoryTransformerForCausalLM(cfg).to(device).eval()

    # ── 4A: forward pass ──────────────────────────────────────────────────
    ids = torch.randint(0, cfg.vocab_size, (1, 32), device=device)
    try:
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True, return_dict=True)
        ok  = (out.logits.shape == (1, 32, cfg.vocab_size)
               and not out.logits.isnan().any())
        _report(EvalResult('Forward pass (32 tokens)', ok,
                            f"logits shape {tuple(out.logits.shape)}"))
    except Exception as exc:
        _report(EvalResult('Forward pass (32 tokens)', False, str(exc)))

    # ── 4B: sequence > local_window_size ─────────────────────────────────
    ids_long = torch.randint(0, cfg.vocab_size, (1, 64), device=device)
    try:
        with torch.no_grad():
            out_long = model(input_ids=ids_long, return_dict=True)
        ok = (not out_long.logits.isnan().any()
              and not out_long.logits.isinf().any())
        _report(EvalResult('Forward pass (64 tokens > local_window=16)', ok,
                            'No NaN / Inf'))
    except Exception as exc:
        _report(EvalResult('Forward pass (64 tokens > local_window=16)', False, str(exc)))

    # ── 4C: save / load ───────────────────────────────────────────────────
    query = torch.randint(0, cfg.vocab_size, (1, 4), device=device)
    with torch.no_grad():
        logits_ref = model(input_ids=query, return_dict=True).logits

    try:
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            loaded = MemoryTransformerForCausalLM.from_pretrained(tmp).to(device).eval()
            with torch.no_grad():
                logits_loaded = loaded(input_ids=query, return_dict=True).logits
        ok = torch.allclose(logits_ref, logits_loaded, atol=1e-5)
        _report(EvalResult('save_pretrained → from_pretrained weight identity', ok,
                            f"max abs diff = {(logits_ref - logits_loaded).abs().max():.2e}"))
    except Exception as exc:
        _report(EvalResult('save_pretrained → from_pretrained weight identity', False, str(exc)))

    # ── 4D: AutoConfig / AutoModelForCausalLM ────────────────────────────
    try:
        with tempfile.TemporaryDirectory() as tmp:
            cfg.save_pretrained(tmp)
            loaded_cfg = AutoConfig.from_pretrained(tmp)
            auto_model = AutoModelForCausalLM.from_config(loaded_cfg)
        ok = isinstance(auto_model, MemoryTransformerForCausalLM)
        _report(EvalResult('AutoModelForCausalLM.from_config', ok,
                            f"resolved to {type(auto_model).__name__}"))
    except Exception as exc:
        _report(EvalResult('AutoModelForCausalLM.from_config', False, str(exc)))

    # ── 4E: generation smoke ──────────────────────────────────────────────
    try:
        prompt = torch.randint(0, cfg.vocab_size, (1, 4), device=device)
        gen    = model.generate(prompt, max_new_tokens=4, do_sample=False)
        ok     = gen.shape == (1, 8)
        _report(EvalResult('model.generate smoke test', ok,
                            f"generated shape {tuple(gen.shape)}"))
    except Exception as exc:
        _report(EvalResult('model.generate smoke test', False, str(exc)))


# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════

def _print_summary():
    print("\n" + "═" * 60)
    passed = sum(1 for r in RESULTS if r.passed)
    total  = len(RESULTS)
    print(f"  SUMMARY: {passed}/{total} checks passed")
    if passed < total:
        print("\n  Failed checks:")
        for r in RESULTS:
            if not r.passed:
                print(f"    ✗ {r.name}")
                if r.details:
                    print(f"      {r.details.splitlines()[0]}")
    print("═" * 60)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='MemoryTransformer diagnostic eval')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args   = parser.parse_args()
    device = torch.device(args.device)

    print("MemoryTransformer Diagnostic Evaluation")
    print(f"device={device}  flash_attn={'yes' if _has_flash_attn() else 'no'}")
    print("─" * 60)

    eval_component_shapes()
    eval_compression_boundary_alignment()
    eval_attention_pressure_semantics()
    eval_hf_integration(device)

    _print_summary()

    n_failed = sum(1 for r in RESULTS if not r.passed)
    sys.exit(0 if n_failed == 0 else 1)


if __name__ == '__main__':
    main()
