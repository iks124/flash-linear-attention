#!/usr/bin/env python3
"""
Save MemoryTransformer Weights
================================
Creates a MemoryTransformerForCausalLM from a config and saves it in
HuggingFace format (config.json + model.safetensors).

Optionally loads standard Transformer weights from an existing HF checkpoint
into the shared components (attention, MLP, norms, embeddings) so that only
the new memory-specific modules need to be trained from scratch.

Usage
-----
1. Save randomly-initialized tiny model:
       python scripts/save_memory_transformer_weights.py \\
           --output_dir ./checkpoints/memory_transformer_tiny

2. Save with a specific config:
       python scripts/save_memory_transformer_weights.py \\
           --output_dir ./checkpoints/memory_transformer_small \\
           --hidden_size 512 --num_layers 8 --num_heads 8

3. Load from a pre-trained standard Transformer checkpoint and inject into
   the shared components (strategy A fine-tuning):
       python scripts/save_memory_transformer_weights.py \\
           --output_dir ./checkpoints/memory_transformer_from_gpt2 \\
           --pretrained_base gpt2

Weight mapping strategy (--pretrained_base)
--------------------------------------------
When a HuggingFace checkpoint is supplied the script attempts to copy weights
for every parameter whose name appears in BOTH the base model and the
MemoryTransformerModel.  The mapping is done by suffix-matching the parameter
name (e.g. "attn.q_proj.weight") so it is architecture-agnostic.

New memory-specific parameters (boundary_detector, compressor, memory_attn,
segment_pos_embed) are left at their random PyTorch initialization.  The
memory_attn.gate_proj and memory_attn.o_proj are zero-initialized so that
at the start of fine-tuning the memory cross-attention contributes a zero
delta and the model is identical to the base Transformer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path when run as a standalone script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn

# ── flash_attn stub for CPU-only weight saving ────────────────────────────────
# Attention.__init__ raises ImportError if flash_attn is absent.
# We only need to *instantiate + save* the model, never to call forward,
# so patching the module-level sentinel is safe here.
def _patch_flash_attn_if_missing():
    import importlib
    _attn_mod = importlib.import_module('fla.layers.attn')
    if getattr(_attn_mod, 'flash_attn_func', None) is None:
        _stub = lambda *a, **kw: (None, None, None)
        _attn_mod.flash_attn_func          = _stub
        _attn_mod.flash_attn_varlen_func   = _stub
        print("  [info] flash_attn not found – using no-op stub for weight initialisation only.")
        print("  [info] The saved model requires flash_attn at runtime (inference/training).")

_patch_flash_attn_if_missing()

try:
    from fla.models.memory_transformer.configuration_memory_transformer import (
        MemoryTransformerConfig,
    )
    from fla.models.memory_transformer.modeling_memory_transformer import (
        MemoryTransformerForCausalLM,
    )
except ImportError as exc:
    sys.exit(f"[ERROR] Cannot import MemoryTransformer: {exc}")


# ── Memory-specific module name substrings that must NOT be overwritten ───────
_MEMORY_ONLY_KEYS = (
    'boundary_detector',
    'compressor',
    'memory_attn',
    'segment_pos_embed',
)


def _zero_init_memory_gates(model: MemoryTransformerForCausalLM) -> None:
    """
    Zero-initialize memory_attn.gate_proj and memory_attn.o_proj so that
    at the start of fine-tuning the memory cross-attention contributes
    nothing (identity residual), preserving the base model's behaviour.
    """
    for name, module in model.named_modules():
        if 'memory_attn' in name:
            for attr in ('gate_proj', 'o_proj'):
                proj = getattr(module, attr, None)
                if proj is not None and isinstance(proj, nn.Linear):
                    nn.init.zeros_(proj.weight)
                    if proj.bias is not None:
                        nn.init.zeros_(proj.bias)


def _load_pretrained_base(
    model: MemoryTransformerForCausalLM,
    base_name_or_path: str,
) -> dict[str, int]:
    """
    Load shared-component weights from a HF checkpoint into `model`.

    Returns a dict with copy statistics:
        {'copied': N, 'skipped_memory': M, 'not_found': K}
    """
    from transformers import AutoModel

    print(f"  Loading base weights from: {base_name_or_path}")
    base = AutoModel.from_pretrained(base_name_or_path, ignore_mismatches_during_load=True)
    base_sd = {k: v for k, v in base.state_dict().items()}

    # Build a name → tensor map keyed by the *suffix* (everything after the
    # first model-specific prefix, e.g. "model.layers.0.attn.q_proj.weight"
    # → "layers.0.attn.q_proj.weight").
    def _strip_prefix(name: str) -> str:
        parts = name.split('.')
        # Drop the leading "model" / "transformer" / "gpt_neox" etc.
        if parts[0] in ('model', 'transformer', 'gpt_neox', 'bert'):
            return '.'.join(parts[1:])
        return name

    base_by_suffix = {_strip_prefix(k): v for k, v in base_sd.items()}

    stats = {'copied': 0, 'skipped_memory': 0, 'not_found': 0}
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Never overwrite memory-specific components
            if any(m in name for m in _MEMORY_ONLY_KEYS):
                stats['skipped_memory'] += 1
                continue

            suffix = _strip_prefix(name)
            if suffix in base_by_suffix:
                src = base_by_suffix[suffix]
                if src.shape == param.shape:
                    param.copy_(src)
                    stats['copied'] += 1
                else:
                    stats['not_found'] += 1   # shape mismatch
            else:
                stats['not_found'] += 1

    del base
    return stats


def build_config(args: argparse.Namespace) -> MemoryTransformerConfig:
    return MemoryTransformerConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads or args.num_heads,
        vocab_size=args.vocab_size,
        max_position_embeddings=args.max_position_embeddings,
        # memory params
        use_memory=True,
        local_window_size=args.local_window_size,
        raw_buffer_max_size=args.raw_buffer_max_size,
        min_segment_length=args.min_segment_length,
        max_segment_length=args.max_segment_length,
        memory_slots_per_segment=args.memory_slots_per_segment,
        max_memory_slots=args.max_memory_slots,
        attention_pressure_window=args.attention_pressure_window,
        # perf
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
    )


def main():
    parser = argparse.ArgumentParser(description='Save MemoryTransformer weights')
    parser.add_argument('--output_dir', required=True, help='Directory to save the model')

    # Architecture
    parser.add_argument('--hidden_size',            type=int, default=512)
    parser.add_argument('--num_layers',             type=int, default=8)
    parser.add_argument('--num_heads',              type=int, default=8)
    parser.add_argument('--num_kv_heads',           type=int, default=None)
    parser.add_argument('--vocab_size',             type=int, default=32000)
    parser.add_argument('--max_position_embeddings',type=int, default=2048)

    # Memory params
    parser.add_argument('--local_window_size',          type=int, default=512)
    parser.add_argument('--raw_buffer_max_size',        type=int, default=256)
    parser.add_argument('--min_segment_length',         type=int, default=8)
    parser.add_argument('--max_segment_length',         type=int, default=64)
    parser.add_argument('--memory_slots_per_segment',   type=int, default=4)
    parser.add_argument('--max_memory_slots',           type=int, default=512)
    parser.add_argument('--attention_pressure_window',  type=int, default=8)

    # Optional weight transfer
    parser.add_argument(
        '--pretrained_base',
        type=str, default=None,
        help='HF model name or path to load shared-component weights from '
             '(e.g. gpt2, facebook/opt-125m).  Memory-specific layers are '
             'kept at random init; gate/output projections of memory_attn '
             'are zero-initialized for a stable fine-tuning start.',
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*60}")
    print("  MemoryTransformer weight saver")
    print(f"{'─'*60}")

    # ── Build model ───────────────────────────────────────────────────────
    cfg   = build_config(args)
    model = MemoryTransformerForCausalLM(cfg)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_mem_params = sum(
        p.numel()
        for n, p in model.named_parameters()
        if any(m in n for m in _MEMORY_ONLY_KEYS)
    ) / 1e6
    print(f"  Total parameters:          {n_params:.1f}M")
    print(f"  Memory-specific parameters:{n_mem_params:.1f}M")
    print(f"  Shared (base) parameters:  {n_params - n_mem_params:.1f}M")

    # ── Optionally inject pretrained base weights ─────────────────────────
    if args.pretrained_base:
        stats = _load_pretrained_base(model, args.pretrained_base)
        print(f"  Weight transfer stats: {stats}")
        # Zero-init memory gates for clean fine-tuning start
        _zero_init_memory_gates(model)
        print("  memory_attn gate/output projections zero-initialized")
    else:
        print("  No pretrained base – all weights randomly initialized")

    # ── Save ──────────────────────────────────────────────────────────────
    model.save_pretrained(output_dir, safe_serialization=True)
    print(f"\n  Saved to: {output_dir.resolve()}")
    print(f"  Files:    {[f.name for f in sorted(output_dir.iterdir())]}")

    # ── Verify round-trip ─────────────────────────────────────────────────
    loaded = MemoryTransformerForCausalLM.from_pretrained(output_dir)
    param_mismatch = any(
        not torch.allclose(p1, p2, atol=1e-6)
        for (_, p1), (_, p2) in zip(
            model.named_parameters(), loaded.named_parameters()
        )
    )
    if param_mismatch:
        print("  [WARNING] Parameter mismatch after reload – check safetensors support")
    else:
        print("  [✓] Round-trip load verified – weights are identical")

    print(f"{'─'*60}\n")


if __name__ == '__main__':
    main()
