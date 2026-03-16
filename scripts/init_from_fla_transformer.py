"""
Initialize MemoryTransformerForCausalLM weights from a FLA-hub TransformerForCausalLM checkpoint.

Source: fla-hub/transformer-340M-4K-0.5B-20480-lr3e-4-cosine
  - hidden_size=1024, num_heads=16, num_kv_heads=None (=16, full MHA), num_layers=24, vocab=32000

Dest:   your MemoryTransformerForCausalLM checkpoint
  - hidden_size=1024, num_heads=16, num_kv_heads=8  (GQA),          num_layers=24, vocab=32000

Copyable weights (exact shape match):
  - model.embeddings.weight, model.norm.weight, lm_head.weight
  - per layer: attn_norm, mlp_norm, attn.q_proj, attn.o_proj, mlp.*

KV projection mismatch (MHA 16 heads → GQA 8 heads):
  - Source k/v_proj: (1024, 1024) = (16*64, 1024)
  - Dest   k/v_proj: (512,  1024) = ( 8*64, 1024)
  - Strategy: average adjacent pairs of head rows  [h0+h1]/2, [h2+h3]/2, ...

Memory-specific components (dest only) are left as random init:
  - boundary_detector.*, compressor.*, memory_attn.*, segment_pos_embed.*
"""

import argparse
import os
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer


def average_kv_heads(weight: torch.Tensor, src_kv_heads: int, dst_kv_heads: int) -> torch.Tensor:
    """
    weight: (src_kv_heads * head_dim, hidden_size)
    Returns: (dst_kv_heads * head_dim, hidden_size)  by averaging groups
    """
    assert src_kv_heads % dst_kv_heads == 0, \
        f"src_kv_heads={src_kv_heads} must be divisible by dst_kv_heads={dst_kv_heads}"
    group = src_kv_heads // dst_kv_heads          # heads per group
    head_dim = weight.shape[0] // src_kv_heads
    # (src_kv_heads, head_dim, hidden_size) -> (dst_kv_heads, group, head_dim, hidden_size)
    w = weight.view(dst_kv_heads, group, head_dim, weight.shape[1])
    return w.mean(dim=1).reshape(dst_kv_heads * head_dim, weight.shape[1])


def load_safetensors(path: str) -> dict:
    from safetensors import safe_open
    sd = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k)
    return sd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="fla-hub/transformer-340M-4K-0.5B-20480-lr3e-4-cosine",
                        help="Source model: HF hub id or local path")
    parser.add_argument("--dst", default="/home/shihoukun/checkpoints/memory_transformer_660m",
                        help="Destination MemoryTransformer checkpoint directory (weights will be updated in-place)")
    parser.add_argument("--out", default="/home/shihoukun/checkpoints/memory_transformer_660m_init",
                        help="Output directory for the new checkpoint")
    parser.add_argument("--src-kv-heads", type=int, default=16,
                        help="Number of KV heads in source model (default 16 = full MHA)")
    parser.add_argument("--dst-kv-heads", type=int, default=8,
                        help="Number of KV heads in destination model")
    args = parser.parse_args()

    # ── 1. Load source weights ──────────────────────────────────────────────
    print(f"Loading source model from: {args.src}")
    if os.path.isdir(args.src):
        src_file = os.path.join(args.src, "model.safetensors")
        src_sd = load_safetensors(src_file)
    else:
        from huggingface_hub import hf_hub_download
        src_file = hf_hub_download(args.src, "model.safetensors")
        src_sd = load_safetensors(src_file)
    print(f"  Source has {len(src_sd)} tensors")

    # ── 2. Load destination weights ─────────────────────────────────────────
    dst_file = os.path.join(args.dst, "model.safetensors")
    print(f"Loading destination model from: {dst_file}")
    dst_sd = load_safetensors(dst_file)
    print(f"  Destination has {len(dst_sd)} tensors")

    # ── 3. Copy weights ──────────────────────────────────────────────────────
    copied, skipped, kv_averaged = [], [], []

    for dst_key in dst_sd.keys():
        if dst_key not in src_sd:
            skipped.append(dst_key)
            continue

        src_w = src_sd[dst_key]
        dst_w = dst_sd[dst_key]

        if src_w.shape == dst_w.shape:
            dst_sd[dst_key] = src_w.to(dst_w.dtype)
            copied.append(dst_key)
        else:
            # Handle KV projection mismatch
            if dst_key.endswith(("attn.k_proj.weight", "attn.v_proj.weight")):
                dst_sd[dst_key] = average_kv_heads(
                    src_w, args.src_kv_heads, args.dst_kv_heads
                ).to(dst_w.dtype)
                kv_averaged.append(dst_key)
            else:
                print(f"  [WARN] Shape mismatch, skipping: {dst_key} "
                      f"src={tuple(src_w.shape)} dst={tuple(dst_w.shape)}")
                skipped.append(dst_key)

    # ── 4. Report ────────────────────────────────────────────────────────────
    print(f"\n=== Transfer summary ===")
    print(f"  Copied (exact):      {len(copied)}")
    print(f"  KV averaged:         {len(kv_averaged)}")
    print(f"  Skipped (rand init): {len(skipped)}")
    print(f"\n  Skipped keys (new memory components):")
    for k in sorted(skipped):
        print(f"    {k}")

    # ── 5. Copy config + tokenizer, save new weights ─────────────────────────
    import shutil
    os.makedirs(args.out, exist_ok=True)

    # Copy config and generation_config from destination (keep MemoryTransformer config)
    for fname in ("config.json", "generation_config.json"):
        src_cfg = os.path.join(args.dst, fname)
        if os.path.exists(src_cfg):
            shutil.copy(src_cfg, os.path.join(args.out, fname))

    # Copy tokenizer files from source if available locally, else save from HF
    print(f"\nSaving tokenizer from source to: {args.out}")
    try:
        tok = AutoTokenizer.from_pretrained(args.src, trust_remote_code=True)
        tok.save_pretrained(args.out)
        print("  Tokenizer saved.")
    except Exception as e:
        print(f"  [WARN] Could not save tokenizer: {e}")

    # Save weights
    out_file = os.path.join(args.out, "model.safetensors")
    print(f"Saving new weights to: {out_file}")
    save_file(dst_sd, out_file)
    print("Done.")


if __name__ == "__main__":
    main()
