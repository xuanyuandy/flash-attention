#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hopper_fwd_minimal import flash_attn_forward  # noqa: E402
from hopper_fwd_minimal.hopper_fwd_minimal.reference import attention_ref  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal Hopper forward kernel on H100.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seqlen-q", type=int, default=128)
    parser.add_argument("--seqlen-k", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--heads-kv", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--no-lse", action="store_true", help="do not return/write LSE")
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 9:
        raise RuntimeError(f"H100/SM90 is required, got compute capability {capability}")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(0)
    q = torch.randn(args.batch, args.seqlen_q, args.heads, args.head_dim, device="cuda", dtype=dtype)
    k = torch.randn(args.batch, args.seqlen_k, args.heads_kv, args.head_dim, device="cuda", dtype=dtype)
    v = torch.randn(args.batch, args.seqlen_k, args.heads_kv, args.head_dim, device="cuda", dtype=dtype)
    return_lse = not args.no_lse

    for _ in range(args.warmup):
        flash_attn_forward(q, k, v, causal=args.causal, return_lse=return_lse)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(args.iters):
        out, _ = flash_attn_forward(q, k, v, causal=args.causal, return_lse=return_lse)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / args.iters

    print(
        "case:",
        f"batch={args.batch}",
        f"seqlen_q={args.seqlen_q}",
        f"seqlen_k={args.seqlen_k}",
        f"heads={args.heads}",
        f"heads_kv={args.heads_kv}",
        f"head_dim={args.head_dim}",
        f"dtype={args.dtype}",
        f"causal={args.causal}",
        f"return_lse={return_lse}",
    )
    print(f"forward latency: {elapsed_ms:.3f} ms")

    if not args.no_check:
        ref = attention_ref(q, k, v, causal=args.causal)
        max_abs = (out - ref).abs().max().item()
        allowed = args.atol + args.rtol * ref.abs().max().item()
        print(f"max_abs_error: {max_abs:.6f} allowed: {allowed:.6f}")
        if max_abs > allowed:
            raise AssertionError(f"forward check failed: max_abs={max_abs:.6f}, allowed={allowed:.6f}")


if __name__ == "__main__":
    main()
