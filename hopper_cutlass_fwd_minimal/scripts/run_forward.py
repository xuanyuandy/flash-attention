#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hopper_cutlass_fwd_minimal import forward  # noqa: E402
from hopper_cutlass_fwd_minimal.reference import attention_ref  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal Hopper CUTLASS FlashAttention forward.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seqlen-q", type=int, default=128)
    parser.add_argument("--seqlen-k", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("H100 / SM90 CUDA device is required")

    torch.manual_seed(0)
    q = torch.randn(args.batch, args.seqlen_q, args.heads, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(args.batch, args.seqlen_k, args.heads, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(args.batch, args.seqlen_k, args.heads, 64, device="cuda", dtype=torch.bfloat16)

    for _ in range(args.warmup):
        forward(q, k, v, args.causal, None)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(args.iters):
        out, _ = forward(q, k, v, args.causal, None)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / args.iters

    ref = attention_ref(q, k, v, causal=args.causal)
    max_abs = (out - ref).abs().max().item()
    allowed = 5e-2 + 5e-2 * ref.abs().max().item()
    print(
        "case:",
        f"batch={args.batch}",
        f"seqlen_q={args.seqlen_q}",
        f"seqlen_k={args.seqlen_k}",
        f"heads={args.heads}",
        "head_dim=64",
        "dtype=bf16",
        f"causal={args.causal}",
    )
    print(f"forward latency: {elapsed_ms:.3f} ms")
    print(f"max_abs_error: {max_abs:.6f} allowed: {allowed:.6f}")
    if max_abs > allowed:
        raise AssertionError(f"forward check failed: max_abs={max_abs:.6f}, allowed={allowed:.6f}")


if __name__ == "__main__":
    main()
