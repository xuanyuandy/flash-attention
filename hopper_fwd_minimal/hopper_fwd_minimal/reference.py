"""Small PyTorch reference for the extracted dense forward path."""

from __future__ import annotations

import math
from typing import Optional

import torch


def attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Reference attention for tensors shaped (batch, seqlen, heads, dim)."""
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    q_heads = q.shape[2]
    kv_heads = k.shape[2]
    assert q_heads % kv_heads == 0
    if q_heads != kv_heads:
        repeat = q_heads // kv_heads
        k = k.repeat_interleave(repeat, dim=2)
        v = v.repeat_interleave(repeat, dim=2)

    scores = torch.einsum("bqhd,bkhd->bhqk", q.float() * softmax_scale, k.float())
    if causal:
        seqlen_q, seqlen_k = q.shape[1], k.shape[1]
        q_idx = torch.arange(seqlen_q, device=q.device)[:, None]
        k_idx = torch.arange(seqlen_k, device=q.device)[None, :]
        causal_mask = k_idx > q_idx + seqlen_k - seqlen_q
        scores = scores.masked_fill(causal_mask, float("-inf"))

    probs = torch.softmax(scores, dim=-1).to(v.dtype)
    return torch.einsum("bhqk,bkhd->bqhd", probs, v)
