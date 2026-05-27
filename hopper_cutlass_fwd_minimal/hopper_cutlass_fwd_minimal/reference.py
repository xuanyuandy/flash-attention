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
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float() * softmax_scale, k.float())
    if causal:
        seqlen_q, seqlen_k = q.shape[1], k.shape[1]
        q_idx = torch.arange(seqlen_q, device=q.device)[:, None]
        k_idx = torch.arange(seqlen_k, device=q.device)[None, :]
        scores = scores.masked_fill(k_idx > q_idx + seqlen_k - seqlen_q, float("-inf"))
    probs = torch.softmax(scores, dim=-1).to(v.dtype)
    return torch.einsum("bhqk,bkhd->bqhd", probs, v)
