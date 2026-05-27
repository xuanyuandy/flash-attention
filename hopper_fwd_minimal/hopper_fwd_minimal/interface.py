"""Forward-only Hopper extraction for dense fixed-length FlashAttention.

This module intentionally keeps one typical SM90 path:
- H100 / SM90 only
- fixed-length tensors shaped (batch, seqlen, heads, dim)
- dense MHA/GQA/MQA
- causal or non-causal
- no varlen, SplitKV, paged KV, block sparsity, score_mod, or mask_mod

The CuTe kernel body is copied into ``hopper_fwd_minimal.kernel``; support
building blocks such as softmax, masking, and schedulers are still imported
from ``flash_attn.cute`` to keep this extraction small.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import torch

import cutlass
import cutlass.cute as cute

from ._bootstrap import ensure_repo_flash_attn_cute

ensure_repo_flash_attn_cute()

from flash_attn.cute.cache_utils import get_jit_cache
from flash_attn.cute.cute_dsl_utils import to_cute_tensor
from flash_attn.cute.testing import is_fake_mode

from .kernel import FlashAttentionForwardSm90

if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
    from flash_attn.cute import cute_dsl_ptxas  # noqa: F401

    cute_dsl_ptxas.patch()


@dataclass(frozen=True)
class FwdConfig:
    m_block_size: int
    n_block_size: int
    mma_pv_is_rs: bool
    intra_wg_overlap: bool


def _parse_arch_str(arch_str: str) -> int:
    match = re.match(r"^(?:sm_?|SM_?)?(\d+)(\d)([af]?)$", arch_str)
    if not match:
        raise ValueError(f"Invalid arch format: {arch_str}")
    major, minor, _ = match.groups()
    return int(major) * 10 + int(minor)


@lru_cache(maxsize=None)
def _get_device_arch() -> int:
    arch_override = os.environ.get("FLASH_ATTENTION_ARCH", None)
    if arch_override is not None:
        return _parse_arch_str(arch_override)
    if is_fake_mode():
        return 90
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + int(minor)


def _validate_head_dims(head_dim: int, head_dim_v: int, alignment: int) -> None:
    is_sm90_range = 8 <= head_dim <= 256 and 8 <= head_dim_v <= 256
    assert is_sm90_range and head_dim % alignment == 0 and head_dim_v % alignment == 0, (
        f"(head_dim, head_dim_v)=({head_dim}, {head_dim_v}) is not supported in this "
        f"SM90 extraction. Both dimensions must be in [8, 256] and divisible by {alignment}."
    )


def _tile_size_fwd_sm90(head_dim: int, head_dim_v: int, is_causal: bool) -> FwdConfig:
    """Subset of the official SM90 tile-size heuristic for dense attention."""
    if head_dim <= 64:
        return FwdConfig(192, 128, True, True)
    if head_dim <= 96:
        return FwdConfig(192, 128 if is_causal else 144, False, True)
    if head_dim <= 128:
        return FwdConfig(128, 128, True, True)
    if head_dim <= 192:
        return FwdConfig(128, 128 if head_dim_v <= 128 else 112, True, True)
    return FwdConfig(128, 80, True, True)


def _maybe_contiguous(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous() if x.stride(-1) != 1 else x


def _validate_tensor(
    t: torch.Tensor,
    name: str,
    expected_shape: tuple[int, ...],
    expected_dtype: torch.dtype,
    expected_device: torch.device,
) -> None:
    assert t.shape == expected_shape, f"{name} shape {tuple(t.shape)} != expected {expected_shape}"
    assert t.dtype == expected_dtype, f"{name} dtype {t.dtype} != expected {expected_dtype}"
    assert t.device == expected_device, f"{name} device {t.device} != expected {expected_device}"
    if not is_fake_mode():
        assert t.is_cuda, f"{name} must be on CUDA"


_TORCH_TO_CUTE_DTYPE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}


@lru_cache(maxsize=1)
def _source_fingerprint() -> str:
    """Include this extracted source in the JIT cache key."""
    h = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).with_name("kernel.py")):
        data = path.read_bytes()
        h.update(path.name.encode())
        h.update(len(data).to_bytes(8, "little"))
        h.update(data)
    return h.hexdigest()


def flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    return_lse: bool = False,
    pack_gqa: Optional[bool] = None,
    tile_mn: Optional[Tuple[int, int]] = None,
    num_threads: int = 384,
    mma_pv_is_rs: Optional[bool] = None,
    intra_wg_overlap: Optional[bool] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run the extracted H100 forward kernel.

    Args:
        q, k, v: CUDA tensors shaped ``(batch, seqlen, heads, dim)``.
        causal: Use causal masking. Sliding-window/local masking is intentionally
            not included in this extraction.
        return_lse: Return the log-sum-exp tensor used by the full implementation
            for backward.
        pack_gqa: Keep the official packed-GQA path when Q heads > KV heads.
        tile_mn: Optional manual ``(tile_m, tile_n)`` override.
    """
    assert q.ndim == k.ndim == v.ndim == 4, "q, k, v must be 4D tensors"
    q, k, v = [_maybe_contiguous(t) for t in (q, k, v)]

    batch_size, seqlen_q, num_head, head_dim = q.shape
    batch_k, seqlen_k, num_head_kv, head_dim_k = k.shape
    batch_v, seqlen_v, num_head_v, head_dim_v = v.shape
    assert batch_size == batch_k == batch_v, "q, k, v batch sizes must match"
    assert seqlen_k == seqlen_v, "k and v sequence lengths must match"
    assert num_head_kv == num_head_v, "k and v head counts must match"
    assert head_dim == head_dim_k, "q and k head dimensions must match"
    assert num_head % num_head_kv == 0, "Q heads must be divisible by KV heads"
    assert q.dtype in _TORCH_TO_CUTE_DTYPE, "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype, "q, k, v must have the same dtype"

    if not is_fake_mode():
        assert q.is_cuda and k.is_cuda and v.is_cuda, "q, k, v must be CUDA tensors"

    arch = _get_device_arch()
    assert arch // 10 == 9, (
        f"hopper_fwd_minimal only selects the SM90/H100 kernel, got compute capability {arch}"
    )

    alignment = 16 // q.element_size()
    _validate_head_dims(head_dim, head_dim_v, alignment)

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1

    base_cfg = _tile_size_fwd_sm90(head_dim, head_dim_v, causal)
    if tile_mn is None:
        fwd_cfg = base_cfg
    else:
        fwd_cfg = FwdConfig(tile_mn[0], tile_mn[1], base_cfg.mma_pv_is_rs, base_cfg.intra_wg_overlap)
    tile_m, tile_n = fwd_cfg.m_block_size, fwd_cfg.n_block_size
    if mma_pv_is_rs is None:
        mma_pv_is_rs = fwd_cfg.mma_pv_is_rs
    if intra_wg_overlap is None:
        intra_wg_overlap = fwd_cfg.intra_wg_overlap

    out_shape = (batch_size, seqlen_q, num_head, head_dim_v)
    lse_shape = (batch_size, num_head, seqlen_q)
    if out is None:
        out = torch.empty(out_shape, dtype=q.dtype, device=q.device)
    else:
        _validate_tensor(out, "out", out_shape, q.dtype, q.device)
    if lse is None:
        lse = torch.empty(lse_shape, dtype=torch.float32, device=q.device) if return_lse else None
    else:
        _validate_tensor(lse, "lse", lse_shape, torch.float32, q.device)

    dtype = _TORCH_TO_CUTE_DTYPE[q.dtype]
    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compile_key = (
        _source_fingerprint(),
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        lse is None,
        tile_m,
        tile_n,
        num_threads,
        pack_gqa,
        mma_pv_is_rs,
        intra_wg_overlap,
        arch,
    )

    if compile_key not in flash_attn_forward.compile_cache:
        q_tensor, k_tensor, v_tensor, o_tensor = [to_cute_tensor(t) for t in (q, k, v, out)]
        lse_tensor = to_cute_tensor(lse, assumed_align=4) if lse is not None else None
        fa_fwd = FlashAttentionForwardSm90(
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            is_causal=causal,
            is_local=False,
            pack_gqa=pack_gqa,
            tile_m=tile_m,
            tile_n=tile_n,
            num_stages=2,
            num_threads=num_threads,
            Q_in_regs=False,
            intra_wg_overlap=intra_wg_overlap,
            mma_pv_is_rs=mma_pv_is_rs,
            score_mod=None,
            mask_mod=None,
            has_aux_tensors=False,
            q_subtile_factor=None,
            paged_kv_non_tma=False,
        )
        flash_attn_forward.compile_cache[compile_key] = cute.compile(
            fa_fwd,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            softmax_scale,
            None,  # cu_seqlens_q
            None,  # cu_seqlens_k
            None,  # seqused_q
            None,  # seqused_k
            None,  # page_table
            None,  # window_size_left
            None,  # window_size_right
            None,  # learnable_sink
            None,  # block sparse tensors
            None,  # aux tensors
            current_stream,
            options="--enable-tvm-ffi",
        )

    if not is_fake_mode():
        flash_attn_forward.compile_cache[compile_key](
            q.detach(),
            k.detach(),
            v.detach(),
            out.detach(),
            lse,
            softmax_scale,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    return out, lse


flash_attn_forward.compile_cache = get_jit_cache("hopper_fwd_minimal")
