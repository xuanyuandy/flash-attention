import os
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

USE_FAKE_TENSOR = int(os.getenv("FLASH_ATTENTION_FAKE_TENSOR", "0")) == 1


def _skip_without_h100() -> bool:
    if USE_FAKE_TENSOR:
        return False
    return not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9


if _skip_without_h100():
    pytest.skip("requires H100/SM90 or fake tensor compile mode", allow_module_level=True)


from hopper_fwd_minimal import flash_attn_forward  # noqa: E402
from flash_attn.cute.testing import is_fake_mode, maybe_fake_tensor_mode  # noqa: E402
from hopper_fwd_minimal.hopper_fwd_minimal.reference import attention_ref  # noqa: E402


@pytest.mark.parametrize(
    "seqlen_q,seqlen_k,head_dim,num_heads,num_heads_kv,causal",
    [
        pytest.param(128, 128, 64, 8, 8, False, id="smoke-mha-d64"),
        pytest.param(128, 128, 64, 8, 8, True, id="smoke-mha-d64-causal"),
        pytest.param(256, 256, 128, 8, 2, False, id="smoke-gqa-d128"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_forward_smoke(seqlen_q, seqlen_k, head_dim, num_heads, num_heads_kv, causal, dtype):
    torch.manual_seed(0)
    device = "cuda"
    batch = 2
    q = torch.randn(batch, seqlen_q, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, device=device, dtype=dtype)

    out, lse = flash_attn_forward(q, k, v, causal=causal, return_lse=True)
    assert lse is not None

    if is_fake_mode():
        return

    ref = attention_ref(q, k, v, causal=causal)
    max_abs = (out - ref).abs().max().item()
    ref_scale = ref.abs().max().item()
    assert max_abs <= 5e-2 + 5e-2 * ref_scale
