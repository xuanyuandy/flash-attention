import os
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _skip_without_h100() -> bool:
    return not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9


if _skip_without_h100():
    pytest.skip("requires H100/SM90 and a built hopper_cutlass_fwd_minimal extension", allow_module_level=True)


from hopper_cutlass_fwd_minimal import forward  # noqa: E402
from hopper_cutlass_fwd_minimal.reference import attention_ref  # noqa: E402


@pytest.mark.parametrize("causal", [False, True])
def test_forward_bf16_hdim64(causal):
    torch.manual_seed(0)
    q = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.bfloat16)

    out, lse = forward(q, k, v, causal, None)
    assert lse.shape == (2, 8, 128)

    ref = attention_ref(q, k, v, causal=causal)
    max_abs = (out - ref).abs().max().item()
    allowed = 5e-2 + 5e-2 * ref.abs().max().item()
    assert max_abs <= allowed
