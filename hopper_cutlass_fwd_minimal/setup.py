from __future__ import annotations

import os
import subprocess
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
CUTLASS_DIR = REPO_ROOT / "csrc" / "cutlass"
CUTLASS_COMMIT = "7127592069c2fe01b041e174ba4345ef9b279671"


def fetch_cutlass_at_pinned_commit() -> None:
    CUTLASS_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=CUTLASS_DIR, check=True)
    subprocess.run(
        ["git", "fetch", "--depth", "1", "https://github.com/NVIDIA/cutlass.git", CUTLASS_COMMIT],
        cwd=CUTLASS_DIR,
        check=True,
    )
    subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=CUTLASS_DIR, check=True)


def ensure_cutlass_headers() -> None:
    header = CUTLASS_DIR / "include" / "cutlass" / "cutlass.h"
    if header.exists():
        return
    if (REPO_ROOT / ".git").exists():
        try:
            subprocess.run(
                ["git", "submodule", "update", "--init", "csrc/cutlass"],
                cwd=REPO_ROOT,
                check=True,
            )
        except Exception:
            pass
    if header.exists():
        return
    CUTLASS_DIR.parent.mkdir(parents=True, exist_ok=True)
    fetch_cutlass_at_pinned_commit()


ensure_cutlass_headers()

feature_args = [
    "-DFLASHATTENTION_DISABLE_BACKWARD",
    "-DFLASHATTENTION_DISABLE_PAGEDKV",
    "-DFLASHATTENTION_DISABLE_SPLIT",
    "-DFLASHATTENTION_DISABLE_APPENDKV",
    "-DFLASHATTENTION_DISABLE_LOCAL",
    "-DFLASHATTENTION_DISABLE_SOFTCAP",
    "-DFLASHATTENTION_DISABLE_PACKGQA",
    "-DFLASHATTENTION_DISABLE_FP16",
    "-DFLASHATTENTION_DISABLE_FP8",
    "-DFLASHATTENTION_DISABLE_VARLEN",
    "-DFLASHATTENTION_DISABLE_CLUSTER",
    "-DFLASHATTENTION_DISABLE_SM8x",
    "-DFLASHATTENTION_DISABLE_HDIM96",
    "-DFLASHATTENTION_DISABLE_HDIM128",
    "-DFLASHATTENTION_DISABLE_HDIM192",
    "-DFLASHATTENTION_DISABLE_HDIM256",
    "-DFLASHATTENTION_DISABLE_HDIMDIFF64",
    "-DFLASHATTENTION_DISABLE_HDIMDIFF192",
]

nvcc_threads = os.getenv("NVCC_THREADS", "2")
nvcc_flags = [
    "--threads",
    nvcc_threads,
    "-O3",
    "-std=c++17",
    "--ftemplate-backtrace-limit=0",
    "--use_fast_math",
    "--resource-usage",
    "-lineinfo",
    "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
    "-DCUTLASS_ENABLE_GDC_FOR_SM90",
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
    "-DNDEBUG",
    "-gencode",
    "arch=compute_90a,code=sm_90a",
]

setup(
    name="hopper_cutlass_fwd_minimal",
    version="0.0.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="hopper_cutlass_fwd_minimal._C",
            sources=[
                str(ROOT / "csrc" / "flash_fwd_minimal.cpp"),
                str(ROOT / "csrc" / "flash_fwd_hdim64_bf16_sm90.cu"),
            ],
            include_dirs=[
                str(ROOT / "csrc" / "hopper"),
                str(CUTLASS_DIR / "include"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"] + feature_args,
                "nvcc": nvcc_flags + feature_args,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
