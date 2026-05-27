from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal


def find_local_repo_root() -> Path:
    path = Path(__file__).resolve()
    for candidate in (path.parent, *path.parents):
        if (candidate / "flash_attn").is_dir() and (candidate / "hopper_fwd_minimal").is_dir():
            return candidate
    return Path.cwd()


LOCAL_REPO_ROOT = find_local_repo_root()
REMOTE_REPO_ROOT = "/root/flash-attention"

app = modal.App("fa4-hopper-fwd-minimal")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential", "git", "ninja-build")
    .run_commands(
        "python -m pip install --upgrade pip",
        "python -m pip install --index-url https://download.pytorch.org/whl/cu124 'torch==2.5.1+cu124'",
        "python -m pip install einops pytest 'nvidia-cutlass-dsl>=4.4.1' apache-tvm-ffi 'quack-kernels>=0.2.10' cuda-python",
    )
    .add_local_dir(str(LOCAL_REPO_ROOT), remote_path=REMOTE_REPO_ROOT)
)


@app.function(gpu="H100", image=image, timeout=60 * 60 * 2)
def run_forward(
    *,
    fake_compile: bool = True,
    actual_run: bool = True,
    causal: bool = False,
    head_dim: int = 64,
    seqlen: int = 128,
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REMOTE_REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    env["FLASH_ATTENTION_ARCH"] = "sm_90a"
    env["CUTE_DSL_ARCH"] = "sm_90a"
    env.setdefault("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "1")

    if fake_compile:
        compile_env = env.copy()
        compile_env["FLASH_ATTENTION_FAKE_TENSOR"] = "1"
        subprocess.run(
            [
                sys.executable,
                "hopper_fwd_minimal/scripts/compile_forward.py",
                "-q",
            ],
            cwd=REMOTE_REPO_ROOT,
            env=compile_env,
            check=True,
        )

    if actual_run:
        cmd = [
            sys.executable,
            "hopper_fwd_minimal/scripts/run_forward.py",
            "--seqlen-q",
            str(seqlen),
            "--seqlen-k",
            str(seqlen),
            "--head-dim",
            str(head_dim),
        ]
        if causal:
            cmd.append("--causal")
        subprocess.run(cmd, cwd=REMOTE_REPO_ROOT, env=env, check=True)


@app.local_entrypoint()
def main(
    fake_compile: bool = True,
    actual_run: bool = True,
    causal: bool = False,
    head_dim: int = 64,
    seqlen: int = 128,
) -> None:
    run_forward.remote(
        fake_compile=fake_compile,
        actual_run=actual_run,
        causal=causal,
        head_dim=head_dim,
        seqlen=seqlen,
    )
