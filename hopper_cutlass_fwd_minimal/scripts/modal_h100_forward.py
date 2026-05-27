from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal


def find_local_project_root() -> Path:
    path = Path(__file__).resolve()
    for candidate in (path.parent, *path.parents):
        if (candidate / "setup.py").is_file() and (candidate / "csrc").is_dir():
            return candidate
    return Path.cwd()


LOCAL_PROJECT_ROOT = find_local_project_root()
REMOTE_REPO_ROOT = "/root/flash-attention"
REMOTE_PROJECT_ROOT = f"{REMOTE_REPO_ROOT}/hopper_cutlass_fwd_minimal"
CUTLASS_COMMIT = "7127592069c2fe01b041e174ba4345ef9b279671"

app = modal.App("fa3-hopper-cutlass-fwd-minimal")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential", "git", "ninja-build")
    .run_commands(
        "python -m pip install --upgrade pip",
        "python -m pip install --index-url https://download.pytorch.org/whl/cu124 'torch==2.5.1+cu124'",
        "python -m pip install packaging ninja pytest wheel numpy",
    )
    .add_local_dir(
        str(LOCAL_PROJECT_ROOT),
        remote_path=REMOTE_PROJECT_ROOT,
        ignore=[
            "**/__pycache__/**",
            "**/*.pyc",
            "**/*.so",
            "**/*.o",
            "**/*.egg-info/**",
            ".pytest_cache/**",
            "build/**",
            "dist/**",
        ],
    )
)


@app.function(gpu="H100", image=image, timeout=60 * 60 * 2)
def run_forward(causal: bool = False, seqlen: int = 128) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REMOTE_REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    env.setdefault("MAX_JOBS", "4")
    env.setdefault("NVCC_THREADS", "2")
    env.setdefault("CC", "gcc")
    env.setdefault("CXX", "g++")

    cutlass_header = Path(REMOTE_REPO_ROOT) / "csrc" / "cutlass" / "include" / "cutlass" / "cutlass.h"
    if not cutlass_header.exists() and (Path(REMOTE_REPO_ROOT) / ".git").exists():
        subprocess.run(
            ["git", "submodule", "update", "--init", "csrc/cutlass"],
            cwd=REMOTE_REPO_ROOT,
            env=env,
            check=False,
        )
    if not cutlass_header.exists():
        cutlass_dir = Path(REMOTE_REPO_ROOT) / "csrc" / "cutlass"
        cutlass_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init"], cwd=cutlass_dir, env=env, check=True)
        subprocess.run(
            ["git", "fetch", "--depth", "1", "https://github.com/NVIDIA/cutlass.git", CUTLASS_COMMIT],
            cwd=cutlass_dir,
            env=env,
            check=True,
        )
        subprocess.run(
            ["git", "checkout", "--detach", "FETCH_HEAD"],
            cwd=cutlass_dir,
            env=env,
            check=True,
        )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "-v",
            "-e",
            "hopper_cutlass_fwd_minimal",
        ],
        cwd=REMOTE_REPO_ROOT,
        env=env,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "hopper_cutlass_fwd_minimal/scripts/run_forward.py",
            "--seqlen-q",
            str(seqlen),
            "--seqlen-k",
            str(seqlen),
            *(["--causal"] if causal else []),
        ],
        cwd=REMOTE_REPO_ROOT,
        env=env,
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "hopper_cutlass_fwd_minimal/tests/test_forward.py"],
        cwd=REMOTE_REPO_ROOT,
        env=env,
        check=True,
    )


@app.local_entrypoint()
def main(causal: bool = False, seqlen: int = 128) -> None:
    run_forward.remote(causal=causal, seqlen=seqlen)
