"""
Usage:
    modal run run_cuda.py --file 0-tma-single-load.cu
    modal run run_cuda.py --file 1-tma-single-store.cu
    modal run run_cuda.py --file 0-tma-single-load.cu --extra-flags="-DDEBUG"
"""

import modal
import subprocess

# ---------- image ----------
cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu24.04",
        add_python="3.12",
    )
    # add_local_dir: 默认运行时挂载（非 bake 进镜像），改文件后无需重建镜像
    .add_local_dir(
        local_path=".",
        remote_path="/workspace",
        ignore=lambda path: path.suffix not in (".cu", ".cuh", ".h", ".hpp"),
    )
)

app = modal.App("tma-lab", image=cuda_image)


@app.function(gpu="H100", timeout=120)
def compile_and_run(file: str, extra_flags: str = ""):
    """编译并运行指定的 .cu 文件"""
    import os
    os.chdir("/workspace")

    binary = "/tmp/a.out"
    cmd = f"nvcc -arch=sm_90 -lcuda {extra_flags} {file} -o {binary}"
    print(f"[compile] {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print("===== COMPILE ERROR =====")
        print(r.stderr)
        return
    if r.stderr:
        print(f"[warnings]\n{r.stderr}")

    print(f"[run] {binary}")
    r = subprocess.run(binary, shell=True, capture_output=True, text=True)
    print(r.stdout)
    if r.stderr:
        print(f"[stderr]\n{r.stderr}")
    if r.returncode != 0:
        print(f"[exit code] {r.returncode}")


@app.local_entrypoint()
def main(file: str, extra_flags: str = ""):
    compile_and_run.remote(file, extra_flags)