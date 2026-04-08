"""
Usage:
    # 普通运行
    modal run run_cuda.py --file 0-tma-single-load.cu

    # 调试模式: 生成 PTX / SASS / lineinfo，产物保存到 Volume
    modal run run_cuda.py --file 0-tma-single-load.cu --debug

    # 自定义编译选项
    modal run run_cuda.py --file 0-tma-single-load.cu --extra-flags="-DDEBUG"

    # 拉取调试产物到本地，提前创建路径
    modal volume get tma-lab-artifacts / ./debug-output/
"""

import modal
import subprocess
import os
import glob

# ---------- image ----------
cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu24.04",
        add_python="3.12",
    )
    .add_local_dir(
        local_path=".",
        remote_path="/workspace",
        ignore=lambda path: path.suffix not in (".cu", ".cuh", ".h", ".hpp"),
    )
)

app = modal.App("tma-lab", image=cuda_image)

# ---------- 持久 Volume: 保存 PTX / SASS 等调试产物 ----------
vol = modal.Volume.from_name("tma-lab-artifacts", create_if_missing=True)


@app.function(gpu="H100", timeout=120, volumes={"/artifacts": vol})
def compile_and_run(file: str, extra_flags: str = "", debug: bool = False):
    """编译并运行指定的 .cu 文件"""
    os.chdir("/workspace")

    # 调试产物输出目录（在 Volume 中按文件名组织）
    stem = file.rsplit(".", 1)[0]          # e.g. "0-tma-single-load"
    artifact_dir = f"/artifacts/{stem}"
    os.makedirs(artifact_dir, exist_ok=True)

    binary = f"{artifact_dir}/a.out"

    # ------ 构造编译命令 ------
    debug_flags = ""
    if debug:
        # --keep     保留所有中间文件 (.ptx, .cubin, 等)
        # --ptx      额外单独生成 .ptx
        # -lineinfo  在 SASS 中嵌入源码行号映射
        # -src-in-ptx 在 PTX 中嵌入源码注释
        # --keep-dir  指定中间文件输出位置
        debug_flags = (
            f"--keep --keep-dir={artifact_dir} "
            f"-lineinfo -src-in-ptx "
        )

    cmd = f"nvcc -arch=sm_90 -lcuda {debug_flags}{extra_flags} {file} -o {binary}"
    print(f"[compile] {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print("===== COMPILE ERROR =====")
        print(r.stderr)
        return
    if r.stderr:
        print(f"[warnings]\n{r.stderr}")

    # ------ 额外生成独立 PTX (方便阅读) ------
    if debug:
        ptx_cmd = f"nvcc -arch=sm_90 -ptx -src-in-ptx {extra_flags} {file} -o {artifact_dir}/{stem}.ptx"
        print(f"[ptx] {ptx_cmd}")
        subprocess.run(ptx_cmd, shell=True, capture_output=True, text=True)

        # cuobjdump 反汇编 SASS
        sass_cmd = f"cuobjdump -sass {binary} > {artifact_dir}/{stem}.sass"
        print(f"[sass] {sass_cmd}")
        subprocess.run(sass_cmd, shell=True, capture_output=True, text=True)

    # ------ 运行 ------
    print(f"\n[run] {binary}")
    r = subprocess.run(binary, shell=True, capture_output=True, text=True)
    print(r.stdout)
    if r.stderr:
        print(f"[stderr]\n{r.stderr}")
    if r.returncode != 0:
        print(f"[exit code] {r.returncode}")

    # ------ 列出产物 ------
    if debug:
        artifacts = glob.glob(f"{artifact_dir}/*")
        print(f"\n[artifacts] saved {len(artifacts)} files to volume 'tma-lab-artifacts/{stem}/':")
        for f in sorted(artifacts):
            size = os.path.getsize(f)
            print(f"  {os.path.basename(f):40s}  {size:>8,} bytes")
        # 确保 Volume 写入持久化
        vol.commit()
        print(f"\n[tip] 拉到本地:\n  modal volume get tma-lab-artifacts {stem}/ ./{stem}-debug/")


@app.local_entrypoint()
def main(file: str, extra_flags: str = "", debug: bool = False):
    compile_and_run.remote(file, extra_flags, debug)