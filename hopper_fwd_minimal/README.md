# Hopper Forward Minimal

这个目录提取了 FA4/CuTeDSL 的一个典型 H100 forward 分支，方便在 Modal H100 上快速改 kernel、编译、验证。

## 当前范围

保留：

- H100 / SM90 forward
- fixed-length dense attention: `q/k/v = (batch, seqlen, heads, dim)`
- MHA / GQA / MQA
- causal 和 non-causal
- fp16 / bf16
- 可选返回 LSE

暂不迁移：

- backward
- varlen / cu_seqlens / seqused
- SplitKV / FlashDecoding
- paged KV
- block sparsity
- score_mod / mask_mod / aux_tensors
- sliding-window local attention

## 文件结构

- `hopper_fwd_minimal/kernel.py`: 从 `flash_attn/cute/flash_fwd_sm90.py` 复制出的 SM90 forward kernel 主体，后续优先改这里。
- `hopper_fwd_minimal/interface.py`: 瘦身后的 forward-only Python 入口，只分发到 SM90。
- `hopper_fwd_minimal/reference.py`: 小型 PyTorch dense attention 参考实现。
- `tests/test_forward.py`: forward smoke tests，支持 FakeTensor 编译 pass。
- `tests/test_backward.py`: backward 占位测试，后续单独迁移。
- `scripts/run_forward.py`: 在 H100 上实际运行并和 PyTorch reference 对比。
- `scripts/compile_forward.py`: FakeTensor 编译 pass，适合先填充 CuTeDSL cache。
- `scripts/modal_h100_forward.py`: Modal H100 运行入口。

## 本地 H100 运行

从仓库根目录执行：

```bash
export PYTHONPATH=$PWD
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
python -c "from hopper_fwd_minimal import flash_attn_forward; print(flash_attn_forward)"
python hopper_fwd_minimal/scripts/run_forward.py --head-dim 64
python hopper_fwd_minimal/scripts/run_forward.py --head-dim 64 --causal
python -m pytest -q hopper_fwd_minimal/tests/test_forward.py
```

## 先编译再运行

FakeTensor 编译 pass 不分配 GPU tensor，适合先并行预热 cache：

```bash
python hopper_fwd_minimal/scripts/compile_forward.py -q
python hopper_fwd_minimal/scripts/run_forward.py --head-dim 64
```

如果你手动改了 `hopper_fwd_minimal/kernel.py` 或 `interface.py`，这个目录会把自身源码 hash 放进 compile key，避免复用旧 cubin。

## Modal H100

```bash
modal run hopper_fwd_minimal/scripts/modal_h100_forward.py
modal run hopper_fwd_minimal/scripts/modal_h100_forward.py --causal --head-dim 128 --seqlen 256
```

Modal 脚本会把当前仓库复制到镜像中，安装 PyTorch CUDA wheel、CuTeDSL、quack-kernels 等依赖，然后先跑 FakeTensor compile，再做一次真实 forward 校验。

## macOS 本地编辑环境

Apple Silicon / macOS 没有可用的 NVIDIA CUDA toolkit 和 CuTeDSL native runtime wheel。为了让 VSCode/Pylance 能解析 `cutlass`、`cutlass.cute`、`quack` 等 Python 包，本目录提供了 `python_vendor/`，其中包含从 Linux CuTeDSL wheel 抽取的 Python 源码和少量 `cuda-python` stub。

在 `cuda-learn` 环境中挂载本地 vendor：

```bash
/Users/dy/miniconda3/envs/cuda-learn/bin/python -m pip install -e hopper_fwd_minimal/python_vendor
```

这只用于本地补全/跳转/静态检查；真实 CuTeDSL 编译和运行仍以 Modal H100 环境为准。

## 下一步迁移 backward

建议 backward 也按同样方式单独开入口：

1. 先复制 `flash_attn/cute/flash_bwd_sm90.py`。
2. 只保留 fixed-length dense causal/non-causal 分支。
3. 把 preprocess / postprocess 和主 backward kernel 拆成独立测试。
4. 最后再接一个最小 autograd wrapper。
