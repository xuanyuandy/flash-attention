# Hopper CUTLASS Forward Minimal

这是一个基于现有 `hopper/` C++ CUTLASS 框架抽出来的最小 FlashAttention forward extension。

## 当前范围

保留：

- H100 / SM90a
- CUTLASS C++ Hopper forward kernel
- fixed-length dense MHA: `q/k/v = (batch, seqlen, heads, 64)`
- `torch.bfloat16`
- causal / non-causal
- 返回 `(out, lse)`

暂不迁移：

- backward
- GQA / MQA / PackGQA
- fp16 / fp8
- head_dim 96 / 128 / 192 / 256
- varlen / paged KV / SplitKV / softcap / local window / append KV / rotary

## 结构

- `csrc/hopper/`: 从原工程 `hopper/` 复制出的 header 依赖。
- `csrc/flash_fwd_minimal.cpp`: PyTorch extension wrapper，直接设置 `Flash_fwd_params` 并调用最小实例化。
- `csrc/flash_fwd_hdim64_bf16_sm90.cu`: 唯一的 CUDA template instantiation。
- `setup.py`: 只编译上述两个源文件。
- `scripts/modal_h100_forward.py`: 在 Modal H100 上安装并验证。

## Modal H100 验证

```bash
modal run hopper_cutlass_fwd_minimal/scripts/modal_h100_forward.py
modal run hopper_cutlass_fwd_minimal/scripts/modal_h100_forward.py --causal --seqlen 128
```

脚本会在远端补齐 `csrc/cutlass`，安装这个最小 extension，然后运行 forward 并和 PyTorch reference 对比。
