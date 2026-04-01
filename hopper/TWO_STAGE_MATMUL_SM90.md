# SM90 两阶段 Matmul：QK→PV 计算与 Layout 转换详解

> 基于 FA3 `mainloop_fwd_sm90_tma_gmma_ws.hpp::mma()` 提取，重点解析
> `convert_layout_acc_Aregs` 如何将 Gemm-I 的 C 寄存器布局重映射为
> Gemm-II 的 RS A 寄存器布局，实现 P 矩阵零 smem 中转。

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [实现文件与编译](#2-实现文件与编译)
3. [核心参数与类型](#3-核心参数与类型)
4. [Smem 布局设计](#4-smem-布局设计)
5. [Gemm-I：QK 矩阵乘（SS 模式）](#5-gemm-i-qk-矩阵乘ss-模式)
6. [Layout 转换：convert_layout_acc_Aregs](#6-layout-转换convert_layout_acc_aregs)
7. [Gemm-II：PV 矩阵乘（RS 模式）](#7-gemm-ii-pv-矩阵乘rs-模式)
8. [完整数据流](#8-完整数据流)

---

## 1 背景与动机

Flash Attention 的前向传播核心是两次连续矩阵乘：

```
S[M, N]   = Q[M, K]  × K^T[K, N]     // Gemm-I
P[M, N]   = softmax(S)                // 逐行 softmax（类型转换同步完成）
O[M, D]   = P[M, N]  × V[N, D]       // Gemm-II
```

**性能关键问题**：P 矩阵如何从 Gemm-I 传递给 Gemm-II？

| 方案 | 路径 | 代价 |
|------|------|------|
| SS 模式 | 寄存器 → smem(P) → smem 描述符 → Gemm-II | smem 读写带宽 + barrier 同步 |
| **RS 模式** | **寄存器 → layout 重映射 → Gemm-II** | **仅寄存器操作，零 smem** |

RS 模式（Register Source）要求 P 的寄存器布局与 wgmma 的 A 操作数规范严格匹配。
`convert_layout_acc_Aregs` 正是完成这一匹配的关键函数。

---

## 2 实现文件与编译

```
hopper/
├── two_stage_matmul_sm90.cu       # 核函数实现（本文档对应代码）
└── compile_two_stage_matmul.sh    # 编译脚本
```

### 编译命令

```bash
/usr/local/cuda/bin/nvcc           \
  -arch=sm_90a                     \  # Hopper，必须用 90a 才能启用 wgmma/TMA
  -std=c++17                       \
  -O3                              \
  --expt-relaxed-constexpr         \  # 允许 device 调用 constexpr host 函数
                                      # (std::min/max in utils.h)，FA3 全系必需
  -DTWO_STAGE_MATMUL_TEST          \  # 启用 §8 测试 main；去掉则只编译 kernel
  -I../csrc/cutlass/include        \  # CUTLASS 头文件（submodule）
  -I.                              \  # hopper/ 目录（utils.h 等）
  two_stage_matmul_sm90.cu         \
  -o two_stage_matmul_sm90
```

> **`-arch=sm_90a` vs `-arch=sm_90`**  
> `sm_90a` 解锁 SM90 专属扩展指令（wgmma、TMA、barrier 等），
> 普通 `sm_90` 不包含这些指令，会导致编译错误。

---

## 3 核心参数与类型

```cpp
// 矩阵维度
static constexpr int kBlockM   = 64;   // M tile = wgmma.m64（单 warp group）
static constexpr int kBlockN   = 128;  // N tile (seqlen_k tile / K_pv)
static constexpr int kHeadDim  = 64;   // Gemm-I  K 维（Q/K head dim）
static constexpr int kHeadDimV = 64;   // Gemm-II N 维（V  head dim）

using Element      = cutlass::half_t;  // fp16
using ElementAccum = float;
```

### TiledMma 类型

```cpp
// ── Gemm-I (QK): SS 模式 ─────────────────────────────────────────────────
// wgmma.m64n128k16_f32.f16.f16
// A=Q(smem 描述符), B=K(smem 描述符), C=S(寄存器 float32)
using TileShape_MNK_QK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
//                              64            128            64
using AtomLayoutQK = Layout<Shape<_1, _1, _1>>;  // 单原子，1 warp group
using TiledMmaQK   = decltype(cute::make_tiled_mma(
    cute::GMMA::ss_op_selector<Element, Element, ElementAccum,
                               TileShape_MNK_QK>(),
    AtomLayoutQK{}));

// ── Gemm-II (PV): RS 模式 ────────────────────────────────────────────────
// wgmma.m64n64k16_f32.f16.f16（RS: A 来自寄存器）
// A=P(寄存器 fp16), B=Vt(smem 描述符), C=O(寄存器 float32)
using TileShape_MNK_PV = Shape<Int<kBlockM>, Int<kHeadDimV>, Int<kBlockN>>;
//                              64            64               128
using AtomLayoutPV = Layout<Shape<_1, _1, _1>>;
using TiledMmaPV   = decltype(cute::make_tiled_mma(
    cute::GMMA::rs_op_selector<Element, Element, ElementAccum,
                               TileShape_MNK_PV,
                               GMMA::Major::K,    // A(P)  K 维连续
                               GMMA::Major::K>(), // B(Vt) K 维连续
    AtomLayoutPV{}));

// 编译期检查: 均恰好需要 128 线程 (1 warp group)
static_assert(size(TiledMmaQK{}) == 128);
static_assert(size(TiledMmaPV{}) == 128);
```

---

## 4 Smem 布局设计

### `ss_smem_selector` 的作用

```cpp
ss_smem_selector<Major, Element, N_rows, K_cols>
```

返回带最优 **swizzle** 的 smem layout，消除 GMMA 的 shared memory bank conflict。
- `Major::K`：K_cols 维连续（row-major 视角）

```cpp
// sQ: [kBlockM=64, kHeadDim=64]，kHeadDim 连续
using SmemLayoutAtomQ = decltype(
    ss_smem_selector<GMMA::Major::K, Element, Int<kBlockM>, Int<kHeadDim>>());
using SmemLayoutQ = decltype(
    tile_to_shape(SmemLayoutAtomQ{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));

// sK: [kBlockN=128, kHeadDim=64]，kHeadDim 连续
using SmemLayoutAtomK = decltype(
    ss_smem_selector<GMMA::Major::K, Element, Int<kBlockN>, Int<kHeadDim>>());
using SmemLayoutK = decltype(
    tile_to_shape(SmemLayoutAtomK{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));

// sVt: [kHeadDimV=64, kBlockN=128]，kBlockN 连续
// V 已转置为 Vt[kHeadDimV, kBlockN]；
// GMMA B 操作数形状 = [N_pv, K_pv] = [kHeadDimV, kBlockN]，K_pv=kBlockN 连续
using SmemLayoutAtomVt = decltype(
    ss_smem_selector<GMMA::Major::K, Element, Int<kHeadDimV>, Int<kBlockN>>());
using SmemLayoutVt = decltype(
    tile_to_shape(SmemLayoutAtomVt{}, Shape<Int<kHeadDimV>, Int<kBlockN>>{}));
```

> **V 的转置说明**  
> 标准 V 在全局内存为 `[kBlockN, kHeadDimV]` row-major。  
> 本实现接受已转置的 `Vt[kHeadDimV, kBlockN]` row-major 作为输入，  
> 与 FA3 `TmaMajorV = GMMA::Major::K` 路径一致，避免核函数内额外转置开销。

---

## 5 Gemm-I：QK 矩阵乘（SS 模式）

### Fragment 分配

```cpp
auto thr_mma_qk = tiled_mma_qk.get_thread_slice(thread_idx);

Tensor tSrQ = thr_mma_qk.partition_fragment_A(sQ);  // smem 描述符（非寄存器数据）
Tensor tSrK = thr_mma_qk.partition_fragment_B(sK);  // smem 描述符
Tensor tSrS = partition_fragment_C(tiled_mma_qk,
                  select<0, 1>(TileShape_MNK_QK{})); // float32 寄存器累加器
```

**`tSrS` 的 layout：**

```
tSrS.layout() = ((2, 2, 16), 1, 1)
                 ──────────  ─  ─
                 mode-0      M  N
                             原子数
每线程元素数: 2 × 2 × 16 × 1 × 1 = 64 个 float32
```

### 执行

```cpp
// flash::gemm<zero_init=true, wg_wait=0>:
//   warpgroup_arrive()
//   → loop k_block: wgmma.mma_async(tSrQ, tSrK[:,k], tSrS)
//   → warpgroup_commit_batch()
//   → warpgroup_wait<0>()     ← 等待全部完成
//   → warpgroup_fence_operand(tSrS)
flash::gemm<true, 0>(tiled_mma_qk, tSrQ, tSrK, tSrS);
```

---

## 6 Layout 转换：convert_layout_acc_Aregs

这是整个两阶段 Matmul 的核心。

### 6.1 Gemm-I C 寄存器的物理布局

SM90 `wgmma.m64n128k16` 输出的 C 寄存器，每线程 64 个 `float32`，
CuTe 将其表示为 `((2, 2, 16), 1, 1)`。

各 mode 的物理含义：

```
mode<0,0> = 2  ──  M 方向：每线程覆盖连续 2 行（stride=1）
mode<0,1> = 2  ──  M 方向：跨 warp 的另 2 行分组（stride=2）
mode<0,2> = 16 ──  N 方向：N/8 = 128/8 = 16 个"8列组"
MMA_M     = 1  ──  M 原子数（单 warp group，无重复）
MMA_N     = 1  ──  N 原子数
```

用矩阵示意（每格 = 1 个 `float32` 寄存器）：

```
         列组0  列组1  列组2  ...  列组15
        [0..7][8..15][16..23]   [120..127]
行对(r0,r1)  [●][●][●]     ...     [●]   } mode<0,1>=0
行对(r2,r3)  [●][●][●]     ...     [●]   } mode<0,1>=1
              ←──────── mode<0,2>=16 ───────→
```

### 6.2 Gemm-II RS A 操作数所需布局

`wgmma.m64n64k16` RS 模式（K_pv=128 → 8 次 k16 迭代）：

```
每次 k16 迭代，每线程需提供: A[m=64, k=16] / 128线程 = 8 个 fp16
共 8 次迭代 → 总计 64 个 fp16（体积等于 64 float32）

所需 layout: (per_iter_regs, MMA_M, k_iter_count)
           = ((2, 2, 2),     1,     8)
              ─────────             ─
              8 fp16/iter           k_iter 维
```

### 6.3 变换步骤（FP16 路径）

```cpp
// utils.h: convert_layout_acc_Aregs<MMA_Traits>(acc_layout)
// FP16 路径: sizeof(ValTypeA) == 2

auto l = logical_divide(get<0,2>(acc_layout), Tile<_2>{});
//        输入: get<0,2> = Int<16>
//        输出: l = Layout<Shape<Shape<_2, _8>>>
//        含义: 将 16 个列组拆为 (内层2, 外层8)
//              内层 2: 每 k16 迭代消耗 2 个列组（2×8=16列 = 1个k16宽度）
//              外层 8: k_iter 数 = K_pv/16 = 128/16 = 8

return make_layout(
    make_layout(get<0,0>(acc_layout),  // 2  (M行对)
                get<0,1>(acc_layout),  // 2  (跨warp行)
                get<0,0>(l)),          // 2  (k16内列分组, 并入mode-0)
    get<1>(acc_layout),                // 1  (MMA_M, 不变)
    coalesce(make_layout(
        get<0,1>(l),                   // 8  (k_iter, 提升为mode-2)
        get<2>(acc_layout)))           // 1  (MMA_N) → coalesce → 8
);
// 输出: ((2, 2, 2), 1, 8)   仍是 64 个元素，mode-2 = k_iter 维已解耦
```

### 6.4 变换前后对比

```
变换前 (Gemm-I C regs):   ((2, 2, 16), 1, 1)
                                     ^^
                                     16 个列组全部混在 mode<0,2>
                                     k_iter 与列组没有显式分离

变换后 (Gemm-II A regs):  ((2, 2,  2), 1, 8)
                                     ^      ^
                                     ┆      └── mode-2 = 8 k_iter
                                     └────────── 每迭代内的列分组 (2个)
                                                 并入 mode-0 → (2,2,2)=8 fp16/iter
```

### 6.5 为什么同一批寄存器能"两用"？

关键对应关系：

```
Gemm-I 输出 S[m, n] 中：
  n 方向的 16 个列组，每组 8 列
  相邻的 2 个列组 = 16 列 = 1 个 k16 宽度

Gemm-II RS A P[m, k] 中：
  k 方向按 k16 分段，每段 16 个 k 维元素

因此：
  S 的"第 k16 段的 16 列" ←→ P 的"第 k16 次迭代的 A 寄存器"
  在物理寄存器上一一对应，convert_layout_acc_Aregs 只是将此对应关系显式化
```

### 6.6 代码实现

```cpp
// ① 用同一物理寄存器构造新 layout（零拷贝）
Tensor tOrP_acc = make_tensor(
    tSrS.data(),                                             // 共享物理寄存器
    flash::convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout()));
// tOrP_acc: layout=((2,2,2),1,8), dtype=float32, 64 元素

// ② 分配 fp16 目标寄存器（相同 layout，不同 dtype）
Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
// tOrP: layout=((2,2,2),1,8), dtype=fp16, 64 元素

// ③ 类型转换: float32 → fp16
//    FragmentSize=2: 以 Array<float32,2>→Array<fp16,2> 向量化，
//    底层使用 CUDA PRMT 指令，编译为 2个 F2F 指令/组
flash::convert_type_out(tOrP_acc, tOrP);
```

### 6.7 验证 layout 等价性

```
TiledMmaPV::partition_fragment_A(sP).layout()
    == convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout())  // 设计保证成立

FA3 源码印证 (mainloop_fwd_sm90_tma_gmma_ws.hpp:1157):
    Tensor tOrP_acc = make_tensor(tSrS.data(),
        flash::convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout()));
    Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
    convert_type_out(tOrP_acc, tOrP);
    flash::gemm<...>(tiled_mma_pv, tOrP, tOrV(...), tOrO);  // tOrP 直接作为 A
```

---

## 7 Gemm-II：PV 矩阵乘（RS 模式）

### RS 模式的 `flash::gemm` 执行流程

```cpp
flash::gemm<true, 0>(tiled_mma_pv, tOrP, tOrVt, tOrO);
```

RS 模式检测（`utils.h::gemm`）：

```cpp
constexpr bool Is_RS =
    !cute::is_base_of<cute::GMMA::DescriptorIterator,
                      typename TiledMma::FrgTypeA>::value;
// SS 模式: FrgTypeA = DescriptorIterator → Is_RS = false
// RS 模式: FrgTypeA = ArrayEngine<fp16>  → Is_RS = true
```

RS 路径额外步骤：

```cpp
// RS 独有: 确保 P 寄存器对 GMMA 硬件可见，防止编译器错误调度
warpgroup_fence_operand(tOrP);

warpgroup_arrive();
tiled_mma_pv.accumulate_ = GMMA::ScaleOut::Zero;  // zero_init=true

for (int k = 0; k < K_pv/16; ++k) {               // 8 次 k16 迭代
    cute::gemm(tiled_mma_pv,
               tOrP(_, _, k),    // ← 按 mode-2 切片，8 fp16/次
               tOrVt(_, _, k),   // ← smem 描述符切片
               tOrO);
    tiled_mma_pv.accumulate_ = GMMA::ScaleOut::One;
}

warpgroup_commit_batch();
warpgroup_wait<0>();
warpgroup_fence_operand(tOrO);
```

`tOrP(_, _, k)` 能正确切片的前提正是 `convert_layout_acc_Aregs` 将 k_iter 解耦为 mode-2。

---

## 8 完整数据流

```
全局内存
  Q[64,64] ──cp.async──► sQ[64,64] (swizzled smem)
  K[128,64]──cp.async──► sK[128,64](swizzled smem)
  Vt[64,128]─cp.async──► sVt[64,128](swizzled smem)
                ↓ cp_async_fence + cp_async_wait<0> + __syncthreads

Gemm-I (WGMMA SS)
  tSrQ(descriptor) × tSrK(descriptor) → tSrS(64×float32, layout=((2,2,16),1,1))

                ↓ convert_layout_acc_Aregs<TiledMmaPV>

  tOrP_acc: 同物理寄存器, layout=((2,2,2),1,8), float32  [零拷贝重解释]

                ↓ convert_type_out (float32→fp16, PRMT指令)

  tOrP:     新寄存器, layout=((2,2,2),1,8), fp16          [64 fp16]
            mode-2=8 ↔ 8次k16迭代，mode-0=(2,2,2) ↔ 每迭代8fp16

Gemm-II (WGMMA RS)
  tOrP(fp16, regs) × tOrVt(descriptor) → tOrO(32×float32, layout=((2,2,8),1,1))
  P 矩阵全程在寄存器中，未经 smem 落地

Epilogue
  convert_layout_acc_rowcol(tOrO.layout()) → 按(row,col)逐元素写入 gO[64,64]
```

### 与 FA3 SS 模式对比

| | RS 模式（本实现）| SS 模式（FA3 可选路径）|
|--|--|--|
| P 的传递方式 | 寄存器 → 寄存器 | 寄存器 → smem → 描述符 |
| layout 转换 | `convert_layout_acc_Aregs` | STSM 写 smem |
| 额外同步 | 无 | `fence_view_async_shared` + barrier |
| 适用场景 | kHeadDimV ≤ 128 | kHeadDimV 大（LargeHeadDimV）|
