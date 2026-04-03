/******************************************************************************
 * two_stage_matmul_sm90.cu
 *
 * SM90 两阶段矩阵乘法：从 FA3 mainloop_fwd_sm90_tma_gmma_ws.hpp::mma() 提取
 * 的 QK→PV 核心计算，重点演示 convert_layout_acc_Aregs 的 layout 转换原理。
 *
 * 计算流程:
 *   Gemm-I  : S[M,N]   = Q[M,K]  × K^T[K,N]    (WGMMA SS 模式, float32 累加)
 *   layout  : convert_layout_acc_Aregs<TiledMmaPV>(S.layout())
 *             → 将 wgmma C-寄存器布局重映射为 wgmma RS A-寄存器布局
 *   convert : float32 → fp16   (convert_type_out, 元素数不变)
 *   Gemm-II : O[M,D]   = P[M,N] × Vt[D,N]^T    (WGMMA RS 模式, P 留在寄存器)
 *
 * 输入张量 (全局内存, row-major fp16):
 *   Q   : [kBlockM=64,  kHeadDim=64 ]
 *   K   : [kBlockN=128, kHeadDim=64 ]   (GMMA B 视角: K[N,K], kHeadDim 连续)
 *   Vt  : [kHeadDimV=64, kBlockN=128]   (V 已转置: Vt[D,N], kBlockN 连续)
 *   O   : [kBlockM=64,  kHeadDimV=64]   float32 输出
 *
 * 运行参数: gridDim=(1,1,1), blockDim=(128,1,1)  —— 单个 warp group
 *
 * 编译:
 *   source /path/to/compile_hopper.sh
 *   /usr/local/cuda/bin/nvcc -arch=sm_90a -std=c++17 -O3           \
 *       -I../../csrc/cutlass/include -I.                            \
 *       two_stage_matmul_sm90.cu -o two_stage_matmul_sm90
 ******************************************************************************/

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/builders/sm90_common.inl"

// FA3 工具: gemm(), convert_layout_acc_Aregs(), convert_layout_acc_rowcol(),
//          convert_type_out(), warpgroup_* intrinsics
#include "utils.h"

using namespace cute;

// ============================================================
// §1  编译期参数
// ============================================================
static constexpr int kBlockM   = 64;   // M tile (= wgmma.m64, 单 warp group)
static constexpr int kBlockN   = 128;  // N tile (seqlen_k / K_pv)
static constexpr int kHeadDim  = 64;   // Gemm-I  K 维 (Q/K head dim)
static constexpr int kHeadDimV = 64;   // Gemm-II N 维 (V  head dim)

using Element      = cutlass::half_t;  // fp16
using ElementAccum = float;

// ============================================================
// §2  TiledMma 类型
// ============================================================

// wgmma is warp group level, not thread block level
// Gemm-I (QK): SS 模式 — A=Q(smem), B=K(smem), C=S(寄存器)
//   wgmma.m64n128k16_f32.f16.f16  (Major::K for both A and B)
using TileShape_MNK_QK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
using AtomLayoutQK     = Layout<Shape<_1, _1, _1>>;  // 单 wgmma 原子
using TiledMmaQK = decltype(cute::make_tiled_mma(
    cute::GMMA::ss_op_selector<Element, Element, ElementAccum,
                               TileShape_MNK_QK>(),
    AtomLayoutQK{}));

// Gemm-II (PV): RS 模式 — A=P(寄存器), B=Vt(smem), C=O(寄存器)
//   wgmma.m64n64k16_f32.f16.f16  Major::K for A(P) and B(Vt)
//   与 FA3 中 MmaMajorV=Major::K 路径对应 (V 在 smem 中已转置为 Vt)
using TileShape_MNK_PV = Shape<Int<kBlockM>, Int<kHeadDimV>, Int<kBlockN>>;
using AtomLayoutPV     = Layout<Shape<_1, _1, _1>>;
using TiledMmaPV = decltype(cute::make_tiled_mma(
    cute::GMMA::rs_op_selector<Element, Element, ElementAccum,
                               TileShape_MNK_PV,
                               GMMA::Major::K,   // A(P)  K 连续
                               GMMA::Major::K>(),// B(Vt) K 连续
    AtomLayoutPV{}));

// 编译期检查: 两个 TiledMma 均恰好需要 128 线程 (1 warp group)
static_assert(size(TiledMmaQK{}) == 128,
    "TiledMmaQK must be exactly 1 warp group (128 threads)");
static_assert(size(TiledMmaPV{}) == 128,
    "TiledMmaPV must be exactly 1 warp group (128 threads)");

// ============================================================
// §3  Smem 布局
//
//   ss_smem_selector<Major, Type, N_rows, K_cols>:
//     Major::K → K_cols 维连续 (row-major 视角)，并附加最优 swizzle
//     选择 swizzle 模式以避免 GMMA shared memory bank conflict
// ============================================================

// sQ: [kBlockM=64, kHeadDim=64], kHeadDim 连续
using SmemLayoutAtomQ =
    decltype(cutlass::gemm::collective::detail::ss_smem_selector<
        GMMA::Major::K, Element, Int<kBlockM>, Int<kHeadDim>>());
using SmemLayoutQ =
    decltype(tile_to_shape(SmemLayoutAtomQ{},
                           Shape<Int<kBlockM>, Int<kHeadDim>>{}));

// sK: [kBlockN=128, kHeadDim=64], kHeadDim 连续
using SmemLayoutAtomK =
    decltype(cutlass::gemm::collective::detail::ss_smem_selector<
        GMMA::Major::K, Element, Int<kBlockN>, Int<kHeadDim>>());
using SmemLayoutK =
    decltype(tile_to_shape(SmemLayoutAtomK{},
                           Shape<Int<kBlockN>, Int<kHeadDim>>{}));

// sVt: [kHeadDimV=64, kBlockN=128], kBlockN 连续
//   Vt 是 V 的转置: V[kBlockN, kHeadDimV] → Vt[kHeadDimV, kBlockN]
//   GMMA B 操作数形状为 [N_pv, K_pv] = [kHeadDimV, kBlockN]，
//   Major::K 意味着 K_pv=kBlockN 是连续维
using SmemLayoutAtomVt =
    decltype(cutlass::gemm::collective::detail::ss_smem_selector<
        GMMA::Major::K, Element, Int<kHeadDimV>, Int<kBlockN>>());
using SmemLayoutVt =
    decltype(tile_to_shape(SmemLayoutAtomVt{},
                           Shape<Int<kHeadDimV>, Int<kBlockN>>{}));

// SmemCopyAtomP: 将 QK 结果 (fp16) 写入 smem 时使用的 Copy Atom
// (本实现使用 RS 模式，P 不落盘 smem；此处仅作类型参考，与 FA3 保持一致)
using SmemCopyAtomP = Copy_Atom<cute::SM90_U32x4_STSM_N, Element>;

// ============================================================
// §4  SharedStorage
// ============================================================
struct SharedStorage {
    cute::array_aligned<Element, cosize_v<SmemLayoutQ>>  smem_q;
    cute::array_aligned<Element, cosize_v<SmemLayoutK>>  smem_k;
    cute::array_aligned<Element, cosize_v<SmemLayoutVt>> smem_vt;
};

// ============================================================
// §5  Gmem → Smem 的 cp.async TiledCopy
//
//   使用 128-bit (uint128_t) 向量化加载，每线程每步加载 8 个 fp16。
//   GmemLayoutAtom 定义 128 线程的二维排布，使 tile 覆盖完整的一行。
// ============================================================
static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element); // 8

// Q/K: kHeadDim=64 列 → 每行需 64/8=8 线程，共 128/8=16 行/步
static constexpr int kGmemThrsPerRowQK = kHeadDim / kGmemElemsPerLoad;  // 8
using GmemLayoutAtomQK =
    Layout<Shape<Int<128 / kGmemThrsPerRowQK>, Int<kGmemThrsPerRowQK>>,
           Stride<Int<kGmemThrsPerRowQK>, _1>>;
using GmemTiledCopyQK = decltype(make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>, Element>{},
    GmemLayoutAtomQK{},
    Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));

// Vt: kBlockN=128 列 → 每行需 128/8=16 线程，共 128/16=8 行/步
static constexpr int kGmemThrsPerRowVt = kBlockN / kGmemElemsPerLoad;   // 16
using GmemLayoutAtomVt =
    Layout<Shape<Int<128 / kGmemThrsPerRowVt>, Int<kGmemThrsPerRowVt>>,
           Stride<Int<kGmemThrsPerRowVt>, _1>>;
using GmemTiledCopyVt = decltype(make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>, Element>{},
    GmemLayoutAtomVt{},
    Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));

// ============================================================
// §6  CUDA 核函数
// ============================================================
__global__ void __launch_bounds__(128)
two_stage_matmul_sm90_kernel(
    Element const* __restrict__ Q_ptr,   // [kBlockM,  kHeadDim]  row-major
    Element const* __restrict__ K_ptr,   // [kBlockN,  kHeadDim]  row-major
    Element const* __restrict__ Vt_ptr,  // [kHeadDimV, kBlockN]  row-major
    float*         __restrict__ O_ptr    // [kBlockM,  kHeadDimV] row-major 输出
) {
    int const thread_idx = threadIdx.x;

    // ------------------------------------------------------------------
    // 6.1  Smem 张量
    // ------------------------------------------------------------------
    extern __shared__ char smem_buf[];
    auto& ss = *reinterpret_cast<SharedStorage*>(smem_buf);

    Tensor sQ  = make_tensor(make_smem_ptr(ss.smem_q.data()),  SmemLayoutQ{});
    Tensor sK  = make_tensor(make_smem_ptr(ss.smem_k.data()),  SmemLayoutK{});
    Tensor sVt = make_tensor(make_smem_ptr(ss.smem_vt.data()), SmemLayoutVt{});

    // ------------------------------------------------------------------
    // 6.2  全局内存张量 (row-major)
    // ------------------------------------------------------------------
    Tensor gQ  = make_tensor(make_gmem_ptr(Q_ptr),
        make_layout(Shape<Int<kBlockM>,  Int<kHeadDim>>{},
                    Stride<Int<kHeadDim>,  _1>{}));
    Tensor gK  = make_tensor(make_gmem_ptr(K_ptr),
        make_layout(Shape<Int<kBlockN>,  Int<kHeadDim>>{},
                    Stride<Int<kHeadDim>,  _1>{}));
    Tensor gVt = make_tensor(make_gmem_ptr(Vt_ptr),
        make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{},
                    Stride<Int<kBlockN>,   _1>{}));
    Tensor gO  = make_tensor(make_gmem_ptr(O_ptr),
        make_layout(Shape<Int<kBlockM>,  Int<kHeadDimV>>{},
                    Stride<Int<kHeadDimV>, _1>{}));

    // ------------------------------------------------------------------
    // 6.3  cp.async: 全局内存 → smem
    // ------------------------------------------------------------------
    GmemTiledCopyQK gmem_tiled_copy_qk;
    GmemTiledCopyVt gmem_tiled_copy_vt;

    auto thr_cpQ  = gmem_tiled_copy_qk.get_thread_slice(thread_idx);
    auto thr_cpK  = gmem_tiled_copy_qk.get_thread_slice(thread_idx);
    auto thr_cpVt = gmem_tiled_copy_vt.get_thread_slice(thread_idx);

    Tensor tQgQ   = thr_cpQ.partition_S(gQ);
    Tensor tQsQ   = thr_cpQ.partition_D(sQ);
    Tensor tKgK   = thr_cpK.partition_S(gK);
    Tensor tKsK   = thr_cpK.partition_D(sK);
    Tensor tVtgVt = thr_cpVt.partition_S(gVt);
    Tensor tVtsVt = thr_cpVt.partition_D(sVt);

    cute::copy(gmem_tiled_copy_qk, tQgQ,   tQsQ);
    cute::copy(gmem_tiled_copy_qk, tKgK,   tKsK);
    cute::copy(gmem_tiled_copy_vt, tVtgVt, tVtsVt);

    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();

    // ------------------------------------------------------------------
    // 6.4  构建 TiledMma 与 fragment
    // ------------------------------------------------------------------
    TiledMmaQK tiled_mma_qk;
    TiledMmaPV tiled_mma_pv;

    auto thr_mma_qk = tiled_mma_qk.get_thread_slice(thread_idx);
    auto thr_mma_pv = tiled_mma_pv.get_thread_slice(thread_idx);

    // Gemm-I fragments
    //   tSrQ, tSrK: SS 模式 — smem 描述符，非寄存器数据
    //   tSrS      : C 寄存器累加器，float32，per-thread
    Tensor tSrQ = thr_mma_qk.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma_qk.partition_fragment_B(sK);
    Tensor tSrS = partition_fragment_C(tiled_mma_qk,
                                       select<0, 1>(TileShape_MNK_QK{}));
    // tSrS 布局: ((2, 2, kBlockN/8), MMA_M, MMA_N) = ((2, 2, 16), 1, 1)
    // 每线程 2×2×16 = 64 个 float32

    // Gemm-II fragments
    //   tOrVt: B smem 描述符
    //   tOrO : C 寄存器累加器，float32
    Tensor tOrVt = thr_mma_pv.partition_fragment_B(sVt);
    Tensor tOrO  = partition_fragment_C(tiled_mma_pv,
                                        select<0, 1>(TileShape_MNK_PV{}));
    // tOrO 布局: ((2, 2, kHeadDimV/8), 1, 1) = ((2, 2, 8), 1, 1)
    // 每线程 2×2×8 = 32 个 float32

    // ------------------------------------------------------------------
    // 6.5  Gemm-I: S = Q × K^T   (WGMMA SS 模式)
    //   flash::gemm<zero_init=true, wg_wait=0> 完成:
    //     warpgroup_arrive → loop{wgmma.mma_async} → warpgroup_commit_batch
    //     → warpgroup_wait<0> (同步等待) → warpgroup_fence_operand(tSrS)
    // ------------------------------------------------------------------
    flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(
        tiled_mma_qk, tSrQ, tSrK, tSrS);

    // ------------------------------------------------------------------
    // 6.6  Layout 转换: Gemm-I C 寄存器 → Gemm-II RS A 寄存器
    //
    //  ┌─────────────────────────────────────────────────────────────┐
    //  │ convert_layout_acc_Aregs<TiledMmaPV> 的数学原理 (FP16 路径)  │
    //  │                                                             │
    //  │ 输入 tSrS.layout() = ((2, 2, N/8), MMA_M, MMA_N)           │
    //  │                    = ((2, 2,  16),   1,    1)               │
    //  │   Mode 0: (2, 2, 16) — wgmma C 寄存器的内部排列             │
    //  │     · 每 wgmma.m64nNk16 原子，C 按如下方式分布到线程:        │
    //  │       [0,1]: 同行相邻两列 (stride=1)                         │
    //  │       [0]: 隔行两行 (stride=2)                               │
    //  │       [2]: N/8 个这样的"2×2"列组                             │
    //  │     · 实际物理寄存器: 每线程 64 个 float32                    │
    //  │                                                             │
    //  │ 转换步骤 (FP16, sizeof(ValTypeA)==2):                        │
    //  │   l = logical_divide(get<0,2>(layout), Tile<_2>{})          │
    //  │     = logical_divide(Int<16>{}, Tile<_2>{})                 │
    //  │     = Layout<Shape<Shape<_2,_8>>>                           │
    //  │   输出 = make_layout(                                        │
    //  │           make_layout(get<0,0>, get<0,1>, get<0,0>(l)),     │
    //  │              ──────────── (2, 2, 2) ─────────────           │
    //  │           get<1>,                                           │
    //  │              ──── 1 ────                                    │
    //  │           coalesce(make_layout(get<0,1>(l), get<2>))        │
    //  │              ──────────── 8 ────────────────                │
    //  │         ) = ((2, 2, 2), 1, 8)    [仍是 64 个 float32]       │
    //  │                                                             │
    //  │ 语义解释:                                                    │
    //  │   原 N/8=16 个列组 → 拆为 8(mode-2) × 2(并入mode-0)         │
    //  │   mode-2 的 8 个分量 ↔ K_pv/16 = 128/16 = 8 次 k16 迭代    │
    //  │   每次 k16 迭代，A 操作数在 mode-0 的 2×2×2=8 个 fp16       │
    //  │   (128-bit per thread) 与 wgmma RS A 寄存器布局完全吻合      │
    //  │                                                             │
    //  │ 结论: 同一批物理寄存器，既是 Gemm-I 的 C 输出，              │
    //  │       又是 Gemm-II 的 A 输入 — 零额外内存访问。              │
    //  └─────────────────────────────────────────────────────────────┘
    Tensor tOrP_acc = make_tensor(
        tSrS.data(),
        flash::convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout()));
    // tOrP_acc: 与 tSrS 共享物理寄存器，布局 ((2,2,2), 1, 8)，dtype=float32

    // 分配 fp16 目标寄存器 (相同 layout，不同 dtype)
    Tensor tOrP = make_tensor_like<Element>(tOrP_acc);

    // convert_type_out: 以 Array<float32,2>→Array<fp16,2> 向量化转换
    // FragmentSize=2: 2 float32 打包为 1 uint32 → 转为 2 fp16 (PRMT 指令)
    // 转换后 tOrP 持有 64 个 fp16，布局 ((2,2,2), 1, 8)
    flash::convert_type_out(tOrP_acc, tOrP);

    // ------------------------------------------------------------------
    // 6.7  Gemm-II: O = P × Vt   (WGMMA RS 模式)
    //   tOrP  作为 A 操作数直接来自寄存器 (RS = Register Source)
    //   tOrVt 作为 B 操作数来自 smem 描述符
    //
    //   flash::gemm 在 RS 模式下额外执行:
    //     warpgroup_fence_operand(tOrP)  — 通知编译器/硬件 P 寄存器即将
    //                                      被 GMMA 读取，防止调度冲突
    // ------------------------------------------------------------------
    flash::gemm</*zero_init=*/true, /*wg_wait=*/0>(
        tiled_mma_pv, tOrP, tOrVt, tOrO);

    // ------------------------------------------------------------------
    // 6.8  Epilogue: tOrO (float32 寄存器) → gO (全局内存)
    //
    //   用 identity tensor + convert_layout_acc_rowcol 获取每个累加器
    //   寄存器对应的 (行, 列) 坐标，然后直接写入全局内存。
    //
    //   convert_layout_acc_rowcol 将 ((2,2,V), MMA_M, MMA_N) 转换为
    //   (nrow=(2, MMA_M), ncol=(2, V, MMA_N))，便于按行列索引。
    // ------------------------------------------------------------------
    __syncthreads();

    // cO: identity tensor，每个元素的"值"即为其 (m,n) 逻辑坐标
    Tensor cO = thr_mma_pv.partition_C(
        make_identity_tensor(select<0, 1>(TileShape_MNK_PV{})));

    // rowcol 视图: shape = (nrow, ncol)
    Tensor cO_rowcol   = make_tensor(cO.data(),
        flash::convert_layout_acc_rowcol(cO.layout()));
    Tensor tOrO_rowcol = make_tensor(tOrO.data(),
        flash::convert_layout_acc_rowcol(tOrO.layout()));

    // 逐元素写回全局内存
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < size<0>(tOrO_rowcol); ++mi) {
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(tOrO_rowcol); ++ni) {
            int m = get<0>(cO_rowcol(mi, ni));
            int n = get<1>(cO_rowcol(mi, ni));
            gO(m, n) = tOrO_rowcol(mi, ni);
        }
    }
}

// ============================================================
// §7  Host 封装
// ============================================================
void two_stage_matmul_sm90(
    Element const* Q_ptr,   // device, [kBlockM,  kHeadDim]
    Element const* K_ptr,   // device, [kBlockN,  kHeadDim]
    Element const* Vt_ptr,  // device, [kHeadDimV, kBlockN]
    float*         O_ptr    // device, [kBlockM,  kHeadDimV]
) {
    size_t smem_size = sizeof(SharedStorage);
    if (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(
            two_stage_matmul_sm90_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size);
    }
    two_stage_matmul_sm90_kernel<<<1, 128, smem_size>>>(
        Q_ptr, K_ptr, Vt_ptr, O_ptr);
    cudaDeviceSynchronize();
}

// ============================================================
// §8  测试 main: 与 CPU 结果对比验证
// ============================================================
#ifdef TWO_STAGE_MATMUL_TEST

#include <random>
#include <cmath>
#include <algorithm>

static void ref_matmul_fp16(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    // C[M,N] = A[M,K] * B[N,K]^T  (K-major B: same as B[k,n]^T)
    // Since B is stored as [N,K] row-major, B[n][k] = B[n*K+k]
    // C[m][n] = sum_k A[m*K+k] * B[n*K+k]
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k)
                acc += A[m*K+k] * B[n*K+k];
            C[m*N+n] = acc;
        }
}

int main() {
    // Host buffers
    const int szQ  = kBlockM  * kHeadDim;
    const int szK  = kBlockN  * kHeadDim;
    const int szVt = kHeadDimV * kBlockN;
    const int szO  = kBlockM  * kHeadDimV;

    std::vector<cutlass::half_t> hQ(szQ), hK(szK), hVt(szVt);
    std::vector<float> hO(szO), hRef(szO, 0.f);
    std::vector<float> hQ_f(szQ), hK_f(szK), hVt_f(szVt), hS_f(kBlockM * kBlockN, 0.f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (auto& v : hQ_f)  v = dist(rng);
    for (auto& v : hK_f)  v = dist(rng);
    for (auto& v : hVt_f) v = dist(rng);

    // Convert to fp16
    for (int i = 0; i < szQ;  ++i) hQ[i]  = cutlass::half_t(hQ_f[i]);
    for (int i = 0; i < szK;  ++i) hK[i]  = cutlass::half_t(hK_f[i]);
    for (int i = 0; i < szVt; ++i) hVt[i] = cutlass::half_t(hVt_f[i]);

    // CPU reference:
    //   S[M, N] = Q[M,K] * K[N,K]^T   (K[N,K] row-major → B[n,k])
    //   O[M, D] = S[M,N] * Vt[D,N]^T  (Vt[D,N] row-major → B[d,n])
    ref_matmul_fp16(hQ_f.data(),  hK_f.data(),  hS_f.data(),
                    kBlockM, kBlockN, kHeadDim);
    ref_matmul_fp16(hS_f.data(),  hVt_f.data(), hRef.data(),
                    kBlockM, kHeadDimV, kBlockN);

    // Device
    void *dQ, *dK, *dVt, *dO;
    cudaMalloc(&dQ,  szQ  * sizeof(Element));
    cudaMalloc(&dK,  szK  * sizeof(Element));
    cudaMalloc(&dVt, szVt * sizeof(Element));
    cudaMalloc(&dO,  szO  * sizeof(float));

    cudaMemcpy(dQ,  hQ.data(),  szQ  * sizeof(Element), cudaMemcpyHostToDevice);
    cudaMemcpy(dK,  hK.data(),  szK  * sizeof(Element), cudaMemcpyHostToDevice);
    cudaMemcpy(dVt, hVt.data(), szVt * sizeof(Element), cudaMemcpyHostToDevice);

    two_stage_matmul_sm90(
        reinterpret_cast<Element*>(dQ),
        reinterpret_cast<Element*>(dK),
        reinterpret_cast<Element*>(dVt),
        reinterpret_cast<float*>(dO));

    cudaMemcpy(hO.data(), dO, szO * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare
    float max_err = 0.f, max_ref = 0.f;
    for (int i = 0; i < szO; ++i) {
        max_err = std::max(max_err, std::abs(hO[i] - hRef[i]));
        max_ref = std::max(max_ref, std::abs(hRef[i]));
    }
    float rel_err = max_ref > 0.f ? max_err / max_ref : max_err;
    printf("[two_stage_matmul_sm90] max_abs_err=%.6f  max_rel_err=%.6f  %s\n",
           max_err, rel_err, rel_err < 1e-2f ? "PASS" : "FAIL");

    cudaFree(dQ); cudaFree(dK); cudaFree(dVt); cudaFree(dO);
    return rel_err < 1e-2f ? 0 : 1;
}
#endif  // TWO_STAGE_MATMUL_TEST
