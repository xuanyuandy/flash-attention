// TL+ {"platform": "h100"}
// TL+ {"header_files": ["tma-interface.cuh"]}
// TL+ {"compile_flags": ["-lcuda"]}

#include <cuda.h>
#include <cuda_bf16.h>
#include <stdio.h>

#include "tma-interface.cuh"

typedef __nv_bfloat16 bf16;

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// Part 4: Bring Your Own Warp Scheduler
////////////////////////////////////////////////////////////////////////////////
constexpr int COLS       = 128;
constexpr int TILE_ROWS  = 64;
constexpr int TILE_ELEMS = TILE_ROWS * COLS;           // 8192
constexpr int TILE_BYTES = TILE_ELEMS * sizeof(bf16);  // 16384

constexpr int NUM_WARPS  = 7;
constexpr int STAGES     = 2;
constexpr int TOTAL_BUFS = NUM_WARPS * STAGES;          // 14
constexpr int DATA_BYTES = TOTAL_BUFS * TILE_BYTES;     // 229376
constexpr int MAX_SMEM   = 232448;                       // H100 max (227KB)

// Layout: [buf0|buf1|...|buf13 | bar0|bar1|...|bar13]

__global__ void
tma_multiwarp_pipeline(__grid_constant__ const CUtensorMap tensor_map,
                       __grid_constant__ const CUtensorMap dest_tensor_map,
                       const int N) {
    extern __shared__ __align__(128) char smem[];
    uint64_t *bars = (uint64_t *)(smem + DATA_BYTES);

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // 只有 7 个 warp 的 lane 0 干活
    if (warp_id >= NUM_WARPS || lane_id != 0) return;

    int total_tiles = N / TILE_ELEMS;

    // ── Block 级工作划分（连续块）──
    int tiles_per_blk = (total_tiles + gridDim.x - 1) / gridDim.x;
    int blk_start     = blockIdx.x * tiles_per_blk;
    int blk_end       = min(blk_start + tiles_per_blk, total_tiles);

    // ── Warp 级工作划分（块内交错）──
    // warp j 处理: blk_start+j, blk_start+j+NUM_WARPS, ...
    int my_buf = warp_id * STAGES;  // 我的 buffer 起始索引

    // ① 初始化 barrier
    for (int s = 0; s < STAGES; s++)
        init_barrier(&bars[my_buf + s], 1);

    // ② Prefill: 发射前 STAGES 个 load 填满流水线
    for (int s = 0; s < STAGES; s++) {
        int tile = blk_start + warp_id + s * NUM_WARPS;
        if (tile >= blk_end) break;
        expect_bytes_and_arrive(&bars[my_buf + s], TILE_BYTES);
        cp_async_bulk_tensor_2d_global_to_shared(
            smem + (my_buf + s) * TILE_BYTES,
            &tensor_map, 0, tile * TILE_ROWS,
            &bars[my_buf + s]);
    }

    // ③ 主循环
    int iter = 0;
    for (int tile = blk_start + warp_id;
         tile < blk_end;
         tile += NUM_WARPS, iter++)
    {
        int s     = iter % STAGES;
        int phase = (iter / STAGES) % 2;

        // 等当前 tile 的 load 完成
        wait(&bars[my_buf + s], phase);

        // Store: shared → global
        async_proxy_fence();
        cp_async_bulk_tensor_2d_shared_to_global(
            &dest_tensor_map, 0, tile * TILE_ROWS,
            smem + (my_buf + s) * TILE_BYTES);
        tma_commit_group();

        // 为这个 stage 发射下一轮 load（复用 buffer）
        int next_tile = tile + STAGES * NUM_WARPS;
        if (next_tile < blk_end) {
            // 确保此 buffer 上的旧 store 已读完
            tma_wait_until_pending<2>();
            expect_bytes_and_arrive(&bars[my_buf + s], TILE_BYTES);
            cp_async_bulk_tensor_2d_global_to_shared(
                smem + (my_buf + s) * TILE_BYTES,
                &tensor_map, 0, next_tile * TILE_ROWS,
                &bars[my_buf + s]);
        }
    }

    // ④ 等所有 store 完成
    tma_wait_until_pending<0>();
}

void launch_multiwarp_pipeline(bf16 *dest, bf16 *src, const int N) {
    int total_rows = N / COLS;

    uint64_t globalDim[2]      = {(uint64_t)COLS, (uint64_t)total_rows};
    uint64_t globalStrides[1]  = {COLS * sizeof(bf16)};
    uint32_t boxDim[2]         = {(uint32_t)COLS, (uint32_t)TILE_ROWS};
    uint32_t elementStrides[2] = {1, 1};

    CUtensorMap src_map;
    CUDA_CHECK(cuTensorMapEncodeTiled(
        &src_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, (void *)src,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    CUtensorMap dest_map;
    CUDA_CHECK(cuTensorMapEncodeTiled(
        &dest_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, (void *)dest,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    // H100 有 132 个 SM，每个 SM 只能放 1 个 block
    int total_tiles = total_rows / TILE_ROWS;
    int num_blocks  = min(total_tiles, 132);

    // 必须设置最大动态共享内存
    CUDA_CHECK(cudaFuncSetAttribute(
        tma_multiwarp_pipeline,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        MAX_SMEM));

    // 启动: 7 warps = 224 threads, 使用最大共享内存
    tma_multiwarp_pipeline<<<num_blocks, NUM_WARPS * 32, MAX_SMEM>>>(
        src_map, dest_map, N);
}

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

const int elem_per_block = 16384;
__global__ void simple_vector_copy(bf16 *__restrict__ dest,
                                   const bf16 *__restrict__ src, int N) {
    constexpr int VEC_ELEMS = 8;
    using VecT = uint4;

    int total_vecs = elem_per_block / VEC_ELEMS;
    int start_vec = (blockIdx.x * blockDim.x) * total_vecs;

    const VecT *src_vec = reinterpret_cast<const VecT *>(src);
    VecT *dest_vec = reinterpret_cast<VecT *>(dest);

    for (int i = threadIdx.x; i < blockDim.x * total_vecs; i += blockDim.x) {
        dest_vec[start_vec + i] = src_vec[start_vec + i];
    }
}

#define BENCHMARK_KERNEL(kernel_call, num_iters, size_bytes, label)            \
    do {                                                                       \
        cudaEvent_t start, stop;                                               \
        CUDA_CHECK(cudaEventCreate(&start));                                   \
        CUDA_CHECK(cudaEventCreate(&stop));                                    \
        CUDA_CHECK(cudaEventRecord(start));                                    \
        for (int i = 0; i < num_iters; i++) {                                  \
            kernel_call;                                                       \
        }                                                                      \
        CUDA_CHECK(cudaEventRecord(stop));                                     \
        CUDA_CHECK(cudaEventSynchronize(stop));                                \
        float elapsed_time;                                                    \
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));          \
        float time_per_iter = elapsed_time / num_iters;                        \
        float bandwidth_gb_s = (2.0 * size_bytes * 1e-6 / time_per_iter);      \
        printf("%s - Time: %.4f ms, Bandwidth: %.2f GB/s\n", label,            \
               time_per_iter, bandwidth_gb_s);                                 \
        CUDA_CHECK(cudaEventDestroy(start));                                   \
        CUDA_CHECK(cudaEventDestroy(stop));                                    \
    } while (0)

int main() {
    const size_t size = 132 * 10 * 32 * 128 * 128;

    // Allocate and initialize host memory
    bf16 *matrix = (bf16 *)malloc(size * sizeof(bf16));
    const int N = 128;
    for (int idx = 0; idx < size; idx++) {
        int i = idx / N;
        int j = idx % N;
        float val = fmodf((i * 123 + j * 37) * 0.001f, 2.0f) - 1.0f;
        matrix[idx] = __float2bfloat16(val);
    }

    // Allocate device memory
    bf16 *d_src, *d_dest;
    CUDA_CHECK(cudaMalloc(&d_src, size * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&d_dest, size * sizeof(bf16)));
    CUDA_CHECK(
        cudaMemcpy(d_src, matrix, size * sizeof(bf16), cudaMemcpyHostToDevice));

    // Test TMA copy correctness
    printf("Testing TMA copy correctness...\n");
    CUDA_CHECK(cudaMemset(d_dest, 0, size * sizeof(bf16)));
    launch_multiwarp_pipeline(d_dest, d_src, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    bf16 *tma_result = (bf16 *)malloc(size * sizeof(bf16));
    CUDA_CHECK(cudaMemcpy(tma_result, d_dest, size * sizeof(bf16),
                          cudaMemcpyDeviceToHost));

    bool tma_correct = true;
    for (int idx = 0; idx < size; idx++) {
        if (tma_result[idx] != matrix[idx]) {
            printf("First mismatch at [%d]: %.4f != %.4f\n", idx,
                   __bfloat162float(tma_result[idx]),
                   __bfloat162float(matrix[idx]));
            tma_correct = false;
            break;
        }
    }
    printf("TMA Copy: %s\n\n", tma_correct ? "PASSED" : "FAILED");
    free(tma_result);

    // Test simple copy correctness
    printf("Testing simple copy correctness...\n");
    CUDA_CHECK(cudaMemset(d_dest, 0, size * sizeof(bf16)));
    simple_vector_copy<<<size / (elem_per_block * 32), 32>>>(d_dest, d_src,
                                                             size);
    CUDA_CHECK(cudaDeviceSynchronize());

    bf16 *simple_result = (bf16 *)malloc(size * sizeof(bf16));
    CUDA_CHECK(cudaMemcpy(simple_result, d_dest, size * sizeof(bf16),
                          cudaMemcpyDeviceToHost));

    bool simple_correct = true;
    for (int idx = 0; idx < size; idx++) {
        if (simple_result[idx] != matrix[idx]) {
            printf("First mismatch at [%d]: %.4f != %.4f\n", idx,
                   __bfloat162float(tma_result[idx]),
                   __bfloat162float(matrix[idx]));

            simple_correct = false;
            break;
        }
    }
    printf("Simple Copy: %s\n\n", simple_correct ? "PASSED" : "FAILED");
    free(simple_result);

    // Benchmark both kernels
    const int num_iters = 10;
    const size_t size_bytes = size * sizeof(bf16);

    if (tma_correct) {
        BENCHMARK_KERNEL((launch_multiwarp_pipeline(d_dest, d_src, size)),
                         num_iters, size_bytes, "TMA Copy");
    }

    if (simple_correct) {
        BENCHMARK_KERNEL(
            (simple_vector_copy<<<size / (elem_per_block * 32), 32>>>(
                 d_dest, d_src, size),
             cudaDeviceSynchronize()),
            num_iters, size_bytes, "Simple Copy");
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dest));
    free(matrix);
    return 0;
}