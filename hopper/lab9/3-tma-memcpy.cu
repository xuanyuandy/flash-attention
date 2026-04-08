// TL+ {"platform": "h100"}
// TL+ {"header_files": ["tma-interface.cuh"]}
// TL+ {"compile_flags": ["-lcuda"]}
// TL {"workspace_files": []}

#include <cuda.h>
#include <cuda_bf16.h>
#include <stdio.h>

#include "tma-interface.cuh"

typedef __nv_bfloat16 bf16;

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// Part 3: TMA Memcpy
////////////////////////////////////////////////////////////////////////////////
constexpr int COLS       = 128;
constexpr int TILE_ROWS  = 64;
constexpr int TILE_ELEMS = TILE_ROWS * COLS;            // 8192
constexpr int TILE_BYTES = TILE_ELEMS * sizeof(bf16);   // 16384
constexpr int NUM_STAGES = 3;

// 共享内存布局: [stage0 data | stage1 data | stage2 data | bar0 | bar1 | bar2]
constexpr int SMEM_DATA_BYTES = NUM_STAGES * TILE_BYTES; // 49152
constexpr int SMEM_TOTAL = SMEM_DATA_BYTES + NUM_STAGES * sizeof(uint64_t);

__global__ void tma_copy(__grid_constant__ const CUtensorMap tensor_map,
                         __grid_constant__ const CUtensorMap dest_tensor_map,
                         const int N) {
    extern __shared__ __align__(128) char smem_buf[];
    uint64_t *bars = (uint64_t *)(smem_buf + SMEM_DATA_BYTES);

    const int total_tiles    = N / TILE_ELEMS;
    const int tiles_per_blk  = (total_tiles + gridDim.x - 1) / gridDim.x;
    const int tile_start     = blockIdx.x * tiles_per_blk;
    if (tile_start >= total_tiles) return;
    const int tile_end  = min(tile_start + tiles_per_blk, total_tiles);
    const int num_tiles = tile_end - tile_start;

    if (threadIdx.x == 0) {
        // ① 初始化 barriers
        for (int s = 0; s < NUM_STAGES; s++)
            init_barrier(&bars[s], 1);

        // ② Prefill: 发射前 NUM_STAGES 个 load，填满流水线
        int prefill = min(num_tiles, NUM_STAGES);
        for (int i = 0; i < prefill; i++) {
            expect_bytes_and_arrive(&bars[i], TILE_BYTES);
            cp_async_bulk_tensor_2d_global_to_shared(
                smem_buf + i * TILE_BYTES,
                &tensor_map,
                0, (tile_start + i) * TILE_ROWS,
                &bars[i]);
        }

        // ③ 主循环: wait → store → load_next
        for (int i = 0; i < num_tiles; i++) {
            int s     = i % NUM_STAGES;
            int phase = (i / NUM_STAGES) % 2;

            // 等待当前 tile 的 load 完成
            wait(&bars[s], phase);

            // store 当前 tile: shared → global
            async_proxy_fence();
            cp_async_bulk_tensor_2d_shared_to_global(
                &dest_tensor_map,
                0, (tile_start + i) * TILE_ROWS,
                smem_buf + s * TILE_BYTES);
            tma_commit_group();

            // 为这个 stage 发射下一个 load（复用同一 buffer）
            int next = i + NUM_STAGES;
            if (next < num_tiles) {
                // 确保此 buffer 上的旧 store 已完成读取
                tma_wait_until_pending<NUM_STAGES>();
                expect_bytes_and_arrive(&bars[s], TILE_BYTES);
                cp_async_bulk_tensor_2d_global_to_shared(
                    smem_buf + s * TILE_BYTES,
                    &tensor_map,
                    0, (tile_start + next) * TILE_ROWS,
                    &bars[s]);
            }
        }

        // ④ 等待所有 store 完成
        tma_wait_until_pending<0>();
    }
}

void launch_tma_copy(bf16 *dest, bf16 *src, int N) {
    int total_rows = N / COLS;

    // 构造 src / dest tensor map，逻辑形状 = [total_rows, 128]
    uint64_t globalDim[2]      = {(uint64_t)COLS, (uint64_t)total_rows};
    uint64_t globalStrides[1]  = {COLS * sizeof(bf16)};   // 行步长 = 256 bytes
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

    int total_tiles = total_rows / TILE_ROWS;
    int num_blocks  = min(total_tiles, 132 * 4);   // ≈528，覆盖所有 SM

    CUDA_CHECK(cudaFuncSetAttribute(
        tma_copy, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL));
    tma_copy<<<num_blocks, 32, SMEM_TOTAL>>>(src_map, dest_map, N);
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
        // Don't want to use a random number generator, takes too long.
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
    launch_tma_copy(d_dest, d_src, size);
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
        BENCHMARK_KERNEL((launch_tma_copy(d_dest, d_src, size)), num_iters,
                         size_bytes, "TMA Copy");
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