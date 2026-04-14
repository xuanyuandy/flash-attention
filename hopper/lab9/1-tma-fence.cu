// TMA fence demo: demonstrates why async_proxy_fence is necessary
//
// Scenario:
//   1. TMA Load:  global → SMEM  (async proxy WRITES smem)
//   2. Threads:   scale SMEM ×2  (generic proxy WRITES smem)
//   3. TMA Store: SMEM → global  (async proxy READS smem)
//
// Without async_proxy_fence before step 3, the TMA store (async proxy)
// may not see the thread modifications (generic proxy writes).
// Result: destination gets the ORIGINAL data instead of the scaled data.
//
// Toggle ENABLE_FENCE to 0/1 to compare behavior.

#include <cuda.h>
#include <cuda_bf16.h>
#include <random>
#include <stdio.h>

#include "tma-interface.cuh"

typedef __nv_bfloat16 bf16;

#ifndef ENABLE_FENCE
#define ENABLE_FENCE 0
#endif

////////////////////////////////////////////////////////////////////////////////
// Kernel: TMA Load → Thread Modify → TMA Store
////////////////////////////////////////////////////////////////////////////////
template <int TILE_M, int TILE_N>
__global__ void tma_load_modify_store(
    __grid_constant__ const CUtensorMap src_map,
    __grid_constant__ const CUtensorMap dest_map,
    float scale_factor)
{
    __shared__ __align__(128) bf16 smem[TILE_M * TILE_N];
    __shared__ __align__(8) uint64_t bar;

    // ===== Phase 1: TMA Load (async proxy writes SMEM) =====
    if (threadIdx.x == 0) {
        init_barrier(&bar, 1);
        expect_bytes_and_arrive(&bar, TILE_M * TILE_N * sizeof(bf16));
        cp_async_bulk_tensor_2d_global_to_shared(
            smem, &src_map, 0, 0, &bar);
    }
    __syncthreads();
    wait(&bar, 0);

    // ===== Phase 2: Threads modify SMEM (generic proxy writes SMEM) =====
    // 每个线程处理多个元素，模拟真实计算（如 GEMM 累加、softmax 等）
    constexpr int total_elems = TILE_M * TILE_N;
    for (int idx = threadIdx.x; idx < total_elems; idx += blockDim.x) {
        float val = __bfloat162float(smem[idx]);
        val *= scale_factor;
        smem[idx] = __float2bfloat16(val);
    }

    // 确保所有线程的 SMEM 写入完成
    __syncthreads();

    // ===== Critical Point: Fence =====
    // generic proxy 的写入需要通过 fence 对 async proxy 可见
#if ENABLE_FENCE
    async_proxy_fence();  // fence.proxy.async.shared::cta
#endif
    // 如果不加 fence，TMA store (async proxy) 可能读到的是
    // Phase 1 TMA load 写入的原始数据，而非 Phase 2 线程修改后的数据

    // ===== Phase 3: TMA Store (async proxy reads SMEM) =====
    if (threadIdx.x == 0) {
        cp_async_bulk_tensor_2d_shared_to_global(
            &dest_map, 0, 0, smem);
        tma_commit_group();
        tma_wait_until_pending<0>();
    }
}

////////////////////////////////////////////////////////////////////////////////
// Multi-tile version: 更高并发度，更容易暴露问题
////////////////////////////////////////////////////////////////////////////////
template <int TILE_M, int TILE_N, int NUM_TILES_N>
__global__ void tma_load_modify_store_multi(
    __grid_constant__ const CUtensorMap src_map,
    __grid_constant__ const CUtensorMap dest_map,
    float scale_factor)
{
    __shared__ __align__(128) bf16 smem[TILE_M * TILE_N];
    __shared__ __align__(8) uint64_t bar;

    constexpr int total_elems = TILE_M * TILE_N;

    // 遍历多个 tile，增大并发压力
    for (int tile = 0; tile < NUM_TILES_N; tile++) {
        // col_offset is the begin offset of current tile in the global matrix 
        int col_offset = tile * TILE_N;

        // Phase 1: TMA Load
        if (threadIdx.x == 0) {
            init_barrier(&bar, 1);
            expect_bytes_and_arrive(&bar, total_elems * sizeof(bf16));
            cp_async_bulk_tensor_2d_global_to_shared(
                smem, &src_map, col_offset, 0, &bar);
        }
        __syncthreads();
        // wait should after __syncthreads to ensure all threads have arrived before any thread proceeds
        wait(&bar, 0);

        // Phase 2: 线程修改 — 做一些非平凡计算
        for (int idx = threadIdx.x; idx < total_elems; idx += blockDim.x) {
            float val = __bfloat162float(smem[idx]);
            // 混合计算：scale + 偏移 + 截断，模拟真实 kernel 行为
            val = val * scale_factor + 1.0f;
            val = fmaxf(val, -65504.0f);  // clamp to bf16 range
            smem[idx] = __float2bfloat16(val);
        }

        __syncthreads();

#if ENABLE_FENCE
        async_proxy_fence();
#endif

        // Phase 3: TMA Store
        if (threadIdx.x == 0) {
            cp_async_bulk_tensor_2d_shared_to_global(
                &dest_map, col_offset, 0, smem);
            tma_commit_group();
            tma_wait_until_pending<0>();
        }

        // 确保 store 完成后再进入下一个 tile 的 load
        __syncthreads();
    }
}

////////////////////////////////////////////////////////////////////////////////
// Host launcher
////////////////////////////////////////////////////////////////////////////////
template <int TILE_M, int TILE_N>
void launch_single_tile(bf16 *src, bf16 *dest, float scale) {
    CUtensorMap src_map, dest_map;
    uint64_t globalDim[2]      = {TILE_N, TILE_M};
    uint64_t globalStrides[1]  = {TILE_N * sizeof(bf16)};
    uint32_t boxDim[2]         = {TILE_N, TILE_M};
    uint32_t elementStrides[2] = {1, 1};

    CUDA_CHECK(cuTensorMapEncodeTiled(
        &src_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, (void *)src, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    CUDA_CHECK(cuTensorMapEncodeTiled(
        &dest_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, (void *)dest, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    tma_load_modify_store<TILE_M, TILE_N><<<1, 128>>>(src_map, dest_map, scale);
}

template <int TILE_M, int TILE_N, int NUM_TILES_N>
void launch_multi_tile(bf16 *src, bf16 *dest, float scale, int num_blocks) {
    int total_N = TILE_N * NUM_TILES_N;

    CUtensorMap src_map, dest_map;
    uint64_t globalDim[2]      = {(uint64_t)total_N, TILE_M};
    uint64_t globalStrides[1]  = {(uint64_t)total_N * sizeof(bf16)};
    uint32_t boxDim[2]         = {TILE_N, TILE_M};
    uint32_t elementStrides[2] = {1, 1};

    CUDA_CHECK(cuTensorMapEncodeTiled(
        &src_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, (void *)src, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    CUDA_CHECK(cuTensorMapEncodeTiled(
        &dest_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, (void *)dest, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    tma_load_modify_store_multi<TILE_M, TILE_N, NUM_TILES_N>
        <<<num_blocks, 256>>>(src_map, dest_map, scale);
}

////////////////////////////////////////////////////////////////////////////////
// Verification
////////////////////////////////////////////////////////////////////////////////
bool verify(bf16 *src_host, bf16 *dest_device, int M, int N,
            float scale, float bias, const char *test_name) {
    int total = M * N;
    bf16 *result = (bf16 *)malloc(total * sizeof(bf16));
    cudaMemcpy(result, dest_device, total * sizeof(bf16), cudaMemcpyDeviceToHost);

    int mismatches = 0;
    int first_mismatch_idx = -1;
    float first_expected = 0, first_got = 0;

    for (int i = 0; i < total; i++) {
        float src_val = __bfloat162float(src_host[i]);
        float expected = src_val * scale + bias;
        expected = fmaxf(expected, -65504.0f);
        // bf16 往返精度损失，用 bf16 round-trip 比较
        float expected_bf16 = __bfloat162float(__float2bfloat16(expected));
        float got = __bfloat162float(result[i]);

        if (expected_bf16 != got) {
            mismatches++;
            if (first_mismatch_idx == -1) {
                first_mismatch_idx = i;
                first_expected = expected_bf16;
                first_got = got;
            }
        }
    }

    if (mismatches > 0) {
        printf("[%s] FAILED: %d/%d mismatches\n", test_name, mismatches, total);
        printf("  First mismatch at index %d: expected %.4f, got %.4f\n",
               first_mismatch_idx, first_expected, first_got);
        printf("  (original value: %.4f)\n",
               __bfloat162float(src_host[first_mismatch_idx]));
    } else {
        printf("[%s] PASSED: all %d elements correct\n", test_name, total);
    }

    free(result);
    return mismatches == 0;
}

////////////////////////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////////////////////////
int main() {
    printf("=== TMA async_proxy_fence Demo ===\n");
    printf("ENABLE_FENCE = %d\n\n", ENABLE_FENCE);

    // ---- Test 1: Single tile, single block ----
    {
        const int M = 64, N = 128;
        const int total = M * N;
        const float scale = 2.0f;

        bf16 *h_src = (bf16 *)malloc(total * sizeof(bf16));
        bf16 *d_src, *d_dest;
        cudaMalloc(&d_src, total * sizeof(bf16));
        cudaMalloc(&d_dest, total * sizeof(bf16));
        cudaMemset(d_dest, 0, total * sizeof(bf16));

        std::default_random_engine gen(42);
        std::normal_distribution<float> dist(0, 1);
        for (int i = 0; i < total; i++)
            h_src[i] = __float2bfloat16(dist(gen));
        cudaMemcpy(d_src, h_src, total * sizeof(bf16), cudaMemcpyHostToDevice);

        launch_single_tile<M, N>(d_src, d_dest, scale);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());

        // scale=2.0, bias=0.0
        verify(h_src, d_dest, M, N, scale, 0.0f, "Single Tile (64x128, scale=2x)");

        free(h_src);
        cudaFree(d_src);
        cudaFree(d_dest);
    }

    // ---- Test 2: Multi-tile, multi-block (higher pressure) ----
    {
        const int TILE_M = 64, TILE_N = 128;
        const int NUM_TILES = 8;
        const int M = TILE_M;
        const int N = TILE_N * NUM_TILES;  // 64 x 1024
        const int total = M * N;
        const float scale = 3.0f;
        const int num_blocks = 4;  // 多 block 增加 TMA 竞争

        bf16 *h_src = (bf16 *)malloc(total * sizeof(bf16));
        bf16 *d_src, *d_dest;
        cudaMalloc(&d_src, total * sizeof(bf16));
        cudaMalloc(&d_dest, total * sizeof(bf16));
        cudaMemset(d_dest, 0, total * sizeof(bf16));

        std::default_random_engine gen(123);
        std::normal_distribution<float> dist(0, 2);
        for (int i = 0; i < total; i++)
            h_src[i] = __float2bfloat16(dist(gen));
        cudaMemcpy(d_src, h_src, total * sizeof(bf16), cudaMemcpyHostToDevice);

        launch_multi_tile<TILE_M, TILE_N, NUM_TILES>(d_src, d_dest, scale, num_blocks);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());

        // scale=3.0, bias=1.0 (kernel does val * scale + 1.0)
        verify(h_src, d_dest, M, N, scale, 1.0f,
               "Multi Tile (64x1024, 4 blocks, scale=3x+1)");

        free(h_src);
        cudaFree(d_src);
        cudaFree(d_dest);
    }

    // ---- Test 3: Stress test with repeated runs ----
    {
        const int TILE_M = 64, TILE_N = 128;
        const int NUM_TILES = 8;
        const int M = TILE_M;
        const int N = TILE_N * NUM_TILES;
        const int total = M * N;
        const float scale = 0.5f;
        const int REPEATS = 100;

        bf16 *h_src = (bf16 *)malloc(total * sizeof(bf16));
        bf16 *d_src, *d_dest;
        cudaMalloc(&d_src, total * sizeof(bf16));
        cudaMalloc(&d_dest, total * sizeof(bf16));

        std::default_random_engine gen(999);
        std::normal_distribution<float> dist(0, 5);
        for (int i = 0; i < total; i++)
            h_src[i] = __float2bfloat16(dist(gen));
        cudaMemcpy(d_src, h_src, total * sizeof(bf16), cudaMemcpyHostToDevice);

        int failures = 0;
        for (int r = 0; r < REPEATS; r++) {
            cudaMemset(d_dest, 0, total * sizeof(bf16));
            launch_multi_tile<TILE_M, TILE_N, NUM_TILES>(d_src, d_dest, scale, 8);
            cudaDeviceSynchronize();

            bf16 *result = (bf16 *)malloc(total * sizeof(bf16));
            cudaMemcpy(result, d_dest, total * sizeof(bf16), cudaMemcpyDeviceToHost);

            for (int i = 0; i < total; i++) {
                float src_val = __bfloat162float(h_src[i]);
                float expected = fmaxf(src_val * scale + 1.0f, -65504.0f);
                float expected_bf16 = __bfloat162float(__float2bfloat16(expected));
                if (expected_bf16 != __bfloat162float(result[i])) {
                    failures++;
                    break;
                }
            }
            free(result);
        }

        printf("[Stress Test (%dx, 8 blocks)] %d/%d runs correct\n",
               REPEATS, REPEATS - failures, REPEATS);

        free(h_src);
        cudaFree(d_src);
        cudaFree(d_dest);
    }

    return 0;
}
