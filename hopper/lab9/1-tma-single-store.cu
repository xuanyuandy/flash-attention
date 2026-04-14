// TL+ {"platform": "h100"}
// TL+ {"header_files": ["tma-interface.cuh"]}
// TL+ {"compile_flags": ["-lcuda"]}
// TL {"workspace_files": []}

#include <cuda.h>
#include <cuda_bf16.h>
#include <random>
#include <stdio.h>

#include "tma-interface.cuh"

// Type alias for bfloat16
typedef __nv_bfloat16 bf16;

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// Part 1: Single Block, Single Tile TMA Store
////////////////////////////////////////////////////////////////////////////////
template <int TILE_M, int TILE_N>
__global__ void single_tma_store(__grid_constant__ const CUtensorMap src_map,
                                 __grid_constant__ const CUtensorMap dest_map) {
    __shared__ __align__(128) bf16 smem[TILE_M * TILE_N];
    __shared__ __align__(8) uint64_t bar;

    if (threadIdx.x == 0) {
        // ===== Phase 1: TMA Load (global → shared) =====
        init_barrier(&bar, 1);
        expect_bytes_and_arrive(&bar, TILE_M * TILE_N * sizeof(bf16));
        cp_async_bulk_tensor_2d_global_to_shared(
            smem, &src_map, 0, 0, &bar);
    }

    __syncthreads();
    wait(&bar, 0);

    // ===== Phase 2: TMA Store (shared → global) =====
    // 确保 shared memory 的写入对 async proxy 可见
    // async_proxy_fence may not work when TMA store and TMA load are in same async proxy
    async_proxy_fence();

    if (threadIdx.x == 0) {
        cp_async_bulk_tensor_2d_shared_to_global(
            &dest_map, 0, 0, smem);

        // TMA store 用 bulk_group 机制来等待完成
        tma_commit_group();
        tma_wait_until_pending<0>();
    }
}

template <int TILE_M, int TILE_N>
void launch_single_tma_store(bf16 *src, bf16 *dest) {
    // src tensor map (用于 load)
    CUtensorMap src_map;
    uint64_t globalDim[2]      = {TILE_N, TILE_M};
    uint64_t globalStrides[1]  = {TILE_N * sizeof(bf16)};
    uint32_t boxDim[2]         = {TILE_N, TILE_M};
    uint32_t elementStrides[2] = {1, 1};

    CUDA_CHECK(cuTensorMapEncodeTiled(
        &src_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, (void *)src,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));

    // dest tensor map (用于 store)
    CUtensorMap dest_map;
    CUDA_CHECK(cuTensorMapEncodeTiled(
        &dest_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, (void *)dest,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));

    single_tma_store<TILE_M, TILE_N><<<1, 128>>>(src_map, dest_map);
}

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

int main() {
    const int M = 64;
    const int N = 128;
    const uint64_t total_size = M * N;

    // Allocate host and device memory
    bf16 *matrix = (bf16 *)malloc(total_size * sizeof(bf16));
    bf16 *d_matrix;
    bf16 *d_dest;
    cudaMalloc(&d_matrix, total_size * sizeof(bf16));
    cudaMalloc(&d_dest, total_size * sizeof(bf16));

    // Zero out destination buffer
    for (int i = 0; i < total_size; i++) {
        matrix[i] = 0;
    }
    cudaMemcpy(d_dest, matrix, total_size * sizeof(bf16),
               cudaMemcpyHostToDevice);

    // Initialize source matrix on host
    std::default_random_engine generator(0);
    std::normal_distribution<float> dist(0, 1);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float val = dist(generator);
            matrix[i * N + j] = __float2bfloat16(val);
        }
    }
    cudaMemcpy(d_matrix, matrix, total_size * sizeof(bf16),
               cudaMemcpyHostToDevice);

    printf("\n\nRunning TMA store kernel...\n\n");

    // Launch the TMA kernel
    launch_single_tma_store<M, N>(d_matrix, d_dest);

    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    bf16 *final_output = (bf16 *)malloc(total_size * sizeof(bf16));
    cudaMemcpy(final_output, d_dest, total_size * sizeof(bf16),
               cudaMemcpyDeviceToHost);

    // Verify correctness
    bool correct = true;
    for (int x = 0; x < M * N; x++) {
        int i = x / N;
        int j = x % N;
        float ref = (float)matrix[i * N + j];
        float computed = (float)final_output[i * N + j];
        if (ref != computed) {
            correct = false;
            printf("Mismatch at (%d, %d): expected %f, got %f \n", i, j, ref,
                   computed);
            break;
        }
    }

    printf("%s output!\n\n\n", correct ? "Correct" : "Incorrect");

    // Cleanup resources
    cudaFree(d_matrix);
    cudaFree(d_dest);
    free(matrix);
    free(final_output);

    return 0;
}