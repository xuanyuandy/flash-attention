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
// Part 2: Single Block, Single Tile TMA Reduce
////////////////////////////////////////////////////////////////////////////////
// TMA reduce-add 的 PTX 内联汇编
// 对比 store 版本:  cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group
// reduce-add 版本:  cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group
__device__ static __forceinline__ void
cp_async_reduce_add_bulk_tensor_2d_shared_to_global(
    const CUtensorMap *tensor_map, int c0, int c1, const void *src) {
    asm volatile(
        "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group "
        "[%0, {%1, %2}], [%3];\n"
        :
        : "l"(tensor_map), "r"(c0), "r"(c1),
          "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src)))
        : "memory");
}

template <int TILE_M, int TILE_N>
__global__ void
single_tma_reduce(__grid_constant__ const CUtensorMap src_map,
                  __grid_constant__ const CUtensorMap dest_map) {
    __shared__ __align__(128) bf16 smem[TILE_M * TILE_N];
    __shared__ __align__(8) uint64_t bar;

    if (threadIdx.x == 0) {
        // TMA Load: global → shared
        init_barrier(&bar, 1);
        expect_bytes_and_arrive(&bar, TILE_M * TILE_N * sizeof(bf16));
        cp_async_bulk_tensor_2d_global_to_shared(
            smem, &src_map, 0, 0, &bar);
    }

    __syncthreads();
    wait(&bar, 0);

    // TMA Reduce-Add: shared → global (原子加)
    async_proxy_fence();

    if (threadIdx.x == 0) {
        cp_async_reduce_add_bulk_tensor_2d_shared_to_global(
            &dest_map, 0, 0, smem);
        tma_commit_group();
        tma_wait_until_pending<0>();
    }
}

template <int TILE_M, int TILE_N>
void launch_single_tma_reduce(bf16 *src, bf16 *dest) {
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

    single_tma_reduce<TILE_M, TILE_N><<<1, 128>>>(src_map, dest_map);
}
/// <--- /your code here --->

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

    // Copy in 1s for the reduction.
    for (int i = 0; i < total_size; i++) {
        matrix[i] = 1;
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

    printf("\n\nRunning TMA reduce kernel...\n\n");

    // Launch the TMA kernel
    launch_single_tma_reduce<M, N>(d_matrix, d_dest);

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
        float ref = (float)matrix[i * N + j] + 1.0f;
        float computed = (float)final_output[i * N + j];
        float diff = std::fabs(ref - computed);
        if (diff > 0.1) {
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