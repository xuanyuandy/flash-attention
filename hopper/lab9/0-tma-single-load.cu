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
// Part 0: Single Block, Single Tile TMA Load
////////////////////////////////////////////////////////////////////////////////

template <int TILE_M, int TILE_N>
__global__ void single_tma_load(__grid_constant__ const CUtensorMap src_map,
                                bf16 *dest) {
    // 1. 在 shared memory 中分配 tile 空间和 mbarrier
    __shared__ __align__(128) bf16 smem[TILE_M * TILE_N];
    __shared__ __align__(8) uint64_t bar;

    // 2. 只需要一个线程发起 TMA 操作
    if (threadIdx.x == 0) {
        // 初始化 mbarrier，arrival_count = 1（只有 thread 0 会 arrive）
        init_barrier(&bar, 1);

        // 告诉 barrier 期待多少字节的数据，同时完成一次 arrive
        expect_bytes_and_arrive(&bar, TILE_M * TILE_N * sizeof(bf16));

        // 发起 TMA 2D load: global -> shared
        // 坐标 (0, 0) 表示从 tensor 的起始位置开始加载
        cp_async_bulk_tensor_2d_global_to_shared(
            smem, &src_map, /*c0=*/0, /*c1=*/0, &bar);
    }

    // 3. 确保所有线程都能看到 barrier 的初始化
    __syncthreads();

    // 4. 等待 TMA 传输完成（phase parity = 0，第一次使用）
    wait(&bar, /*phaseParity=*/0);

    // 5. 所有线程协作，将 shared memory 数据写回 global memory (dest)
    for (int i = threadIdx.x; i < TILE_M * TILE_N; i += blockDim.x) {
        dest[i] = smem[i];
    }
}

template <int TILE_M, int TILE_N>
void launch_single_tma_load(bf16 *src, bf16 *dest) {
    // 创建 TMA tensor map 描述符
    CUtensorMap tensor_map;

    // TMA 使用 "fastest-dim-first" 的惯例：
    //   对于 row-major [M][N] 矩阵，dim0 = 列(N), dim1 = 行(M)
    uint64_t globalDim[2]     = {TILE_N, TILE_M};
    // globalStrides 只需 rank-1 个值，表示 dim1 方向的字节步长
    uint64_t globalStrides[1] = {TILE_N * sizeof(bf16)};
    // box 大小 = tile 大小（整个矩阵就是一个 tile）
    uint32_t boxDim[2]        = {TILE_N, TILE_M};
    uint32_t elementStrides[2]= {1, 1};

    CUDA_CHECK(cuTensorMapEncodeTiled(
        &tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        /*rank=*/2,
        /*globalAddress=*/(void *)src,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));

    // 启动 1 个 block，128 个线程（用于最后的 store-back）
    single_tma_load<TILE_M, TILE_N><<<1, 128>>>(tensor_map, dest);
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

    printf("\n\nRunning TMA load kernel...\n\n");

    // Launch the TMA kernel
    launch_single_tma_load<M, N>(d_matrix, d_dest);

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