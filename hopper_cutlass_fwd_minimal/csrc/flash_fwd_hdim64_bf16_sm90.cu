// Minimal SM90 bf16 hdim64 forward instantiation.

#include "flash_fwd_launch_template.h"

template void run_mha_fwd_<90, cutlass::bfloat16_t, 64, 64, false, false, false, false>(
    Flash_fwd_params &params,
    cudaStream_t stream);
