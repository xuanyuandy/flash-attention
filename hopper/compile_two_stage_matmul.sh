#!/bin/bash
# compile_two_stage_matmul.sh
# 编译 two_stage_matmul_sm90.cu
#
# 关键 flag 说明:
#   -arch=sm_90a              Hopper 扩展指令集 (wgmma/TMA), 必须用 90a 而非 90
#   --expt-relaxed-constexpr  允许 device 调用 constexpr host 函数 (std::min/max),
#                             CUTLASS/FA3 全系必需
#   -DTWO_STAGE_MATMUL_TEST   编译测试 main; 去掉则只编译 kernel + host wrapper

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUTLASS_INC="${SCRIPT_DIR}/../csrc/cutlass/include"
HOPPER_INC="${SCRIPT_DIR}"

nvcc \
  -arch=sm_90a \
  -std=c++17 \
  -O3 \
  --expt-relaxed-constexpr \
  -DTWO_STAGE_MATMUL_TEST \
  -I"${CUTLASS_INC}" \
  -I"${HOPPER_INC}" \
  "${SCRIPT_DIR}/two_stage_matmul_sm90.cu" \
  -o "${SCRIPT_DIR}/two_stage_matmul_sm90"

echo "build exit: $?"
