#include <cmath>
#include <cstdint>
#include <optional>
#include <tuple>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <cutlass/numeric_types.h>

#include "cuda_check.h"
#include "flash.h"

namespace {

#define CHECK_DEVICE(x) TORCH_CHECK((x).is_cuda(), #x " must be on CUDA")
#define CHECK_LASTDIM_CONTIGUOUS(x) TORCH_CHECK((x).stride(-1) == 1, #x " must have contiguous last dimension")

int round_multiple(int x, int m) {
    return (x + m - 1) / m * m;
}

void set_minimal_fwd_params(
        Flash_fwd_params &params,
        const at::Tensor &q,
        const at::Tensor &k,
        const at::Tensor &v,
        at::Tensor &out,
        at::Tensor &lse,
        float softmax_scale,
        bool causal) {
    params = {};

    const int batch = q.size(0);
    const int seqlen_q = q.size(1);
    const int num_heads = q.size(2);
    const int head_dim = q.size(3);
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);
    const int head_dim_v = v.size(3);

    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.o_ptr = out.data_ptr();
    params.softmax_lse_ptr = lse.data_ptr();

    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = k.stride(0);
    params.v_batch_stride = v.stride(0);
    params.o_batch_stride = out.stride(0);

    params.q_row_stride = q.stride(1);
    params.k_row_stride = k.stride(1);
    params.v_row_stride = v.stride(1);
    params.o_row_stride = out.stride(1);

    params.q_head_stride = q.stride(2);
    params.k_head_stride = k.stride(2);
    params.v_head_stride = v.stride(2);
    params.o_head_stride = out.stride(2);
    params.v_dim_stride = v.stride(3);

    params.b = batch;
    params.h = num_heads;
    params.h_k = num_heads_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = round_multiple(seqlen_q, 128);
    params.seqlen_k_rounded = round_multiple(seqlen_k, 128);
    params.d = head_dim;
    params.dv = head_dim_v;
    params.d_rounded = 64;
    params.dv_rounded = 64;

    params.scale_softmax = softmax_scale;
    params.softcap = 0.0f;
    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = uint8_t(255);
    params.rp_dropout = 1.0f;

    params.is_bf16 = true;
    params.is_fp32 = false;
    params.is_e4m3 = false;
    params.is_causal = causal;
    params.is_local = false;
    params.window_size_left = seqlen_k - 1;
    params.window_size_right = causal ? 0 : seqlen_q - 1;
    params.attention_chunk = 0;

    params.num_splits = 1;
    params.pack_gqa = false;
    params.pagedkv_tma = false;
    params.page_size = 1;
    params.varlen_sort_batches = true;
    params.head_swizzle = causal;

    const cudaDeviceProp *props = at::cuda::getCurrentDeviceProperties();
    params.arch = props->major * 10 + props->minor;
    params.num_sm = props->multiProcessorCount;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> fwd(
        at::Tensor q,
        at::Tensor k,
        at::Tensor v,
        bool causal,
        std::optional<double> softmax_scale_) {
    CHECK_DEVICE(q);
    CHECK_DEVICE(k);
    CHECK_DEVICE(v);
    CHECK_LASTDIM_CONTIGUOUS(q);
    CHECK_LASTDIM_CONTIGUOUS(k);
    CHECK_LASTDIM_CONTIGUOUS(v);

    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q, k, v must be 4D");
    TORCH_CHECK(q.scalar_type() == at::ScalarType::BFloat16, "minimal CUTLASS fwd only supports bf16");
    TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(), "q, k, v dtype mismatch");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "batch size mismatch");
    TORCH_CHECK(k.size(1) == v.size(1), "K/V sequence length mismatch");
    TORCH_CHECK(q.size(2) == k.size(2) && q.size(2) == v.size(2), "minimal CUTLASS fwd only supports MHA, not GQA/MQA");
    TORCH_CHECK(q.size(3) == 64 && k.size(3) == 64 && v.size(3) == 64, "minimal CUTLASS fwd only supports head_dim=64");

    const c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(q.get_device()));
    const cudaDeviceProp *props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(props->major == 9, "minimal CUTLASS fwd requires Hopper / SM90");

    const int batch = q.size(0);
    const int seqlen_q = q.size(1);
    const int num_heads = q.size(2);
    const int head_dim = q.size(3);
    const float softmax_scale = softmax_scale_.has_value()
        ? static_cast<float>(softmax_scale_.value())
        : 1.0f / std::sqrt(static_cast<float>(head_dim));

    at::Tensor out = at::empty({batch, seqlen_q, num_heads, head_dim}, q.options());
    at::Tensor lse = at::empty({batch, num_heads, seqlen_q}, q.options().dtype(at::kFloat));

    Flash_fwd_params params;
    set_minimal_fwd_params(params, q, k, v, out, lse, softmax_scale, causal);
    at::Tensor tile_count_semaphore;
    if (params.is_causal || params.is_local) {
        tile_count_semaphore = at::zeros({1}, q.options().dtype(at::kInt));
        params.tile_count_semaphore = tile_count_semaphore.data_ptr<int>();
        params.tile_count_semaphore_offset = 0;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    run_mha_fwd_<90, cutlass::bfloat16_t, 64, 64, false, false, false, false>(params, stream);
    CHECK_CUDA_KERNEL_LAUNCH();

    return {out, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fwd, "Minimal Hopper CUTLASS FlashAttention forward");
}
