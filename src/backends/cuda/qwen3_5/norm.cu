// RMSNorm variants of Qwen3.5.
#include "common.cuh"
#include "ops.h"

namespace cs1 {
namespace {

constexpr int NORM_THREADS = 256;

// Qwen3_5RMSNorm: float32 normalization, times (1 + w), rounded once to bfloat16.
__global__ void __launch_bounds__(NORM_THREADS)
    rms_norm_kernel(const bf16* __restrict__ x, const bf16* __restrict__ w, bf16* __restrict__ out, int D,
                    float eps) {
    __shared__ float scratch[32];
    const size_t row = blockIdx.x;
    x += row * D;
    out += row * D;
    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float v = f32(x[i]);
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss, scratch) / D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) out[i] = to_bf16(f32(x[i]) * inv * (1.f + f32(w[i])));
}

// The residual add of a decoder layer (bfloat16 + bfloat16, rounded) fused with the
// following Qwen3_5RMSNorm. Each thread rereads only the elements it wrote.
__global__ void __launch_bounds__(NORM_THREADS)
    add_rms_norm_kernel(bf16* __restrict__ residual, const bf16* __restrict__ delta, const bf16* __restrict__ w,
                        bf16* __restrict__ out, int D, float eps) {
    __shared__ float scratch[32];
    const size_t row = blockIdx.x;
    residual += row * D;
    delta += row * D;
    out += row * D;
    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const bf16 r = to_bf16(f32(residual[i]) + f32(delta[i]));
        residual[i] = r;
        const float v = f32(r);
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss, scratch) / D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        out[i] = to_bf16(f32(residual[i]) * inv * (1.f + f32(w[i])));
}

// Keep each thread's rounded residuals in registers across the reduction. Retain
// the scalar kernel's element assignment, sum order, and 256-thread block.
template <int D>
__global__ void __launch_bounds__(NORM_THREADS)
    add_rms_norm_cached_kernel(bf16* __restrict__ residual, const bf16* __restrict__ delta,
                               const bf16* __restrict__ w, bf16* __restrict__ out, float eps) {
    __shared__ float scratch[32];
    const size_t row = (size_t)blockIdx.x * D;
    float r[D / NORM_THREADS];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < D / NORM_THREADS; i++) {
        const int col = threadIdx.x + i * NORM_THREADS;
        const bf16 value = to_bf16(f32(residual[row + col]) + f32(delta[row + col]));
        residual[row + col] = value;
        r[i] = f32(value);
        ss += r[i] * r[i];
    }
    const float inv = rsqrtf(block_sum(ss, scratch) / D + eps);
#pragma unroll
    for (int i = 0; i < D / NORM_THREADS; i++) {
        const int col = threadIdx.x + i * NORM_THREADS;
        out[row + col] = to_bf16(r[i] * inv * (1.f + f32(w[col])));
    }
}

// Vectorize memory traffic for 4096-wide rows while retaining the scalar
// kernel's FP32 reduction order. The shared transpose gives each thread its
// original strided elements; rounded BF16 values remain in registers for output.
__global__ void __launch_bounds__(NORM_THREADS)
    add_rms_norm_vector_4096_kernel(Pack8* __restrict__ residual, const Pack8* __restrict__ delta,
                                    const Pack8* __restrict__ w, Pack8* __restrict__ out, float eps) {
    constexpr int D = 4096, PACKS = D / (8 * NORM_THREADS);
    __shared__ float scratch[32];
    __shared__ __align__(16) bf16 shared_values[D];
    const size_t base = (size_t)blockIdx.x * (D / 8);
    Pack8 values[PACKS];
#pragma unroll
    for (int i = 0; i < PACKS; ++i) {
        const int col = threadIdx.x + i * NORM_THREADS;
        const Pack8 a = residual[base + col], b = delta[base + col];
#pragma unroll
        for (int e = 0; e < 8; ++e) values[i].v[e] = to_bf16(f32(a.v[e]) + f32(b.v[e]));
        residual[base + col] = values[i];
        reinterpret_cast<Pack8*>(shared_values)[col] = values[i];
    }
    __syncthreads();
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < D / NORM_THREADS; ++i) {
        const float v = f32(shared_values[threadIdx.x + i * NORM_THREADS]);
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss, scratch) / D + eps);
#pragma unroll
    for (int i = 0; i < PACKS; ++i) {
        const int col = threadIdx.x + i * NORM_THREADS;
        const Pack8 weight = w[col];
        Pack8 result;
#pragma unroll
        for (int e = 0; e < 8; ++e)
            result.v[e] = to_bf16(f32(values[i].v[e]) * inv * (1.f + f32(weight.v[e])));
        out[base + col] = result;
    }
}

// Qwen3_5RMSNormGated with D = 128, one warp per row: the normalized value is
// rounded to bfloat16, multiplied by w in bfloat16, then by silu(z) in float32.
__global__ void gated_rms_norm_kernel(const bf16* __restrict__ x, const bf16* __restrict__ z, int ldz,
                                      const bf16* __restrict__ w, bf16* __restrict__ out, int rows, int H,
                                      float eps) {
    constexpr int D = 128, PER = D / 32;
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (row >= rows) return;
    const size_t base = (size_t)row * D + lane * PER;
    const size_t zbase = (size_t)(row / H) * ldz + (row % H) * D + lane * PER;
    float v[PER];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < PER; i++) {
        v[i] = f32(x[base + i]);
        ss += v[i] * v[i];
    }
    const float inv = rsqrtf(warp_sum(ss) / D + eps);
#pragma unroll
    for (int i = 0; i < PER; i++) {
        const float normed = round_bf16(v[i] * inv);
        const float scaled = round_bf16(f32(w[lane * PER + i]) * normed);
        out[base + i] = to_bf16(scaled * silu(f32(z[zbase + i])));
    }
}

}  // namespace
}  // namespace cs1

using namespace cs1;

extern "C" int cs1_rms_norm(const void* x, const void* w, void* out, int rows, int D, float eps, void* stream) {
    if (rows <= 0) return cudaSuccess;
    rms_norm_kernel<<<rows, NORM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const bf16*>(x), static_cast<const bf16*>(w), static_cast<bf16*>(out), D, eps);
    return cudaGetLastError();
}

extern "C" int cs1_add_rms_norm(void* residual, const void* delta, const void* w, void* out, int rows, int D,
                                float eps, void* stream) {
    if (rows <= 0) return cudaSuccess;
    if (D == 4096 && ((reinterpret_cast<uintptr_t>(residual) | reinterpret_cast<uintptr_t>(delta) |
                       reinterpret_cast<uintptr_t>(w) | reinterpret_cast<uintptr_t>(out)) & 15) == 0) {
        add_rms_norm_vector_4096_kernel<<<rows, NORM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<Pack8*>(residual), static_cast<const Pack8*>(delta), static_cast<const Pack8*>(w),
            static_cast<Pack8*>(out), eps);
        return cudaGetLastError();
    }
    switch (D) {
#define CACHED_NORM(width) \
    case width: \
        add_rms_norm_cached_kernel<width><<<rows, NORM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>( \
            static_cast<bf16*>(residual), static_cast<const bf16*>(delta), static_cast<const bf16*>(w), \
            static_cast<bf16*>(out), eps); \
        return cudaGetLastError()
        CACHED_NORM(2560);
        CACHED_NORM(5120);
#undef CACHED_NORM
    }
    add_rms_norm_kernel<<<rows, NORM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<bf16*>(residual), static_cast<const bf16*>(delta), static_cast<const bf16*>(w),
        static_cast<bf16*>(out), D, eps);
    return cudaGetLastError();
}

extern "C" int cs1_gated_rms_norm(const void* x, const void* z, int ldz, const void* w, void* out, int T, int H,
                                  int D, float eps, void* stream) {
    if (D != 128 || ldz < H * D) return cudaErrorInvalidValue;
    const int rows = T * H;
    if (rows <= 0) return cudaSuccess;
    constexpr int WARPS = 8;
    gated_rms_norm_kernel<<<(rows + WARPS - 1) / WARPS, WARPS * 32, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const bf16*>(x), static_cast<const bf16*>(z), ldz, static_cast<const bf16*>(w),
        static_cast<bf16*>(out), rows, H, eps);
    return cudaGetLastError();
}
