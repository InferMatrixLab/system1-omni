// Embedding lookup, the Gated DeltaNet conv and gates, and the elementwise
// activations of Qwen3.5.
#include "common.cuh"
#include "ops.h"

namespace cs1 {
namespace {

constexpr int THREADS = 256;

__global__ void embed_kernel(const int32_t* __restrict__ ids, const Pack8* __restrict__ table,
                             Pack8* __restrict__ out, int packs) {
    const size_t t = blockIdx.x;
    const size_t id = ids[t];
    for (int i = threadIdx.x; i < packs; i += blockDim.x) out[t * packs + i] = table[id * packs + i];
}

// F.conv1d in bfloat16 (float32 accumulation, rounded), then SiLU (rounded again),
// written to three contiguous outputs.
__global__ void gdn_conv_kernel(const bf16* __restrict__ qkv, int ld, const bf16* __restrict__ w,
                                bf16* __restrict__ q, bf16* __restrict__ k, bf16* __restrict__ v, int T,
                                int key_dim, int value_dim) {
    const int channels = 2 * key_dim + value_dim;
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)T * channels) return;
    const int t = idx / channels, c = idx % channels;
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int s = t - 3 + j;
        if (s >= 0) acc = fmaf(f32(w[c * 4 + j]), f32(qkv[(size_t)s * ld + c]), acc);
    }
    const bf16 y = to_bf16(silu(round_bf16(acc)));
    if (c < key_dim)
        q[(size_t)t * key_dim + c] = y;
    else if (c < 2 * key_dim)
        k[(size_t)t * key_dim + c - key_dim] = y;
    else
        v[(size_t)t * value_dim + c - 2 * key_dim] = y;
}

// Identical four-tap accumulation with the preceding three projection rows.
__global__ void gdn_conv_prefix_kernel(const bf16* __restrict__ qkv, const bf16* __restrict__ history, int ld, const bf16* __restrict__ w,
                                bf16* __restrict__ q, bf16* __restrict__ k, bf16* __restrict__ v, int T,
                                int key_dim, int value_dim) {
    const int channels = 2 * key_dim + value_dim;
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)T * channels) return;
    const int t = idx / channels, c = idx % channels;
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int s = t - 3 + j;
        const bf16* source = s < 0 ? history + (size_t)(s + 3) * ld : qkv + (size_t)s * ld;
        acc = fmaf(f32(w[c * 4 + j]), f32(source[c]), acc);
    }
    const bf16 y = to_bf16(silu(round_bf16(acc)));
    if (c < key_dim)
        q[(size_t)t * key_dim + c] = y;
    else if (c < 2 * key_dim)
        k[(size_t)t * key_dim + c - key_dim] = y;
    else
        v[(size_t)t * value_dim + c - 2 * key_dim] = y;
}

// Eight adjacent channels retain the original four FMAs and both BF16 roundings.
// Separate token/channel grid dimensions avoid per-element integer division.
template<bool Prefix>
__global__ void gdn_conv_vector_kernel(const Pack8* __restrict__ qkv,
                                        const Pack8* __restrict__ history, int ld,
                                        const Pack8* __restrict__ w, Pack8* __restrict__ q,
                                        Pack8* __restrict__ k, Pack8* __restrict__ v,
                                        int key_packs, int value_packs) {
    const int t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= 2 * key_packs + value_packs) return;
    const Pack8 weights[4] = {w[c * 4], w[c * 4 + 1], w[c * 4 + 2], w[c * 4 + 3]};
    float acc[8] = {};
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int row = t - 3 + j;
        if (Prefix || row >= 0) {
            const Pack8 value = row < 0 ? history[(size_t)(row + 3) * ld + c]
                                        : qkv[(size_t)row * ld + c];
#pragma unroll
            for (int e = 0; e < 8; e++)
                acc[e] = fmaf(f32(weights[e / 2].v[(e % 2) * 4 + j]), f32(value.v[e]), acc[e]);
        }
    }
    Pack8 out;
#pragma unroll
    for (int e = 0; e < 8; e++) out.v[e] = to_bf16(silu(round_bf16(acc[e])));
    if (c < key_packs)
        q[(size_t)t * key_packs + c] = out;
    else if (c < 2 * key_packs)
        k[(size_t)t * key_packs + c - key_packs] = out;
    else
        v[(size_t)t * value_packs + c - 2 * key_packs] = out;
}

// beta = sigmoid(b) in bfloat16; g = -exp(A_log) * softplus(a + dt_bias) in float32
// (F.softplus with threshold 20).
__global__ void gdn_gates_kernel(const bf16* __restrict__ b, const bf16* __restrict__ a, int ld,
                                 const bf16* __restrict__ A_log, const bf16* __restrict__ dt_bias,
                                 bf16* __restrict__ beta, float* __restrict__ g, int n, int H) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int h = i % H;
    const size_t src = (size_t)(i / H) * ld + h;
    beta[i] = to_bf16(sigmoid(f32(b[src])));
    const float x = f32(a[src]) + f32(dt_bias[h]);
    const float sp = x > 20.f ? x : log1pf(expf(x));
    g[i] = -expf(f32(A_log[h])) * sp;
}

// attn_output * torch.sigmoid(gate), both bfloat16.
__global__ void sigmoid_gate_kernel(bf16* __restrict__ x, const bf16* __restrict__ gate, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = to_bf16(f32(x[i]) * round_bf16(sigmoid(f32(gate[i]))));
}

// act_fn(gate_proj(x)) * up_proj(x), both bfloat16; gate and up are the two halves of
// each row of gate_up.
__global__ void silu_mul_kernel(const bf16* __restrict__ gate_up, int ld, bf16* __restrict__ out, int I,
                                size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t t = i / I, j = i % I;
    const bf16* row = gate_up + t * ld;
    out[i] = to_bf16(round_bf16(silu(f32(row[j]))) * f32(row[I + j]));
}

__global__ void silu_mul_packed_kernel(const Pack8* __restrict__ gate_up, int ld,
                                       Pack8* __restrict__ out, int I, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t t = i / I, j = i % I;
    const Pack8* row = gate_up + t * ld;
    const Pack8 gate = row[j], up = row[I + j];
    Pack8 result;
#pragma unroll
    for (int e = 0; e < 8; e++)
        result.v[e] = to_bf16(round_bf16(silu(f32(gate.v[e]))) * f32(up.v[e]));
    out[i] = result;
}

__global__ void bf16_finite_kernel(const uint16_t* x, size_t n, uint32_t* flag) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        if ((x[i] & 0x7f80u) == 0x7f80u) atomicExch(flag, 1u);
    }
}

unsigned blocks(size_t n) { return (unsigned)((n + THREADS - 1) / THREADS); }

}  // namespace
}  // namespace cs1

using namespace cs1;

extern "C" int cs1_embed(const int32_t* ids, const void* table, void* out, int T, int D, void* stream) {
    if (D % 8 != 0) return cudaErrorInvalidValue;
    if (T <= 0) return cudaSuccess;
    embed_kernel<<<T, THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        ids, static_cast<const Pack8*>(table), static_cast<Pack8*>(out), D / 8);
    return cudaGetLastError();
}

extern "C" int cs1_gdn_conv(const void* qkv, int ld, const void* w, void* q, void* k, void* v, int T, int key_dim,
                            int value_dim, void* stream) {
    if (T < 0 || key_dim < 0 || value_dim < 0 || ld < 2 * key_dim + value_dim) return cudaErrorInvalidValue;
    const size_t n = (size_t)T * (2 * key_dim + value_dim);
    if (n == 0) return cudaSuccess;
    gdn_conv_kernel<<<blocks(n), THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const bf16*>(qkv), ld, static_cast<const bf16*>(w), static_cast<bf16*>(q),
        static_cast<bf16*>(k), static_cast<bf16*>(v), T, key_dim, value_dim);
    return cudaGetLastError();
}

extern "C" int cs1_gdn_gates(const void* b, const void* a, int ld, const void* A_log, const void* dt_bias,
                             void* beta, float* g, int T, int H, void* stream) {
    if (T < 0 || H < 0 || ld < H) return cudaErrorInvalidValue;
    const int n = T * H;
    if (n == 0) return cudaSuccess;
    gdn_gates_kernel<<<blocks(n), THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const bf16*>(b), static_cast<const bf16*>(a), ld, static_cast<const bf16*>(A_log),
        static_cast<const bf16*>(dt_bias), static_cast<bf16*>(beta), g, n, H);
    return cudaGetLastError();
}

extern "C" int cs1_sigmoid_gate(void* x, const void* gate, size_t n, void* stream) {
    if (n == 0) return cudaSuccess;
    sigmoid_gate_kernel<<<blocks(n), THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<bf16*>(x), static_cast<const bf16*>(gate), n);
    return cudaGetLastError();
}

extern "C" int cs1_silu_mul(const void* gate_up, int ld, void* out, int T, int I, void* stream) {
    if (T < 0 || I < 0 || ld < 2 * I) return cudaErrorInvalidValue;
    const size_t n = (size_t)T * I;
    if (n == 0) return cudaSuccess;
    if (I % 8 == 0 && ld % 8 == 0 &&
        ((reinterpret_cast<uintptr_t>(gate_up) | reinterpret_cast<uintptr_t>(out)) & 15) == 0) {
        silu_mul_packed_kernel<<<blocks(n / 8), THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const Pack8*>(gate_up), ld / 8, static_cast<Pack8*>(out), I / 8, n / 8);
    } else {
        silu_mul_kernel<<<blocks(n), THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const bf16*>(gate_up), ld, static_cast<bf16*>(out), I, n);
    }
    return cudaGetLastError();
}

extern "C" int cs1_bf16_finite(const void* x, size_t n, uint32_t* flag, void* stream) {
    auto st = static_cast<cudaStream_t>(stream);
    cudaError_t e = cudaMemsetAsync(flag, 0, sizeof(uint32_t), st);
    if (e != cudaSuccess) return e;
    if (n) bf16_finite_kernel<<<static_cast<unsigned int>(((n - 1) / 256 + 1) < 65535 ? ((n - 1) / 256 + 1) : 65535), 256, 0, st>>>(
        static_cast<const uint16_t*>(x), n, flag);
    return cudaGetLastError();
}

extern "C" int cs1_gdn_conv_prefix(const void* qkv, const void* history, int ld, const void* w,
                                   void* q, void* k, void* v, int T, int key_dim, int value_dim,
                                   void* stream) {
    if (!qkv || !history || !w || !q || !k || !v || T <= 0 || key_dim <= 0 || value_dim <= 0 ||
        ld < 2 * key_dim + value_dim) return cudaErrorInvalidValue;
    const size_t n = (size_t)T * (2 * key_dim + value_dim);
    const uintptr_t alignment = reinterpret_cast<uintptr_t>(qkv) | reinterpret_cast<uintptr_t>(history) |
        reinterpret_cast<uintptr_t>(w) | reinterpret_cast<uintptr_t>(q) | reinterpret_cast<uintptr_t>(k) |
        reinterpret_cast<uintptr_t>(v);
    if (T >= 64 && T <= 65535 && key_dim % 8 == 0 && value_dim % 8 == 0 && ld % 8 == 0 && !(alignment & 15)) {
        const int packs = (2 * key_dim + value_dim) / 8;
        gdn_conv_vector_kernel<true><<<dim3((packs + THREADS - 1) / THREADS, T), THREADS, 0,
                                       static_cast<cudaStream_t>(stream)>>>(
            static_cast<const Pack8*>(qkv), static_cast<const Pack8*>(history), ld / 8,
            static_cast<const Pack8*>(w), static_cast<Pack8*>(q), static_cast<Pack8*>(k),
            static_cast<Pack8*>(v), key_dim / 8, value_dim / 8);
    } else {
    gdn_conv_prefix_kernel<<<blocks(n), THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const bf16*>(qkv), static_cast<const bf16*>(history), ld, static_cast<const bf16*>(w),
        static_cast<bf16*>(q), static_cast<bf16*>(k), static_cast<bf16*>(v), T, key_dim, value_dim);
    }
    return cudaGetLastError();
}

// Additive opt-in entry point; unsupported layouts preserve the scalar path.
extern "C" int cs1_gdn_conv_vector(const void* qkv, int ld, const void* w, void* q,
                                   void* k, void* v, int T, int key_dim, int value_dim,
                                   void* stream) {
    const uintptr_t alignment = reinterpret_cast<uintptr_t>(qkv) | reinterpret_cast<uintptr_t>(w) |
        reinterpret_cast<uintptr_t>(q) | reinterpret_cast<uintptr_t>(k) | reinterpret_cast<uintptr_t>(v);
    if (T < 64 || T > 65535 || key_dim <= 0 || value_dim <= 0 ||
        key_dim % 8 || value_dim % 8 || ld % 8 || ld < 2 * key_dim + value_dim || (alignment & 15))
        return cs1_gdn_conv(qkv, ld, w, q, k, v, T, key_dim, value_dim, stream);
    const int packs = (2 * key_dim + value_dim) / 8;
    gdn_conv_vector_kernel<false><<<dim3((packs + THREADS - 1) / THREADS, T), THREADS, 0,
                                    static_cast<cudaStream_t>(stream)>>>(
        static_cast<const Pack8*>(qkv), nullptr, ld / 8, static_cast<const Pack8*>(w),
        static_cast<Pack8*>(q), static_cast<Pack8*>(k), static_cast<Pack8*>(v), key_dim / 8, value_dim / 8);
    return cudaGetLastError();
}
