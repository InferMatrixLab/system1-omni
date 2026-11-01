// Frozen measurement program for the 128-query attention tile (see ../plan.json).
// Emits one JSON object per line on stdout. One process is one trial.
//
//   attn_record <trial-number> [tokens...]
//
// Arms are interleaved A B B A per repetition so drift affects both equally.
// Both arms compute the same per-query arithmetic; the program also checks that
// their outputs are bit-identical at every measured length.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <random>
#include <vector>
#include "common.cuh"
#include "mma.cuh"

using namespace cs1;

namespace {

constexpr int D = 256, BN = 32;
constexpr int LDS = D + 8;

template <int BM, int THREADS, bool Gated>
__device__ __forceinline__ void flash_body(const bf16* __restrict__ q, const bf16* __restrict__ k,
                                           const bf16* __restrict__ v, int ldv, const bf16* __restrict__ gate,
                                           bf16* __restrict__ out, int T, int Hq, int Hk, float scale_log2) {
    extern __shared__ __align__(16) unsigned char smem[];
    bf16* qs = reinterpret_cast<bf16*>(smem);
    bf16* ks = qs + BM * LDS;
    bf16* vs = ks + BN * LDS;
    const int h = blockIdx.y, hk = h / (Hq / Hk);
    const int q0 = (gridDim.x - 1 - blockIdx.x) * BM;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int g = lane / 4, t = lane % 4;
    const int row0 = q0 + warp * 16;

    for (int c = tid; c < BM * (D / 8); c += THREADS) {
        const int r = c / (D / 8), col = (c % (D / 8)) * 8, row = q0 + r;
        cp_async16(qs + r * LDS + col, q + ((size_t)min(row, T - 1) * Hq + h) * D + col, row < T);
    }
    cp_async_commit();

    float o[D / 8][4];
#pragma unroll
    for (int n = 0; n < D / 8; n++) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};

    const int kv_end = min(T, q0 + BM);
    for (int k0 = 0; k0 < kv_end; k0 += BN) {
        for (int c = tid; c < BN * (D / 8); c += THREADS) {
            const int r = c / (D / 8), col = (c % (D / 8)) * 8, s = k0 + r;
            cp_async16(ks + r * LDS + col, k + ((size_t)min(s, T - 1) * Hk + hk) * D + col, s < T);
        }
        cp_async_commit();
        for (int c = tid; c < BN * (D / 8); c += THREADS) {
            const int r = c / (D / 8), col = (c % (D / 8)) * 8, s = k0 + r;
            cp_async16(vs + r * LDS + col, v + (size_t)min(s, T - 1) * ldv + (size_t)hk * D + col, s < T);
        }
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();

        const bool active = k0 <= row0 + 15;
        float sc[BN / 8][4];
#pragma unroll
        for (int n = 0; n < BN / 8; n++) sc[n][0] = sc[n][1] = sc[n][2] = sc[n][3] = 0.f;
        if (active) {
#pragma unroll
            for (int kk = 0; kk < D; kk += 16) {
                uint32_t a[4];
                load_a(a, qs, LDS, warp * 16, kk, lane);
#pragma unroll
                for (int n = 0; n < BN / 8; n += 2) {
                    uint32_t b[4];
                    load_b_nk(b, ks, LDS, kk, n * 8, lane);
                    mma16816(sc[n], a, b[0], b[1]);
                    mma16816(sc[n + 1], a, b[2], b[3]);
                }
            }
        }
        uint32_t p[BN / 16][4];
        if (active) {
            float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
            for (int n = 0; n < BN / 8; n++) {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int key = k0 + n * 8 + 2 * t + (e & 1), row = row0 + g + (e >> 1) * 8;
                    const bool visible = key <= row && key < T;
                    sc[n][e] = visible ? sc[n][e] * scale_log2 : -INFINITY;
                    mx[e >> 1] = fmaxf(mx[e >> 1], sc[n][e]);
                }
            }
            float alpha[2], base[2];
#pragma unroll
            for (int r = 0; r < 2; r++) {
                mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
                mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
                const float mn = fmaxf(m[r], mx[r]);
                base[r] = mn == -INFINITY ? 0.f : mn;
                alpha[r] = exp2f(m[r] - base[r]);
                m[r] = mn;
                l[r] *= alpha[r];
            }
#pragma unroll
            for (int n = 0; n < BN / 8; n++) {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    sc[n][e] = exp2f(sc[n][e] - base[e >> 1]);
                    l[e >> 1] += sc[n][e];
                }
            }
#pragma unroll
            for (int n = 0; n < D / 8; n++) {
                o[n][0] *= alpha[0];
                o[n][1] *= alpha[0];
                o[n][2] *= alpha[1];
                o[n][3] *= alpha[1];
            }
#pragma unroll
            for (int j = 0; j < BN / 16; j++) {
                p[j][0] = pack_bf16(sc[2 * j][0], sc[2 * j][1]);
                p[j][1] = pack_bf16(sc[2 * j][2], sc[2 * j][3]);
                p[j][2] = pack_bf16(sc[2 * j + 1][0], sc[2 * j + 1][1]);
                p[j][3] = pack_bf16(sc[2 * j + 1][2], sc[2 * j + 1][3]);
            }
        }
        cp_async_wait<0>();
        __syncthreads();
        if (active) {
#pragma unroll
            for (int j = 0; j < BN / 16; j++) {
#pragma unroll
                for (int n = 0; n < D / 8; n += 2) {
                    uint32_t b[4];
                    load_b_kn(b, vs, LDS, j * 16, n * 8, lane);
                    mma16816(o[n], p[j], b[0], b[1]);
                    mma16816(o[n + 1], p[j], b[2], b[3]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < 2; r++) {
        l[r] += __shfl_xor_sync(0xffffffffu, l[r], 1);
        l[r] += __shfl_xor_sync(0xffffffffu, l[r], 2);
    }
    const float inv[2] = {1.f / l[0], 1.f / l[1]};
#pragma unroll
    for (int r = 0; r < 2; r++) {
        const int row = row0 + g + r * 8;
        if (row >= T) continue;
        bf16* dst = out + ((size_t)row * Hq + h) * D + 2 * t;
#pragma unroll
        for (int n = 0; n < D / 8; n++) {
            float a = o[n][2 * r] * inv[r], b = o[n][2 * r + 1] * inv[r];
            if constexpr (Gated) {
                const size_t idx = ((size_t)row * Hq + h) * D + 2 * t + n * 8;
                a = round_bf16(a) * round_bf16(sigmoid(f32(gate[idx])));
                b = round_bf16(b) * round_bf16(sigmoid(f32(gate[idx + 1])));
            }
            *reinterpret_cast<uint32_t*>(dst + n * 8) = pack_bf16(a, b);
        }
    }
}

template <int BM, int THREADS>
__global__ void __launch_bounds__(THREADS)
flash_kernel(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v, int ldv,
             const bf16* __restrict__ gate, bf16* __restrict__ out, int T, int Hq, int Hk, float scale_log2) {
    flash_body<BM, THREADS, true>(q, k, v, ldv, gate, out, T, Hq, Hk, scale_log2);
}

}  // namespace

static void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::printf("{\"kind\":\"error\",\"what\":\"%s\",\"msg\":\"%s\"}\n", what, cudaGetErrorString(e)); std::exit(1); }
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char** argv) {
    const int trial = argc > 1 ? atoi(argv[1]) : 0;
    std::vector<int> tokens = {512, 640, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192};
    if (argc > 2) {
        tokens.clear();
        for (int i = 2; i < argc; i++) tokens.push_back(atoi(argv[i]));
    }
    const int Hq = 24, Hk = 4, ldv = Hk * D;
    int cc_major = 0, cc_minor = 0, driver = 0;
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0), "prop");
    check(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0), "ccmaj");
    check(cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, 0), "ccmin");
    check(cudaDriverGetVersion(&driver), "drv");
    std::printf("{\"kind\":\"device\",\"trial\":%d,\"name\":\"%s\",\"cc\":\"%d.%d\",\"sm_count\":%d,"
                "\"smem_per_sm\":%zu,\"max_smem_optin\":%zu,\"driver_api\":%d,\"cuda_runtime\":%d}\n",
                trial, prop.name, cc_major, cc_minor, prop.multiProcessorCount, prop.sharedMemPerMultiprocessor,
                prop.sharedMemPerBlockOptin, driver, CUDART_VERSION);

    const int smem64 = (64 + 2 * BN) * LDS * 2, smem128 = (128 + 2 * BN) * LDS * 2;
    check(cudaFuncSetAttribute(flash_kernel<64, 128>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem64), "a64");
    check(cudaFuncSetAttribute(flash_kernel<128, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem128), "a128");
    cudaFuncAttributes a64, a128;
    check(cudaFuncGetAttributes(&a64, flash_kernel<64, 128>), "at64");
    check(cudaFuncGetAttributes(&a128, flash_kernel<128, 256>), "at128");
    int b64 = 0, b128 = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b64, flash_kernel<64, 128>, 128, smem64), "o64");
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b128, flash_kernel<128, 256>, 256, smem128), "o128");
    std::printf("{\"kind\":\"occupancy\",\"trial\":%d,\"arm\":\"bm64\",\"regs\":%d,\"smem\":%d,\"blocks_per_sm\":%d,\"warps_per_sm\":%d}\n",
                trial, a64.numRegs, smem64, b64, b64 * 4);
    std::printf("{\"kind\":\"occupancy\",\"trial\":%d,\"arm\":\"bm128\",\"regs\":%d,\"smem\":%d,\"blocks_per_sm\":%d,\"warps_per_sm\":%d}\n",
                trial, a128.numRegs, smem128, b128, b128 * 8);

    const int maxt = *std::max_element(tokens.begin(), tokens.end());
    bf16 *q, *k, *v, *gate, *out0, *out1;
    check(cudaMalloc(&q, (size_t)maxt * Hq * D * 2), "q");
    check(cudaMalloc(&k, (size_t)maxt * Hk * D * 2), "k");
    check(cudaMalloc(&v, (size_t)maxt * ldv * 2), "v");
    check(cudaMalloc(&gate, (size_t)maxt * Hq * D * 2), "g");
    check(cudaMalloc(&out0, (size_t)maxt * Hq * D * 2), "o0");
    check(cudaMalloc(&out1, (size_t)maxt * Hq * D * 2), "o1");
    // Incompressible seeded inputs; both arms read the same buffer.
    std::mt19937 rng(20261101);
    std::vector<unsigned char> qb((size_t)maxt * Hq * D * 2), kvb((size_t)maxt * ldv * 2);
    for (auto& b : qb) b = (unsigned char)(rng() & 0xff);
    for (auto& b : kvb) b = (unsigned char)(rng() & 0xff);
    check(cudaMemcpy(q, qb.data(), qb.size(), cudaMemcpyHostToDevice), "upq");
    check(cudaMemcpy(gate, qb.data(), qb.size(), cudaMemcpyHostToDevice), "upg");
    check(cudaMemcpy(k, kvb.data(), kvb.size(), cudaMemcpyHostToDevice), "upk");
    check(cudaMemcpy(v, kvb.data(), kvb.size(), cudaMemcpyHostToDevice), "upv");
    cudaStream_t s;
    check(cudaStreamCreate(&s), "stream");
    const float scale = 0.0625f * 1.4426950408889634f;

    auto arm = [&](int T, bool wide) {
        if (wide)
            flash_kernel<128, 256><<<dim3((T + 127) / 128, Hq), 256, smem128, s>>>(q, k, v, ldv, gate, out1, T, Hq, Hk, scale);
        else
            flash_kernel<64, 128><<<dim3((T + 63) / 64, Hq), 128, smem64, s>>>(q, k, v, ldv, gate, out0, T, Hq, Hk, scale);
    };
    auto batch = [&](int T, bool wide) {
        cudaEvent_t a, b;
        cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a, s);
        for (int i = 0; i < 5; i++) arm(T, wide);
        cudaEventRecord(b, s);
        cudaEventSynchronize(b);
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        cudaEventDestroy(a); cudaEventDestroy(b);
        return (double)ms / 5;
    };

    for (int T : tokens) {
        // parity: both arms over the same buffer
        check(cudaMemset(out0, 0, (size_t)T * Hq * D * 2), "m0");
        check(cudaMemset(out1, 0, (size_t)T * Hq * D * 2), "m1");
        arm(T, false);
        arm(T, true);
        check(cudaStreamSynchronize(s), "sync");
        std::vector<unsigned char> x((size_t)T * Hq * D * 2), y(x.size());
        check(cudaMemcpy(x.data(), out0, x.size(), cudaMemcpyDeviceToHost), "d0");
        check(cudaMemcpy(y.data(), out1, x.size(), cudaMemcpyDeviceToHost), "d1");
        std::printf("{\"kind\":\"parity\",\"trial\":%d,\"tokens\":%d,\"identical\":%s}\n", trial, T,
                    x == y ? "true" : "false");

        arm(T, false); arm(T, true);
        check(cudaStreamSynchronize(s), "warm");
        std::vector<double> a, b;
        for (int rep = 0; rep < 10; rep++) {  // A B B A
            a.push_back(batch(T, false));
            b.push_back(batch(T, true));
            b.push_back(batch(T, true));
            a.push_back(batch(T, false));
        }
        for (size_t i = 0; i < a.size(); i++) {
            std::printf("{\"kind\":\"sample\",\"trial\":%d,\"tokens\":%d,\"arm\":\"bm64\",\"batch\":%zu,\"ms\":%.6f}\n",
                        trial, T, i, a[i]);
            std::printf("{\"kind\":\"sample\",\"trial\":%d,\"tokens\":%d,\"arm\":\"bm128\",\"batch\":%zu,\"ms\":%.6f}\n",
                        trial, T, i, b[i]);
        }
        std::printf("{\"kind\":\"median\",\"trial\":%d,\"tokens\":%d,\"bm64\":%.6f,\"bm128\":%.6f,\"delta_pct\":%.3f}\n",
                    trial, T, median(a), median(b), (median(b) - median(a)) / median(a) * 100.0);
    }
    return 0;
}
