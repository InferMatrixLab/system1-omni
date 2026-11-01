// Library-level recorder for the same change: calls the public cs1_attention_gated
// entry point and writes the output bits for cross-build comparison.
// The caller swaps libqwen3_5_cuda.so in the loader directory between arms.
//
//   attn_lib_record <revision-label> <tokens> <output-file>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <random>
#include <vector>
#include "ops.h"

#define CK(x)                                                                     \
    do {                                                                          \
        int _rc = (x);                                                            \
        if (_rc) {                                                                \
            std::printf("{\"kind\":\"error\",\"what\":\"%s\",\"msg\":\"%s\"}\n", #x, cs1_error_string(_rc)); \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <revision-label> <tokens> <output-file> [trial]\n", argv[0]);
        return 2;
    }
    const char* label = argv[1];
    const int T = atoi(argv[2]);
    const char* path = argv[3];
    const int trial = argc > 4 ? atoi(argv[4]) : 0;
    const int Hq = 24, Hk = 4, Dh = 256, ldv = Hk * Dh;
    int cc_major = 0, cc_minor = 0;
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    CK(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0));
    CK(cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, 0));
    int driver = 0;
    CK(cudaDriverGetVersion(&driver));
    std::printf("{\"kind\":\"device\",\"trial\":%d,\"revision\":\"%s\",\"name\":\"%s\",\"cc\":\"%d.%d\","
                "\"sm_count\":%d,\"driver_api\":%d}\n",
                trial, label, prop.name, cc_major, cc_minor, prop.multiProcessorCount, driver);

    CK(cs1_set_device(0));
    void* s = nullptr;
    CK(cs1_stream_create(&s));
    std::mt19937 rng(20261101);
    auto fill = [&](size_t n) {
        std::vector<unsigned char> v(n);
        for (auto& b : v) b = (unsigned char)(rng() & 0xff);
        return v;
    };
    const size_t qbytes = (size_t)T * Hq * Dh * 2, kvbytes = (size_t)T * ldv * 2;
    void *q, *k, *v, *gate, *out;
    CK(cs1_malloc(&q, qbytes)); CK(cs1_malloc(&k, kvbytes)); CK(cs1_malloc(&v, kvbytes));
    CK(cs1_malloc(&gate, qbytes)); CK(cs1_malloc(&out, qbytes));
    auto qd = fill(qbytes), kvd = fill(kvbytes);
    CK(cs1_upload(q, qd.data(), qd.size(), s));
    CK(cs1_upload(gate, qd.data(), qd.size(), s));
    CK(cs1_upload(k, kvd.data(), kvd.size(), s));
    CK(cs1_upload(v, kvd.data(), kvd.size(), s));
    CK(cs1_stream_sync(s));
    auto run = [&] { CK(cs1_attention_gated(q, k, v, ldv, gate, out, T, Hq, Hk, Dh, 0.0625f, s)); };
    run();
    CK(cs1_stream_sync(s));
    std::vector<unsigned char> result(qbytes);
    CK(cs1_download(result.data(), out, qbytes, s));
    FILE* f = std::fopen(path, "wb");
    if (!f) { std::printf("{\"kind\":\"error\",\"what\":\"open\"}\n"); return 1; }
    std::fwrite(result.data(), 1, result.size(), f);
    std::fclose(f);

    for (int rep = 0; rep < 3; rep++) run();
    CK(cs1_stream_sync(s));
    std::vector<double> samples;
    for (int rep = 0; rep < 10; rep++) {
        cudaEvent_t a, b;
        cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a, (cudaStream_t)s);
        for (int i = 0; i < 5; i++) run();
        cudaEventRecord(b, (cudaStream_t)s);
        cudaEventSynchronize(b);
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        samples.push_back(ms / 5.0);
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    for (size_t i = 0; i < samples.size(); i++)
        std::printf("{\"kind\":\"lib_sample\",\"trial\":%d,\"revision\":\"%s\",\"tokens\":%d,\"batch\":%zu,\"ms\":%.6f}\n",
                    trial, label, T, i, samples[i]);
    std::sort(samples.begin(), samples.end());
    std::printf("{\"kind\":\"lib_median\",\"trial\":%d,\"revision\":\"%s\",\"tokens\":%d,\"ms\":%.6f}\n",
                trial, label, T, samples[samples.size() / 2]);
    cs1_stream_destroy(s);
    return 0;
}
