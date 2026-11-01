// Checks cs1_attention_gated_prefix against the full pass for one prompt, for a
// 128-aligned q_base (wide block) and a 64-aligned one (64-query block).
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "ops.h"

#define CK(x)                                                                     \
    do {                                                                          \
        int _rc = (x);                                                            \
        if (_rc) {                                                                \
            std::printf("FAIL %s rc=%d (%s)\n", #x, _rc, cs1_error_string(_rc));  \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

int main(int argc, char** argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 2048;
    const int Hq = 24, Hk = 4, Dh = 256, ldv = Hk * Dh;
    CK(cs1_set_device(0));
    void* s = nullptr;
    CK(cs1_stream_create(&s));
    std::mt19937 rng(4242);
    auto fill = [&](size_t n) {
        std::vector<unsigned char> v(n);
        for (auto& b : v) b = (unsigned char)(rng() & 0xff);
        return v;
    };
    const size_t qbytes = (size_t)T * Hq * Dh * 2, kvbytes = (size_t)T * ldv * 2;
    void *q, *k, *v, *gate, *full, *win;
    CK(cs1_malloc(&q, qbytes)); CK(cs1_malloc(&k, kvbytes)); CK(cs1_malloc(&v, kvbytes));
    CK(cs1_malloc(&gate, qbytes)); CK(cs1_malloc(&full, qbytes)); CK(cs1_malloc(&win, qbytes));
    auto qd = fill(qbytes), gd = fill(qbytes), kvd = fill(kvbytes);
    CK(cs1_upload(q, qd.data(), qd.size(), s));
    CK(cs1_upload(gate, gd.data(), gd.size(), s));
    CK(cs1_upload(k, kvd.data(), kvd.size(), s));
    CK(cs1_upload(v, kvd.data(), kvd.size(), s));
    CK(cs1_stream_sync(s));
    CK(cs1_attention_gated(q, k, v, ldv, gate, full, T, Hq, Hk, Dh, 0.0625f, s));
    CK(cs1_stream_sync(s));

    for (int q_base : {64, 128, 1024}) {
        if (T - q_base < 640) continue;
        CK(cudaMemsetAsync(win, 0, qbytes, (cudaStream_t)s));
        CK(cs1_attention_gated_prefix(q, k, v, ldv, gate, win, T, Hq, Hk, Dh, 0.0625f, q_base, s));
        CK(cs1_stream_sync(s));
        const size_t row = (size_t)Hq * Dh * 2;
        std::vector<unsigned char> a((size_t)(T - q_base) * row), b(a.size());
        CK(cs1_download(a.data(), (char*)full + (size_t)q_base * row, a.size(), s));
        CK(cs1_download(b.data(), (char*)win + (size_t)q_base * row, a.size(), s));
        std::printf("T=%d q_base=%d rows=%d %s\n", T, q_base, T - q_base, a == b ? "exact" : "DIFF");
    }
    cs1_stream_destroy(s);
    return 0;
}
