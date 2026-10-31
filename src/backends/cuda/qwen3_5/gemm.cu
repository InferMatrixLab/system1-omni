// bfloat16 GEMMs through cuBLASLt, float32 accumulation.
//
// Row-major y [M, N] = x [M, K] * w [N, K]^T is the column-major product
// y^T [N, M] = (w viewed as [K, N])^T * (x viewed as [K, M]); y's rows may be
// strided (ldy >= N), so one GEMM can fill a slice of a wider buffer.
//
// Each shape uses cuBLASLt's first heuristic choice, excluding split-K reductions that
// accumulate into the output in place, since their order, and so the rounding, is not
// fixed (vision's FP32 and biased GEMMs exclude all split-K). That choice depends on M,
// so a row's result can change with the number of rows in the call. A handle from
// cs1_gemm_create_fixed instead keeps one algorithm per weight shape (N, K, ldy, and for
// vision's GEMMs the data type and bias) for every M: the heuristic's first choice at a
// reference M among algorithms without split-K. Each output row then takes the same
// path whatever the other rows, so its result does not depend on M or on its row index.
// cuBLASLt does not document this; tests/qwen3_5/kernels.rs checks it on the GPU it
// runs on.
#include <cublasLt.h>

#include <map>
#include <list>
#include <iterator>
#include <tuple>
#include <cstring>
#include <limits>
#include <vector>

#include "ops.h"
#include "gemm_profile.h"

namespace {

struct Plan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t algo{};
};

using Key = std::tuple<int, int, int, int, bool, bool>;  // M, N, K, ldy, FP32, bias
using PackedKey = std::tuple<std::vector<size_t>, int, int>;

struct PackedSegment {
    Plan plan;
    size_t offset = 0;
    int rows = 0;
    bool enabled = false;
};

using PackedSegments = std::vector<PackedSegment>;
constexpr size_t MAX_PACKED_PLANS = 512;
struct PackedPlan {
    PackedSegments segments;
    std::list<PackedKey>::iterator age;
};

using ReuseKey = std::tuple<int, int, int, int>;
constexpr size_t MAX_REUSED_PLANS = 512;
struct ReusedPlan {
    Plan plan;
    std::list<ReuseKey>::iterator age;
};

struct Gemm {
    cublasLtHandle_t handle = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    bool equivalent_tiles = false;
    bool reference = false;
    bool matched = false;
    int reference_m = 0;  // > 0: one algorithm per weight shape, chosen at this M
    std::map<Key, Plan> plans;
    std::map<std::tuple<int, int, int, bool, bool>, cublasLtMatmulAlgo_t> fixed;  // N, K, ldy, FP32, bias
    std::map<ReuseKey, ReusedPlan> reused_plans;
    std::list<ReuseKey> reused_order;
    std::map<PackedKey, PackedPlan> packed_plans;
    std::list<PackedKey> packed_order;
};

template<typename T>
bool config_value(const cublasLtMatmulAlgo_t& algo, cublasLtMatmulAlgoConfigAttributes_t attr, T& value) {
    size_t written = 0;
    return cublasLtMatmulAlgoConfigGetAttribute(&algo, attr, &value, sizeof(value), &written)
        == CUBLAS_STATUS_SUCCESS && written == sizeof(value);
}

// Empirically validated SM90/cuBLASLt 13.1.1 long-prefill profile. cuBLAS
// does not promise arithmetic equivalence between custom options; the optional
// capability is version/device scoped and covered by full-row/packed/graph and
// model precision tests. Keep split-K and all unknown configurations separate.
bool validated_long_profile(const Plan& p, bool short_reuse = false) {
    for (auto pair : {std::make_pair(CUBLASLT_ALGO_CONFIG_ID, 66u),
                      std::make_pair(CUBLASLT_ALGO_CONFIG_SPLITK_NUM, 1u),
                      std::make_pair(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, 0u),
                      std::make_pair(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, 0u),
                      std::make_pair(CUBLASLT_ALGO_CONFIG_STAGES_ID, 35u)}) {
        uint32_t value = 0;
        if (!config_value(p.algo, pair.first, value) || value != pair.second) return false;
    }
    uint32_t option = 0;
    uint16_t inner = 0, cluster = 0;
    if (!config_value(p.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, option) ||
        !config_value(p.algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, inner) ||
        !config_value(p.algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, cluster)) return false;
    return (option == 0 || option == 1 || option == 2 || option == 4 || option == 5 || option == 6 || option == 7 || (short_reuse && option == 3)) && inner == 0 &&
           (cluster == CUBLASLT_CLUSTER_SHAPE_2x1x1 || cluster == CUBLASLT_CLUSTER_SHAPE_4x1x1 ||
            cluster == CUBLASLT_CLUSTER_SHAPE_1x2x1 ||
            (short_reuse && cluster == CUBLASLT_CLUSTER_SHAPE_4x2x1));
}

bool compatible_algorithms(const Gemm& g, const Plan& a, const Plan& b, size_t ma, size_t mb, bool short_reuse = false) {
    if (std::memcmp(&a.algo, &b.algo, sizeof(a.algo)) == 0) return true;
    if (!g.equivalent_tiles || ma > 4096 || mb > 4096) return false;
    if (ma >= 512 && mb >= 512) return validated_long_profile(a) && validated_long_profile(b);
    // Only retained prefix/suffix projections can use the tested short profiles.
    // Keep ordinary packed GEMM grouping and algorithm selection unchanged.
    return short_reuse && ma >= 64 && mb >= 64 &&
           validated_long_profile(a, true) && validated_long_profile(b, true);
}

int status(cublasStatus_t s) { return s == CUBLAS_STATUS_SUCCESS ? 0 : 1000 + (int)s; }

void destroy(Plan& p) {
    if (p.a) cublasLtMatrixLayoutDestroy(p.a);
    if (p.b) cublasLtMatrixLayoutDestroy(p.b);
    if (p.c) cublasLtMatrixLayoutDestroy(p.c);
    if (p.op) cublasLtMatmulDescDestroy(p.op);
    p = Plan{};
}

int describe(int M, int N, int K, int ldy, Plan& p, bool fp32, bool bias) {
    cublasStatus_t s = cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (s != CUBLAS_STATUS_SUCCESS) return status(s);
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
    if (bias) {
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        if ((s = cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue))) != CUBLAS_STATUS_SUCCESS) return status(s);
    }
    const cudaDataType_t dtype = fp32 ? CUDA_R_32F : CUDA_R_16BF;
    if ((s = cublasLtMatrixLayoutCreate(&p.a, dtype, K, N, K)) != CUBLAS_STATUS_SUCCESS) return status(s);
    if ((s = cublasLtMatrixLayoutCreate(&p.b, dtype, K, M, K)) != CUBLAS_STATUS_SUCCESS) return status(s);
    if ((s = cublasLtMatrixLayoutCreate(&p.c, dtype, N, M, ldy)) != CUBLAS_STATUS_SUCCESS) return status(s);
    return 0;
}

// The heuristic's first choice among the given reduction schemes.
int first_choice(Gemm& g, Plan& p, uint32_t schemes) {
    cublasLtMatmulPreference_t pref;
    cublasStatus_t s = cublasLtMatmulPreferenceCreate(&pref);
    if (s != CUBLAS_STATUS_SUCCESS) return status(s);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g.workspace_bytes,
                                         sizeof(g.workspace_bytes));
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &schemes,
                                         sizeof(schemes));
    cublasLtMatmulHeuristicResult_t r{};
    int found = 0;
    s = cublasLtMatmulAlgoGetHeuristic(g.handle, p.op, p.a, p.b, p.c, p.c, pref, 1, &r, &found);
    cublasLtMatmulPreferenceDestroy(pref);
    if (s != CUBLAS_STATUS_SUCCESS) return status(s);
    if (found == 0 || r.state != CUBLAS_STATUS_SUCCESS) return status(CUBLAS_STATUS_NOT_SUPPORTED);
    p.algo = r.algo;
    return 0;
}

// The weight shape's algorithm, chosen at the reference M on first use. An M it cannot
// serve is an error, not a reason to switch algorithms.
int fixed_choice(Gemm& g, int N, int K, int ldy, bool fp32, bool bias, Plan& p) {
    const std::tuple<int, int, int, bool, bool> key{N, K, ldy, fp32, bias};
    auto it = g.fixed.find(key);
    if (it == g.fixed.end()) {
        Plan r;
        int rc = describe(g.reference_m, N, K, ldy, r, fp32, bias);
        const uint32_t schemes = g.matched
            ? CUBLASLT_REDUCTION_SCHEME_MASK & ~CUBLASLT_REDUCTION_SCHEME_INPLACE
            : CUBLASLT_REDUCTION_SCHEME_NONE;
        if (rc == 0) rc = first_choice(g, r, schemes);
        if (rc == 0) {
            // Legacy fixed excludes split-K; matched retains ordinary deterministic reductions.
            int32_t splits = 0;
            uint32_t scheme = 0;
            size_t written = 0;
            const bool read =
                cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splits,
                                                     sizeof(splits), &written) == CUBLAS_STATUS_SUCCESS &&
                cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &scheme,
                                                     sizeof(scheme), &written) == CUBLAS_STATUS_SUCCESS;
            if (!read || (!g.matched && (splits != 1 || scheme != CUBLASLT_REDUCTION_SCHEME_NONE)) ||
                scheme == CUBLASLT_REDUCTION_SCHEME_INPLACE)
                rc = status(CUBLAS_STATUS_NOT_SUPPORTED);
        }
        if (rc == 0) it = g.fixed.emplace(key, r.algo).first;
        destroy(r);
        if (rc != 0) return rc;
    }
    p.algo = it->second;
    cublasLtMatmulHeuristicResult_t check{};
    const cublasStatus_t s = cublasLtMatmulAlgoCheck(g.handle, p.op, p.a, p.b, p.c, p.c, &p.algo, &check);
    if (s != CUBLAS_STATUS_SUCCESS) return status(s);
    if (g.matched && check.state != CUBLAS_STATUS_SUCCESS) return status(check.state);
    if (check.workspaceSize > g.workspace_bytes) return status(CUBLAS_STATUS_NOT_SUPPORTED);
    return 0;
}

// The plan for a shape, created on first use.
int plan_for(Gemm& g, int M, int N, int K, int ldy, Plan*& out, bool fp32 = false, bool bias = false) {
    const Key key{M, N, K, ldy, fp32, bias};
    auto it = g.plans.find(key);
    if (it != g.plans.end()) {
        out = &it->second;
        return 0;
    }
    Plan p;
    int rc = describe(M, N, K, ldy, p, fp32, bias);
    if (rc == 0 && g.reference_m > 0) {
        rc = fixed_choice(g, N, K, ldy, fp32, bias, p);
    } else if (rc == 0) {
        // The reference handle allows every reduction scheme. Otherwise vision's FP32 and
        // biased GEMMs exclude all split-K to avoid BF16 intermediate reductions, and the
        // language GEMMs exclude only in-place reductions.
        const uint32_t schemes = g.reference    ? CUBLASLT_REDUCTION_SCHEME_MASK
                                 : fp32 || bias ? CUBLASLT_REDUCTION_SCHEME_NONE
                                                : CUBLASLT_REDUCTION_SCHEME_MASK & ~CUBLASLT_REDUCTION_SCHEME_INPLACE;
        rc = first_choice(g, p, schemes);
    }
    if (rc != 0) {
        destroy(p);
        return rc;
    }
    out = &g.plans.emplace(key, p).first->second;
    return 0;
}

// Optional shape changes retain the original arithmetic profile. The original
// algorithm must still support the target shape; only opted-in compatible profiles
// may use the target-row heuristic. Return -1 keeps the complete-forward fallback.
int reused_plan(Gemm& g, int M, int N, int K, int original_M, Plan*& out) {
    const auto key = std::make_tuple(M, N, K, original_M);
    auto it = g.reused_plans.find(key);
    if (it != g.reused_plans.end()) {
        if (std::next(it->second.age) != g.reused_order.end())
            g.reused_order.splice(g.reused_order.end(), g.reused_order, it->second.age);
        out = &it->second.plan;
        return 0;
    }
    Plan* original = nullptr;
    int rc = plan_for(g, original_M, N, K, N, original);
    if (rc) return rc;
    Plan p;
    rc = describe(M, N, K, N, p, false, false);
    if (rc) { destroy(p); return rc; }
    p.algo = original->algo;
    cublasLtMatmulHeuristicResult_t checked{};
    const auto result = cublasLtMatmulAlgoCheck(g.handle, p.op, p.a, p.b, p.c, p.c, &p.algo, &checked);
    if (result != CUBLAS_STATUS_SUCCESS || checked.state != CUBLAS_STATUS_SUCCESS ||
        checked.workspaceSize > g.workspace_bytes) {
        destroy(p);
        return -1;
    }
    // Prefix sharing changes the executed row count, sometimes by a large factor.
    // Keep the original eligibility check above, then choose the actual-row plan
    // only inside the empirically validated SM90/Lt profile. Other handles and
    // split-K/unknown algorithms retain the exact original choice.
    if (g.equivalent_tiles && M != original_M) {
        Plan* actual = nullptr;
        if (plan_for(g, M, N, K, N, actual) == 0 &&
            compatible_algorithms(g, *original, *actual, original_M, M, true)) {
            cublasLtMatmulHeuristicResult_t candidate{};
            const auto valid = cublasLtMatmulAlgoCheck(
                g.handle, p.op, p.a, p.b, p.c, p.c, &actual->algo, &candidate);
            if (valid == CUBLAS_STATUS_SUCCESS && candidate.state == CUBLAS_STATUS_SUCCESS &&
                candidate.workspaceSize <= g.workspace_bytes) p.algo = actual->algo;
        }
    }
    if (g.reused_plans.size() == MAX_REUSED_PLANS) {
        const auto oldest = g.reused_plans.find(g.reused_order.front());
        destroy(oldest->second.plan);
        g.reused_plans.erase(oldest);
        g.reused_order.pop_front();
    }
    g.reused_order.push_back(key);
    out = &g.reused_plans.emplace(key, ReusedPlan{p, std::prev(g.reused_order.end())}).first->second.plan;
    return 0;
}

}  // namespace

extern "C" void* cs1_gemm_create_fixed(size_t workspace_bytes, int reference_m) {
    if (reference_m <= 0) return nullptr;
    Gemm* g = static_cast<Gemm*>(cs1_gemm_create(workspace_bytes));
    if (g) g->reference_m = reference_m;
    return g;
}

extern "C" void* cs1_gemm_create(size_t workspace_bytes) {
    Gemm* g = new Gemm();
    if (cublasLtCreate(&g->handle) != CUBLAS_STATUS_SUCCESS ||
        (workspace_bytes > 0 && cudaMalloc(&g->workspace, workspace_bytes) != cudaSuccess)) {
        if (g->handle) cublasLtDestroy(g->handle);
        delete g;
        return nullptr;
    }
    g->workspace_bytes = workspace_bytes;
    return g;
}

extern "C" void cs1_gemm_destroy(void* gemm) {
    Gemm* g = static_cast<Gemm*>(gemm);
    if (!g) return;
    for (auto& kv : g->plans) destroy(kv.second);
    for (auto& kv : g->reused_plans) destroy(kv.second.plan);
    for (auto& kv : g->packed_plans)
        for (auto& segment : kv.second.segments) destroy(segment.plan);
    if (g->workspace) cudaFree(g->workspace);
    cublasLtDestroy(g->handle);
    delete g;
}

extern "C" int cs1_gemm(void* gemm, const void* x, const void* w, void* y, int M, int N, int K, int ldy,
                        void* stream) {
    Gemm* g = static_cast<Gemm*>(gemm);
    if (!g || M < 0 || N <= 0 || K <= 0 || ldy < N) return cudaErrorInvalidValue;
    if (M == 0) return cudaSuccess;
    Plan* p = nullptr;
    const int rc = plan_for(*g, M, N, K, ldy, p);
    if (rc != 0) return rc;
    const float alpha = 1.f, beta = 0.f;
    return status(cublasLtMatmul(g->handle, p->op, &alpha, w, p->a, x, p->b, &beta, y, p->c, y, p->c, &p->algo,
                                 g->workspace, g->workspace_bytes, static_cast<cudaStream_t>(stream)));
}

// Additive opt-in for merged language models. Keep each original shape's
// algorithm, including split-K and its reduction scheme. Pack only consecutive
// shapes with compatible algorithms, and split groups to fit the existing workspace.
extern "C" int cs1_gemm_packed(void* gemm, const void* x, const void* w, void* y,
                                const size_t* lengths, size_t count, int N, int K,
                                void* stream) {
    auto* g = static_cast<Gemm*>(gemm);
    if (!g || !lengths || count == 0 || N <= 0 || K <= 0) return cudaErrorInvalidValue;
    size_t total = 0;
    for (size_t i = 0; i < count; ++i) {
        if (lengths[i] == 0 || lengths[i] > static_cast<size_t>(std::numeric_limits<int>::max()) - total)
            return cudaErrorInvalidValue;
        total += lengths[i];
    }
    if (count == 1) return cs1_gemm(g, x, w, y, static_cast<int>(total), N, K, N, stream);
    const PackedKey key{std::vector<size_t>(lengths, lengths + count), N, K};
    auto it = g->packed_plans.find(key);
    if (it == g->packed_plans.end()) {
        std::vector<Plan*> originals(count);
        for (size_t i = 0; i < count; ++i) {
            const int rc = plan_for(*g, static_cast<int>(lengths[i]), N, K, N, originals[i]);
            if (rc) return rc;
        }
        PackedSegments packed;
        size_t offset = 0;
        for (size_t begin = 0; begin < count;) {
            size_t end = begin + 1;
            size_t rows = lengths[begin];
            // Equality of the opaque value is conservative: differences in
            // reserved bytes only decline packing. Original map pointers are stable.
            while (!g->reference && end < count &&
                   compatible_algorithms(*g, *originals[begin], *originals[end], lengths[begin], lengths[end])) {
                rows += lengths[end++];
            }
            PackedSegment segment;
            segment.offset = offset;
            // Try the largest compatible group first. A smaller group may fit
            // when split-K workspace or an algorithm shape limit rejects it.
            while (end > begin + 1) {
                const int rc = describe(static_cast<int>(rows), N, K, N, segment.plan, false, false);
                if (rc) {
                    destroy(segment.plan);
                    for (auto& previous : packed) destroy(previous.plan);
                    return rc;
                }
                auto& p = segment.plan;
                p.algo = originals[begin]->algo;
                cublasLtMatmulHeuristicResult_t checked{};
                const auto result = cublasLtMatmulAlgoCheck(g->handle, p.op, p.a, p.b, p.c, p.c,
                                                            &p.algo, &checked);
                segment.enabled = result == CUBLAS_STATUS_SUCCESS && checked.state == CUBLAS_STATUS_SUCCESS
                                  && checked.workspaceSize <= g->workspace_bytes;
                if (segment.enabled) break;
                destroy(segment.plan);
                rows -= lengths[--end];
            }
            segment.rows = static_cast<int>(rows);
            packed.push_back(segment);
            offset += rows;
            begin = end;
        }
        if (g->packed_plans.size() == MAX_PACKED_PLANS) {
            const auto oldest = g->packed_plans.find(g->packed_order.front());
            // Descriptors are host-side cuBLAS inputs; captured graphs own their
            // launch arguments. The eviction/replay regression covers this lifetime.
            for (auto& segment : oldest->second.segments) destroy(segment.plan);
            g->packed_plans.erase(oldest);
            g->packed_order.pop_front();
        }
        g->packed_order.push_back(key);
        it = g->packed_plans.emplace(key, PackedPlan{std::move(packed), std::prev(g->packed_order.end())}).first;
    }
    if (std::next(it->second.age) != g->packed_order.end())
        g->packed_order.splice(g->packed_order.end(), g->packed_order, it->second.age);
    const float alpha = 1.f, beta = 0.f;
    for (auto& segment : it->second.segments) {
        const auto* input = static_cast<const char*>(x) + segment.offset * K * 2;
        auto* output = static_cast<char*>(y) + segment.offset * N * 2;
        int rc;
        if (segment.enabled) {
            auto& p = segment.plan;
            rc = status(cublasLtMatmul(g->handle, p.op, &alpha, w, p.a, input, p.b, &beta, output, p.c,
                                       output, p.c, &p.algo, g->workspace, g->workspace_bytes,
                                       static_cast<cudaStream_t>(stream)));
        } else {
            rc = cs1_gemm(g, input, w, output, segment.rows, N, K, N, stream);
        }
        if (rc) return rc;
    }
    return 0;
}

extern "C" int cs1_gemm_reuse_check(void* gemm, const size_t* original_rows, size_t originals,
                                     const size_t* target_rows, size_t targets, int N, int K) {
    auto* g = static_cast<Gemm*>(gemm);
    if (!g || !original_rows || !target_rows || originals == 0 || targets == 0 || N <= 0 || K <= 0)
        return cudaErrorInvalidValue;
    if (g->reference) return -1;
    for (size_t i = 0; i < originals; ++i)
        if (!original_rows[i] || original_rows[i] > static_cast<size_t>(std::numeric_limits<int>::max()))
            return cudaErrorInvalidValue;
    for (size_t i = 0; i < targets; ++i)
        if (!target_rows[i] || target_rows[i] > static_cast<size_t>(std::numeric_limits<int>::max()))
            return cudaErrorInvalidValue;
    Plan* original = nullptr;
    int rc = plan_for(*g, static_cast<int>(original_rows[0]), N, K, N, original);
    if (rc) return rc;
    for (size_t i = 1; i < originals; ++i) {
        Plan* other = nullptr;
        rc = plan_for(*g, static_cast<int>(original_rows[i]), N, K, N, other);
        if (rc) return rc;
        if (!compatible_algorithms(*g, *original, *other, original_rows[0], original_rows[i], true)) return -1;
    }
    for (size_t i = 0; i < targets; ++i) {
        Plan* target = nullptr;
        rc = reused_plan(*g, static_cast<int>(target_rows[i]), N, K, static_cast<int>(original_rows[0]), target);
        if (rc) return rc;
    }
    return 0;
}

extern "C" int cs1_gemm_reuse(void* gemm, const void* x, const void* w, void* y,
                              int M, int N, int K, int original_M, void* stream) {
    auto* g = static_cast<Gemm*>(gemm);
    if (!g || M <= 0 || N <= 0 || K <= 0 || original_M <= 0) return cudaErrorInvalidValue;
    if (g->reference) return -1;
    Plan* p = nullptr;
    const int rc = reused_plan(*g, M, N, K, original_M, p);
    if (rc) return rc;
    const float alpha = 1.f, beta = 0.f;
    return status(cublasLtMatmul(g->handle, p->op, &alpha, w, p->a, x, p->b, &beta, y, p->c, y, p->c,
                               &p->algo, g->workspace, g->workspace_bytes, static_cast<cudaStream_t>(stream)));
}

// Vision's biased BF16 linears round only after adding bias. The patch
// convolution passes zero bias here and applies its bias after BF16 rounding.
extern "C" int cs1_vision_linear(void* gemm, const void* x, const void* w, const void* bias,
                                  void* y, int M, int N, int K, void* stream) {
    Gemm* g = static_cast<Gemm*>(gemm);
    if (!g || !bias || M <= 0 || N <= 0 || K <= 0) return cudaErrorInvalidValue;
    Plan* p = nullptr;
    int rc = plan_for(*g, M, N, K, N, p, false, true);
    if (rc) return rc;
    auto s = cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    if (s != CUBLAS_STATUS_SUCCESS) return status(s);
    const float alpha = 1.f, beta = 0.f;
    return status(cublasLtMatmul(g->handle, p->op, &alpha, w, p->a, x, p->b, &beta, y, p->c, y, p->c, &p->algo,
                                 g->workspace, g->workspace_bytes, static_cast<cudaStream_t>(stream)));
}
// Separate unmerged LoRA matrices and intermediates stay FP32. No TF32 fast compute.
extern "C" int cs1_gemm_f32(void* gemm, const float* x, const float* w, float* y,
                             int M, int N, int K, void* stream) {
    Gemm* g = static_cast<Gemm*>(gemm);
    if (!g || M <= 0 || N <= 0 || K <= 0) return cudaErrorInvalidValue;
    Plan* p = nullptr;
    int rc = plan_for(*g, M, N, K, N, p, true, false);
    if (rc) return rc;
    const float alpha = 1.f, beta = 0.f;
    return status(cublasLtMatmul(g->handle, p->op, &alpha, w, p->a, x, p->b, &beta, y, p->c, y, p->c, &p->algo,
                                 g->workspace, g->workspace_bytes, static_cast<cudaStream_t>(stream)));
}

// JEMM matches the pinned framework heuristic, including all reduction schemes.
extern "C" void* cs1_reference_gemm_create(size_t workspace_bytes) {
    auto* g=static_cast<Gemm*>(cs1_gemm_create(workspace_bytes));
    if(g) g->reference=true;
    return g;
}

// Additive ABI 9 entry points. Ordinary and fixed-M64 handles keep their policy.
extern "C" void* cs1_gemm_create_matched(size_t bytes, int reference_m) {
    auto* g = static_cast<Gemm*>(cs1_gemm_create_fixed(bytes, reference_m));
    if (g) g->matched = true;
    return g;
}

// ordinary == nullptr checks that the selected algorithm supports a subwindow.
extern "C" int cs1_gemm_compatible(void* ordinary, void* matched, int M, int N, int K) {
    if (!matched || M <= 0 || N <= 0 || K <= 0) return 0;
    Plan *a = nullptr, *b = nullptr;
    if (plan_for(*static_cast<Gemm*>(matched), M, N, K, N, b)) return 0;
    if (!ordinary) return 1;
    if (plan_for(*static_cast<Gemm*>(ordinary), M, N, K, N, a)) return 0;
    return std::memcmp(&a->algo, &b->algo, sizeof(a->algo)) == 0;
}

// Additive ABI 9 capability. Select before creating plans/capturing any graph.
// 1 = enabled, 0 = unsupported profile (unchanged fallback), negative = error.
extern "C" int cs1_gemm_enable_equivalent_tiles(void* gemm) {
    auto* g = static_cast<Gemm*>(gemm);
    if (!g || g->reference || g->matched || g->reference_m || !g->plans.empty()) return -1;
    int device = 0;
    cudaDeviceProp prop{};
    auto error = cudaGetDevice(&device);
    if (error == cudaSuccess) error = cudaGetDeviceProperties(&prop, device);
    if (error != cudaSuccess) return -static_cast<int>(error);
    g->equivalent_tiles = qwen3_5::equivalent_tiles_device(
        prop.name, prop.major, prop.minor, prop.multiProcessorCount, cublasLtGetVersion());
    return g->equivalent_tiles ? 1 : 0;
}
