// C interface of libqwen3_5_cuda.so: the Qwen3.5 prefill operations and the few
// CUDA runtime calls a caller needs, so that it can load this one library at run time.
//
// Tensors are row-major and bfloat16 unless noted. Every operation queues work on
// `stream` (a cudaStream_t) and returns a cudaError_t, or 1000 + a cublasStatus_t for
// the GEMMs. The norm, elementwise and attention-prep operations round to bfloat16
// where the Transformers reference (modeling_qwen3_5.py) does; attention and the
// Gated DeltaNet prefill keep some intermediate results in bfloat16, as FlashAttention
// and flash-linear-attention do (see attention.cu and gdn_prefill.cu).
#pragma once

#include <stddef.h>
#include <stdint.h>

// Bumped whenever the required interface below changes.
#define CS1_ABI_VERSION 9

#ifdef __cplusplus
extern "C" {
#endif

// ---- runtime ----

uint32_t cs1_abi_version(void);
const char* cs1_error_string(int code);
int cs1_set_device(int device);
int cs1_get_device(int* device);
// Set flag to zero, then atomically set to one if any BF16 value is non-finite.
int cs1_bf16_finite(const void* x, size_t n, uint32_t* flag, void* stream);
int cs1_malloc(void** ptr, size_t bytes);
int cs1_free(void* ptr);
int cs1_stream_create(void** stream);
int cs1_stream_sync(void* stream);
int cs1_stream_destroy(void* stream);
int cs1_graph_begin(void* stream);
int cs1_graph_end(void* stream, void** exec);
int cs1_graph_launch(void* exec, void* stream);
int cs1_graph_destroy(void* exec);
// Copy and wait for the copy.
int cs1_upload(void* dst, const void* src, size_t bytes, void* stream);
int cs1_download(void* dst, const void* src, size_t bytes, void* stream);
// Device-to-device copy of `bytes` (queue-only completion).
int cs1_copy_dd(void* dst, const void* src, size_t bytes, void* stream);
// Pitched copy both ways (default direction = device-to-device):
// height rows of `width`, output pitch dpitch, input pitch spitch, in bytes.
int cs1_copy2d(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, void* stream);

// ---- operations ----

// out[t] = table[ids[t]], rows of D.
int cs1_embed(const int32_t* ids, const void* table, void* out, int T, int D, void* stream);

// Zero-centred RMSNorm over rows of D: out = x / rms(x) * (1 + w), in float32.
int cs1_rms_norm(const void* x, const void* w, void* out, int rows, int D, float eps, void* stream);

// residual = residual + delta (rounded to bfloat16), then out = cs1_rms_norm(residual).
int cs1_add_rms_norm(void* residual, const void* delta, const void* w, void* out, int rows, int D,
                     float eps, void* stream);

// Gated RMSNorm of the Gated DeltaNet output x [T, H, D] (D = 128), with z [T, H*D]
// in rows of ldz: out = (w * (x / rms(x))) * silu(z).
int cs1_gated_rms_norm(const void* x, const void* z, int ldz, const void* w, void* out, int T, int H,
                       int D, float eps, void* stream);

// Depthwise causal conv1d (kernel 4, no bias) and SiLU over qkv [T, ld], split into
// q [T, key_dim], k [T, key_dim] and v [T, value_dim]. w is [key_dim*2 + value_dim, 4].
int cs1_gdn_conv(const void* qkv, int ld, const void* w, void* q, void* k, void* v, int T, int key_dim,
                 int value_dim, void* stream);
// Optional vectorized convolution with exact scalar-layout fallback.
int cs1_gdn_conv_vector(const void* qkv, int ld, const void* w, void* q, void* k, void* v, int T, int key_dim,
                 int value_dim, void* stream);

// Optional compact suffix convolution seeded by three immutable pre-conv rows [3,ld].
int cs1_gdn_conv_prefix(const void* qkv, const void* history, int ld, const void* w,
                        void* q, void* k, void* v, int T, int key_dim, int value_dim, void* stream);
// Optional compact suffix attention. Prefix K/V are contiguous [prefix,Hk,Dh];
// suffix Q/gate/output start at row zero and use absolute query position prefix.
int cs1_attention_gated_shared(const void* q, const void* k, const void* v, int ldv,
                               const void* gate, void* out, const void* prefix_k,
                               const void* prefix_v, int suffix, int prefix, int Hq,
                               int Hk, int Dh, float scale, void* stream);

// beta = sigmoid(b) (bfloat16) and g = -exp(A_log) * softplus(a + dt_bias) (float32), [T, H];
// b and a are [T, H] in rows of ld.
int cs1_gdn_gates(const void* b, const void* a, int ld, const void* A_log, const void* dt_bias,
                  void* beta, float* g, int T, int H, void* stream);

// Chunked gated delta rule, q and k L2-normalized inside, q scaled by `scale`.
// q, k [T, HK, 128], v [T, H, 128], g float [T, H], beta [T, H], o [T, H, 128].
// Optional independent sequence batching, with separate chunk padding/workspaces.
size_t cs1_gdn_batch_workspace_floats(int tokens, int H, int count);
int cs1_gdn_prefill_batch(const void* q, const void* k, const void* v, const float* g,
                         const void* beta, void* o, float* workspace, size_t workspace_bytes,
                         const size_t* lengths, size_t count, int H, int HK, float scale, void* stream);
size_t cs1_gdn_workspace_floats(int T, int H);
// Optional no-FLA text precision mode: FP32 normalized Q/K and recurrent state,
// BF16 output. Context and scratch remain model-owned; prefix continuation is unsupported.
void* cs1_gdn_fp32_create(size_t workspace_bytes);
void cs1_gdn_fp32_destroy(void* context);
size_t cs1_gdn_fp32_workspace_floats(int T,int H);
int cs1_gdn_prefill_fp32(void* context,const void* q, const void* k, const void* v, const float* g,
                       const void* beta, void* o, float* workspace,
                       int T, int H, int HK, float scale, void* stream);
int cs1_gdn_prefill(const void* q, const void* k, const void* v, const float* g, const void* beta,
                    void* o, float* workspace, int T, int H, int HK, float scale, void* stream);

// Optional wider preparation/output launch for the opt-in 27B text path. Same layout,
// workspace, arithmetic and output as cs1_gdn_prefill; other shapes fall back.
int cs1_gdn_prefill_wide(const void* q, const void* k, const void* v, const float* g, const void* beta,
                         void* o, float* workspace, int T, int H, int HK, float scale, void* stream);

// Same scan with an explicit state: s_in is the [H, K, V] float32 state before
// this window (null = zero), s_out receives the [H, K, V] float32 state after
// it (null = skip). The float32 state makes a continuation starting exactly at
// a chunk boundary repeat the arithmetic of one full pass bit for bit. Used
// for the cached-prefix continuation and for its capture.
int cs1_gdn_prefill_x(const void* q, const void* k, const void* v, const float* g, const void* beta,
                      void* o, float* workspace, int T, int H, int HK, float scale,
                      const void* s_in, void* s_out, void* stream);

// Attention inputs: q and gate from qg [T, Hq, 2*Dh], k from kr [T, Hk, Dh], both in rows
// of ld; per-head zero-centred RMSNorm, then rotary embedding on the first 2*half dims
// using cos/sin [T, half] (bfloat16). Writes q [T, Hq, Dh], gate [T, Hq*Dh], k [T, Hk, Dh].
int cs1_attn_prep(const void* qg, const void* kr, int ld, const void* qw, const void* kw,
                  const void* cos, const void* sin, void* q, void* gate, void* k, int T, int Hq, int Hk,
                  int Dh, int half, float eps, void* stream);

// Causal attention with grouped KV heads, Dh = 256: q [T, Hq, Dh], k [T, Hk, Dh], v
// [T, Hk, Dh] in rows of ldv; out [T, Hq, Dh].
int cs1_attention(const void* q, const void* k, const void* v, int ldv, void* out, int T, int Hq,
                  int Hk, int Dh, float scale, void* stream);

// Same attention, with the sigmoid gate fused into its epilogue. gate [T, Hq, Dh].
// Both the attention result and sigmoid are rounded to bfloat16 before multiplying,
// exactly as cs1_attention followed by cs1_sigmoid_gate. gate must be non-null if T > 0.
int cs1_attention_gated(const void* q, const void* k, const void* v, int ldv, const void* gate,
                        void* out, int T, int Hq, int Hk, int Dh, float scale, void* stream);

// Additive, optional at ABI 9. Same tensors as cs1_attention_gated, but every query
// reads every key in the prompt. Missing on libraries built before this entry point;
// causal callers do not resolve it. q_base-style prefix continuation is not supported.
int cs1_attention_gated_bidirectional(const void* q, const void* k, const void* v, int ldv,
                                      const void* gate, void* out, int T, int Hq, int Hk, int Dh,
                                      float scale, void* stream);

// Windowed variant for the cached prefix: k and v cover rows [0, T) (the cached
// prefix must already be in place); q, gate and out cover rows [q_base, T) only.
// q_base must be a multiple of 64; q_base = 0 reduces to cs1_attention_gated.
int cs1_attention_gated_prefix(const void* q, const void* k, const void* v, int ldv, const void* gate,
                               void* out, int T, int Hq, int Hk, int Dh, float scale, int q_base,
                               void* stream);

// x = x * sigmoid(gate), n elements.
int cs1_sigmoid_gate(void* x, const void* gate, size_t n, void* stream);

// out [T, I] = silu(gate) * up, from gate_up [T, 2*I] (gate first) in rows of ld.
int cs1_silu_mul(const void* gate_up, int ld, void* out, int T, int I, void* stream);

// y [M, N] (rows of ldy) = x [M, K] * w [N, K]^T through cuBLASLt, float32 accumulation,
// with cuBLASLt's first heuristic choice for each shape (see gemm.cu).
void* cs1_gemm_create(size_t workspace_bytes);
// A handle that keeps one algorithm per weight shape (N, K, ldy) for every M, chosen at
// reference_m rows among algorithms without split-K, so that a row's result depends
// neither on M nor on its row index (checked by tests/qwen3_5/kernels.rs on the GPU it
// runs on). Null if reference_m <= 0 or setup fails. cs1_gemm with this handle returns a
// cuBLAS status for an M the algorithm cannot serve, rather than switching algorithms.
void* cs1_gemm_create_fixed(size_t workspace_bytes, int reference_m);
// Optional additive ABI 9: choose the ordinary reduction mask at reference_m.
// Compatibility returns 1 for equal complete descriptors, else 0. A null ordinary
// handle tests support only (used for prefix/suffix windows); no GEMM executes.
void* cs1_gemm_create_matched(size_t workspace_bytes, int reference_m);
int cs1_gemm_compatible(void* ordinary, void* matched, int M, int N, int K);
void cs1_gemm_destroy(void* gemm);
int cs1_gemm(void* gemm, const void* x, const void* w, void* y, int M, int N, int K, int ldy,
             void* stream);

// Optional packed projections retaining original per-sequence algorithms.
int cs1_gemm_packed(void* gemm, const void* x, const void* w, void* y,
                    const size_t* lengths, size_t count, int N, int K, void* stream);

// Optional retained-algorithm projections. Check returns 0 if compatible, -1 if
// unsupported; other values are errors. Check all windows before computing any.
int cs1_gemm_reuse_check(void* gemm, const size_t* original_rows, size_t originals,
                        const size_t* target_rows, size_t targets, int N, int K);
int cs1_gemm_reuse(void* gemm, const void* x, const void* w, void* y,
                  int M, int N, int K, int original_M, void* stream);

// ---- vision: single image, 1024 hidden, 16 heads of 64; all BF16 except explicit float pointers ----
int cs1_vision_linear(void* gemm, const void* x, const void* w, const void* bias, void* y, int M, int N, int K, void* stream);
int cs1_gemm_f32(void* gemm, const float* x, const float* w, float* y, int M, int N, int K, void* stream);
int cs1_vision_norm(const void* x, const void* w, const void* b, void* y, int rows, int d, void* stream);
// indices/weights [N,4], 48x48 learned table; FP32 rotary cos/sin [N,32].
int cs1_vision_position(void* x, const void* table, const int* indices, const float* weights, int n, void* stream);
int cs1_vision_rope(const void* qkv, const float* co, const float* si, void* q, void* k, int n, void* stream);
// q/k [N,1024], V is a slice in qkv [N,3072]. No causal mask, O(N) memory.
int cs1_vision_attention(const void* q, const void* k, const void* v, void* out, int n, void* stream);
// Additive vision endpoints, optional within the required CUDA ABI. Supports the exact 4B/27B layouts.
// Rotary cos/sin [N,head_dim/2]. Attention workspace: 4*N*16*80 BF16 elements for head72.
int cs1_vision_position_v2(void* x, const void* table, const int* indices, const float* weights, int n, int hidden, void* stream);
int cs1_vision_rope_v2(const void* qkv, const float* co, const float* si, void* q, void* k, int n, int hidden, int head_dim, void* stream);
int cs1_vision_attention_v2(const void* q, const void* k, const void* v, void* out, int n, int heads, int head_dim, void* workspace, void* stream);
int cs1_vision_bias(void* x, const void* bias, size_t n, int d, void* stream);
int cs1_vision_gelu(void* x, size_t n, int exact, void* stream);
int cs1_vision_add(void* x, const void* delta, size_t n, void* stream);
int cs1_vision_to_float(const void* x, float* out, size_t n, void* stream);
int cs1_vision_lora_add(void* x, const float* delta, size_t n, float scale, void* stream);


// ---- Optional Kev Torch 2.8 default-service profile (additive; old ABI unchanged) ----
// Torch 2.8 reduction order; only 4B hidden width 2560 / GDN dimension 128.
int cs1_torch28_rms_norm(const void* x,const void* w,void* out,int rows,int D,float eps,void* stream);
int cs1_torch28_add_rms_norm(void* residual,const void* delta,const void* w,void* out,int rows,int D,float eps,void* stream);
int cs1_torch28_gated_rms_norm(const void* x,const void* z,int ldz,const void* w,void* out,int T,int H,int D,float eps,void* stream);
int cs1_torch28_attn_prep(const void* qg,const void* kr,int ld,const void* qw,const void* kw,const void* cos,const void* sin,
                        void* q,void* gate,void* k,int T,int Hq,int Hk,int Dh,int half,float eps,void* stream);
void* cs1_gdn_fp32_create_torch28(size_t workspace_bytes);
// Null initial state means zeros; optional final_state receives [H,128,128] FP32.
// Context and every allocation remain alive until the supplied stream drains.
int cs1_gdn_prefill_fp32_x(void* context,const void* q,const void* k,const void* v,const float* g,const void* beta,
                         void* out,float* workspace,int T,int H,int HK,float scale,const void* initial,void* final_state,void* stream);
// Q/gate/out [nq,H,256]; K [nk,HK,256]; V [nk,ld]. prefix = nk-nq.
// Mask allows real prefix keys [left,prefix) and causal question keys [prefix,prefix+real),
// plus each query's own key. Workspace bytes include a BF16 bias with 8-element row alignment.
size_t cs1_service_attention_bytes(int nq,int nk,int H);
int cs1_service_attention(const void* q,const void* k,const void* v,int ld,const void* gate,void* out,
                          int nq,int nk,int H,int HK,int prefix,int left,int real,void* workspace,void* stream);
int cs1_service_zero(void* ptr,size_t bytes,void* stream);

// ---- Optional JEMM framework numerics (ABI 6 additive) ----
void* cs1_reference_gemm_create(size_t workspace_bytes);
int cs1_reference_rms_norm(const void* x, const void* w, void* out, int rows, int D, float eps, void* stream);
int cs1_reference_add_rms_norm(void* residual, const void* delta, const void* w, void* out, int rows, int D,
                     float eps, void* stream);
size_t cs1_reference_gdn_workspace_floats(int T, int H);
int cs1_reference_gdn_prefill(const void* q, const void* k, const void* v, const float* g, const void* beta,
                    void* o, float* workspace, int T, int H, int HK, float scale, void* stream);
int cs1_reference_attn_prep(const void* qg, const void* kr, int ld, const void* qw, const void* kw,
                  const void* cos, const void* sin, void* q, void* gate, void* k, int T, int Hq, int Hk,
                  int Dh, int half, float eps, void* stream);
int cs1_reference_vision_norm(const void* x, const void* w, const void* b, void* y, int rows, int d, void* stream);
int cs1_reference_vision_rope(const void* qkv, const float* co, const float* si, void* q, void* k, int n, int hidden, int head_dim, void* stream);
int cs1_reference_vision_attention(const void* q, const void* k, const void* v, void* out, int n, int heads, int head_dim, void* workspace, void* stream);
// Rotary's co parameter is FP32 angles, si is unused; unlike the legacy cos/sin API.
// Language workspace is measured in FP32 elements and covers every length <= capacity.
// Vision workspace adds 4*capacity*16*80 BF16 padding elements before that FP32 region.
int cs1_reference_available(void);
size_t cs1_reference_language_attention_workspace_floats(int capacity,int heads);
size_t cs1_reference_vision_attention_workspace_floats(int capacity,int heads);
int cs1_reference_attention(const void* q,const void* k,const void* v,int ldv,const void* gate,void* out,int tokens,int heads,int kv_heads,float scale,void* workspace,void* stream);
int cs1_reference_vision_attention_padded(const void* q,const void* k,const void* v,void* out,int tokens,int heads,void* workspace,void* stream);
// Create/destroy outside capture, on the owning device. The caller synchronizes before destroy.
void* cs1_reference_patch_create(int rows,void* stream);
int cs1_reference_patch(void* plan,const void* x,const void* weight,void* out);
void cs1_reference_patch_destroy(void* plan);

#ifdef __cplusplus
}
#endif
