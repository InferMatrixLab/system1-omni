// Host-only dispatch spies. Arithmetic is deliberately not implemented.
static int causal_calls, bidir_calls, prefix_calls, gdn_calls, last_tokens, last_state;
int dispatch_causal(void) { return causal_calls; }
int dispatch_bidir(void) { return bidir_calls; }
int dispatch_prefix(void) { return prefix_calls; }
int dispatch_gdn(void) { return gdn_calls; }
int dispatch_tokens(void) { return last_tokens; }
int dispatch_state(void) { return last_state; }
int cs1_download(void* dst, const void* src, size_t n, void* stream) {
    memset(dst, 0, n); return 0;
}
int cs1_embed(const int* ids, const void* table, void* out, int t, int d, void* stream) { return 0; }
int cs1_rms_norm(const void* x, const void* w, void* out, int rows, int d, float eps, void* stream) { return 0; }
int cs1_add_rms_norm(void* residual, const void* delta, const void* w, void* out, int rows, int d, float eps, void* stream) { return 0; }
int cs1_gated_rms_norm(const void* x, const void* z, int ldz, const void* w, void* out, int t, int h, int d, float eps, void* stream) { return 0; }
int cs1_gdn_conv(const void* qkv, int ld, const void* w, void* q, void* k, void* v, int t, int kd, int vd, void* stream) { return 0; }
int cs1_gdn_gates(const void* b, const void* a, int ld, const void* alog, const void* bias, void* beta, float* g, int t, int h, void* stream) { return 0; }
size_t cs1_gdn_workspace_floats(int t, int h) { return 1; }
int cs1_gdn_prefill(const void* q, const void* k, const void* v, const float* g, const void* beta, void* o, float* workspace, int t, int h, int hk, float scale, void* stream) { ++gdn_calls; return 0; }
int cs1_attn_prep(const void* qg, const void* kr, int ld, const void* qw, const void* kw, const void* co, const void* si, void* q, void* gate, void* k, int t, int hq, int hk, int dh, int half, float eps, void* stream) { return 0; }
int cs1_attention_gated(const void* q, const void* k, const void* v, int ldv, const void* gate, void* out, int t, int hq, int hk, int dh, float scale, void* stream) {
    ++causal_calls; last_tokens=t; return 0;
}
int cs1_attention_gated_bidirectional(const void* q, const void* k, const void* v, int ldv, const void* gate, void* out, int t, int hq, int hk, int dh, float scale, void* stream) {
    ++bidir_calls; last_tokens=t; return 0;
}
int cs1_attention_gated_prefix_bidirectional(const void* q, const void* k, const void* v, int ldv, const void* gate, void* out, int t, int state, int hq, int hk, int dh, float scale, void* stream) {
    ++prefix_calls; last_tokens=t; last_state=state; return 0;
}
int cs1_silu_mul(const void* gate_up, int ld, void* out, int t, int i, void* stream) { return 0; }
int cs1_gemm(void* gemm, const void* x, const void* w, void* y, int m, int n, int k, int ldy, void* stream) { return 0; }
