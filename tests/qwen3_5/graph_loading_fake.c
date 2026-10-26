// Host-only CUDA boundary for model loading and stream ownership; no kernels run.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
static int live, streams, destroyed, early, gemms, pending, unsynced_destroy;
void loading_submit(void) { pending=1; }
int loading_pending(void) { return pending; }
int loading_unsynced_destroy(void) { return unsynced_destroy; }
int loading_live(void) { return live; }
int loading_streams(void) { return streams; }
int loading_destroyed(void) { return destroyed; }
int loading_early(void) { return early; }
int loading_gemms(void) { return gemms; }
uint32_t cs1_abi_version(void) { return LOADING_ABI; }
const char* cs1_error_string(int code) { (void)code; return "unexpected host-test CUDA call"; }
int cs1_get_device(int* device) { *device=0; return 0; }
int cs1_set_device(int device) { return device == 0 ? 0 : 1; }
int cs1_malloc(void** p, size_t n) { *p=malloc(n); if (!*p) return 1; ++live; return 0; }
int cs1_free(void* p) { --live; free(p); return 0; }
int cs1_stream_create(void** p) { *p=(void*)1; ++streams; return 0; }
int cs1_stream_sync(void* stream) { (void)stream; pending=0; return 0; }
int cs1_stream_destroy(void* stream) { (void)stream; ++destroyed; if (live || gemms) ++early; return 0; }
int cs1_upload(void* dst, const void* src, size_t n, void* stream) { (void)stream; memcpy(dst,src,n); pending=1; return 0; }
void* cs1_gemm_create(size_t n) { (void)n; ++gemms; return (void*)2; }
void* cs1_gemm_create_fixed(size_t n, int m) { (void)n; if(m<=0) return NULL; ++gemms; return (void*)3; }
void cs1_gemm_destroy(void* p) { (void)p; if (pending) ++unsynced_destroy; --gemms; }

static int compatible = 1, window_supported = 1;
static int reject_ordinary=-1, reject_m, reject_n, reject_k;
void loading_reject(int ordinary, int m, int n, int k) {
    reject_ordinary=ordinary; reject_m=m; reject_n=n; reject_k=k;
}
void loading_window_supported(int value) { window_supported = value; }
void loading_compatible(int value) { compatible = value; }
void* cs1_gemm_create_matched(size_t n, int m) { return cs1_gemm_create_fixed(n, m); }
int cs1_gemm_compatible(void* ordinary, void* matched, int m, int n, int k) {
    (void)matched;
    if (reject_ordinary == (ordinary != NULL) && m == reject_m &&
        (!reject_n || n == reject_n) && (!reject_k || k == reject_k)) return 0;
    return ordinary ? compatible : window_supported;
}
