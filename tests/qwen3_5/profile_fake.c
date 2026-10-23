// CPU stand-in for the CUDA library that counts every call, for the profile test.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
static int calls;
int profile_fake_calls(void) { return calls; }
uint32_t cs1_abi_version(void) { ++calls; return PROFILE_FAKE_ABI; }
const char* cs1_error_string(int code) { (void)code; ++calls; return "unexpected fake CUDA call"; }
int cs1_get_device(int* device) { ++calls; if (!device) return 1; *device=0; return 0; }
int cs1_set_device(int device) { (void)device; ++calls; return 0; }
int cs1_malloc(void** p, size_t n) { ++calls; *p=malloc(n ? n : 1); return *p ? 0 : 1; }
int cs1_free(void* p) { ++calls; free(p); return 0; }
int cs1_stream_create(void** stream) { ++calls; *stream=(void*)1; return 0; }
int cs1_stream_sync(void* stream) { (void)stream; ++calls; return 0; }
int cs1_stream_destroy(void* stream) { (void)stream; ++calls; return 0; }
int cs1_upload(void* dst, const void* src, size_t n, void* stream) { (void)stream; ++calls; memcpy(dst,src,n); return 0; }
void* cs1_gemm_create(size_t n) { (void)n; ++calls; return &calls; }
void cs1_gemm_destroy(void* gemm) { (void)gemm; ++calls; }
void* cs1_gdn_fp32_create(size_t n) { (void)n; ++calls; return &calls; }
void cs1_gdn_fp32_destroy(void* handle) { (void)handle; ++calls; }
