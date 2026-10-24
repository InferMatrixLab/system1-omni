#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
static size_t live_bytes;
static int pending, in_capture, unsafe_frees, fail_download;
struct Allocation { size_t bytes; max_align_t alignment; };
static int live, linear_count, attention_count, captures, launches, destroyed, fail_capture;
static int rows[16384], outs[16384], lens[16384];
static uintptr_t qkv_base;
static uintptr_t qs[16384], ks[16384], vs[16384], os[16384];
unsigned cs1_abi_version(void){return 9;}
const char *cs1_error_string(int c){(void)c;return "fake CUDA error";}
int cs1_get_device(int *d){if(!d)return 1;*d=0;return 0;}
int cs1_set_device(int d){(void)d;return 0;}
int cs1_malloc(void **p,size_t n){struct Allocation *a=calloc(1,sizeof(*a)+n);if(!a)return 1;a->bytes=n;*p=a+1;live++;live_bytes+=n;return 0;}
int cs1_free(void *p){struct Allocation *a=(struct Allocation*)p-1;if(pending)unsafe_frees++;live_bytes-=a->bytes;free(a);live--;return 0;}
int cs1_stream_create(void **p){*p=(void*)1;return 0;}
int cs1_stream_sync(void *s){(void)s;pending=0;return 0;}
int cs1_stream_destroy(void *s){(void)s;return 0;}
int cs1_upload(void *d,const void *s,size_t n,void *stream){(void)stream;memcpy(d,s,n);pending=0;return 0;}
int cs1_download(void *d,const void *s,size_t n,void *stream){if(fail_download){pending=1;return 1;}(void)stream;memcpy(d,s,n);pending=0;return 0;}
void *cs1_gemm_create(size_t n){(void)n;return (void*)1;}
void cs1_gemm_destroy(void *g){(void)g;}
int cs1_vision_linear(void *g,const void *x,const void *w,const void *bias,void *y,int m,int n,int k,void *s){
 (void)g;(void)x;(void)w;(void)bias;(void)y;(void)k;(void)s;
 if(!in_capture)pending=1;
 rows[linear_count]=m;outs[linear_count]=n;linear_count++;return 0;
}
int cs1_vision_norm(const void*x,const void*w,const void*b,void*y,int n,int h,void*s){(void)x;(void)w;(void)b;(void)y;(void)n;(void)h;(void)s;return 0;}
int cs1_vision_bias(void*x,const void*b,size_t n,int h,void*s){(void)x;(void)b;(void)n;(void)h;(void)s;return 0;}
int cs1_vision_gelu(void*x,size_t n,int e,void*s){(void)x;(void)n;(void)e;(void)s;return 0;}
int cs1_vision_add(void*x,const void*d,size_t n,void*s){(void)x;(void)d;(void)n;(void)s;return 0;}
int cs1_vision_position_v2(void*x,const void*t,const int*i,const float*w,int n,int h,void*s){(void)x;(void)t;(void)i;(void)w;(void)n;(void)h;(void)s;return 0;}
int cs1_vision_rope_v2(const void*x,const float*c,const float*si,void*q,void*k,int n,int h,int d,void*s){if(attention_count==0)qkv_base=(uintptr_t)x;(void)c;(void)si;(void)q;(void)k;(void)n;(void)h;(void)d;(void)s;return 0;}
int cs1_vision_attention_v2(const void*q,const void*k,const void*v,void*o,int n,int h,int d,void*w,void*s){
 (void)h;(void)d;(void)w;(void)s;
 lens[attention_count]=n;qs[attention_count]=(uintptr_t)q;ks[attention_count]=(uintptr_t)k;vs[attention_count]=(uintptr_t)v;os[attention_count]=(uintptr_t)o;attention_count++;return 0;
}
int cs1_graph_begin(void*s){(void)s;in_capture=1;return 0;}
int cs1_graph_end(void*s,void**p){(void)s;in_capture=0;if(fail_capture){*p=NULL;return 1;}*p=malloc(1);captures++;return 0;}
int cs1_graph_launch(void*g,void*s){(void)g;(void)s;pending=1;launches++;return 0;}
int cs1_graph_destroy(void*g){free(g);destroyed++;return 0;}
int batch_live(void){return live;}
int batch_linear_count(void){return linear_count;}
int batch_linear_rows(int i){return rows[i];}
int batch_linear_out(int i){return outs[i];}
int batch_attention_count(void){return attention_count;}
int batch_attention_rows(int i){return lens[i];}
uintptr_t batch_q(int i){return qs[i];}
uintptr_t batch_k(int i){return ks[i];}
uintptr_t batch_v(int i){return vs[i];}
uintptr_t batch_out(int i){return os[i];}
int batch_captures(void){return captures;}
int batch_launches(void){return launches;}
int batch_destroyed(void){return destroyed;}

uintptr_t batch_qkv_base(void){return qkv_base;}

void batch_fail_capture(int value){fail_capture=value;}

size_t batch_live_bytes(void){return live_bytes;}
int batch_unsafe_frees(void){return unsafe_frees;}
void batch_fail_download(int value){fail_download=value;}

int cs1_vision_position(void*x,const void*t,const int*i,const float*w,int n,void*s){(void)x;(void)t;(void)i;(void)w;(void)n;(void)s;return 0;}
int cs1_vision_rope(const void*x,const float*c,const float*si,void*q,void*k,int n,void*s){(void)x;(void)c;(void)si;(void)q;(void)k;(void)n;(void)s;return 0;}
int cs1_vision_attention(const void*q,const void*k,const void*v,void*o,int n,void*s){(void)q;(void)k;(void)v;(void)o;(void)n;(void)s;return 0;}
int cs1_vision_to_float(const void*x,float*y,size_t n,void*s){(void)x;(void)y;(void)n;(void)s;return 0;}
int cs1_gemm_f32(void*g,const float*x,const float*w,float*y,int m,int n,int k,void*s){(void)g;(void)x;(void)w;(void)y;(void)m;(void)n;(void)k;(void)s;return 0;}
int cs1_vision_lora_add(void*x,const float*d,size_t n,float scale,void*s){(void)x;(void)d;(void)n;(void)scale;(void)s;return 0;}
