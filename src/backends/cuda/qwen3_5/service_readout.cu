// Kev's masked SDPA: PyTorch 2.8.0 + CUTLASS ad7b2f5 (see recipe/kev/build_cuda.sh).
// Header-only kernel instantiation; no libtorch or Python runtime dependency.
// Other workers retain their existing attention path and build dependencies.
#ifdef CS1_KEV_SDPA
#include <tuple>
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>
using Kernel=PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t,cutlass::arch::Sm80,true,32,128,65536,true,true>;
static int cutlass_attention(const void* q,const void* k,const void* v,const void* mask,void* out,void* accum,int nq,int nk,int heads,void* stream) {
  Kernel::Params p;
  p.query_ptr=(const cutlass::bfloat16_t*)q;
  p.key_ptr=(const cutlass::bfloat16_t*)k;
  p.value_ptr=(const cutlass::bfloat16_t*)v;
  p.attn_bias_ptr=(const cutlass::bfloat16_t*)mask;
  p.output_ptr=(cutlass::bfloat16_t*)out;
  p.output_accum_ptr=(float*)accum;
  p.scale=.0625f;p.head_dim=256;p.head_dim_value=256;
  p.num_queries=nq;p.num_keys=nk;p.num_keys_absolute=nk;
  p.q_strideM=p.k_strideM=p.v_strideM=p.o_strideM=heads*256;
  p.q_strideH=p.k_strideH=p.v_strideH=256;
  p.bias_strideM=(nk+7)/8*8;p.bias_strideH=0;p.bias_strideB=0;
  p.q_strideB=(int64_t)nq*heads*256;p.k_strideB=p.v_strideB=(int64_t)nk*heads*256;
  p.num_batches=1;p.num_heads=heads;
  auto fn=PyTorchMemEffAttention::attention_kernel_batched_impl<Kernel>;
  constexpr int smem=sizeof(Kernel::SharedStorage);
  auto err=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
  if(err) return err;
  fn<<<p.getBlocksGrid(),p.getThreadsGrid(),smem,(cudaStream_t)stream>>>(p);
  return cudaGetLastError();
}
#include "common.cuh"
#include "ops.h"
namespace {
__global__ void expand_kv(const cs1::bf16* k,const cs1::bf16* v,int ld,cs1::bf16* kr,cs1::bf16* vr,int nk,int h,int hk) {
 size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x,n=(size_t)nk*h*256;
 if(i>=n)return;int d=i%256,head=i/256%h,t=i/(256*h),kh=head/(h/hk);
 kr[i]=k[((size_t)t*hk+kh)*256+d];vr[i]=v[(size_t)t*ld+kh*256+d];
}
__global__ void mask_rows(cs1::bf16* mask,int nq,int nk,int prefix,int left,int real) {
 int maskld=(nk+7)/8*8;
 size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=(size_t)nq*maskld)return;
 int q=i/maskld,k=i%maskld;
 bool visible=(k<nk && k>=left && k<=prefix+q && (k<prefix || k-prefix<real)) || k==prefix+q;
 mask[i]=cs1::to_bf16(visible?0.f:-3.3895313892515355e38f);
}
}
extern "C" size_t cs1_service_attention_bytes(int nq,int nk,int h) {
 if(nq<=0 || nq>65536 || nk<nq || nk>65536 || h<=0 || h>32)return 0;
 return (size_t)nq*h*256*4+2*(size_t)nk*h*256*2+(size_t)nq*((nk+7)/8*8)*2;
}
extern "C" int cs1_service_attention(const void* q,const void* k,const void* v,int ld,const void* gate,void* out,int nq,int nk,int h,int hk,int prefix,int left,int real,void* workspace,void* stream) {
 if(!q || !k || !v || !gate || !out || !workspace || !cs1_service_attention_bytes(nq,nk,h)
    || hk<=0 || h%hk || ld<hk*256 || prefix!=nk-nq || left<0 || left>nk
    || real<0 || real>nq)return cudaErrorInvalidValue;
 char* bytes=(char*)workspace;size_t accum=(size_t)nq*h*256*4,kv=(size_t)nk*h*256*2;
 auto kr=(cs1::bf16*)(bytes+accum);auto vr=(cs1::bf16*)(bytes+accum+kv);auto mask=(cs1::bf16*)(bytes+accum+2*kv);
 expand_kv<<<((size_t)nk*h*256+255)/256,256,0,(cudaStream_t)stream>>>((const cs1::bf16*)k,(const cs1::bf16*)v,ld,kr,vr,nk,h,hk);
 mask_rows<<<((size_t)nq*((nk+7)/8*8)+255)/256,256,0,(cudaStream_t)stream>>>(mask,nq,nk,prefix,left,real);
 auto error=cudaGetLastError();if(error)return error;
 int rc=cutlass_attention(q,kr,vr,mask,out,workspace,nq,nk,h,stream);
 if(rc)return rc;
 return cs1_sigmoid_gate(out,gate,(size_t)nq*h*256,stream);
}
extern "C" int cs1_service_zero(void* ptr,size_t n,void* stream) {return cudaMemsetAsync(ptr,0,n,(cudaStream_t)stream);}

#endif // CS1_KEV_SDPA
