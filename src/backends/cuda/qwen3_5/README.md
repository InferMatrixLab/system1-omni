# Qwen3.5/3.8 prefill operations

CUDA kernels for a prefill-only Qwen3.5/3.8 forward pass, built into `libqwen3_5_cuda.so` with a C interface ([`ops.h`](ops.h)), so that a Rust model engine loads it at run time and builds without a CUDA toolkit. The Cua-S1, Open-Jev and experimental JEV-VL native workers share the layer loop and buffers in [`src/models/qwen3_5/native/`](../../../models/qwen3_5/native/).

The backend manifest remains `experimental`. Maintained CI checks source/build
metadata and CPU tests; it does not compile this backend with NVCC or establish
GPU numerical parity. Current-head CUDA compilation and GPU regressions remain
separate release checks.

```sh
src/backends/cuda/qwen3_5/build.sh <output dir> [compute capability, default 89]
```

The norm, elementwise and q/k preparation kernels round to bfloat16 where Transformers (`modeling_qwen3_5.py`) does. Attention (FlashAttention-2 style, on tensor cores) and the chunked gated delta rule keep some intermediate results in bfloat16, as FlashAttention and flash-linear-attention do. GEMMs go through cuBLASLt with its first heuristic choice. Tensor-core kernels need sm_80 or newer; PR #19 validated the original kernels on sm_89. The current reference tests, including fused gating, cached residual RMSNorm and packed SiLU, passed on H200 (sm_90). Compilation passed for sm_80, sm_89 and sm_90; execution of the modified GDN kernel on sm_80/sm_89 remains unverified.

`cs1_attention_gated` fuses the sigmoid gate into the attention epilogue, preserving
the BF16 rounding of both attention and sigmoid before multiplication. The native
workers use this entry point; the separate operations remain available for kernel
comparisons. `cs1_attention_gated_bidirectional` is an optional ABI 9 symbol with
the same argument list. It is the Instant prompt mask: every query reads every
key in that prompt, and a non-zero `q_base` is rejected. Causal
`cs1_attention_gated` and prefix continuation stay causal. The second template
parameter defaults to false, so existing causal launches keep that mask. Causal
callers do not resolve the new symbol. StartLux Instant fails closed when it is
missing. Rebuild this library before starting that worker. This tree does not
claim that the causal specialization's SASS is bit-identical to the previous
build; GPU revalidation of Cua-S1 and Open-Jev was not run with this change.

HF Qwen3.5 folders store `linear_attn.A_log` and `linear_attn.norm.weight` as
FP32. The shared loader rounds those two suffixes to BF16 at upload because the
kernels consume BF16. Other non-BF16 language tensors still fail. That rounding
is a parity gap versus an FP32 `exp(A_log)` and GDN norm in Transformers. Rebuild the library and all native Rust consumers together for
**ABI version 9**, even with resident features disabled. ABI 9 retains the vision,
CUDA Graph, gated attention and prefix-continuation entry points and ABI 8's
`cs1_get_device` and `cs1_bf16_finite` for device feature validation, and adds the
fixed-algorithm GEMM handle below. Older libraries are rejected at load. See the
[Cua-S1 resident-feature recipe](../../../../recipe/cua_s1/native_vision_graph.md#gpu-resident-vision-features)
for the opt-in device feature path, lifetime contract and regression commands.

cuBLASLt's heuristic picks a GEMM algorithm per M, so a row's result can change with
the number of rows in the call: a prefix and its branch, run separately, round
differently from the same tokens in one pass. `cs1_gemm_create_fixed` returns a handle
that keeps one algorithm per weight shape for every M, the heuristic's first choice at a
reference M among algorithms without split-K, so each row's result is the same whatever
M is and wherever the row sits. cuBLASLt doesn't document this property;
`fixed_gemm_rows_do_not_depend_on_m` checks it, so run it on a new GPU before relying
on it (it passed on sm_89). It is slower for some shapes and faster for others: on an
RTX 6000 Ada, the down and output projections take up to about 4 times as long at small
M without split-K. `cs1_gemm_create` remains the default.

Gated DeltaNet preparation stores converted TF32 operands in three-byte component planes, preserves the original four-term TF32 accumulation, and writes U/W fragments directly as bfloat16. Dynamic shared memory is 72 KiB per block. The [H200 comparison](../../../../benchmarks/gdn/README.md) records complete GDN call latency, numerical checks, and the small end-to-end change measured with the Open-Jev worker from PR #55.

The shared Rust model can pack independent sequences for input and gate/up GEMMs.
Output/down GEMMs retain each prompt's original shape and reduction order;
attention, convolution and GDN calls remain sequence-local. Packing adds no CUDA
entry points. Open-Jev uses this path within requests; Cua-S1 keeps single-prompt calls. Decider optionally packs complete
question rows within one admitted request.

Prefix continuation retains full-attention KV, three convolution input rows,
and FP32 Gated DeltaNet state at 64-token chunk boundaries. It is used by the
experimental JEV-VL worker and by the shared executor's `forward_shared`, which
runs prompts that share prefixes once per prefix with the fixed-algorithm GEMM
handle ([#85](https://github.com/ThinkFlowLab/system1-omni/issues/85)); existing
full-prefill callers keep their own paths. Fixed/shared execution rejects
unmerged adapters, reference arithmetic and FP32 GDN before allocating execution
scratch or selecting the fixed GEMM handle.
ABI 6 adds mandatory copy and prefix entry points after ABI 5 native vision,
and the original ABI 7 branch adds the fixed-algorithm GEMM handle. ABI 9
combines this handle with the current ABI 8 device-feature contract.
Rebuild the library and **all** Qwen worker binaries together; older libraries
are rejected. Current-branch GPU regression is required before release.

Decider request-local prefix execution uses the same public `PrefixState` and
`run_window` continuation path as JEV-VL, with fixed-GEMM selection for its
independent/shared controls. No separate cached-attention, convolution-history
or GDN-state ABI is required. See the [Decider recipe](../../../../recipe/decider/README.md).
The Rust loader checks ABI version before resolving required symbols, so older
libraries report the rebuild requirement. Decider Graph statistics and synchronized
stream disposal are preserved; device validation of this consolidated head remains
unverified. Historical results retain their original implementation provenance.

## Optional JEMM reference arithmetic

JEMM enables the additive `cs1_reference_*` API group after model load and before
any forward. It requires cuDNN 9; build with `CUDNN_INCLUDE_DIR` and
`CUDNN_LIB_DIR`. Builds without those variables retain the current ABI-9 functions;
JEMM reports a missing-capability error rather than silently selecting older numerics.

The reference path uses a per-handle GEMM reduction policy, separate normalization,
attention and GDN functions, and shape-owned cuDNN Conv3D plans. Attention and
convolution workspaces allocate before capture and retire with their model/shape.
The rotary entry point consumes FP32 angles and computes trigonometry on device;
legacy rotary functions still consume cos/sin tables. Prefix caching is unsupported
on this opt-in path. The shared workers' default arithmetic remains unchanged.

The [JEMM numerical record](../../../../docs/benchmarks/jemm-reference-20261011/README.md)
contains a frozen 1–4-image corpus and reproducible operator/HTTP checks.

## Optional Cua-S1 no-FLA text arithmetic

The unmerged Cua-S1 text export reuses the reference norm, attention and GEMM
operations without requiring cuDNN. Norm additionally supports hidden size 2560.
The additive `cs1_gdn_fp32_*` group owns a cuBLAS handle and 32 MiB internal
workspace; the caller supplies shape-dependent FP32 buffers. It implements
64-token chunked GDN, FP32 normalization, triangular solves and recurrent state,
Torch 2.14 cumulative-decay association, and a final BF16 output cast. Each call
starts with zero state; prefix continuation is rejected by the model adapter.
The context drains its last stream before destruction, and is retired before
that stream. Rebuild the library and all consuming workers together for the
current ABI 9, including this optional group. The independent A800 qualification
at `27f77e49868d84b091aa804f02ccb4d3474a9a00` used ABI 6; the earlier cumulative
campaign used ABI 8. Neither campaign qualifies the current integrated source.
Builds now link both cuBLAS and cuBLASLt. Every consumer of the rebuilt shared
library must provide both runtime libraries (`libcublas.so` and
`libcublasLt.so`), even when it does not enable the optional FP32 GDN group.
An existing cuBLASLt-only deployment must add cuBLAS before upgrading.

`tests/qwen3_5/test_gdn_fp32.py` compares against the pinned Transformers 5.17
no-FLA implementation. On the cumulative A800 sm_80/CUDA 13.0/Torch 2.14 build, all 14 cases passed
with identical BF16 outputs, covering chunk boundaries, grouped heads,
changed-input replay/state reset, invalid dimensions/scales and context scratch
release. The fixed relative-L2 gate is 1e-4. This does not qualify other GPUs or
Torch versions; the original approximate GDN entry points remain available.

## Shared vision cache and operator checks

The generic shared vision constructor keeps a four-entry exact-grid FIFO for
compatibility. JEMM also retains a separate four-entry FIFO keyed by the ordered
multi-image grids. Their caps count entries, not device bytes; four large grids may
retain substantially more memory than four small grids. A borrowed resident
output prevents mutation or eviction until its consumer is finished.

[Cua-S1's bounded-cache integration in PR #127](https://github.com/ThinkFlowLab/system1-omni/pull/127)
uses a separate explicit policy: count/byte LRU limits, default one entry and
256 MiB, with zero retention available. The Cua wrapper owns environment parsing;
generic Qwen/JEMM users retain their existing policy. Multi-image and JEMM
reference execution require the generic constructor: the bounded policy cannot
account for opaque cuDNN patch-plan allocations through ABI 8. Retention-disabled or
oversized Cua output stays model-owned through both producer and consumer stream
drains. Graph enablement and scratch retention are separate policies: turning
Graph off alone does not release retained scratch.

The weight-free 27B operator regressions directly check 1152-wide positional
interpolation, head-72 rotary pairing and noncausal attention, including strided
V, non-tile-aligned row counts, changed inputs and poisoned padding workspace:

```sh
CUA_S1_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
  cargo test --release --locked -p omni-qwen3-5-native --test kernels \
  vision_v2:: -- --ignored --test-threads=1
```

Position/rotary assert exact BF16 readouts on dyadic inputs. Attention compares
against a host float64 softmax reference with a fixed 0.004 absolute gate. These
are explicit device tests; ordinary CPU CI compiles and skips them. Their addition
does not establish a fresh device pass or whole-model 27B qualification.

Shared preprocessing and vision adaptation notices accompany the Qwen crate in
[THIRD_PARTY_NOTICES.md](../../../models/qwen3_5/native/THIRD_PARTY_NOTICES.md).

## Optional full-prompt-matched GEMMs

`cs1_gemm_create_matched(workspace_bytes, reference_m)` and
`cs1_gemm_compatible(ordinary, matched, M, N, K)` are additive ABI 9 symbols.
Existing consumers do not resolve them. The matched handle retains the ordinary
BF16 reduction mask and chooses one algorithm per weight shape at `reference_m`;
legacy `cs1_gemm_create_fixed` still excludes split-K reductions. Compatibility
returns 1 only for equal complete algorithm descriptors and supported layouts
within the workspace budget. Passing a null ordinary handle checks window
support without requiring another heuristic to choose the same algorithm.
Neither compatibility check executes a GEMM.

Rust owns admission, request scope and the bounded workspace cache. Descriptor
matching is not a guarantee of row-invariant arithmetic on arbitrary hardware;
run the matched projection regression in `tests/qwen3_5/kernels.rs` and the
[Cua screenshot qualification](../../../../recipe/cua_s1/native_screenshot.md#full-prompt-matched-algorithms)
for each supported GPU/library/model combination. Other numerical profiles,
vision GEMMs and the existing default/fixed policies stay on their own handles.
