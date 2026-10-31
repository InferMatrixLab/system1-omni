# Kev BF16 service parity repair (v3)

This records the v3 repair and its then-incomplete coverage. The subsequent
[v4 qualification](noul-boundary.md) adds 33 Noul boundary questions and passes
all frozen gates without changing production code or discarding these results.

The same 3,215 requests / 3,393 questions now pass the probability, Score and
far-boundary decision gates. Maximum probability error is **0.0024709404 <=0.01**
(`choice-255`); maximum raw Score error is **0.00000305218 <=0.1 level**
(`score-255`), with zero returned-Score difference. All observed decisions,
including the 31 near-boundary questions, agree. The 896 labeled questions have
identical 76.5625% accuracy; all frozen task-quality gates pass.

**At the end of v3, acceptance remained incomplete:** the fixed pool still contains only
3 near-boundary Noul questions against the requirement of 10. Choice has 28.
The [v2 protocol and failures](service-acceptance.md) are retained. No gate,
reference output, request, candidate order or label changed during this repair.
No additional boundary search or performance campaign was run.

## Where the error starts

The diagnostic starts from the original worst probability case
`hard-v1:hard-v1/probability/development/00023` and `score-255`.
Official observers retain the default BF16 service's real state/branch passes,
padding, cache and graph behavior. An additional complete-row forward isolates
arithmetic from segmentation; it never replaces the primary service reference.

| Isolated difference | Evidence and correction |
| --- | --- |
| GDN accumulation | Same-input projections and convolution matched. The original BF16 GDN core differed from the official no-FLA FP32 core (relative RMS about 0.002). Use FP32 normalized Q/K, solve and recurrent state, then round its output to BF16. |
| Framework reduction order | The existing FP32 path followed Torch 2.14. Torch 2.8 uses different lane assignments at dimension 128 and ascending shuffle offsets. Match its RMSNorm, gated norm, Q/K norm and GDN normalization with optional symbols; preserve existing consumers' arithmetic. |
| Projection shape | A fixed 64-row readout GEMM selected different arithmetic from the reference's actual row count. Use the reference GEMM profile. Multi-question requests additionally require the official length groups, padding, batch count and row offset. |
| Rotary buffer precision | Official checkpoint loading casts the backbone, including `inv_freq`, to BF16. Rounding this buffer before constructing angles makes the first ten complete-row layers exact. Native's prior FP32 frequencies encoded a different model computation. |
| Export device | CPU and official GPU FP32 LoRA merge differ in nine BF16 scalar weights across nine of 426 tensors. The next divergence begins at layer 10. GPU FP32 merge of the same pinned inputs eliminates all 426 tensor differences. This is precision alignment, not a new quantization scheme. |
| State/branch attention | A complete causal row and official padded state/continuation passes select different SDPA kernels and GDN chunks. Execute those segments eagerly, with temporary per-call state, and instantiate the exact Torch 2.8 CUTLASS efficient-attention kernel for the masked path. |

With corrected export and arithmetic, both original worst samples have **exact
embeddings, all 32 residual layer outputs and final normalization** against the
corresponding official default-service state/question passes. Their probability
errors are 1.19209e-7 and 1.02445e-8 after the separate FP32 pointer head.

An intermediate full regression still found six probability failures, max
0.01862866. All were multi-question requests. Equalizing question padding alone
fixed two. In the remaining worst request, every layer-0 projection, GDN output
and gated norm matched exactly; the output projection was the first difference
(341 BF16 elements, max 0.00048828125). Official GEMM had 160 rows; native had 80.
Keeping the official matrix shape and each row's position fixes all six, max
probability difference 9.23872e-7. Other rows of the temporary projection matrix
are zero: matrix rows are independent, and only the current row's result is read.
This conservative baseline spends extra computation; it is not an optimization.

## Implementation and build boundary

`planning.rs` belongs to Kev: it reproduces official question length grouping,
padding and mapping. The shared Qwen executor provides an opt-in segmented
selected-position readout. A call computes state then its question, retains
attention KV / GDN state only until that question finishes, and synchronizes
before freeing temporary allocations on success or failure. There is no retained
cross-request prefix and no native CUDA Graph. The scheduler and HTTP API remain
unchanged. The trained FP32 head and calibration remain unchanged.

The optional profile requires `recipe/kev/build_cuda.sh`. It compiles a
header-only instance of PyTorch 2.8.0's BSD-licensed attention kernel with the
exact CUTLASS submodule `ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e` (BSD-3-Clause).
The archive SHA-256 is checked before extraction. Torch headers are a build-time
dependency; the worker and CUDA library have no libtorch/Python runtime dependency.
The ordinary Qwen build still works without these headers. Missing service
symbols fail at Kev startup, rather than silently selecting another precision.

The worker now requires the CUDA FP32 merge metadata. A CPU diagnostic export is
still available for head tests but is rejected for serving. Re-export to a new
directory; do not overwrite historical artifacts. See [export](README.md) and
[worker](native.md) commands. The original export/head review boundary is retained;
the CUDA export correction can be reviewed separately from worker execution.

## Validation and limits

- All 3,215 requests assert exact token IDs, positions, readouts and input usage.
  Three repeated mixed responses remain identical.
- Real startup/warmup and 32 direct-worker/frontend/native responses match exactly;
  eight malformed/unsupported-input HTTP cases match.
- 83 operator checks pass exactly: Torch 2.8 norms, GDN chunk boundaries and seeded
  state, masked SDPA including non-aligned keys, and old/new legacy CUDA entry
  points. The GDN test retains its 1e-4 relative-L2 gate; observed error is zero.
- The labeled population remains 336 Choice, 280 Noul, 280 Score. Brier delta is
  +8.76751e-9 and normalized Score MAE delta -2.57951e-9. Paired record-bootstrap
  intervals and per-source metrics remain in the machine-readable report.
- Existing independent head gates and unmerged/merged FP32 diagnostics remain
  separate. No claim is made that arbitrary optional FLA installations, GPU
  architectures or concurrent official batching are qualified by this run.
- The pinned official server can differ between cold eager fallback and a warmed
  state cache. Native uses deterministic state buckets for short states; the
  fixed original request order remains the primary comparator. The residual
  `choice-255` error is below the unchanged gate; complete bitwise identity for
  every possible serving history is not claimed.

Local evidence root: `/tmp/system1-issue-27/evidence/layer-diagnosis-v3/`.
`comparison-25.json` is the complete fixed-corpus result; the intermediate
`comparison-22.json`, original v2 reports, failed probes and all layer tensors
are preserved. `layer-comparison-service-20.json` has all 32-layer comparisons;
`service-kernels-27.json` has operator/legacy regression results. The handoff
records final source/build hashes and verification commands.

The retained FP32 unmerged control is a different computation from the official
BF16 default service. Comparing the corrected native outputs with that historical
13-case control gives probability max 0.00680214 and Score-255 difference
0.28167720 levels (FP32 about 96.76453 versus official BF16 about 96.48286).
This is separately recorded in `fp32-controls-v3.json`; the unchanged 0.1 Score
gate applies to the primary official BF16 service. No FP32 results are relabeled
as BF16 acceptance, and the original 0.1149 diagnostic remains preserved.
