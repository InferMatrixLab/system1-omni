# Cua-S1 text native/reference parity on A800

This investigation follows the five decision flips reported in
[PR #141](https://github.com/ThinkFlowLab/system1-omni/pull/141).
The original L20X run used a different GPU. On A800, the frozen 139-request corpus
produced two stable flips, not the original five. Two independent HTTP passes
per worker reproduced both flips. This report qualifies the A800 environment;
it does not establish that all five original L20X cases have been reproduced.

## PR dependencies and evidence provenance

This independent change is stacked on [#118](https://github.com/ThinkFlowLab/system1-omni/pull/118)
(`codex/jemm-native`, `ed9f9d6eb566d226e2441c5e9d84187abe1e4d63`) for the
unmerged FP32 language adapter loader and optional reference norm/GEMM/attention
operations. #118 is itself stacked on [#117](https://github.com/ThinkFlowLab/system1-omni/pull/117).
The frozen corpus comes from #141; merging #141 is not a runtime dependency.
Decider execution modes and Cua screenshot cache/resident-feature PRs are outside
this change's dependency chain and diff.

The GPU results below were measured on cumulative candidate
`76af07996f60905082b8d48199d4a7003f33f4f6`, whose parent is
`18e1a5a6f68b9d11d3e11c7ea7b939af87272f53`. That integrated build used ABI 8.
The independently rebased branch retains #118's ABI 6 and adds only optional
FP32 GDN symbols. Rebuild the library with the worker; no cuDNN is needed for
this text path. Results from the cumulative build are historical evidence for
the pinned implementation, not a fresh GPU qualification of the rebased PR head.
The GPU server was shut down after the campaign; no new device run is implied
by packaging this PR. CPU/build/docs checks are run again on the independent head.

## Frozen inputs and environment

- Baseline source: `3e9cc33a61ddf110d97512fb6f296cf540377c61`.
- Corpus: PR #141 `869260649f7dd5b56cac4d91eaae9087f2e985ee`,
  `benchmarks/e2e/choice.jsonl`; 139 requests/questions, 172–3988 prompt tokens.
- Historical GPU candidate parent: cumulative ABI-8/unmerged-LoRA shared executor,
  `18e1a5a6f68b9d11d3e11c7ea7b939af87272f53`.
- GPU: A800 PCIe 80 GB, sm_80; CUDA toolkit 13.0.88; NVIDIA driver 595.71.05.
- Python 3.12, Torch 2.14.0+cu130, Transformers 5.17.0, PEFT 0.21.0;
  TF32 disabled in the independent reference; no FLA or causal-conv1d installed.
- Base `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`;
  text adapter `cua-ai/cua-s1-4b-0.2@16818868b0cc7813808aae4e87b417657046ab79`.
  Pinned files were SHA-256 checked. The new export copies the original adapter
  unchanged, with hash `9b59c5aed96171a50b26526613766bbf44347a5c7af70f81efe6bcc6e9dbfb0e`.

The reference worker uses the repository's independent Transformers/PEFT text
implementation. No reference logits, labels or probabilities are modified.
Requests are sent through the Rust frontend to a real worker. The missing model
field in corpus rows is filled with the required `cua-s1-4b-0.2`, as the benchmark
wire adapter does. Raw request IDs, bodies, statuses, usage and model identities
are retained with source/build/launch manifests in the local investigation archive.

## Confirmed numerical causes

Token IDs matched on all 139 requests. The differences begin in execution:

1. The old export merges FP32 LoRA into BF16 base weights. PEFT instead computes
   the unmerged LoRA branch in FP32 and casts the combined projection to BF16.
2. The old GEMM reduction policy and normalization reductions differ from the
   pinned framework. With identical first-layer MLP down-projection input,
   244,845 BF16 base outputs differed; the existing reference GEMM handle matched.
   With identical normalization input, two BF16 values differed; reference norm
   matched after adding the 2560-hidden shape.
3. The old GDN uses TF32 pair products and BF16 chunk intermediates/state;
   the no-FLA Torch implementation uses FP32 chunked triangular solves and state.
   On identical first-layer input, old relative L2 error was 0.00217711 and
   326,687 of 954,368 BF16 outputs differed. Keeping recurrence FP32 alone did not
   match the reference's chunk arithmetic. Matching the chunked computation and
   Torch's cumulative-decay scan association produced identical BF16 outputs.
4. Native letter projection previously emitted FP32 logits. Reference `lm_head`
   emits BF16 before casting to FP32 softmax. Restoring that BF16 boundary matters
   for tied options and removes the remaining readout differences.

The fix selects the precision path using `cua-s1-text-unmerged/1`. Legacy merged
exports remain loadable. Other model adapters retain their existing numerical
mode. Prefix continuation is rejected in the new mode; eager is the default.
The probability gate of 0.02 was not relaxed.

## Results

| Real HTTP mode | Passes × requests | Flips vs reference | Maximum probability difference | Correct |
| --- | --- | --- | --- | --- |
| Baseline native | 2 × 139 | 2 | 0.04872608184814453 | 97/139 |
| Fixed eager | 2 × 139 | 0 | 5.960464477539063e-8 | 97/139 |
| Fixed Graph | 2 × 139 | 0 | 5.960464477539063e-8 | 97/139 |

Every fixed request returned HTTP 200. Each mode's two passes produced identical
response bytes; model identity, usage and option/question order matched reference.
Graph trace independently records 207 captures and 72 replays, with no fallback.
The two baseline flips were `original-routing-06-1` (tools/general) and
`hard-sol-b-temporal_numeric-03` (9_credits/12_credits); both reference maxima
are ties. Accuracy staying at 97/139 is expected: this repairs implementation
parity and does not retrain or improve the model.

A controlled full-model probe additionally matched all 139 final BF16 hidden
vectors and selected BF16 logit rows exactly. The slight HTTP probability drift
comes after that BF16 boundary in FP32 softmax/response finishing.

Fourteen maintained GPU cases compare the new GDN with the pinned no-FLA Torch
implementation: 1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 233, 513 and 2049 tokens
with 32 value/16 key heads, plus 129 tokens with 4 value/2 key heads. All BF16
outputs matched exactly. Repeated changed-input execution resets state, invalid
shapes/scales fail, and repeated context creation/destruction releases scratch.

## Reproduction and limits

Use the [native text recipe](../../../recipe/cua_s1/native.md) to build sm_80,
export original FP32 LoRA separately and start the worker. Use the pinned
[reference text recipe](../../../recipe/cua_s1/text.md) in an environment without
FLA/causal-conv1d, then replay the frozen corpus twice through each worker/frontend.
Run the maintained operator regression in that reference environment:

```sh
.venv/bin/python tests/qwen3_5/test_gdn_fp32.py \
  --library target/release/libqwen3_5_cuda.so
```

The historical integrated candidate passed Cargo fmt, full-workspace Clippy
with warnings denied, full-workspace tests and
release build with `--locked --features omni-laya/serve` where applicable.
Backend contract checker/tests, 21 pinned Python text contract/server tests, CUDA
operator regressions and strict MkDocs build also passed. The checkpoint-dependent
prefix test used canonical merged text weights with the original pinned multimodal
config for a zero-image contract fixture; it is not numerical-reference evidence.

Independent-PR-head GPU execution, other GPUs, framework versions, FLA execution
and screenshot-model numerical
parity require separate qualification. This corpus does not establish general
GUI accuracy or a performance improvement.
