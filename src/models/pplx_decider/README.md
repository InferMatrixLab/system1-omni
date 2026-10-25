# pplx-decider-v1.1-27b

Reference worker for the public checkpoint
[perplexity-ai/pplx-decider-v1.1-27b](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b)
revision `14c9d90137962d300f3de95633f7b6535e80c5ce` (Apache-2.0). The base
model named inside that checkpoint is `Qwen/Qwen3.8-27B` at
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. License and attribution are the
checkpoint `LICENSE` and `NOTICE`. This tree does not vendor the weights or
the training code.

Startup, the example request and the Public231 concurrency-1 procedure are
in [recipe/pplx_decider/README.md](../../../recipe/pplx_decider/README.md).

## What is implemented

`contract.py` admits `/v1/systemone` requests and builds prompts and typed
answers the way `source/src/autojev/model.py` and `server.py` do at this
revision. `engine.py` hashes those files, the tokenizer, the chat template
and `readout.safetensors` before import, then runs the checkpoint's
`DecisionModel`. `src/frontend/pplx_decider.py` is the HTTP process. The Rust
frontend only forwards.

CPU fixtures in `tests/pplx_decider/fixtures/reference_cases.json` lock prompt
text and typed answers for `choice`, `noul` and `score`, including a JSON
integer above 2^64. Rejection tests cover 256 options, score length, unknown
fields, bad images and the token-limit sentence. They do not load weights.

## Request and answers

The body is `model`, `state`, `questions` and optional `images`. Unknown
fields are rejected. Top-level `thinking` is admitted only when it is
`{"enabled": false}`; that object is ignored. The checkpoint template is
still applied with `enable_thinking=False`. Any other `thinking` value is
HTTP 422 `thinking.enabled must be false`. `model` must be `autojev-qwen3.8-27b`, `autojev`,
`jev-latest`, `jev-preview` or `jev-1.13.0`. The response `model` is always
`autojev-qwen3.8-27b`.

`state` is a string, object or array. JSON integers stay integers, including
values outside the float range, and `json.dumps` writes them in decimal.
Duplicate keys keep the last value, which is the standard-library parse the
checkpoint server uses.

`choice` has 1 to 255 options, in request order. `noul` may only describe
`true` and `false`, and the probability reported is the renormalized `true`
mass (the second option). `score` has 2 to 10 levels; keys are `"0"`, `"1"`,
… in that order; `score` is the expectation. Ties in `choice` and `score`
keep the earliest maximum. Probabilities are not rounded.

At most four inline PNG, JPEG or WebP data URLs. Each string is at most
12,000,000 characters, each decoded image at most 8,000,000 bytes and
16,000,000 pixels. There is no video field.

A question whose prompt exceeds 8,192 tokens raises the checkpoint sentence
`Question branch exceeds the 8192-token limit; no input was truncated.` The
worker returns that as HTTP 422. Options beyond the limits above are also
422. The worker does not drop options or clip token ids.

Inside one request the checkpoint code scores questions in batches of 8.
Separate HTTP requests do not share a batch. One request holds the worker
lock; a second gets HTTP 503.

The served response is:

```json
{"model": "autojev-qwen3.8-27b", "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}
```

`input_tokens` is the checkpoint's attention-mask sum. `output_tokens` is 0.

## Executor boundary

The processor (`contract.py`) owns admission, prompt text and the CPU answer
formulas. The executor owns the loaded module, the CUDA (or explicit CPU)
device, and the forward. `evaluate` follows `server.predict`: `prepare`,
divide by the saved temperature `1.0087417621345625`, float32 softmax, slice
to the valid rows, then the checkpoint's `answer`, which renormalizes.

Full-attention layers run in the checkpoint's noncausal mode. Linear-attention
layers stay recurrent. Pooling is the last token. The readout is the separate
BF16 `[255, 5120]` tensor in `readout.safetensors`, not an `lm_head`.

After load, `attention.py` marks full-attention modules `is_causal = False`
and replaces the checkpoint pre-hook. An all-true padding mask omits the
explicit full-attention mask so Transformers 5.17 SDPA can use its
grouped-query kernel (`head_dim` 256). Any false padding bit keeps
`padding[:, None, None, :].bool()`. The linear-attention mask is unchanged. SDPA stays on the checkpoint
order, which leaves cuDNN off. `PPLX_SDPA=cudnn-first` and
`PPLX_MLP=fused-gate-up` are refused. The three measured compile tips
raised p50, including `34ba5de`, whose 136-token probe was eager and
added no compiled frames. Startup leaves the forward eager.
`PPLX_COMPILE=dynamic` and `PPLX_COMPILE=dynamic-tail` are rejected
opt-ins.
Inductor is not bit-exact with eager. CUDA startup requires
`causal-conv1d` 1.7.0, imported before Qwen3.5 modeling, and refuses the
`F.conv1d` fallback. The build keeps an sm_90 cubin and an sm_89 cubin. It
also requires `flash-linear-attention` 0.5.2. Lab GPUs labeled NVIDIA L20X
are the H200 target. Capability 8.9 reports `linear_attention` as `fla-chunk` and is ready.
Capability 9.0 with `flashinfer-python` 0.6.16.post3 uses all-sequence
FlashInfer and keeps 48 distinct `cu_seqlens` buffers per device and
length, one per linear layer. The kept primary reused one buffer.
`PPLX_GDN_PREFILL=fla` reproduces the prior `f968e576` chunk.
`PPLX_GDN_PREFILL=short` is the rejected one-sequence ≤256 dispatch.
CuteDSL and CUDA graphs are not used. These kernels are not bit-exact with
the torch fallbacks. The kept primary is
`ffae182cc80eaeb023f5e3501fb2dd36b1c65004`: 231/231, p50 92.844 ms,
p95 296.458 ms. Versus the prior primary `f968e576` 94.077 / 300.200
that is p50 −1.31% and p95 −1.25%. `/health` reported `compile=default`,
`compile_median=eager`, `sdpa=default`, `mlp=default`,
`fused_gate_up_layers=0`, `linear_attention=flashinfer+fla+causal_conv1d`,
`flashinfer_status=bound`, and `flashinfer=0.6.16.post3`. Quality:
`choice_acc=0.9065`, `noul_acc=0.8784`, `score_mae=0.2035`. Warmup,
concurrency, the latency metric, and compute capability were not
restated in that paste. The prior primary
`f968e576aa733a0e6e4b2bc46bb11035e7662c8f` is the remeasure on the lab
L20X-labeled GPU: warmup 5, concurrency 1, stock Public231 with
`thinking.enabled=false`, metric `successful_latency`, 231/231, p50
94.077 ms, p95 300.200 ms, `linear_attention=fla-chunk`,
`flashinfer=null`, `compute_capability=9.0`. Against `1ba07d15` run1
that remeasure is p50 −4.700 ms (−4.76%) and p95 −41.002 ms (−12.02%).
An earlier paste of 98.037 / 302.569 is not a primary.
Decision disagreements were not in the remeasure. All-sequence FlashInfer
on `3c72f02` missed the gate (p50 88.808 ms, p95 300.751 ms) and is not
the primary. Short-sequence FlashInfer on `a50b9aa` missed the gate
(p50 95.446 ms, p95 300.871 ms, `choice_acc=0.9065`, `noul_acc=0.8784`)
and is not the primary. cuDNN-first SDPA on `0dcaa0ac` missed the gate
(p50 138.501 ms, p95 332.859 ms, `choice_acc=0.9065`, `noul_acc=0.8784`,
`score_mae=0.2050`) and is not the primary. Fused gate-up on `8b4ee326`
missed the gate (p50 102.437 ms, p95 304.970 ms, `choice_acc=0.9065`,
`noul_acc=0.8784`, `score_mae=0.2027`) and is not the primary.
Always-on dynamic compile on `e36b32ed` missed the both-percentiles gate
(p50 97.751 ms, p95 263.124 ms, `choice_acc≈0.906`, `noul_acc≈0.878`,
`score_mae≈0.203`) and is not the primary. The 512-token split on
`7e3b1be` missed the gate (p50 100.037 ms, p95 257.084 ms,
`choice_acc=0.906`, `noul_acc=0.878`, `score_mae=0.200`) and is not the
primary. Isolated-median compile on `34ba5de` missed the gate (p50
97.084 ms, p95 256.679 ms, `choice_acc=0.9065` (139), `noul_acc=0.8784`
(74), `score_mae=0.2003` (18)) and is not the primary. The 48-slot
`cu_seqlens` ring has no sample.

The shared Qwen3.5 CUDA language kernels in this repository are causal
(`src/backends/cuda/qwen3_5/attention.cu`). Their vision-operator tests mention
noncausal attention for a different kernel. This worker does not call that
backend. A native executor is not in this change.

## Status

| Path | State |
| --- | --- |
| CPU contract and HTTP rejection | Implemented; tests do not need weights |
| CUDA reference worker | Checkpoint `device="cuda"` loader, unpadded full attention, checkpoint SDPA order, separate gate and up GEMMs, an eager language-model forward, `causal-conv1d` 1.7.0 with sm_90 and sm_89 cubins, FLA 0.5.2 chunk, and all-sequence FlashInfer on capability 9.0 with 48 `cu_seqlens` slots per device and length. `PPLX_GDN_PREFILL=fla` holds an installed pin and reproduces the prior `f968e576` chunk. `PPLX_GDN_PREFILL=short` is the rejected length dispatch. Load of head `1ba07d1` was reported on 1× lab L20X (sm_89, ~140 GiB): BF16 resident ~50333 MiB, no OOM |
| Public231 concurrency 1 | Kept primary is `ffae182` on the lab L20X-labeled GPU: 231/231, p50 92.844 ms, p95 296.458 ms, all-sequence FlashInfer, one reused `cu_seqlens`. Versus prior primary `f968e576` 94.077 / 300.200: p50 −1.31%, p95 −1.25%. Quality `choice_acc=0.9065`, `noul_acc=0.8784`, `score_mae=0.2035`. Uncached all-sequence FlashInfer `3c72f02` is not the primary: p50 88.808 ms, p95 300.751 ms. Short-sequence FlashInfer `a50b9aa` failed its gate: p50 95.446 ms, p95 300.871 ms. cuDNN-first SDPA `0dcaa0ac` failed its gate: p50 138.501 ms, p95 332.859 ms. Fused gate-up `8b4ee326` failed its gate: p50 102.437 ms, p95 304.970 ms. Always-on dynamic compile `e36b32ed` failed the both-percentiles gate: p50 97.751 ms, p95 263.124 ms. The 512-token split `7e3b1be` failed its gate: p50 100.037 ms, p95 257.084 ms. Isolated-median compile `34ba5de` failed its gate: p50 97.084 ms, p95 256.679 ms. None of those is the primary. The 48-slot `cu_seqlens` ring has no sample |
| CPU `device="cpu"` | The checkpoint switches the backbone to float32. Not the measured path |
| Images | Admitted. No image parity run |
| Apple GPU | Not supported |

Weight shards were not re-hashed here. `release-manifest.json` at this
revision records the 11 shards plus `readout.safetensors` as 52,173,413,928
bytes together. `readout.safetensors` itself was hashed (2,611,288 bytes).
The card asks for roughly 49 GiB of weights plus working memory. The reported
load on 1× lab L20X (~140 GiB) was ~50333 MiB resident. That run does not
measure a 48 GB card.
