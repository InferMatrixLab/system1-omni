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
`PPLX_MLP=fused-gate-up` are refused. CUDA startup compiles the
language-model forward with a dynamic sequence length unless
`PPLX_COMPILE=default`, and warms that compile before it is ready.
Inductor is not bit-exact with eager. CUDA startup requires
`causal-conv1d` 1.7.0, imported before Qwen3.5 modeling, and refuses the
`F.conv1d` fallback. The build keeps an sm_90 cubin and an sm_89 cubin. It
also requires `flash-linear-attention` 0.5.2. Lab GPUs labeled NVIDIA L20X
are the H200 target. Capability 8.9, and capability 9.0 by default, report
`linear_attention` as `fla-chunk` and are ready. An installed
`flashinfer-python` 0.6.16.post3 pin is held unless
`PPLX_GDN_PREFILL=short`, which is the rejected one-sequence ≤256
dispatch. The all-sequence FlashInfer bind is not the startup path.
CuteDSL and CUDA graphs are not used. These kernels are not bit-exact with
the torch fallbacks. The kept primary is the remeasure of
`f968e576aa733a0e6e4b2bc46bb11035e7662c8f` on the lab L20X-labeled GPU:
warmup 5, concurrency 1, stock Public231 with `thinking.enabled=false`,
metric `successful_latency`, 231/231, p50 94.077 ms, p95 300.200 ms.
`/health` reported `causal_conv1d=1.7.0`, `fla=0.5.2`, `flashinfer=null`,
`linear_attention=fla-chunk`, and `compute_capability=9.0`. Against
`1ba07d15` run1 that is p50 −4.700 ms (−4.76%) and p95 −41.002 ms
(−12.02%). An earlier paste of 98.037 / 302.569 is not this primary.
Decision disagreements were not in the remeasure. All-sequence FlashInfer
on `3c72f02` missed the gate (p50 88.808 ms, p95 300.751 ms) and is not
the primary. Short-sequence FlashInfer on `a50b9aa` missed the gate
(p50 95.446 ms, p95 300.871 ms, `choice_acc=0.9065`, `noul_acc=0.8784`)
and is not the primary. cuDNN-first SDPA on `0dcaa0ac` missed the gate
(p50 138.501 ms, p95 332.859 ms, `choice_acc=0.9065`, `noul_acc=0.8784`,
`score_mae=0.2050`) and is not the primary. Fused gate-up on `8b4ee326`
missed the gate (p50 102.437 ms, p95 304.970 ms, `choice_acc=0.9065`,
`noul_acc=0.8784`, `score_mae=0.2027`) and is not the primary. The
dynamic-compile candidate has no sample.

The shared Qwen3.5 CUDA language kernels in this repository are causal
(`src/backends/cuda/qwen3_5/attention.cu`). Their vision-operator tests mention
noncausal attention for a different kernel. This worker does not call that
backend. A native executor is not in this change.

## Status

| Path | State |
| --- | --- |
| CPU contract and HTTP rejection | Implemented; tests do not need weights |
| CUDA reference worker | Checkpoint `device="cuda"` loader, unpadded full attention, checkpoint SDPA order, separate gate and up GEMMs, dynamic `torch.compile` of the language-model forward unless `PPLX_COMPILE=default`, `causal-conv1d` 1.7.0 with sm_90 and sm_89 cubins, and FLA 0.5.2 chunk. An installed FlashInfer pin is held unless `PPLX_GDN_PREFILL=short`. Load of head `1ba07d1` was reported on 1× lab L20X (sm_89, ~140 GiB): BF16 resident ~50333 MiB, no OOM |
| Public231 concurrency 1 | Kept primary is the `f968e576` remeasure on the lab L20X-labeled GPU, capability 9.0, `fla-chunk`, `flashinfer` null: 231/231, p50 94.077 ms, p95 300.200 ms. Versus `1ba07d1` run1 98.777 / 341.202: p50 −4.76%, p95 −12.02%. All-sequence FlashInfer `3c72f02` failed the gate: p50 88.808 ms, p95 300.751 ms. Short-sequence FlashInfer `a50b9aa` failed the gate: p50 95.446 ms, p95 300.871 ms. cuDNN-first SDPA `0dcaa0ac` failed the gate: p50 138.501 ms, p95 332.859 ms. Fused gate-up `8b4ee326` failed the gate: p50 102.437 ms, p95 304.970 ms. None of those is the primary |
| CPU `device="cpu"` | The checkpoint switches the backbone to float32. Not the measured path |
| Images | Admitted. No image parity run |
| Apple GPU | Not supported |

Weight shards were not re-hashed here. `release-manifest.json` at this
revision records the 11 shards plus `readout.safetensors` as 52,173,413,928
bytes together. `readout.safetensors` itself was hashed (2,611,288 bytes).
The card asks for roughly 49 GiB of weights plus working memory. The reported
load on 1× lab L20X (~140 GiB) was ~50333 MiB resident. That run does not
measure a 48 GB card.
