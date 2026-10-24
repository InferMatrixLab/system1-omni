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
fields are rejected. `model` must be `autojev-qwen3.8-27b`, `autojev`,
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

The shared Qwen3.5 CUDA language kernels in this repository are causal
(`src/backends/cuda/qwen3_5/attention.cu`). Their vision-operator tests mention
noncausal attention for a different kernel. This worker does not call that
backend. A native executor is not in this change.

## Status

| Path | State |
| --- | --- |
| CPU contract and HTTP rejection | Implemented; tests do not need weights |
| CUDA reference worker | Code path is the checkpoint's `device="cuda"` loader. Not run here |
| Public231 concurrency 1 on sm_89 | Procedure only. No p50 or p95 |
| CPU `device="cpu"` | The checkpoint switches the backbone to float32. Not the measured path |
| Images | Admitted. No image parity run |
| Apple GPU | Not supported |

Weight shards were not re-hashed here. `release-manifest.json` at this
revision records the 11 shards plus `readout.safetensors` as 52,173,413,928
bytes together. `readout.safetensors` itself was hashed (2,611,288 bytes).
The card asks for roughly 49 GiB of weights plus working memory. A 48 GB
device cannot be assumed to load that resident set.
