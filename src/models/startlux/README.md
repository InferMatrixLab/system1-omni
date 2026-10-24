# StartLux-Decision v1.1 4B Instant

Native text worker for one pinned checkpoint. It answers `choice`, `noul`,
and `score` on `POST /v1/systemone`. AutoThink, Instant 27B, and images are
out of scope.

The weights are **CC BY-NC 4.0**. They are not in this repository. The
adapted prompt and readout rules come from Apache-2.0 code; see
[NOTICE](NOTICE).

| Pin | Value |
| --- | --- |
| Weights | `startlux-models/StartLux-Decision-v1.1-4B-Instant` |
| Weights revision | `416088f77cfce09633906a3675c7cd2e5dabcf10` |
| Prompt/readout source | `StartLuxLabs/StartLux-Decision` `a99977d9332decc9fb27a2488b4a83b533b6cd5f` |
| Served model id | `StartLux-Decision-v1.1-4B-Instant` |
| Decision format | `lodestar-letter-readout-v1` |
| Prompt attention | bidirectional full attention; linear attention stays causal |

Build and serve from [the recipe](../../../recipe/startlux/README.md). The
L20X latency recipe is [latency.md](../../../recipe/startlux/latency.md).
Contract tests live in `tests/startlux/contract.rs`.

## What this worker is

Instant is Qwen3.5-4B with bidirectional full attention on one of every four
layers and a letter readout. It is not an Open-Jev LoRA plus a scalar head.
The worker reuses the shared Qwen3.5 loader, GDN, MLP, norms, RoPE, and
GEMMs. Full-attention layers call the additive
`cs1_attention_gated_bidirectional` symbol. Gated DeltaNet stays causal.
There is no prefix cache and no CUDA graph on this path.

Processing (`processing.rs`) loads the tokenizer once and shares it.
A request encodes through that share.
The load checks that A–Z are the pinned token ids 32..=57 immediately
after the thinking-off prompt, and compiles the request. The executor loads the checkpoint, requires the
bidirectional symbol, and returns FP32 letter logits from a CPU dot of the
tied `model.language_model.embed_tokens.weight` rows. Softmax, per-type
temperature, none-check, and wide-choice combination stay in the contract
module. One request is one `SerialScheduler` admission. Prompts inside that
admission run one at a time. There is no candidate packing.

## Request contract

Defaults match `startlux-decide serve`: `none_check` on, `allow_none` off,
context cap 65536 tokens (the vLLM release `max-model-len`). The model
context is 262144; `STARTLUX_MAX_MODEL_LEN` can raise the cap up to that.
Prompts over the cap return 422. Nothing is truncated.

`think_mode` `never` and `plain` are accepted. `auto`, `always`, `hybrid`,
and `think` return 422. This folder has no `auto_think` block. Images,
`return_details`, `return_thought`, and `thought_generation: stream` return
422. Bad JSON is 400. The request `model` field is ignored; the response
model id is the pin above. `/v1/decisions` is not served.

Choice lists keep request order. A single listed option with none-check off
skips the forward and returns probability 1 with `input_tokens` 0. Lists
that do not fit in 26 letters (25 when the extra "None of the above" option
is present) use the pinned knockout groups (25 / keep 3 / residual 0.001).
Group rounds do not include None. The final round does. Wide-choice
probability keys follow the original option order. Upstream's combined dict
is finalists then the rest; the values are the same combination, and a tie
goes to the earlier listed option. Choice entries that are not strings or
numbers are rejected.

Score and choice probabilities are full precision. `benchmarks/bench.py`
`--score-decimals 4` accepts a score that equals the expectation of those
probabilities. Noul is P(true), shown in the prompt as yes/no.

`input_tokens` is the sum of prompt tokens, including knockout group prompts
and the final round. `output_tokens` is 0.

## Checkpoint load

Language tensors are BF16 except `linear_attn.A_log` and
`linear_attn.norm.weight`, which are FP32 in this folder (48 tensors). The
existing kernels read BF16, so those two suffixes are rounded to BF16 at
upload. Any other non-BF16 language tensor still fails the load. Vision and
`mtp.*` tensors are not uploaded. The letter head is the tied embedding.
That CPU dot is not yet compared with vLLM or Transformers.

## Status

CPU contract tests cover prompts, option order, none-check, wide knockout,
score limits, and 422/400 rejections. GPU numerical parity and HTTP latency
are **unverified**. The build target is Ada sm_89 / L20X. Do not
read an H200 or H800 leaderboard number as a result for this model.

## Gaps versus Open-Jev native

- Bidirectional full attention, not causal full attention.
- Letter logits and a temperature per question type, not one trained scalar
  head and one temperature.
- No candidate packing. Prompts are serial, as in Open-Jev-9B rather than
  the 27B packed path.
- No prefix cache and no CUDA graph.
- Text only. None-check adds an extra option.
- FP32 `A_log` and GDN norm are rounded to BF16.
- No GPU parity yet. AutoThink and 27B are not implemented.
- Wide-choice probability keys follow request order.
