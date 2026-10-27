# Clef and Clef-flash

Cloudflare Clef on `POST /v1/systemone`. The default bring-up is Clef-flash
(9B). Clef (27B) uses the same loader, tokenizer and request contract with
its own backbone and joint-head config. Image and video inputs are rejected.

The default server is the Rust binary `omni-clef`. It admits one request at
a time through `omni-runtime` and forwards the body to one Python executor
process. That process loads the pinned joint-schema checkpoint and scores
every request on its main thread, the thread that initialized CUDA. The
joint head is still the checkpoint's Python module. A native reimplementation
on the shared Qwen CUDA executor is not this slice.

The checkpoint is not an Open-Jev export. `joint_schema_model.py` loads
`Qwen3_5ForConditionalGeneration` and a separate `JointSchemaHead`. The
safetensors index contains the Qwen language model and vision tower and has
no LoRA keys; the joint head is `joint_head.safetensors`. Shared Qwen CUDA
kernels in this repository score a different head and are not used here.

## Pinned artifacts

Both public snapshots were read on 2026-10-24. The loader and tokenizer bytes
match across the two revisions. Config and joint-head bytes do not.

| | Clef-flash | Clef |
| --- | --- | --- |
| Repo | [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) | [Cloudflare/clef](https://huggingface.co/Cloudflare/clef) |
| Revision | `f0b7d3cd65296abfe459e6c228b618f5d8b26d2f` | `a30162b99d68a481d8bd75d38df069a0cf83df40` |
| Base model | `Qwen/Qwen3.5-9B` | `Qwen/Qwen3.8-27B` |
| License | Apache-2.0 | Apache-2.0 |
| Backbone | hidden 4096, 32 layers | hidden 5120, 64 layers |
| Joint head | hidden 4096, width 1024, 2 routing layers, 4 decoder layers | hidden 5120, same width and depth |
| Safetensors, including the joint head | about 19.1 GB | about 55.0 GB |
| `config.json` SHA-256 | `66f87f6fb2616b46604daf2a9c67ddc87938296d07156efa34d59b5be49e3238` | `c42e88892bd3fd84e8276b2ad90df58c1c3b797676ea161035006a72ad468c58` |
| `joint_head_config.json` SHA-256 | `77efe959a38b5b17b241543e129e695f3c77465ece55a25985279bd8176279a0` | `890be585d75b981201eb96a35f98a9967afe37bc8d220cfdea2e72d56507534f` |

Shared file SHA-256 values, identical at both revisions:

| File | SHA-256 |
| --- | --- |
| `joint_schema_model.py` | `0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3` |
| `tokenizer.json` | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| `tokenizer_config.json` | `91a08f825d370d085d692e04cf117cdd7faad7bf18e996f1e6031b6dab03db72` |
| `processor_config.json` | `d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1` |

Weight shards are not hashed by the worker. Download the snapshot at the
revision above so the shards belong to the same commit as the hashed files.
The card names torch 2.11 and transformers 5.10.2. Those packages were not
imported in this change.

Workers AI model names (`@cf/cloudflare/clef`, `@cf/cloudflare/clef-flash`)
are a hosted API. This worker does not call them.

## Request and response

`choice`, `noul` and `score` are served for text `state`. `state` may be any
JSON value. Strings are inserted as text; every other value is compact JSON
with sorted keys. The response echoes `model`, returns one answer per question
in request order, and sets `usage.output_tokens` to 0. `usage.input_tokens`
is the untruncated prompt length.

The encoder's decision suffix is the empty think block
`<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:`. That is the non-thinking
path. The chat templates differ between the two revisions and can open a
think block; `systemone` does not apply those templates. `/health` reports
`prompt_mode: non-thinking`.

A request may include `thinking` only as an object whose `enabled` value is
the JSON boolean `false`. That object is removed before encoding, including
any other keys on it, so the prompt matches a request that never had the
field. `enabled: true`, a non-object, and an object without `enabled: false`
are rejected. This worker does not serve a thinking prompt.

Choice options are sorted by id inside the prompt. Answer `probabilities`
keep the request's criteria order. Ties use that same request order. The
chosen id is selected from the unrounded probabilities; the wire values are
then rounded to four decimal places. `noul` is the rounded probability of
`true` only. `score` is the rounded expected index, with `legend` holding
the original criteria values. Confidence for `choice` and `score` is the
rounded maximum probability, not an entropy.

Noul criteria may mention `true` and `false`. Any other noul key remains in
the request object and is not an option. A score criterion list must be
non-empty; a one-level list is accepted because the pinned encoder accepts
it. The Public231 runner separately requires at least two score levels.

## Limits

The shipped `systemone` default is 16384 tokens. This worker uses that limit
and rejects a longer prompt. The pinned encoder would otherwise drop the tail
of the state. The blog's 64k window and the backbone
`max_position_embeddings` of 262144 are not admission limits here.

HTTP bodies larger than 8 MiB are rejected with the same 413 detail as the
previous handler. A negative `Content-Length` is rejected by the HTTP stack
with status 400 and an empty body before that check. The cap is larger than
a 16384 token text request and smaller than an unbounded upload. Unknown
request fields are rejected. `thinking` is a known field and is admitted only by the
rule above. Non-empty `images` or `videos` are rejected with a message that
those modalities are not served. Empty lists are accepted and add no media
tokens.

One request is scored at a time. `omni-clef` admits it through
`SerialScheduler`, so a second request waits instead of receiving HTTP 503.
The queue is not bounded. The executor process holds the loaded weights and
the joint head for its lifetime. HTTP acceptance does not create a thread
that touches CUDA: warmup and every later `serve` run on that process's
main thread. There is no cross-request batch and no prefix cache.

## Processor and executor

[prompt.py](prompt.py) builds the text pieces in concatenation order.
[preprocess.py](preprocess.py) checks question shape and the token budget.
[postprocess.py](postprocess.py) mirrors `systemone_answer` for CPU tests.
The live worker calls the checkpoint's `systemone_answer` after one
`encode_record`. It does not call the checkpoint's `systemone` wrapper,
because that wrapper slices state to the token limit.

The forward is the pinned backbone, then the pinned joint head, in bf16 as
loaded, with a float32 softmax. The head reads mean-pooled question and
option spans plus a lexical prior from the output embedding. Those layouts
belong to the checkpoint file. This repository does not reimplement them.

## Validation

CPU tests cover the envelope, the three answer formulas, the 16384-token
refusal, media refusal, the hash gate, and the checked-in prompt fixture.
They also cover the frame loop and the Rust server: concurrent requests
share one executor process and its main thread, and
`python -m frontend.clef` execs `omni-clef` instead of binding
`ThreadingHTTPServer`.
The fixture's token ids were produced from the pinned tokenizer with
`tokenizers.Tokenizer.encode(..., add_special_tokens=False)` and checked
against `encode_record` from the pinned loader. Regenerate it from the
repository root:

```sh
PYTHONPATH=src python recipe/clef/export_prompt_fixture.py \
  --loader /path/to/joint_schema_model.py \
  --tokenizer /path/to/tokenizer.json \
  --check
```

No weight was loaded in this environment. The operator-reported Public231
concurrency-1 campaign below was not re-run here. The measurement steps are
in the [recipe](../../../recipe/clef/reference.md).

## Latency

Operator-reported Public231 concurrency 1 on NVIDIA L20X, compute capability
8.9, Clef-flash, worker HEAD
`903006530fb7772bb668bbf192ee4278e273e583`, warmup 5. Stock Public231 carries
a `thinking` field and that HEAD returned HTTP 422 `Unknown request fields`.
The measured file was `public231-no-thinking.jsonl` (the `thinking` key
stripped). Its SHA-256 was not reported. Do not cite the leaderboard export
hash `00d1f10ca1142499a598b5f1c6602d329cce1cef04c961c46fc0318daa2c67d3` as
the hash of that file. Driver, CUDA version and hostname were not reported.
Do not read the H800 Public231 board as a result for this worker, and do not
label this run as H200. Percentiles are not averaged across rounds.

| Round | p50 (ms) | p95 (ms) | Successful |
| --- | --- | --- | --- |
| run1, primary L20X round | 247.518 | 413.025 | 231/231 |
| run2 | 247.404 | 426.231 | 231/231 |
| run3 | 242.421 | 410.369 | 231/231 |

The `thinking.enabled: false` admission strips that field before encoding.
These L20X rounds do not include a measured p50 or p95 for that admission.
The operator named the
metric `successful_latency` and the clock
`post_to_body_json_and_schema_validation`. Answer parity against another
implementation was not part of this report.

`causal-conv1d>=1.5`, `flash-linear-attention==0.5.2`, and
`flashinfer-python==0.6.16.post3` are installed by the recipe. The worker
imports `causal_conv1d` and `fla.ops.gated_delta_rule` before Transformers
binds Qwen3.5. The default prefill is the FLA chunk. `--gdn-prefill
flashinfer` replaces the chunk global, with `use_cp` false, only when that
flag is set, the reported capability major is 9, 10, or 12, and that
package loads. A reported 8.9, including the lab name `NVIDIA L20X`, keeps
the FLA chunk and still becomes ready. Startup does not require the name
`NVIDIA H200`. `/health` `linear_attention` is `fla+causal_conv1d` by
default, or `flashinfer+fla+causal_conv1d` when the opt-in flag bound and
the loaded layers copied that wrapper. `CLEF_PYTHON` selects the recipe
interpreter. Answer parity is `recipe/clef/answer_parity.py`. A GPU pass
is not recorded here, so the default stays `fla`.
The process runs one text forward before it binds, so `/health` is not
served until that forward returns.
`torch_chunk_gated_delta_rule` and `F.conv1d` are not served.

Operator-reported Public231 C1 remeasure of worker
`594528a9f9c1ca38ce0dbb58a549191dfbde5bf3` on the L20X-labeled lab host.
The paste states warmup 5, concurrency 1, stock Public231 with
`thinking.enabled` false, and metric `successful_latency`. 231/231, p50
164.969 ms, p95 216.755 ms, `/health` `linear_attention`
`fla+causal_conv1d`. Against `9030065` run1: Δp50 −82.549 ms (−33.35%),
Δp95 −196.270 ms (−47.52%). Not re-run here. The paste does not restate
the clock name, manifest hash, driver, or CUDA version. Answer parity
was not reported. The measured bind is the FLA chunk. This pair is the
primary a later tip has to beat.

| Id | Change | Expected effect | Measured result |
| --- | --- | --- | --- |
| H1 | One `encode_record` per request, then the pinned forward. The handler does not tokenize the schema again before that encode. | Removes a second BPE pass from the critical path. Prefill and the joint head still dominate. | Included in HEAD `9030065`. Isolated delta unknown |
| H2 | Fuse the joint head's per-question Python loop. | Fewer small launches after prefill. Needs a forked head and numerical parity against the pinned module. | Not implemented. No measured H200 delta |
| H3 | `torch.compile` or CUDA graphs on the Qwen3.5 backbone. | Uncertain on the linear-attention layers. | Not implemented. No measured H200 delta |
| H4 | Bind `flash-linear-attention` 0.5.2 and `causal-conv1d` >= 1.5 before the Qwen3.5 layers copy the chunk global. The default prefill is that FLA chunk. `--gdn-prefill flashinfer` uses FlashInfer GDN prefill 0.6.16.post3 with `use_cp` false only when the flag is set, the package loads, and the reported capability major is 9, 10, or 12. Reported 8.9, including nvidia-smi name `NVIDIA L20X`, stays on the FLA chunk and still starts. Startup is not ready on `torch_chunk` or `F.conv1d`. vLLM CuteDSL, CUDA graphs, and `torch.compile` are not used. | Replaces the PyTorch chunk loop. `/health` names the bind that started. | Tip `594528a9`, L20X-labeled host, warmup 5, concurrency 1, stock Public231 with `thinking.enabled` false, metric `successful_latency`, `/health` `fla+causal_conv1d`: p50 164.969 ms, p95 216.755 ms, 231/231. Versus `9030065` run1: Δp50 −82.549 ms (−33.35%), Δp95 −196.270 ms (−47.52%). This is the primary. FlashInfer remains opt-in. `recipe/clef/answer_parity.py` is the gate; a GPU pass is not recorded here, so it is not the default |

`omni-clef` removes the `ThreadingHTTPServer` path. An operator diagnosis on
the L20X-labeled host put the first CUDA touch on a fresh HTTP thread near
110 ms: HTTP about 162–175 ms, the same executor on a persistent thread
about 48–65 ms. That diagnosis is not a Public231 paste. Against
the vLLM p50 cited with that diagnosis (~32 ms), a persistent-thread serve
of ~48–65 ms would leave about 1.55×. That remainder is unverified here.
The primary to beat remains tip `594528a9`. Operator-reported Public231 C1
of `omni-clef` tip `f9127a03b732aaf6e88c99a0212e02acfdb01880` on that host,
not re-run here: smoke 4/4, `/health` `fla+causal_conv1d`, R1 p50 169.605 ms
and p95 220.003 ms (231/231), R2 53.580 / 97.031, R3 52.950 / 97.608,
quality `choice_acc=0.8417`, `noul_acc=0.7973`, `score_mae=0.2488` (matches
the `594528a9` baseline). This is a serve-path measurement, not a latency
KEEP. R1 does not pass the dual gate against 164.969 / 216.755. R2 and R3
are not a new primary. Start the binary with `CLEF_PYTHON` set to the
recipe venv; the command is in the recipe. System `python3` lacks
`causal-conv1d` and the executor dies before the port opens.

## Status

| Path | Clef-flash | Clef |
| --- | --- | --- |
| Text `choice` / `noul` / `score`, non-thinking prompt | Contract tested. Kept primary is the previous `ThreadingHTTPServer` worker: `9030065` run1 p50 247.518 ms, p95 413.025 ms; tip `594528a9` p50 164.969 ms, p95 216.755 ms, 231/231, `/health` `fla+causal_conv1d`, Δp50 −82.549 ms, Δp95 −196.270 ms. `omni-clef` `f9127a0` is recorded above and is not a KEEP | Same loader pin; the recorded latency campaigns were flash only |
| Image and video | Rejected | Rejected |
| Rust HTTP + persistent Python executor | Default serve path. CPU tests, plus the operator `f9127a0` paste above (not a KEEP). `CLEF_PYTHON` must be the recipe venv | Same server, `--variant clef` |
| Joint-head native CUDA | Not implemented | Not implemented |
| Workers AI proxy | Out of scope | Out of scope |
