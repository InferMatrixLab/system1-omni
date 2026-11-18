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
The card names torch 2.11 and transformers 5.10.2. The [flash acceptance](../../../recipe/clef/reference.md#controlled-migration-acceptance)
uses the complete pinned stack in the recipe; 27B remains unverified.

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

The flash checkpoint was loaded in the recorded 2026-10-27 run on one NVIDIA
L20X reporting PyTorch capability 9.0 (nvidia-smi reports 8.9). Both HTTP paths
completed three Public231 C1 rounds, 231/231 each, with exact parity under identical
frozen FLA autotune choices. Startup, first/repeated requests, concurrency,
TCP disconnect recovery and child exit were exercised. The
[recipe](../../../recipe/clef/reference.md#controlled-migration-acceptance)
records the six latency results, controls, failed initial cache experiment
and limitations; its historical section retains the earlier operator reports.
The contributor confirmed the raw evidence and frozen cache were lost after a
host restart. These historical GPU results cannot currently be independently
verified; a new campaign with durable artifacts is needed for acceptance.
CPU coverage is 18 Rust lifecycle/serving tests and 36 Python contract tests. The
fixed tokenizer fixture also matches the pinned loader.

## Status

| Path | Clef-flash | Clef |
| --- | --- | --- |
| Text `choice` / `noul` / `score`, non-thinking prompt | Historical GPU run recorded 693 exact paired responses; raw evidence lost, new acceptance pending; see recipe | Same loader interface; no 27B inference run |
| Image and video | Rejected | Rejected |
| Rust HTTP + persistent Python executor | Default serve path with CPU lifecycle regression tests; previous GPU checks are recorded, but their raw evidence is lost | Same server, `--variant clef`; GPU unverified |
| Joint-head native CUDA | Not implemented | Not implemented |
| Workers AI proxy | Out of scope | Out of scope |

### Failure and cancellation ownership

The Rust `Executor` owns child liveness and publishes IPC errors or unwinding
exchange panics while its
scheduled blocking exchange still holds admission. This keeps failure state
correct even after the HTTP caller is cancelled. Health checks reap an exited
idle child and return unavailable instead of serving stale ready data. The
scheduler owns FIFO admission; its closure retains the executor until the full
response has been read. See the [service lifecycle checks](../../../recipe/clef/reference.md#service-lifecycle-acceptance)
for error statuses, cancellation boundaries and remaining queue/deadline limits.
