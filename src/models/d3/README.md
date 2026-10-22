# D3 reference contract

This directory contains the first System1-Omni integration slice for
D3 (Decision 3.0, [`vllm-sr/d3`](https://huggingface.co/vllm-sr/d3)). It is a
reference-serving boundary only: a Python worker that runs the checkpoint's
own pinned runtime for text state and the three typed question kinds
(`choice`, `noul`, `score`) on `/v1/systemone`. Native CUDA execution and
multimodal inputs are not part of this slice.

## Pinned artifacts

| Artifact | Value |
| --- | --- |
| Package | [vllm-sr/d3](https://huggingface.co/vllm-sr/d3) |
| Pinned revision | `cbff3a3cb0b074b6585fe9f747b9f9f5557a0e75` |
| Manifest schema | `d3-package-manifest/1` (built 2026-10-11T03:30:34Z) |
| Model identity | `model_name: "d3"`, 26.09B parameters (manifest-reported) |
| License | Apache-2.0 (Qwen3.8-27B base and d3 weights/readout/runtime) |
| Runtime dependencies | the checkpoint's own `requirements.txt` (torch>=2.8, transformers==5.17.0, flash-linear-attention==0.5.2) |

The checkpoint ships its runtime as flat Python files next to the weights
(`d3_runtime.py` and siblings). Upstream `verify_package` walks only the
39-entry `files_sha256` table and, in fast mode, skips hashing files over
64 MiB — so by default it never verifies the runtime table it executes.
This worker additionally hashes every file of
`runtime.files_sha256` (8 runtime modules plus `requirements.txt`) in full
**before importing anything** from the package directory, and refuses to
serve packages without a manifest. It also rejects importable entries the
runtime table does not name — an extra `*.py` file or `__init__.py` package
in the package root — because that root is placed first on `sys.path`, where
such an entry would shadow a dependency of the pinned runtime. Because hash
agreement with a manifest
proves only internal consistency — a later snapshot ships a self-consistent
table of its own — the gate first requires the manifest to carry exactly
the `repo_id` and `runtime.files_sha256` table checked in for the pinned
revision in `engine.py`; a foreign manifest is refused before any file is
hashed. Shard and config verification uses the
upstream `verify` mechanism (`fast` default, `full` via CLI).

## Current serving scope

- The request envelope is upstream's five-field form:
  `{"model", "state", "questions", "images", "videos"}`. Unknown fields are
  rejected with HTTP 422. `model` is accepted and ignored.
- `state` is any JSON value; the pinned runtime renders it (empty and `""`
  render as `(empty)`).
- `questions` is a nonempty mapping of nonempty string IDs to questions of
  three kinds, validated by the pinned runtime:
  - `choice`: `criteria` maps 1–255 nonempty string keys to descriptions
    (any JSON value);
  - `noul`: `criteria` is optional and may only describe `"false"` and
    `"true"`;
  - `score`: `criteria` is an ordered list of 2–10 distinct nonempty levels.
- `images`/`videos` are accepted only as absent, `null` or empty lists;
  anything else is rejected with HTTP 422. Serving media is stage 2; when it
  opens, the limits re-align to the upstream per-item budgets (8,000,000
  bytes per image, 32,000,000 per video) instead of this slice's 8 MiB
  transport cap.
- No token generation: one forward pass per question scores 255 single-token
  answer codes at the last prompt position.

## Answers

Each answer is built by the pinned runtime from per-option probabilities
(mirrored as pure functions in `postprocess.py` for the CPU contract tests):

- `choice`: `{"type", "choice", "probabilities", "confidence"}` —
  ties go to the earliest option in request order; `probabilities` are
  renormalized per option.
- `noul`: `{"type": "noul", "noul": p_true}` — the renormalized probability
  of `true`; no probabilities or confidence.
- `score`: `{"type", "score", "probabilities", "confidence",
  "legend"}` — `score` is the 0-based expectation over the renormalized
  level distribution, the answer keys are the stringified level indices, and
  `legend` maps each key to its canonical level text.
- `confidence` is one minus the normalized entropy clipped to [0, 1]
  (1.0 for fewer than two options) — the upstream "Decision 2.0 measure",
  not max-p.

`usage.input_tokens` sums each runnable question's prompt tokens;
`usage.output_tokens` is 0. The pinned package reports no
`max_input_tokens` limit (the manifest records `null`), so
`/health` reports `max_input_tokens: null` and `max_length_exceeded` cannot
trigger for it; the admission path stays for packages that set a limit.

## Errors

Three layers, matching the upstream server:

1. Transport follows the shared Python-worker discipline: `400` (unusable
   JSON body, mismatched or invalid `Content-Length`), `408` body timeout,
   `411` without exactly one `Content-Length`, `413` over the slice-scoped
   8 MiB body limit, `415` non-JSON content type, `503` while another
   inference holds the lock, `404` unknown routes, `500` anything else
   (generic detail, no request content in logs).
2. Before execution, the pinned runtime's per-question validation rejects
   the whole request with HTTP 422: invalid questions produce
   `{"invalid_questions": {name: message}}` in request order, and an
   over-limit question produces a plain `detail` naming the actual token
   count and limit — nothing is ever truncated. `max_length_exceeded` wins
   over invalid questions.
3. After execution, a question whose probabilities cannot be decoded
   (non-finite, negative, or all zero) keeps its HTTP 200 response with
   `{"type", "error": "invalid_model_output", "message"}` embedded in that
   question's answer; the other questions are still answered. This partial
   success is new to the repository, so the worker logs one warning per
   embedded failure — naming the question and the message, never request
   content — to keep it detectable.

`/health` returns `{"status", "model", "max_input_tokens", "modalities"}`
after loading and warmup; the upstream `authentication` field is dropped
(this worker has no API key).

## Ownership boundary

- `protocol.py` owns the envelope: body parsing, unknown-field and
  slice-scoped media rejection, question-ID admission.
- `preprocess.py` owns the mapping of the pinned runtime's per-question
  preparation errors to the two pre-execution 422 shapes.
- `engine.py` owns the pinned runtime, the identity pin and the pre-import
  manifest gate (full runtime hashes, no unlisted importables), device
  state, and warmup.
- `postprocess.py` owns the pure answer formulas, response finishing, and
  the embedded-failure warning.
- `src/frontend/d3.py` owns HTTP lifecycle and orchestration.
- The Rust frontend owns transport and forwarding, not model state.

The server-side error surface and envelope naming are recorded as this
repository's stance and remain coordinated with
[system1-omni#61](https://github.com/ThinkFlowLab/system1-omni/issues/61);
the noncausal full-attention hook inside the pinned runtime is adapted from
[perplexity-ai/pplx-decider-v1.1-27b](https://github.com/perplexity-ai/pplx-decider-v1.1-27b)
(Apache-2.0), as its module docstring records.

The current tests are CPU-only and do not require model weights, torch or a
GPU (`PYTHONPATH=src python -m pytest tests/d3 -q`). The real worker path
additionally requires the pinned package and a CUDA-capable environment;
the procedure is [recipe/d3/reference.md](../../recipe/d3/reference.md) and
measured-run evidence is recorded in the integration pull request, not in
the recipe.
