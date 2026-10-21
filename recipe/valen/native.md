# Valen-Preview-0923 native worker (spec 03)

The native worker ([`src/models/valen/native/`](../../src/models/valen/native/))
serves Valen's `choice` questions on the shared Qwen3.5 native backends — the
unmerged 2B language model with the FP32 LoRA of the spec-02 export, the
manifest-verified vision tower, and a CPU FP32 decision head — with no Python
or PyTorch in the serving path. It speaks the reference worker's
`/v1/systemone` contract ([`reference.md`](reference.md)) byte for byte.

This page covers startup, the request examples, the validation scope and the
declared parity gates. The contract table at the end is the promotion artifact
for the tracker.

## Startup

Run from the repository root. Build the CUDA kernels for your GPU's compute
capability (120 for Blackwell consumer, 89 for Ada, 80 for A100, 90 for H100),
then the worker:

```sh
src/backends/cuda/qwen3_5/build.sh target/cuda 120   # needs nvcc, cuBLAS and cuBLASLt
cargo build --release --locked -p omni-valen-native
```

Artifacts: the spec-02 export (five files under `weights/valen-export/`, see
[`export.md`](export.md)) and the pinned Qwen3.5-2B base directory (the
tokenizer source). The worker validates the export policy — manifest format,
LoRA profile (rank 32 / alpha 64 / scale 2 / FP32 / 186 targets / 372
tensors), the head's shape and hash, and the vision provenance pins — before
any CUDA loading.

```sh
VALEN_MODEL=weights/valen-export VALEN_BASE=weights/Qwen3.5-2B \
VALEN_CUDA_LIB=target/cuda/libqwen3_5_cuda.so \
VALEN_HOST=127.0.0.1 VALEN_PORT=8000 target/release/omni-valen-native
```

`VALEN_CUDA_LIB` defaults to the library next to the executable; `VALEN_HOST`
and `VALEN_PORT` default to `127.0.0.1:8000`. The listener binds only after
one real image and one real text warmup inference (the reference worker's
warmup requests), so `/health`'s `ready` means a full round trip per modality
has completed.

The frontend is the byte-forwarding proxy it always was:

```sh
OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 target/release/omni-jev
```

## Examples

The recipe's image choice example:

```sh
curl -X POST http://127.0.0.1:8080/v1/systemone \
  -H 'content-type: application/json' \
  --data-binary @recipe/valen/example-request.json
```

```json
{"model": "valen-preview-0923", "answers": {"color": {"type": "choice",
"choice": "white", "probabilities": {"white": 0.5223, "black": 0.4777},
"confidence": 0.0445}}, "usage": {"input_tokens": 108, "output_tokens": 0},
"internal_usage": {"compute_tokens": 108}}
```

(token counts above are illustrative — the recorded fixture in
`fixtures/choice-parity-reference.json` carries the exact accounting.)

A text choice request:

```sh
curl -X POST http://127.0.0.1:8000/v1/systemone \
  -H 'content-type: application/json' -d '{
  "model": "valen-preview-0923",
  "state": "The QA log lists three runs: r1 passed in 12s, r2 failed on a timeout, r3 passed after a retry.",
  "questions": {"which_passed": {"type": "choice", "instructions": "Which runs passed?",
    "criteria": {"r1_only": "Only run r1 passed.", "r1_and_r3": "Runs r1 and r3 passed."}}}}'
```

## What is served (milestone scope)

| Question type | Status | Notes |
| --- | --- | --- |
| `choice` | **served** | text and single-image states, 1–8 questions, 1–255 options |
| `noul` | rejected 422 (N5) | compiled by the wire/compiler layers already; finishing is N5 |
| `score` | rejected 422 (N5) | per-level branches compile; aggregation is N5 |

The rejection body names the milestone explicitly:
`{"detail": "question 'n': the noul question type is not served by this worker
milestone; see recipe/valen/native.md"}`. This is the one deliberate
deviation from the reference contract in this milestone and is recorded in
the contract table below.

## Execution shape

`choice` compiles one branch per question over the chat-template state
(N3's compiler-equivalent processing). A text state runs the branch tokens
through the unmerged 2B language model; an image state decodes and
preprocesses the wire image once, runs the vision tower once, and splices the
adapted rows into the language forward at the placeholder positions with the
meshgrid mrope positions (the HF `get_rope_index` port the `jev_vl` worker
gates). Both paths read the final-norm hidden state at the compiler's
captured candidate-end and decision-end positions and hand those rows to the
CPU FP32 head: `logit_k = (W_cand·h_cand_k) · (W_dec·h_dec) / 16`. The
finisher is the reference postprocessor: FP64 softmax, Valen's confidence
formula, and `json.dumps` byte semantics (CPython `repr` floats) so response
bodies are byte-identical to the reference's for the same logits.

Admission is serial — one request at a time, `503 {"detail": "worker busy"}`
while one is running, exactly as the reference's `inference_lock`. Each
request is prepare → execute → finish under one admission.

## Validation scope and gates

- **CPU (CI, no GPU)**: compiler equivalence (token ids, branch plans,
  positions, rejection corpus — N3's `tests/valen/native*.rs`), the finisher
  byte parity and the HTTP contract over a stub inference
  (`tests/valen/worker.rs`), the head's readout formula on a deterministic
  synthetic head (`tests/valen/data/head-readout.json`), and the frontend's
  byte-forwarding against the native worker (`tests/frontend/native_worker.rs`).
- **GPU (env-gated)**: the vision tower vs the recorded reference output
  (N1, declared max-abs 1.25, measured 1.03125), the unmerged text forward's
  final hidden state (N2, declared 0.5, measured 0.25), and this ticket's
  end-to-end choice parity vs the pinned reference — recorded in
  `fixtures/choice-parity-reference.json` by `dump_choice_parity.py`
  (usage: like `dump_text_reference.py`; `--tolerance-probability`,
  `--tolerance-confidence` and `--head-tolerance` declare the measured gates):

  ```sh
  VALEN_EXPORT=weights/valen-export VALEN_BASE=weights/Qwen3.5-2B \
  VALEN_CUDA_LIB=target/cuda/libqwen3_5_cuda.so \
  cargo test -p omni-valen-native --test worker_gpu -- --ignored
  ```

  **Recorded verdict (RTX 5060 Laptop GPU, sm_120, BF16, torch
  2.11.0+cu128, eager, TF32 off, 2026-10-21; native: the same GPU, the
  `libqwen3_5_cuda.so` built at ABI 8 for sm_120 by
  `src/backends/cuda/qwen3_5/build.sh`):**

  | Case | Answer | Max probability Δ | Max confidence Δ |
  | --- | --- | --- | --- |
  | text choice (`which_passed`) | **exact** (`r1_and_r3`, reference margin 0.514) | 8.713e-3 | 1.743e-2 |
  | image choice (`color`, 2×2 PNG) | recorded near-tie flip (reference margin 0.108) | 7.629e-2 | 6.349e-2 |
  | head probe (CPU head on the reference's own readout rows) | — | 4.530e-6 max logit Δ | — |

  **Declared tolerances (in the fixture, measured first):** probability 0.1,
  confidence 0.1, head probe 1e-4. The text case alone sits at the predicted
  0.02 class (measured 8.7e-3); the image case adds the vision tower's own
  declared drift (N1: 1.03125 max-abs measured, 1.25 declared, on this very
  fixture), which raises the combined end-to-end figure. The decomposition
  diagnostic (`image_path_decomposition_diagnostic` in `worker_gpu.rs`)
  attributes it precisely: pixel preprocessing is **bit-exact** against the
  recorded HF-processed values (0/393216 BF16 elements differ); feeding the
  native language path the *reference's* tower rows lands within 9.2e-3 of
  the reference logits — so the mrope positions, the row splice, the
  captured-position readout and the head are all correct, and the flip is
  the tower's cross-kernel BF16 drift closing a reference decision margin of
  0.108 on a nearly-tied example (the reference itself answers with
  confidence 0.108). That flip is **declared in the fixture**
  (`declared_flips.image`) and accepted by the gate exactly as recorded;
  every probability and confidence delta stays gated. On a deployment where
  a decision margin that thin matters, the component gates (N1/N2) are the
  levers, not this worker's plumbing.

### Known limitations (this milestone)

- `noul` and `score` are not served (N5); they are rejected with a
  deterministic 422, not silently answered.
- Video and multi-image states are out of scope (spec 04 contract pending);
  the wire rejects them as text states containing control tokens or as
  invalid image states, as the reference does.
- Performance is **not** characterized here — no throughput, latency or
  memory claims are made for this milestone; eager correctness only. Batching,
  CUDA Graph replay and prefix reuse are separate later hypotheses (note: the
  #143 numeric profile rejects prefix continuation, so branch reuse must be
  re-qualified against it when its time comes).
- Transport corners where the HTTP library answers before the worker's gates:
  hyper (axum's HTTP/1 parser) merges duplicate equal `Content-Length` headers
  and rejects conflicting or non-numeric ones at the protocol level, where the
  reference worker answers `411 one Content-Length is required` / `400 invalid
  Content-Length` with a JSON body. Only non-conformant clients reach this;
  every gate reachable through a conforming client is pinned byte-identical in
  `tests/valen/worker.rs`.
- Image decoding: PNG (all colour modes PIL supports) decodes
  deterministically across stacks; JPEG baseline decode differences between
  libjpeg-turbo and the Rust decoder are not gated — the parity fixtures are
  PNG. If byte-exact JPEG parity matters for a deployment, record a JPEG
  fixture before relying on it.
- The response header set differs cosmetically (hyper adds its own
  `Server`/`Date`); statuses, bodies and `Content-Type` are identical.

## Contract table (RFC-style, for promotion)

Extracted from spec 03 and the reference worker; the promotion tracker's
material.

| Area | Contract | Native status (N4) |
| --- | --- | --- |
| Route | `POST /v1/systemone`, `GET /health` | identical |
| Unknown route/method | `404 {"detail": "unknown route"}` | identical (GET/POST; other methods differ, see below) |
| Chunked upload | `411 {"detail": "Content-Length is required; chunked requests are unsupported"}` | identical |
| Duplicate Content-Length | `411 {"detail": "one Content-Length is required"}` | unreachable (hyper normalizes; recorded deviation) |
| Invalid Content-Length | `400 {"detail": "invalid Content-Length"}` | unreachable (hyper rejects at parse; recorded deviation) |
| Content-Length > 8 MiB or negative | `413 {"detail": "request exceeds body limit"}` | identical |
| Content-Type ≠ application/json | `415 {"detail": "Content-Type must be application/json"}` | identical (parameters allowed) |
| Concurrent request | `503 {"detail": "worker busy"}` | identical |
| Incomplete body | `400 {"detail": "incomplete body"}` | identical |
| Slow body (> 15 s) | `408 {"detail": "request body timed out"}` | identical |
| Body > 8 MiB at parse | `413 {"detail": "request body exceeds 8 MiB"}` | identical |
| Malformed JSON / non-object / duplicates / non-finite | `400 {"detail": …}` (N3 wire layer corpus, 70 cases) | identical |
| Wire validation (model/state/questions) | `422 {"detail": …}` (N3 corpus) | identical |
| Token budget (max_length 8192) | `422 "request rejected by the pinned compiler: …"` | identical |
| `noul` / `score` questions | served by the reference | **422 milestone rejection (N5 serves them)** |
| Inference failure | `500 {"detail": "inference failed"}` | identical (no internal detail leaks) |
| Health body | `{"status": "ready", "model": "valen-preview-0923", "modality": "multimodal"}` | identical bytes |
| Readiness | ready only after real warmup (image + text) before the listener binds | identical |
| Response body | reference field order and float repr; probabilities FP64 softmax; Valen confidence formula | byte-identical for equal logits |
| `usage` | `{"input_tokens": logical, "output_tokens": 0}` | identical semantics (base + every suffix) |
| `internal_usage` | `{"compute_tokens": compute}` | identical semantics (every full branch) |
| Serial admission | one request at a time | identical |
| Unsupported HTTP method on served routes | reference: 404 (do_GET/do_POST fallthrough) / 501 (no handler) | 404 via fallback for GET/POST; 405 for other methods (cosmetic deviation) |
| Frontend | byte-forwarding proxy, unchanged | pinned by `tests/frontend/native_worker.rs` |

## Parity recording protocol

`recipe/valen/dump_choice_parity.py` rebuilds the pinned reference stack
(Valen @ 750bfcf, checkpoint sha256 `836622ef…`, base 15852e8c, eager BF16,
TF32 off) exactly as the worker does, serves the two fixtures through
`frontend.valen.decide` with a recording executor, and also records the
reference head's readout rows and logits (the head probe). Re-run it on the
GPU host when the reference moves; tolerances are measured first and then
declared — a `null` tolerance fails the Rust gate closed.
