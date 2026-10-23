# Valen-Preview-0923 native worker (spec 03)

The native worker ([`src/models/valen/native/`](../../src/models/valen/native/))
serves Valen's `choice`, `noul` and `score` questions on the shared Qwen3.5
native backends — the unmerged 2B language model with the FP32 LoRA of the
spec-02 export, the manifest-verified vision tower, and a CPU FP32 decision
head — with no Python or PyTorch in the serving path. It speaks the reference
worker's `/v1/systemone` contract ([`reference.md`](reference.md)) byte for
byte.

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

The recipe's noul example (two noul questions — one plain, one with custom
true/false wording), served verbatim from
[`example-noul-request.json`](example-noul-request.json):

```sh
curl -X POST http://127.0.0.1:8000/v1/systemone \
  -H 'content-type: application/json' \
  --data-binary @recipe/valen/example-noul-request.json
```

```json
{"model": "valen-preview-0923", "answers": {"duplicate": {"type": "noul",
"noul": 0.1944, "probabilities": {"true": 0.1944, "false": 0.8056},
"confidence": 0.6112}, "authorized": {"type": "noul", "noul": 0.3849,
"probabilities": {"true": 0.3849, "false": 0.6151}, "confidence": 0.2303}},
"usage": {"input_tokens": 109, "output_tokens": 0}, "internal_usage":
{"compute_tokens": 123}}
```

The recipe's score example (three severity levels),
[`example-score-request.json`](example-score-request.json):

```sh
curl -X POST http://127.0.0.1:8000/v1/systemone \
  -H 'content-type: application/json' \
  --data-binary @recipe/valen/example-score-request.json
```

```json
{"model": "valen-preview-0923", "answers": {"severity": {"type": "score",
"probabilities": {"0": 0.1746, "1": 0.6223, "2": 0.2032}, "score": 1.0286,
"legend": {"0": "No billing error: every charge is correct.", "1": "Minor
error: a single duplicated charge.", "2": "Major error: repeated duplicated
charges."}, "confidence": 0.4334}}, "usage": {"input_tokens": 107,
"output_tokens": 0}, "internal_usage": {"compute_tokens": 135}}}
```

Values above are the recorded reference responses rounded for reading; the
parity fixture carries the exact figures, and mixed bodies (any combination
of the three types, up to 8 questions) answer in request order.

## What is served (milestone scope)

| Question type | Status | Notes |
| --- | --- | --- |
| `choice` | **served** | text and single-image states, 1–8 questions, 1–255 options |
| `noul` | **served** | fixed `true`/`false` candidates; optional custom wording compiles as the equivalent choice record and still answers `type: noul` with `P(true)` |
| `score` | **served** | 2–10 ordered levels; one isolated branch per level, cross-branch softmax, expected level + legend + concentration confidence |

All three types mix freely in one request. The N4 milestone rejection (a
deterministic 422 for `noul`/`score`) is gone; the wire layer's validation
422s are unchanged.

## Execution shape

`choice` and `noul` compile one branch per question over the chat-template
state (N3's compiler-equivalent processing); `score` compiles one isolated
branch per level, and the K branches' logits are concatenated before the
cross-branch softmax. A text state runs the branch tokens through the
unmerged 2B language model; an image state decodes and preprocesses the wire
image once, runs the vision tower once, and splices the adapted rows into
the language forward at the placeholder positions with the meshgrid mrope
positions (the HF `get_rope_index` port the `jev_vl` worker gates). Both
paths read the final-norm hidden state at the compiler's captured
candidate-end and decision-end positions and hand those rows to the CPU FP32
head: `logit_k = (W_cand·h_cand_k) · (W_dec·h_dec) / 16`. The finisher is
the reference postprocessor: FP64 softmax, Valen's per-type confidence
formulas (choice/noul: `(max(p) - 1/n)/(1 - 1/n)` clamped; score:
`1 - MAD(p)/MAD(uniform)` clamped), the score questions' expected level
`sum(i·p)`, and `json.dumps` byte semantics (CPython `repr` floats) so
response bodies are byte-identical to the reference's for the same logits.

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
  final hidden state (N2, declared 0.5, measured 0.25), and the end-to-end
  parity for all three question types vs the pinned reference — recorded in
  `fixtures/choice-parity-reference.json` by `dump_choice_parity.py`
  (usage: like `dump_text_reference.py`; `--tolerance-probability`,
  `--tolerance-confidence`, `--tolerance-score` and `--head-tolerance`
  declare the measured gates):

  ```sh
  VALEN_EXPORT=weights/valen-export VALEN_BASE=weights/Qwen3.5-2B \
  VALEN_CUDA_LIB=target/cuda/libqwen3_5_cuda.so \
  cargo test -p omni-valen-native --test worker_gpu -- --ignored --test-threads 1
  ```

  **Recorded verdict (RTX 5060 Laptop GPU, sm_120, BF16, torch
  2.11.0+cu128, eager, TF32 off, 2026-10-21; native: the same GPU, the
  `libqwen3_5_cuda.so` built for sm_120 by
  `src/backends/cuda/qwen3_5/build.sh` — measured at ABI 8 on the
  pre-sync tree, re-verified at ABI 9 on ced49901, see below):**

  | Case | Answer | Max probability Δ | Max confidence Δ | Max score Δ |
  | --- | --- | --- | --- | --- |
  | text choice (`which_passed`) | **exact** (`r1_and_r3`, reference margin 0.514) | 8.713e-3 | 1.743e-2 | — |
  | image choice (`color`, 2×2 PNG) | declared near-tie flip (reference margin 0.108) | 7.629e-2 | 6.349e-2 | — |
  | noul (`duplicate` + `authorized`, text) | no discrete field — all floats gated | 2.852e-3 | 5.704e-3 | — |
  | score (`severity`, 3 levels, text) | no discrete field — legend exact, floats gated | 2.454e-2 | 3.681e-2 | 1.511e-2 |
  | mixed (choice + noul + score, one body) | **exact** choice (`void`, margin 0.608) | 2.454e-2 | 3.681e-2 | 1.511e-2 |
  | head probe (CPU head on the reference's own readout rows) | — | 4.530e-6 max logit Δ | — | — |

  **Declared tolerances (in the fixture, measured first):** probability 0.1,
  confidence 0.1, score 0.05, head probe 1e-4. The text/noul/score/mixed
  cases sit at the predicted 0.02 class (max 2.5e-2 probability, 1.5e-2
  score — the noul text path is the tightest at 2.9e-3); the image case adds
  the vision tower's own declared drift (N1: 1.03125 max-abs measured, 1.25
  declared, on this very fixture), which raises the combined end-to-end
  figure. The score tolerance is declared at 0.05 over the 1.5e-2 measured
  maximum (3.3× headroom): the expected level is a probability-weighted
  average of the level indices, so per-level drift partially cancels — the
  measured figure is the gate's own assertion, and the worst-case linear
  bound at the probability gate (Σ i·0.1 = 0.3 for K = 3) is deliberately
  not what is declared. The decomposition diagnostic
  (`image_path_decomposition_diagnostic` in `worker_gpu.rs`) attributes the
  image case precisely: pixel preprocessing is **bit-exact** against the
  recorded HF-processed values (0/393216 BF16 elements differ); feeding the
  native language path the *reference's* tower rows lands within 9.2e-3 of
  the reference logits — so the mrope positions, the row splice, the
  captured-position readout and the head are all correct, and the flip is
  the tower's cross-kernel BF16 drift (max abs 1.03125 over the 64 rows)
  closing a reference decision margin of 0.108 on a nearly-tied example
  (the reference itself answers with confidence 0.108). That flip is
  **declared in the fixture** (`declared_flips.image`) and accepted by the
  gate exactly as recorded; every probability and confidence delta stays
  gated. `noul` and `score` answers carry no discrete answer field — a mode
  flip there moves only gated floats — so the flip mechanism applies to
  `choice` answers only. On a deployment where a decision margin that thin
  matters, the component gates (N1/N2) are the levers, not this worker's
  plumbing.

  **Re-verified on ced49901 at ABI 9 (2026-10-23, RTX 5060 Laptop GPU,
  sm_120, `libqwen3_5_cuda.so` rebuilt from the synced source — 36
  upstream commits, ABI 8 → 9 with #123/#98/#143 — by
  `src/backends/cuda/qwen3_5/build.sh target/cuda-abi9 120`; all gates
  re-run serially, eager, one model at a time). Every layer reproduced
  its recorded figure verbatim, so no tolerance moved and nothing was
  re-declared:**

  | Case | Answer | Max probability Δ | Max confidence Δ | Max score Δ |
  | --- | --- | --- | --- | --- |
  | text choice (`which_passed`) | **exact** (`r1_and_r3`) | 8.713363e-3 | 1.742673e-2 | — |
  | image choice (`color`) | the same declared near-tie flip (native `white`, reference `black`) | 7.628613e-2 | 6.348610e-2 | — |
  | noul (`duplicate` + `authorized`) | floats gated | 2.852246e-3 | 5.704493e-3 | — |
  | score (`severity`) | floats gated, legend exact | 2.453974e-2 | 3.680961e-2 | 1.511225e-2 |
  | mixed | **exact** choice (`void`) | 2.453974e-2 | 3.680961e-2 | 1.511225e-2 |
  | head probe | — | 4.529953e-6 max logit Δ | — | — |

  The component gates re-verified with the same identity: the vision
  tower max abs 1.03125 (3361/131072 bit-exact, mean abs 1.004489e-2)
  under the declared 1.25; the end-to-end text hidden max abs 0.25
  (1765/2048 elements differ) under 0.5; every projection family at its
  recorded 0..1 BF16-ulp figure (in_proj_qkv 0.03125, q_proj 0.015625,
  out_proj 0.007812, o_proj and down 0.000488, fused gate/up bit-exact
  over 98304 elements) under 0.0625. The image decomposition diagnostic
  repeats unchanged: pixel preprocessing bit-exact (0/393216 BF16
  elements), tower drift 1.03125 over the 64 rows, language+head on the
  *reference's* tower rows within 9.2e-3 of the reference logits, and
  the same declared flip on the native rows — the flip mechanism and
  its attribution carry over to ABI 9 as recorded. The shared backend's
  eager numeric path (default `cs1_gemm_create`, no CUDA Graph, no
  fixed-algorithm handle on this worker) is unchanged by the sync; the
  ABI 9 additions sit on code paths this worker does not take.

### Known limitations (this milestone)

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
- Slow-body timeout semantics: the reference arms a per-operation 15 s socket
  timeout (`connection.settimeout(15)`), so a body that keeps dribbling in with
  gaps under 15 s is read indefinitely; the native worker budgets one 15 s
  window for the whole body read, so the same drip-fed body eventually gets
  `408 request body timed out`. A body that stalls for 15 s mid-read gets the
  same 408 on both. Pinned in `tests/valen/worker.rs` (the total-budget
  reading); recorded here as a deliberate semantic difference.
- Non-finite JSON constants in a body that is malformed *elsewhere*: CPython's
  scanner errors the instant it reaches the first constant — `{"a": NaN}xyz`,
  `[NaN` and `{"a": NaN "b": NaN}` all get the reference's specific
  `non-finite JSON number … is not supported` detail — while the native wire
  layer answers with that detail only when the whole body with the constants
  substituted is well-formed, and with the generic malformed-JSON detail
  otherwise. The reverse ordering also differs: a duplicate-key object that
  closes *before* a later constant makes the reference report the duplicate
  (its `object_pairs_hook` fires at object close), where the native pre-check
  must tolerate duplicates and reaches the constant instead. Both are detail-
  text-only differences on doubly-invalid bodies (the status is 400 either
  way); exact parity would mean porting the CPython scanner's error ordering,
  so the behavior is pinned as the recorded deviation in
  `src/models/valen/native/src/wire.rs` tests and the contract table above.
- Integer literals beyond `u64` in an object or array `state` (or other
  rendered text fields): `serde_json` parses one that fits neither `i64` nor
  `u64` straight to a double — `18446744073709551616` re-serializes as
  `1.8446744073709552e+19` — where Python's `json.loads` keeps an
  arbitrary-precision `int` and `json.dumps` prints every digit, so the
  compiled text (and therefore the tokens) silently differ on such states.
  Inherited from the shared #143 JSON layer (`src/models/qwen3_5/native/src/json.rs`),
  which this worker reuses verbatim; not fixed here per the shared-code scope.
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

| Area | Contract | Native status (N5) |
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
| Slow body (> 15 s) | `408 {"detail": "request body timed out"}` | identical for a stalled body; a drip-fed body under the reference's per-recv timeout is read indefinitely there but hits the native total 15 s budget (see limitations) |
| Body > 8 MiB at parse | `413 {"detail": "request body exceeds 8 MiB"}` | identical |
| Malformed JSON / non-object / duplicates / non-finite | `400 {"detail": …}` (N3 wire layer corpus, 70 cases) | identical |
| Non-finite constant in a body that is also malformed elsewhere | the scanner errors at the first constant it reaches — the specific non-finite detail even when what follows is garbage (a duplicate-key object closing earlier wins instead) | generic malformed-JSON detail unless the whole body with constants substituted is well-formed; the earlier-closed-duplicate ordering is reversed (recorded deviation; the status is 400 either way, only the detail text differs, and only on doubly-invalid bodies) |
| Wire validation (model/state/questions) | `422 {"detail": …}` (N3 corpus) | identical |
| Token budget (max_length 8192) | `422 "request rejected by the pinned compiler: …"` | identical |
| `noul` / `score` questions | served by the reference | identical (N5; finishing byte-identical for equal logits) |
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
TF32 off) exactly as the worker does, serves the five parity fixtures — the
N2 text choice state, the example image choice request, the example noul
request, the example score request and the mixed-type body — through
`frontend.valen.decide` with a recording executor, and also records the
reference head's readout rows and logits (the head probe). Re-run it on the
GPU host when the reference moves; tolerances are measured first and then
declared — a `null` tolerance fails the Rust gate closed. Re-records are
deterministic: two consecutive recordings on the same host produced
identical logits, responses and probe rows (verified when the score
tolerance was declared).
