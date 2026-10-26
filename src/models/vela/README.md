# Vela 2.0 0.3B reference contract

This directory contains the System1-Omni integration slice for
[Vela 2.0 0.3B](https://huggingface.co/vllm-sr/Vela-2.0-0.3B)
(`vela2-unified`). It is a reference-serving boundary only: a Python
worker that runs the checkpoint's own pinned runtime for text state and
all five typed question kinds (`choice`, `noul`, `score`, `span`, `set`)
on `/v1/systemone`. Native execution is not part of this slice; this
worker freezes the text/token_ids input, the position-level final-hidden
readout contract, the error/length boundaries and the parity baseline
for later ModernBERT-profile work. The CPU half of that native route —
the 0.3B ModernBERT profile registration, the checkpoint census and the
YaRN tables — lives in [`native/`](native/README.md), whose
`cuda`-gated executor runs the encoder on the regenerated bundle.

The shared Vela 2.0 wire module is deliberately shaped for the family:
the envelope admission (`protocol`), the pinned-runtime error mapping
(`preprocess`) and response finishing (`postprocess`) are member-neutral;
this slice ships the 0.3B pinned executor (`engine`). The hybrid members
(0.8B/4B/9B) reuse the same wire layer with their own executors.

## Pinned artifacts

| Artifact | Value |
| --- | --- |
| Package | [vllm-sr/Vela-2.0-0.3B](https://huggingface.co/vllm-sr/Vela-2.0-0.3B) |
| Pinned revision | `d6f03aa9baca4f9017fa8f8e9c2764d3bed3f409` |
| Identity | `vela2-unified` / `Vela2Model`, 309,114,369 parameters, all FP32 (1,236,473,252-byte safetensors) |
| Runtime | the checkpoint's own flat files: `vela2_inference.py`, `modeling_vela2.py`, `configuration_vela2.py`, `vela2_serve.py` |
| Calibration | the package's own `calibration.json` (never cross-copied between members) |
| License | Apache-2.0; the inherited tokenizer material carries Gemma Terms of Use |
| Runtime dependencies | [recipe/vela/requirements.txt](../../../recipe/vela/requirements.txt) (torch, transformers, safetensors, tokenizers, numpy) |

The package ships no manifest, so the executor's startup gate checks the
config's structural identity (model type, architecture, `max_length`
8,192, the 8 marker ids), then hashes every small file the runtime reads
or executes — the four runtime modules, `config.json`,
`calibration.json` and `tokenizer.json` — against the table checked in
for the pinned revision, before anything is imported, and refuses
importable entries the table does not name plus any existing
`__pycache__` bytecode cache or directory symlink (`model.safetensors` is
checked by exact size; the runtime's strict safetensors load rejects
truncated or structurally mismatched weights, but not same-size
bit-level corruption).
A later self-consistent snapshot rewrites its own files and fails the
checked-in pin.

## Current serving scope

- The request envelope is upstream's three-field form, exactly:
  `{"model", "state", "questions"}`. Any other key is a whole-request
  422 `extra_forbidden`; a missing key is a 422 `missing`; `model` must
  be non-blank and equal the served name (`vllm-sr/Vela-2.0-0.3B`, or
  `VELA2_MODEL_NAME`/`--name`), otherwise 422 `value_error.model`.
- `state` is a plain string or a JSON object/array. An object is mapped
  onto typed parts (`user`/`context`/`answer`) by the pinned runtime's
  key roles (request/user/prompt → user; source/context/document/tools/
  previous_answer → context; answer/response → answer; unknown keys →
  context); non-string values render as canonical JSON. A string holding
  a JSON object is read as that object.
- `questions` is a non-empty object of question ids to questions. All
  five kinds are served forward-untouched; per-question limits (choice
  2–255 options, score 2–10 distinct levels, span/set 1–255 criteria)
  and `over` resolution are the pinned runtime's validation, verbatim.
- `over` names a state key, a part name, or a list of them; default is
  every part (span: the answer part if present, else user, else the
  first part). `threshold` is allowed only on span/set and must be a
  number in [0, 1]. Set-label Noul views (`"<qid>.<label>"`) must not
  collide with real question ids.
- No token generation: one bidirectional forward over the assembled
  schema+state sequence scores every question's readout.

## Answers and the response envelope

The response envelope is the pinned runtime's, passed through verbatim
(full float precision, no rounding anywhere):

- `answers` holds, per question in request order: `noul` →
  `{"type", "noul"}` with `P(yes)`; `choice` → `{"type", "choice",
  "choice", "confidence", "probabilities"}`; `score` → `{"type",
  "score", "score", "confidence", "legend", "probabilities"}` with
  `score` the expected zero-based level and keys `"0".."n-1"`. A `set`
  question contributes no entry of its own but one
  `answers["<qid>.<label>"]` Noul view per label; a `span` question's
  entry is the Noul view `{"type": "noul", "noul": max word
  probability}` taken before span decoding.
- `choice` confidence is the top-two probability margin clamped to
  [0, 1] (1.0 for a single option); `score` confidence is one minus the
  level variance normalized by the uniform variance — a statistic of
  the distribution, not a calibrated probability of being right. Both
  formulas are mirrored as pure functions in `postprocess.py` for the
  CPU contract tests.
- Top-level `spans: {qid: [{label, start, end, text, probability}]}`
  with Unicode code-point offsets into the targeted field's text
  (clipped and re-trimmed to the field when it shares a part),
  `sets: {qid: {"selected", "probabilities"}}` with strict `>`
  threshold selection, and `thresholds: {qid: t}` echoing the applied
  value. Tie-breaking inside a question is insertion order (argmax over
  raw logits).
- Thresholds come from the package's own `calibration.json` by question
  id (`set:*` 0.3, PII length rule + sparse gate, `span:halu` 0.55,
  `span:toxic` 0.1, …) unless the question's own `threshold` overrides
  them, in which case the applied value is echoed (`rule "given"` is
  the pinned predict-level marker on overridden span answers; set
  answers echo the value without a rule field). Serving-time
  temperatures divide the logits per question type; the learned
  `tau`/`tau_span` live inside the model.
  The hybrid-only fields (`span_heads`, `noul_calibration`) do not
  exist on this member.
- `usage.input_tokens` is the sum over every rendered sequence — span
  window rows and extra span-question rows included, nothing
  deduplicated; `usage.output_tokens` is always 0.

## Errors and length boundaries

Whole-request only, matching the pinned server exactly; there are no
per-question errors on this path:

1. Transport follows the shared Python-worker discipline (`400`, `408`,
   `411`, slice-scoped `413` over the 8 MiB body cap, `415`, `503`
   while another inference holds the single-flight lock, `404`, `500`).
2. The pinned runtime rejects the whole request with HTTP 422
   `{"detail": [{"loc", "msg", "type"}]}` on any invalid field — the
   worker's envelope admission reproduces the preamble triples
   (`state|model|questions`) itself and defers everything deeper to the
   runtime so error precedence stays upstream's (state errors before
   question-container errors, question walk in request order). A
   non-`SystemOneError` `ValueError` from the runtime also maps to a
   422 client error here — worker policy, not runtime semantics (the
   upstream server would answer 500); no such path is reachable on the
   pinned runtime's request path.
3. HTTP 413 (`too_long`) only when the questions plus schema alone
   exceed the 8,192-token window.
4. HTTP 401 (`{"detail": "Missing or invalid API key"}`) only when
   `VELA2_API_KEY` is set and the request does not carry exactly
   `Authorization: Bearer <key>` (applies to `/v1/systemone` and
   `/v1/models`; `/health` and `/ready` stay open). The transport
   checks above (411/413/415) run before this gate, so a request that
   is both malformed and unauthenticated answers with its transport
   status, not 401 — upstream FastAPI checks auth first; a recorded
   slice-scoped divergence.

**Over-limit state is truncated, not refused** — a recorded divergence
from the "no silent truncation" convention this repository's other
text workers use. The pinned truncation order (quoted from the runtime's
`assemble`): if the assembled sequence exceeds `max_length`, the part no
question is `over` is cut first (never below 64 tokens, then never below
1), then the protected (targeted) parts are cut the same way, and only
when the questions plus schema alone still do not fit does the request
413. Windowing then re-covers a cut labelled part (windows of the part's
kept length, overlap up to 512 tokens, per-type aggregation: choice/score
mean, set max, span word-logit mean; uncovered words score −30).
**Known risk, kept deliberately**: a guardrail consumer can silently
miss text beyond the truncation point — an injection payload hidden
past the cut still evaluates against the truncated state. Tightening to
reject is a future explicit deviation, not this slice.

`GET /health` → `{"status": "ok"}`, `GET /ready` → `{"ready": true}`,
`GET /v1/models` → the pinned entry — upstream bodies, verbatim. The
worker binds its port only after load and warmup (a real inference of
every question kind), so answering at all means ready.

## Ownership boundary

- `protocol.py` owns the JSON transport discipline and the upstream
  preamble admission (missing/extra fields, model name) with upstream's
  exact detail triples.
- `preprocess.py` owns the mapping of the pinned runtime's
  `SystemOneError` to HTTP (422, or 413 for `too_long`) and of its
  other ValueErrors to client errors.
- `engine.py` owns the pinned runtime, the pre-import package gate, the
  served name, device state and the all-kind warmup.
- `postprocess.py` owns the pure confidence/score formulas (mirrored
  for the CPU tests) and response finishing: answer-order and key-set
  reconstruction including set views, usage accounting, and refusal of
  hybrid-only fields on this member.
- `src/frontend/vela.py` owns HTTP lifecycle, the single-flight lock,
  the API-key gate and orchestration.
- The Rust frontend owns transport and forwarding, not model state.

The server-side error surface and envelope semantics are recorded as
this repository's local stance for Vela 2.0; nothing is proposed into
the shared worker-contract issue, and nothing is posted upstream.

## Validation

- CPU contract tests, no weights, no torch:
  `PYTHONPATH=src python -m pytest tests/vela -q` (envelope triples,
  transport discipline, package gate, formula mirrors, response
  finishing, the HTTP surface with a contract fake).
- Reference parity is exact match on recorded fixtures under the pinned
  versions: `recipe/vela/compare_reference.py` drives the recipe
  fixtures plus five synthetic rejections (invalid-span 422,
  over-limit 413, and the missing-field / extra-field / wrong-model
  envelope 422s) through both the pinned `vela2_serve.py` app and the
  production worker pipeline over one shared engine, and freezes the
  oracle fixture for the native encoder slices. CPU FP32 (no
  flash-linear-attention) is the oracle setting; GPU bf16-autocast is
  a deployment option outside this slice's evidence.
- The serving procedure, hardware requirements and the recorded
  evidence pointer are [recipe/vela/reference.md](../../../recipe/vela/reference.md).
