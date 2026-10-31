# Vela 2.0 reference workers (0.3B, 0.8B hybrid)

This recipe serves the Vela 2.0 members through the Python reference
worker and the existing Rust frontend: the 0.3B encoder and the 0.8B
hybrid, one pinned member per worker process. The model-side
implementation is under
[`src/models/vela/`](../../src/models/vela/) (0.3B in `engine.py`, the
0.8B hybrid in `hybrid.py`), and the worker is
[`src/frontend/vela.py`](../../src/frontend/vela.py), which picks the
executor from the package's declared `model_type`.

Run the commands from the repository root with Python 3.12. The default
and evidence-backed setting is CPU FP32 — the reference oracle; a CUDA
device works for serving but its bf16-autocast backbone is a deployment
option, not the parity baseline. The recipe records the procedure;
measured results and verbatim run output live in the pull request
description, not here.

## Scope

Scope:

- packages: [`vllm-sr/Vela-2.0-0.3B`](https://huggingface.co/vllm-sr/Vela-2.0-0.3B)
  and [`vllm-sr/Vela-2.0-0.8B`](https://huggingface.co/vllm-sr/Vela-2.0-0.8B)
  at the pinned revisions below;
- the Python reference worker behind the Rust frontend, executing each
  checkpoint's own pinned runtime (`vela2_inference.py` and siblings)
  after a full-hash package gate;
- text/JSON state and all five typed kinds — `choice`, `noul`, `score`,
  `span`, `set` — in one request, with the span/set response extensions
  (`spans`/`sets`/`thresholds`, `span_heads` on the hybrid, and their
  Noul views) served forward-untouched;
- FP32 parameters, one bidirectional forward per assembled sequence
  (0.3B) or one parts pass with per-question continuation (hybrid),
  calibrated readouts; zero generated tokens.

Out of scope: the larger hybrid members (4B/9B), the ONNX backend of
the 0.3B, native execution for the reference workers (the ModernBERT
profile's own `cuda`-gated route under
[`native/`](../../src/models/vela/native/README.md) is a separate
slice), and `noul_calibration` (off by default; documented as a
reference-server feature only).

## Architecture and ownership

~~~text
Rust frontend
  -> Python worker (executor chosen by config.json model_type)
     -> admit: JSON transport + upstream envelope preamble (protocol)
     -> verify: config identity pin, full SHA-256 of the pinned table,
        exact weights size, then import it (engine / hybrid)
     -> execute: the checkpoint's own vela2_inference system_one_request
     -> finish: response-order/key-set checks + formula mirrors (postprocess)
~~~

The ownership boundary follows
[`docs/architecture.md`](../../docs/architecture.md) and the
[model contract](../../src/models/vela/README.md):

- `protocol.py` owns the envelope preamble with upstream's exact 422
  triples; deeper validation (state parts, question semantics, the 413)
  is the pinned runtime's, verbatim;
- `preprocess.py` maps the runtime's `SystemOneError` to 422/413;
- `engine.py` owns the 0.3B pinned runtime and its package gate;
  `hybrid.py` mirrors it for the 0.8B (a 10-file pin table including
  the shard index and the side-loaded `broad_head.safetensors`);
- `postprocess.py` owns the pure formulas and response finishing,
  member-aware for the hybrid's `span_heads` field;
- `frontend.vela` owns HTTP lifecycle, member detection, the
  single-flight 503 lock, the `VELA2_API_KEY` 401 gate and
  warmup-before-readiness;
- The Rust frontend forwards `/v1/systemone` and `/health` unchanged.

## Pinned artifacts

| Artifact | 0.3B | 0.8B hybrid |
| --- | --- | --- |
| Package | [vllm-sr/Vela-2.0-0.3B](https://huggingface.co/vllm-sr/Vela-2.0-0.3B) | [vllm-sr/Vela-2.0-0.8B](https://huggingface.co/vllm-sr/Vela-2.0-0.8B) |
| Pinned revision | `d6f03aa9baca4f9017fa8f8e9c2764d3bed3f409` | `326b01d81f61fff7fd4ae570e54b87a49119e490` |
| Package gate | config structural identity + a 7-file SHA-256 table in `engine.py`; no unlisted importables, `__pycache__` or directory symlinks in the package root | config structural identity (incl. `model_name`) + a 10-file SHA-256 table in `hybrid.py`, cross-checked against the package's own SHA256SUMS |
| Weights check | exact 1,236,473,252 bytes | exact 2,022,189,488 bytes + hashed `broad_head.safetensors` |
| Served name | `vllm-sr/Vela-2.0-0.3B` | `vllm-sr/Vela-2.0-0.8B` (from `config.json`) |
| Input window | 8,192 tokens | 16,384 tokens |
| License | Apache-2.0; tokenizer material under Gemma Terms of Use | Apache-2.0; Qwen3.5-0.8B lineage notices in the package |

Each gate's file table is checked in (never read from the package), so
a later snapshot — self-consistent or not — cannot pass as the pinned
revision. Re-check before changing a pin:

~~~sh
curl -sS "https://huggingface.co/api/models/vllm-sr/Vela-2.0-0.3B/revision/d6f03aa9baca4f9017fa8f8e9c2764d3bed3f409" | jq -r '.sha'
sha256sum weights/vela-2.0-0.3b/{vela2_inference.py,modeling_vela2.py,configuration_vela2.py,vela2_serve.py,config.json,calibration.json,tokenizer.json}
curl -sS "https://huggingface.co/api/models/vllm-sr/Vela-2.0-0.8B/revision/326b01d81f61fff7fd4ae570e54b87a49119e490" | jq -r '.sha'
sha256sum weights/vela-2.0-0.8b/{vela2_inference.py,modeling_vela2.py,configuration_vela2.py,vela2_serve.py,config.json,calibration.json,tokenizer.json,tokenizer_config.json,model.safetensors.index.json,broad_head.safetensors}
~~~

## Install and prepare the artifacts

~~~sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r recipe/vela/requirements.txt pytest
~~~

~~~sh
export VELA_PKG="$PWD/weights/vela-2.0-0.3b"
export VELA_REV=d6f03aa9baca4f9017fa8f8e9c2764d3bed3f409
export VELA_HYBRID_PKG="$PWD/weights/vela-2.0-0.8b"
export VELA_HYBRID_REV=326b01d81f61fff7fd4ae570e54b87a49119e490
mkdir -p weights
.venv/bin/hf download vllm-sr/Vela-2.0-0.3B --revision "$VELA_REV" --local-dir "$VELA_PKG"
.venv/bin/hf download vllm-sr/Vela-2.0-0.8B --revision "$VELA_HYBRID_REV" --local-dir "$VELA_HYBRID_PKG"
~~~

The 0.3B package is about 4.4 GB (the 1.24 GB FP32 safetensors, a 34 MB
tokenizer, the ONNX exports and doc assets); serving needs roughly
2 GB of RAM for the FP32 model plus working memory for the batch rows.
The 0.8B package is about 2.1 GB (a 2.02 GB shard plus the 4.5 MB
broad-head file); FP32 parameters take about 3 GB of RAM on CPU (about
3 GB of GPU memory on CUDA, where the backbone runs under bf16
autocast). A worker refuses to start when the config is not its pinned
member's, when any gated file fails its hash, when the package root
holds an importable entry the table does not name, an existing
`__pycache__` or a directory symlink, or when the weights file is not
the pinned size — that
gate runs before any package code is imported.

## Start the worker

~~~sh
PYTHONPATH=src .venv/bin/python -m frontend.vela \
  --model "$VELA_PKG" --device cpu --port 8000          # 0.3B
PYTHONPATH=src .venv/bin/python -m frontend.vela \
  --model "$VELA_HYBRID_PKG" --device cpu --port 8001    # 0.8B hybrid
~~~

The worker detects the member from the package's `model_type`, verifies
the package, loads the model, runs a real warmup request covering every
question kind (the hybrid's warmup also drives both span heads), and
only then binds the port:

~~~sh
curl -sS -i http://127.0.0.1:8000/health    # {"status": "ok"}  (upstream verbatim)
curl -sS -i http://127.0.0.1:8000/ready     # {"ready": true}
curl -sS -i http://127.0.0.1:8000/v1/models
~~~

Optional: set `VELA2_API_KEY` to require `Authorization: Bearer
<$VELA2_API_KEY>` on `/v1/systemone` and `/v1/models` (401 with the
upstream message otherwise), and `VELA2_MODEL_NAME` (or `--name`) to
serve under a different name. One request is served at a time; a
concurrent one gets 503.

## Example requests

Four complete runnable requests ship with the recipe:

- [`example-request.json`](example-request.json) — the model card's PII
  example: a span question over the 17 trained PII labels (the PII
  length-rule threshold is applied and echoed), a hallucination span
  question over the answer part, and a domain `choice` question;
- [`example-request-set-threshold.json`](example-request-set-threshold.json)
  — a `set` question with a per-question `threshold` override (the
  applied value 0.15 is echoed in `thresholds`; the pinned predict
  level marks overrides with rule `given` on span answers);
- [`example-request-cjk.json`](example-request-cjk.json) — CJK state
  exercising Unicode code-point offsets (one unit per CJK character);
- [`example-request-score.json`](example-request-score.json) — two
  `score` questions (three and four ordered levels) covering the fifth
  answer kind in the default parity run.

~~~sh
curl -sS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @recipe/vela/example-request.json
~~~

Expected response shape (a serving smoke test, not an accuracy result;
the exact numbers come from the checkpoint and are frozen by the parity
tool below):

~~~json
{
  "model": "vllm-sr/Vela-2.0-0.3B",
  "answers": {
    "pii": {"type": "noul", "noul": 0.0},
    "halu": {"type": "noul", "noul": 0.0},
    "domain": {"type": "choice", "choice": "…", "confidence": 0.0, "probabilities": {"…": 0.0}}
  },
  "usage": {"input_tokens": 0, "output_tokens": 0},
  "spans": {"pii": [{"label": "…", "start": 0, "end": 0, "text": "…", "probability": 0.0}]},
  "thresholds": {"pii": 0.0, "halu": 0.55}
}
~~~

## Start and verify the Rust frontend

~~~sh
cargo build --release --locked
OMNI_JEV_BIND=127.0.0.1:8080 \
OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 \
  target/release/omni-jev
~~~

Send the same request through port 8080 and compare byte-for-byte:

~~~sh
curl -fsS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @recipe/vela/example-request.json > /tmp/vela-worker.json
curl -fsS http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @recipe/vela/example-request.json > /tmp/vela-frontend.json
cmp /tmp/vela-worker.json /tmp/vela-frontend.json
~~~

Worker responses are forwarded unchanged. Frontend-generated 502/504
responses are frontend errors, not model responses.

## Error behavior

The request, response and error contract — the two error layers, the
five answer shapes, usage counting and the state-truncation order — is
the [model contract](../../src/models/vela/README.md). In short:
transport errors follow the shared worker discipline; every invalid
field rejects the whole request with the upstream 422
`{"detail": [{"loc", "msg", "type"}]}`; 413 `too_long` fires only when
questions plus schema alone exceed the member's input window (8,192 on
0.3B, 16,384 on the hybrid); and over-limit state text is **truncated
as the reference truncates it** (shrink part first, floor 64 then 1,
then protected parts; windowed re-coverage of cut labelled parts) — a
deliberate, loudly documented divergence from the repository's
no-silent-truncation convention, with the guardrail risk recorded in
the model contract.

## Repository-side validation

~~~sh
PYTHONPATH=src .venv/bin/python -m pytest tests/vela -q
PYTHONPATH=src .venv/bin/python -m compileall -q src/models/vela src/frontend/vela.py tests/vela
.venv/bin/python recipe/vela/test_compare_reference.py
cargo fmt --all --check
cargo clippy --workspace --locked --all-targets --features omni-laya/serve -- -D warnings
cargo test --workspace --locked --features omni-laya/serve
cargo build --workspace --release --locked --features omni-laya/serve
~~~

Run these from the repository root. The recorded outputs of a run belong
in the pull request description, not in this recipe.

## Reference parity (author-side, opt-in)

[`compare_reference.py`](compare_reference.py) drives the fixture set —
the four example requests and five synthetic rejections (an
invalid-span 422, an over-limit 413, and the missing-field /
extra-field / wrong-model envelope 422s) — through two paths that
share one loaded engine: the pinned
`vela2_serve.py` app via FastAPI's test client, and the production
`frontend.vela.decide` pipeline. The criterion is reference parity:
**exact match** including JSON key order for 200s,
and agreement on status and body for rejections (needs `pip install
fastapi uvicorn httpx` on top of the requirements file). The run also
compares the worker's verbatim endpoint bodies — `/health`, `/ready`,
`/v1/models` and the unauthorized 401 — against the live reference app,
so a drift in either direction fails the run.
`--export tests/vela/fixtures/oracle-<case>.json` freezes each passing
200 case — request, SystemOne response, predict()-level answers and the
generating run's provenance — as
the frozen comparison baseline for the future native encoder
slices, and writes it only after a passing comparison.

~~~sh
PYTHONPATH=src .venv/bin/python recipe/vela/compare_reference.py \
  --model "$VELA_PKG" --device cpu \
  --export tests/vela/fixtures/
PYTHONPATH=src .venv/bin/python recipe/vela/compare_reference.py \
  --model "$VELA_HYBRID_PKG" --device cpu \
  --export tests/vela/fixtures/
~~~

Run it by hand and record the verbatim output, host, package revision
and dependency versions in the pull request description. CPU FP32 is
the oracle setting; keep flash-linear-attention out of the fixture path
(it changes the hybrid's numbers and its parity against the fallback
path is not documented upstream) and do not substitute GPU
bf16-autocast numbers for the baseline. The tool rewrites the example
files' `model` field to the served member and exports the hybrid's
oracle fixtures under an `oracle-hybrid-` prefix.

## Troubleshooting and limits

- The package directory must be the pinned revision's download; the
  worker refuses foreign configs, hash mismatches, unlisted importables,
  `__pycache__` bytecode caches, directory symlinks and wrong weights
  sizes at startup, before importing anything.
- The 8 MiB body cap is slice-scoped (upstream sets no total-body
  limit); a valid request bounded by either member's input window is
  far below it.
- `/health` and `/ready` are upstream's minimal bodies on both
  members, not the richer shape of some other workers in this
  repository.
- Long states are truncated (see Error behavior); if a consumer must
  never miss tail text, check `usage.input_tokens` against the expected
  budget or enforce limits upstream of the worker.
- GPU bf16-autocast serving (`--device cuda`) is untested in this
  evidence; treat it as a deployment option to validate separately.
- GPU evidence for the supported-models table is recorded in the
  integration pull request, not in this recipe.
