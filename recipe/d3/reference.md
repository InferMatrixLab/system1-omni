# D3 reference worker

This recipe serves D3 (Decision 3.0) through the Python reference worker and
the existing Rust frontend. The model-side implementation is under
[`src/models/d3/`](../../src/models/d3/), and the worker is
[`src/frontend/d3.py`](../../src/frontend/d3.py).

Run the commands from the repository root on Linux with an NVIDIA GPU and
Python 3.12. **No GPU serving run has been recorded for this recipe yet**:
everything under "Repository-side validation" below was executed on a
CPU-only host; the worker launch, health check, example request and parity
blocks describe the procedure and are waiting for an author-side GPU host.

## Scope

The slice is:

- package: [`vllm-sr/d3`](https://huggingface.co/vllm-sr/d3) at the pinned
  revision below;
- the Python reference worker behind the Rust frontend, executing the
  checkpoint's own pinned runtime (`d3_runtime.py` and siblings) after a
  full-hash manifest gate;
- text/JSON state and the three typed kinds — `choice`, `noul`, `score` —
  in one request;
- BF16 backbone with FP32 readout, noncausal full attention, last-position
  pooling, one forward pass per question.

Out of scope for this slice: `images`/`videos` (rejected with 422; upstream
supports them per-item at 8,000,000 bytes per image and 32,000,000 per
video), upstream's FastAPI server features (API key, request ids, server
timing), and any native Rust/CUDA executor.

## Architecture and ownership

~~~text
Rust frontend
  -> Python worker
     -> admit: envelope (protocol), pinned-runtime question errors (preprocess)
     -> verify: full SHA-256 of the manifest runtime table, then import it (engine)
     -> execute: the checkpoint's own d3_runtime prepare/run/respond
     -> finish: response validation + embedded-failure warning (postprocess)
~~~

The ownership boundary follows
[`docs/architecture.md`](../../docs/architecture.md) and the
[model contract](../../src/models/d3/README.md):

- `protocol.py` owns the five-field envelope and the slice-scoped media
  refusal;
- `preprocess.py` owns the pre-execution 422 mapping
  (`invalid_questions` map, plain `max_length_exceeded` message);
- `engine.py` owns the pinned runtime and the manifest gate stronger than
  the upstream default;
- `postprocess.py` owns the pure answer formulas and the warning on
  embedded `invalid_model_output` answers (partial success stays HTTP 200
  but never passes silently);
- `frontend.d3` owns HTTP lifecycle and orchestration;
- The Rust frontend forwards `/v1/systemone` and `/health` unchanged.

## Pinned artifacts

| Artifact | Value |
| --- | --- |
| Package | [vllm-sr/d3](https://huggingface.co/vllm-sr/d3) |
| Pinned revision | `cbff3a3cb0b074b6585fe9f747b9f9f5557a0e75` |
| Manifest | `d3-package-manifest/1`, built 2026-10-11T03:30:34Z |
| Runtime gate | `runtime.files_sha256`, 9 files, full SHA-256 before import |
| License | Apache-2.0 (package metadata) |

The pinned revision predates later upstream commits (the repository's
`lastModified` is later than the manifest build time), which is exactly why
the revision pin plus the two manifest hash tables exist. Re-check before
changing the pin:

~~~sh
curl -sS "https://huggingface.co/api/models/vllm-sr/d3/revision/cbff3a3cb0b074b6585fe9f747b9f9f5557a0e75" | jq -r '.sha'
.venv/bin/python - <<'PY'
import json, hashlib, pathlib
root = pathlib.Path("weights/d3")
manifest = json.loads((root / "MODEL_MANIFEST.json").read_text())
for name, digest in manifest["runtime"]["files_sha256"].items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    assert actual == digest, f"{name}: {actual} != {digest}"
print("runtime table: 9 files verified")
PY
~~~

## Install and prepare the artifacts

The checkpoint carries its own dependency pins in `requirements.txt`
(torch>=2.8, transformers==5.17.0, flash-linear-attention==0.5.2,
torchvision). PyTorch wheels must match the host CUDA runtime.

~~~sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
~~~

~~~sh
export D3_PKG="$PWD/weights/d3"
export D3_REV=cbff3a3cb0b074b6585fe9f747b9f9f5557a0e75
mkdir -p weights
.venv/bin/hf download vllm-sr/d3 --revision "$D3_REV" --local-dir "$D3_PKG"
.venv/bin/python -m pip install -r "$D3_PKG/requirements.txt" pytest
~~~

The package is about 52 GB (26.09B BF16 parameters plus the FP32
(255, 5120) readout). The worker refuses to start when
`MODEL_MANIFEST.json` is missing or when any of the 9 runtime files fails
its full-hash check — that gate runs before any package code is imported.

## Start the worker

~~~sh
PYTHONPATH=src .venv/bin/python -m frontend.d3 \
  --model "$D3_PKG" --device cuda:0 --port 8000
~~~

The worker verifies the runtime table, loads the package through the
pinned runtime's own `verify="fast"` path (pass `--verify full` to hash
every file, a minutes-level operation over 52 GB), compiles the kernels via
warmup, and only then binds the port:

~~~sh
curl -sS -i http://127.0.0.1:8000/health
~~~

Expected shape (the pinned package reports no input limit):

~~~json
{"status":"ready","model":"d3","max_input_tokens":null,"modalities":["text","image","video"]}
~~~

## Example request

The complete runnable request is in
[`example-request.json`](example-request.json): one text state with one
`choice`, one `noul` and one `score` question.

~~~sh
curl -sS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @recipe/d3/example-request.json
~~~

Expected response shape:

~~~json
{
  "model": "d3",
  "answers": {
    "refund": {"type": "choice", "choice": "...", "probabilities": {"...": 0.0}, "confidence": 0.0},
    "urgent": {"type": "noul", "noul": 0.0},
    "severity": {"type": "score", "score": 0.0, "probabilities": {"0": 0.0}, "confidence": 0.0,
                  "legend": {"0": "minor", "1": "moderate", "2": "major", "3": "critical"}}
  },
  "usage": {"input_tokens": 0, "output_tokens": 0}
}
~~~

This is a serving smoke test, not an accuracy result; the exact numbers come
from the checkpoint and are recorded by the parity tool below.

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
  --data-binary @recipe/d3/example-request.json > /tmp/d3-worker.json
curl -fsS http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @recipe/d3/example-request.json > /tmp/d3-frontend.json
cmp /tmp/d3-worker.json /tmp/d3-frontend.json
~~~

Worker responses are forwarded unchanged. Frontend-generated 502/504
responses are frontend errors, not model responses.

## Error behavior

The request, response and error contract — the three error layers, the
partial-success warning, answer shapes and usage counting — is the
[model contract](../../src/models/d3/README.md). In short: transport errors
follow the shared worker discipline; invalid or over-limit questions reject
the whole request with 422 (`{"invalid_questions": {...}}` or a plain
`detail` naming the token count); a question with undecodable model output
keeps its 200 with the failure embedded and a worker warning logged.

## Repository-side validation

Executed on the integration branch (CPU-only host, Python 3.12):

~~~sh
PYTHONPATH=src .venv/bin/python -m pytest tests/d3 -q
PYTHONPATH=src .venv/bin/python -m compileall -q src/models/d3 src/frontend/d3.py tests/d3
.venv/bin/python recipe/d3/test_compare_reference.py
cargo fmt --all --check
cargo clippy --workspace --locked --all-targets -- -D warnings
cargo test --workspace --locked
cargo build --workspace --release --locked
~~~

Recorded results:

~~~text
pytest tests/d3: 47 passed
compileall: passed
compare_reference tests: ok
cargo gates: see the pull request checks
~~~

## Reference parity (author-side, opt-in)

[`compare_reference.py`](compare_reference.py) drives one fixture through
two paths that share the loaded pinned model: the upstream
`d3_server.py` decide body (prepare, 422 admission, run, respond) and the
production `frontend.d3.decide` pipeline. The criteria are exact decision
agreement including embedded `invalid_model_output` shapes, equal token
accounting, and a numeric tolerance of 1e-6. `--export` freezes the oracle
fixture (token ids, probabilities, answers) for the future native worker.

~~~sh
PYTHONPATH=src .venv/bin/python recipe/d3/compare_reference.py \
  --model "$D3_PKG" --device cuda:0 \
  --export tests/d3/fixtures/oracle.json
~~~

**No GPU parity run is recorded yet.** Record the verbatim output here when
it runs, with the host and package revision. The flash-linear-attention
kernels and the PyTorch reference GDN implementation are not assumed to
agree numerically; measure the delta on this fixture before freezing any
tolerance for a native worker.

## Troubleshooting and limits

- The package directory must be the pinned revision's download; the worker
  refuses missing manifests and any runtime-file hash mismatch at startup.
- Upstream moved after the manifest was built — do not track `main` of the
  repository; re-pin deliberately and re-record the revision.
- `images`/`videos` non-empty lists are rejected with 422 in this slice;
  the 8 MiB body cap is slice-scoped and will re-align to the upstream
  per-item budgets when media opens (stage 2).
- `/health` has no `authentication` field: this worker has no API key
  (a deliberate difference from upstream, which reads `DECISION_API_KEY`).
- The noncausal full-attention hook inside the pinned runtime is adapted
  from perplexity-ai/pplx-decider-v1.1-27b (Apache-2.0).
- GPU validation status for the supported-models table stays "unverified"
  until the author-side run above is recorded.
