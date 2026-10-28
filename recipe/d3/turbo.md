# D3 turbo reference worker

This profile serves the fixed `vllm-sr/d3-turbo` package through the existing
D3 Python worker and Rust frontend. It delegates execution to the public
`AutoModel.system_one`, including the package's state-once path. It is a
reference integration, with no native CUDA executor or acceleration claim.
The [legacy D3 profile](reference.md) remains the default and keeps its own pin.

## Artifacts, dependencies and resources

Run commands from the repository root. Use one fresh process per profile and
package; closing an executor does not permit switching packages in that process.

| Input | Pin |
| --- | --- |
| Package | `vllm-sr/d3-turbo` |
| Revision | `a0f3e9e59ebd1202fe1d4de1c4c0f9a986758cbb` |
| `MODEL_MANIFEST.json` SHA-256 | `54a65af0871c21b3440cbdadaccfd752a8e2173b86334d1b3290f005e49699fe` |
| Execution | BF16 backbone, separate FP32 255-code readout, SDPA, `prefix_bidir` |
| Dependencies | Package requirements, including Transformers 5.17.0 and FLA 0.5.2 |
| License | Apache-2.0; keep the package's attribution files |

The weights take approximately 52.2 GB before loading scratch and activations.
Provide enough host RAM and disk space for the download and loading. An 80 GB
GPU was sufficient for the bounded A800 experiment; a 24 GB GPU cannot hold
this full BF16 package. Full 32K-context memory capacity is not established by
the small fixtures or Public231. The intended production target is H200 /
SM90; this does not claim validation on that target.

Use stable Rust, Python 3.12 and a CUDA-compatible PyTorch/torchvision pair. The A800
environment used torch 2.12.1+cu130, torchvision 0.27.1+cu130,
Transformers 5.17.0 and flash-linear-attention 0.5.2. Install the matching
PyTorch wheels for your environment first, then:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.12.1 torchvision==0.27.1 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r recipe/d3/requirements.txt
python -m pip install pytest httpx==0.28.1
export D3_TURBO_PACKAGE="$PWD/weights/d3-turbo"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    "vllm-sr/d3-turbo",
    revision="a0f3e9e59ebd1202fe1d4de1c4c0f9a986758cbb",
    local_dir=os.environ["D3_TURBO_PACKAGE"],
)
PY
PYTHONPATH=src python - <<'PY'
import os
from pathlib import Path
from models.d3.turbo import verify_package
verify_package(Path(os.environ["D3_TURBO_PACKAGE"]), verify="full")
print("full package verification passed")
PY
```

Before importing package code, both verification modes authenticate the exact
manifest, hash the runtime/wrapper/compose closure, and hash every non-weight
load input, including configuration, tokenizer, processor and readout. `fast`
also checks all shard sizes and hashes shards up to 64 MiB; `full` hashes all
shards. Turbo refuses `--verify none`. Unlisted importable code and package
bytecode caches are rejected. The worker isolates copied Transformers modules
from stale dynamic-module caches and refuses old/new package mixing.

The pinned manifest's README size is stale. Like the package's own
`loaded_files`, this gate excludes Markdown, artwork, license texts and
packaging metadata from load verification; runtime `requirements.txt` remains
fully hashed. This exception does not exclude executable or model inputs.

## Launch and health

```sh
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export D3_STATE_ONCE=1
PYTHONPATH=src python -m frontend.d3 --profile turbo \
  --model "$D3_TURBO_PACKAGE" --device cuda:0 --verify fast --port 8000
```

The port binds only after the bundled warmup and a successful real typed
inference. Readiness therefore includes model/kernel initialization. In another
terminal, activate the environment and run:

```sh
curl -fsS http://127.0.0.1:8000/health
cargo build --release --locked -p omni-jev
OMNI_JEV_BIND=127.0.0.1:8080 \
OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 \
  target/release/omni-jev
```

Send a real request directly and through the frontend:

```sh
curl -fsS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @recipe/d3/example-request.json > /tmp/d3-turbo-direct.json
curl -fsS http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @recipe/d3/example-request.json > /tmp/d3-turbo-proxy.json
cmp /tmp/d3-turbo-direct.json /tmp/d3-turbo-proxy.json
```

The request's `model` field is accepted and ignored, as in the reference.
The default response model is `d3-turbo`. Interrupt the worker with Ctrl-C
for its normal cleanup path. A busy worker responds 503 rather than queueing
another request; the existing frontend has its own transport timeouts.

## Scope and deliberate wire restrictions

The envelope accepts only `model`, `state`, `questions` and `images`.
`questions` is a nonempty ordered mapping with nonempty IDs. Supported types
are `choice`, `noul` and `score`; their fields are `type`, `instructions` and
`criteria`. Text and JSON descriptions retain the package's rendering rules.
There is no silent truncation: the default per-question cap is 32,768 tokens.

`images` may be absent, null or a list of zero or one inline PNG/JPEG base64
data URL. To make a deterministic example image request:

```sh
python - <<'PY'
import base64, json
from pathlib import Path
request = json.loads(Path("recipe/d3/example-request.json").read_text())
request["images"] = ["data:image/png;base64," + base64.b64encode(
    Path("tests/d3/fixtures/turbo-scene.png").read_bytes()).decode()]
Path("/tmp/d3-turbo-image.json").write_text(json.dumps(request))
PY
curl -fsS http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @/tmp/d3-turbo-image.json
```

Strict decoding uses the pinned runtime: at most 8,000,000 decoded image
bytes and 16,000,000 source pixels, with integrity checks and the runtime's
processor resize budget. The existing total HTTP body limit is 64 MiB.
Non-inline paths/URLs, WebP, multiple images, videos, assets, question/option
image placements, Box/Track, `long_context` and additional fields are rejected
explicitly with 422, even when an unsupported envelope field is empty.
Recognizable embedded media objects, media data URLs and literal media
placeholder tokens are also refused rather than interpreted as text.
Quantization and additional hardware backends are outside this profile.

Malformed transport/JSON retains the worker's 400/411/413/415 behavior.
Supported-kind questions with invalid criteria or excessive token length keep
the public AutoModel's **HTTP 200 per-question typed errors**, with its own
reason/message and usage. This differs deliberately from legacy D3's
whole-request pre-execution 422 policy. Confidence, calibration, legal-code
order, question order and token accounting are not recomputed by the adapter.

## Validation and frozen fixtures

CPU checks do not load weights or establish GPU parity:

```sh
PYTHONPATH=src python -m pytest tests/d3 -q
python benchmarks/bench.py validate benchmarks/smoke.jsonl
```

The opt-in fixture tool is under root `tests/`, alongside its tests and image
fixtures. It compares the public `AutoModel.system_one` with production
`frontend.d3.decide` on the same pinned loaded model. It records prompt text,
expanded token IDs, exact state/question boundaries, legal code order,
processed tensor bytes/grids/positions, probabilities, typed answers/errors
and usage. The two mode-specific JSON oracles retain their own probabilities,
answers, usage and generation provenance. They reference one SHA-256-pinned
`turbo-preparation.json`, which stores the common requests/preparation and
references one `turbo-inputs.npz` containing all 83 input tensor arrays. Token
arrays stay inline in JSON; state/question token slices are reconstructed
from the complete token IDs and exact state boundary. The loader verifies
both references and reconstructs the original comparison structure.

This storage normalization preserves the original oracle values and tensor
bytes. The historical GPU evidence retains the original representation.
Preparation and decisions/order/usage remain exact; numerical probability/Score
tolerance is still 1e-6, and Noul threshold flips fail even within that tolerance.

```sh
D3_TURBO_CHECK_DIR="$(mktemp -d)"
D3_STATE_ONCE=1 PYTHONPATH=src python tests/d3/turbo_reference.py freeze \
  --model "$D3_TURBO_PACKAGE" --device cuda:0 \
  --source-revision "$(git rev-parse HEAD)" \
  --check tests/d3/fixtures/turbo-oracle-state-once.json \
  --output "$D3_TURBO_CHECK_DIR/turbo-oracle-state-once.json"
D3_STATE_ONCE=0 PYTHONPATH=src python tests/d3/turbo_reference.py freeze \
  --model "$D3_TURBO_PACKAGE" --device cuda:0 \
  --source-revision "$(git rev-parse HEAD)" \
  --check tests/d3/fixtures/turbo-oracle-full-row.json \
  --output "$D3_TURBO_CHECK_DIR/turbo-oracle-full-row.json"
```

The two commands share only the fresh output directory; conflicting shared
inputs are rejected rather than overwriting an existing mode's fixtures.
Each command uses a separate fresh reference process. Do not compare
state-once numerics or usage against the full-row fixture as if they were
identical: the two modes can differ. Full-row uses the same `prefix_bidir`
visibility. Turbo's state attends to the whole state; each question attends
to that state and its own complete question block. GDN remains causal.
Native state reuse will require an exact semantic state-end cut, identical
state reuse and separate numerical qualification. The existing causal
64-token prefix planner does not establish that equivalence.

Public231 C1 and current measured-source evidence are recorded in the
integration PR. Use the [benchmark protocol](../../benchmarks/README.md) and
the model's frozen Public231 manifest: contract smoke, warmup 5, concurrency 1,
three measured rounds, exact source SHA, measured device/capability, complete
serving counts and pooled median/nearest-rank p95. Keep image/many-question
experiments separate. No native, optimization, throughput, H200 or Metal
claim follows from this reference integration.


## Measured reference baseline

The 2026-10-28 campaign measured source
`821d6f93336c45c61576a2787e26d3897aae9f27` on one **NVIDIA A800 80GB PCIe**,
compute capability **8.0**, driver 595.71.05, torch 2.12.1+cu130 and the
versions listed above. GDN used FLA 0.5.2; `causal_conv1d` was absent, so its
operation used the package's PyTorch fallback on both paths. This is a
reference baseline, with no native or optimized execution claim.

| Deployment | Successful / planned | Pooled p50 | Pooled p95 |
| --- | --- | --- | --- |
| Official public AutoModel on bundled HTTP transport | 693 / 693 | 236.63 ms | 927.31 ms |
| System1-Omni turbo worker through Rust frontend | 693 / 693 | 233.83 ms | 930.21 ms |

Each deployment ran three rounds of the frozen 231-task JevBench Public231
workload, with warmup 5, concurrency 1 and no retries. Timings include POST,
complete body receipt, JSON decoding and schema validation. Pooled p50 is the
median of 693 successful samples and p95 is nearest rank; round percentiles
were not averaged. Loading, readiness, supplementary images/many-question
requests and the independent full-row process are excluded.

All 693 paired response bodies were semantically exact: maximum numerical
difference 0, no classification flips, exact question ordering and usage.
Each round had Choice 128/139, Noul 67/74 and Score MAE
0.1253665383 over 18 questions. The services ran serially in fixed order;
server order was not randomized. System1's round medians were 222.81,
216.44 and 254.00 ms. The mixed percentile changes and this variation do not
establish an acceleration benefit.

The largest Public231 prompt was 3970 tokens. PNG/JPEG and 1/4/16-question
checks are supplementary, not image-quality or full-context benchmarks.
Both six-case oracle fixture sets (83 tensor arrays each) were regenerated
and compared separately at the measured source. Client-rejection recovery,
first/repeated requests, direct/frontend semantic parity and graceful worker
shutdown passed. CPU checks do not qualify legacy D3's GPU path on this head.

The [reference evidence release](https://github.com/Levius-Fubuki/system1-omni/releases/tag/d3-turbo-reference-821d6f933-20261028)
contains raw repeated-run records, dependency/identity evidence, source and
input hashes, independent recomputation instructions and the shutdown receipt.
The integration remains [draft PR #198](https://github.com/InferMatrixLab/system1-omni/pull/198).
Later documentation/CI-only commits retain the measured inference code;
ready-for-review and merge require separate explicit owner approval.
