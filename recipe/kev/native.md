# Kev-4B eager native text worker

The worker implements Kev's `choice`, `noul`, and `score` contract in Rust and runs
its Qwen3.5 language backbone on CUDA. It uses the [preparation export](README.md),
the shared Qwen selected-position readout, and Kev's trained FP32 pointer head.
No Python runs in the worker. Other Kev sizes, media encoding, prefix reuse,
CUDA Graphs, and additional kernel optimization are outside this baseline.

Run commands from the repository root. Build the backend for the actual GPU's
compute capability; the build itself does not use a GPU. The acceptance below
used the `89` build (including its PTX); see the hardware note before generalizing:

```sh
src/backends/cuda/qwen3_5/build.sh target/release 89
cargo build --release --locked -p omni-kev-native -p omni-jev
# On hosts requiring reservations, keep gpu run outside the worker process.
gpu run --gpus 1 --nonblock --timeout 2h --note 'Kev eager worker' -- \
  env KEV_MODEL="$PWD/weights/kev-native" \
      KEV_CUDA_LIB="$PWD/target/release/libqwen3_5_cuda.so" \
      KEV_PORT=8000 target/release/omni-kev-native
```

`KEV_HOST` defaults to `127.0.0.1`; `KEV_PORT` defaults to 8000. The library defaults
to `libqwen3_5_cuda.so` beside the executable. Startup verifies the pinned manifest
and every exported file, loads the model, and runs a real request containing all
three question types. Only then does it bind the listener. `/health` reports
`ready`, the checkpoint ID, eager execution, and disabled prefix reuse.

In a second terminal, start the model-independent frontend:

```sh
OMNI_JEV_BIND=127.0.0.1:8080 OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 \
  target/release/omni-jev
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' -d '{
    "model":"kev-4b", "state":"The light is green. The door is open.",
    "questions":{
      "color":{"type":"choice","instructions":"What color is the light?",
               "criteria":{"red":null,"green":null,"blue":null}},
      "open":{"type":"noul","instructions":"The door is open."},
      "safety":{"type":"score","instructions":"How safe is entering?",
                "criteria":["unsafe","uncertain","safe"]}
    }}'
```

The worker accepts `jaredpalmer/kev-4b`, `kev-4b`, and `kev-latest` (default), and
returns the requested alias. It preserves question/candidate insertion order,
rounds probabilities/derived scalars to four decimals, and counts the serialized
answers for `output_tokens`, as the pinned reference does. `input_tokens` counts
the state once plus every question branch, even though this eager baseline
recomputes state for each question. It generates no text tokens.

Limits: 4 MiB request body; 64 questions; 1–255 Choice candidates or Score levels;
8192 state tokens including its marker; 16384 tokens per complete question row;
65536 summed computation-row tokens per request; 16 admitted requests. A request
is validated completely before GPU dispatch. There is no silent truncation.
These context/request budgets are deliberately below the official server's 64k
state allowance. JSON state/instructions/descriptions use the official text
rendering, including booleans, numbers, null, arrays, and objects. Media URLs or
objects are rendered as text; this worker has no image/audio/video encoder.

Malformed JSON (including duplicate keys) returns 400; oversized bodies 413;
non-JSON content types 415; invalid/unsupported requests 422; capacity exhaustion
503 with `Retry-After: 1`; execution failure 500 without internal details. Unknown
fields are ignored as in the reference. Unlike the official server, arbitrary
model names are refused and timing fields are omitted.

## Reproduce correctness checks

Ordinary checks require no checkpoint or GPU:

```sh
cargo test --locked -p omni-kev-native -- --list
cargo test --locked -p omni-kev-native
python -m unittest discover -s tests/kev -p 'test_*.py' -v
```

Pinned tokenizer and released-head checks are CPU-only:

```sh
KEV_EXPORT="$PWD/weights/kev-native" cargo test --locked -p omni-kev-native \
  --test processing -- --ignored --nocapture
KEV_HEAD="$PWD/weights/kev-native/head.safetensors" cargo test --release --locked \
  -p omni-kev-native --test head -- --ignored --nocapture
```

Create a separate Python 3.12 reference environment with
[`reference-requirements.txt`](reference-requirements.txt). Keep the CPU export
environment separate if it uses a CPU-only Torch wheel. Check out
`jaredpalmer/kev@2074cc44256200bb37540103fa4572901faa10f3` as `kev-reference`.
The fixture generator verifies the official source hashes before importing it:

```sh
.venv-kev-reference/bin/python tests/kev/generate_processing_fixture.py \
  --reference kev-reference --base weights/kev-base --out /tmp/kev-processing.json
cmp tests/kev/data/processing.json /tmp/kev-processing.json
```

Reserve at least 40 GiB free host RAM, 24 GiB device memory and space for the base,
merged export, reference environment, and compiled artifacts. The FP32 unmerged
reference is larger than the BF16 worker. Use a new result directory for each run;
preserve failures and the exact source diff. This is a correctness campaign,
not a latency/throughput benchmark.

```sh
gpu run --gpus 1 --nonblock --timeout 2h --note 'Kev reference parity' -- \
  .venv-kev-reference/bin/python recipe/kev/validate.py reference \
  --reference kev-reference --base weights/kev-base --adapter weights/kev-adapter \
  --export weights/kev-native --out /tmp/kev-run-01

gpu run --gpus 1 --nonblock --timeout 2h --note 'Kev native parity' -- \
  env KEV_EXPORT="$PWD/weights/kev-native" \
      KEV_CUDA_LIB="$PWD/target/release/libqwen3_5_cuda.so" \
      KEV_NATIVE_OUT=/tmp/kev-run-01/native.json \
  cargo test --release --locked -p omni-kev-native --lib \
  native_corpus_capture -- --ignored --nocapture
python recipe/kev/validate.py compare --out /tmp/kev-run-01

gpu run --gpus 1 --nonblock --timeout 2h --note 'Kev direct/frontend parity' -- \
  python recipe/kev/validate_http.py \
  --worker-bin "$PWD/target/release/omni-kev-native" \
  --frontend-bin "$PWD/target/release/omni-jev" \
  --export "$PWD/weights/kev-native" \
  --cuda-lib "$PWD/target/release/libqwen3_5_cuda.so" \
  --native /tmp/kev-run-01/native.json --out /tmp/kev-http-01
```

The comparator first isolates pure FP32 PEFT merge error, then BF16 export/compute
drift. It also reports official unmerged → PyTorch merged, PyTorch
merged → native, and official unmerged → native. The corpus covers 1/2/7/31/255
candidates, 1/3/255 Score levels, mixed question types, Unicode/control-token
spellings, structured values, reordering, and three additional repeated mixed
requests. HTTP acceptance compares complete direct/frontend/native responses,
checks the first external request after warmup, and checks error status/body.

Fixed gates: exact rendering/tokens/readout positions; isolated head logit error
`<= 1e-4 + 1e-5*abs(reference)` and probability error `<= 1e-5`; full-model maximum
probability absolute error `<= 0.01`; matching Choice decisions when the reference
top-two margin is `> 0.02`; matching Noul decisions at threshold 0.5 when the
reference is farther than 0.01 from it. Every near-boundary disagreement is listed.
A failed merge baseline is reported separately and never justifies widening a gate.

## Recorded acceptance

Local author-run at source `57e239b681f84d56609ed7cef45889a115381172`, over 13
requests / 23 questions (10 Choice, 7 Noul, 6 Score):

| Comparison | Maximum probability absolute error |
| --- | ---: |
| Official FP32 unmerged → FP32 merged | 0.00000298023 |
| FP32 merged → exported BF16 PyTorch | 0.00543183 |
| Official FP32 unmerged → exported BF16 PyTorch | 0.00543272 |
| Exported BF16 PyTorch → native | 0.00607389 |
| Official FP32 unmerged → native | 0.00553775 |

All declared probability/decision gates passed. All 10 Choice and 7 Noul decisions
agree; three additional native mixed requests are identical. There were no
near-boundary disagreements; this corpus's smallest nontrivial Choice margin is
0.121829 and smallest Noul threshold distance is 0.023158, so it does not establish
real near-boundary coverage. Synthetic checker tests verify boundary reporting.

For the 255-level Score, the expected level is 96.7645 officially versus 96.6496
natively (absolute difference 0.1149 on a 0–254 scale). Its maximum probability
error is 0.000169017. The declared gate is on probabilities; no scalar Score
threshold was specified. Other datasets and longer contexts need their own
qualification; this small deterministic corpus is not a model-quality benchmark.

The exact processing/head CPU gates passed. Workspace tests: 397 passed, 91
ignored, zero failed; relevant ignored tokenizer, released-head and native
capture checks were explicitly run. Real warmup and the first external request
passed. Sixteen paired HTTP requests matched direct worker, frontend and native
capture exactly; eight error cases matched status/body. No performance claim is
made from these checks.

The lab device is labelled NVIDIA L20X: the reserved Torch runtime reports
capability 9.0 while `nvidia-smi` reports 8.9. The tested backend was built with
CUDA 13.0 for sm_89 plus compute_89 PTX; its SHA-256 is
`9a29aae12a35368481f61ccbe9ecf318230b658224ee1d46211bbe2a30703c71`.
Reference Torch uses CUDA 12.8. Do not infer another hardware model from the label
or claim the untested sm_90 build passed. Exact package pins are above; raw JSON,
commands, hashes, and the retained outer-wrapper interruption are supplied in
the local handoff archive. No public artifact upload was performed.
