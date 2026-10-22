# Decider-2B v11 native text worker

The native worker serves plain, state-first text decisions with independent
questions and isolated Score levels. The CPU processor owns prompt construction,
token accounting, calibration and response reconstruction; the Rust executor
uses the shared Qwen CUDA backend and the Decider LM head. This branch supports
eager execution and bounded packing within one request. It does not combine
separate requests into GPU batches. Keep `CUA_S1_GRAPH` unset or `0` for this
worker; the separate Decider Graph/prefix branches require their own recipes.

## Build and start

Run from the repository root on a CUDA host with NVCC, cuBLASLt and an NVIDIA
GPU supported by the [Qwen backend](../../src/backends/cuda/qwen3_5/README.md).
The second build-script argument is the GPU compute capability without a dot
(`89` below is for an RTX 4090). Build the library from the same source as the
worker; this branch requires Qwen ABI 8. The Rust build itself needs no CUDA
toolkit, but serving requires the compiled library and a GPU.

```sh
python3 recipe/decider/download_weights.py /path/to/decider-2b-v11
src/backends/cuda/qwen3_5/build.sh /path/to/decider-cuda 89
cargo build --release --locked -p omni-decider-native --bin omni-decider

DECIDER_MODEL=/path/to/decider-2b-v11 \
DECIDER_CUDA_LIB=/path/to/decider-cuda/libqwen3_5_cuda.so \
DECIDER_HOST=127.0.0.1 DECIDER_PORT=8000 \
DECIDER_MAX_PENDING_REQUESTS=8 \
DECIDER_MAX_PENDING_TOKENS=1048576 \
DECIDER_REQUEST_TIMEOUT_MS=5000 \
  target/release/omni-decider
```

The downloader verifies all five artifacts against fixed sizes and SHA-256
hashes at checkpoint `533964dae8be954c5b5e19fa4948e48408094c1e`. Loading verifies
checkpoint/configuration/calibration compatibility and runs a real model warmup
before binding the listening socket. The reference runtime revision is
`50d0be0d7cb43d2066965ce5fa7f3fe4e489a60f`.

The three capacity/deadline values above are examples, not measured deployment
recommendations. They are independent and optional; omit any variable to leave
that limit unset. Explicit values must be positive integers; invalid values
fail startup before model loading. These HTTP controls do not apply to startup
warmup or direct `Engine::predict` calls.

```sh
curl --fail-with-body http://127.0.0.1:8000/health
curl --fail-with-body http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data '{"state":"The worker is initialized.","questions":{"ready":{"type":"choice","instructions":"Choose the next action.","criteria":{"continue":"Continue","stop":"Stop"}}}}'
```

## Capacity and deadline semantics

| Variable | What it bounds |
| --- | --- |
| `DECIDER_MAX_PENDING_REQUESTS` | Outstanding HTTP requests, including body upload, preparation, execution queueing, execution and response reconstruction. A slot is reserved before reading the body. |
| `DECIDER_MAX_PENDING_TOKENS` | The sum of complete prepared-row token lengths across admitted requests. Reservation follows preprocessing; repeated state and isolated Score rows count in full. This differs from API `usage.input_tokens` and does not change it. |
| `DECIDER_REQUEST_TIMEOUT_MS` | One absolute deadline from request admission through response reconstruction, including slow body upload and waiting. It does not cover startup, `/health` or network delivery of the response. |

Request/token capacity uses shared
[`BoundedAdmission`](../../src/runtime/README.md#optional-request-capacity-and-execution-control).
The separate FIFO scheduler still admits one complete Decider execution request
at a time. Request capacity also bounds preprocessing; a token-only deployment
must prepare a request before it can determine its token cost.

Temporary request/token contention returns `503`, `Content-Type: application/json`,
`Retry-After: 1`, and `{"detail":"inference capacity exhausted"}`. Use bounded
retries with backoff and an overall client deadline for this response.

A single request whose processed-token cost exceeds the entire token ceiling
returns JSON `422` with `{"detail":"request exceeds processed-token capacity"}`
and no `Retry-After` header. It is rejected before execution and cannot succeed
unchanged under that configuration: reduce its prepared work or raise the
configured budget and restart the worker. A request costing exactly the ceiling
is accepted when that token capacity is free.

Deadline expiry returns JSON `504` with `{"detail":"request timed out"}`.
The deadline is strict: work that completes after it may still yield `504`
instead of a successful response. Timeout/disconnect signals cancellation and
prevents work that has not started from running at the next context check;
it does not preempt already-running CPU/CUDA work. Detached preparation and
execution retain their capacity until they finish, and submitted device work
must synchronize before its resources are released. A `504` therefore does
not imply an immediately free request slot.

Existing content-type, payload and validation checks remain in force: JSON is
required (`415` otherwise), the request body limit is 8 MiB (`413`), and processor
validation errors return `422`. A not-ready worker returns `503` with
`{"detail":"model unavailable"}` rather than the capacity-refusal response.

Within-request packing is configured separately with `DECIDER_BATCH_MAX_ROWS`
(default `1`, allowed `1..=4`) and `DECIDER_BATCH_MAX_TOKENS` (default `4096`,
allowed `1..=4096`). These select packed groups within an admitted request;
they do not replace the outstanding-token budget. Rows above the packing token
budget run alone, subject to the processor/executor's separate input limits.

## Health and validation

`GET /health` is outside request admission and returns `200` for a ready
executor or `503` for an unavailable one. Existing model, checkpoint, execution,
batch and calibration metadata remain present. When any HTTP runtime limit is
set, `runtime_limits` reports `requests`, `processed_tokens` and `timeout_ms`;
unset members are `null`.

Only a configured request-count bound adds top-level `depth`, `capacity` and
`rejected`. `depth` includes all outstanding request reservations, including
blocking work retained after a timeout; `capacity` is the request-count ceiling.
`rejected` counts request-capacity and token-capacity refusals, including requests
larger than the entire token ceiling. It excludes timeouts and processor
validation failures. Token-only or timeout-only deployments do not publish
these three fields. Token occupancy is not exposed by this health interface.

The configuration and HTTP control checks use no checkpoint or GPU:

```sh
cargo test --locked -p omni-runtime
cargo test --locked -p omni-decider-native --test runtime
cargo test --locked -p omni-decider-native --lib serve::runtime_tests
```

These checks cover admission, cancellation, capacity retention and synthetic
HTTP deadlines. They do not establish model parity, performance, GPU-memory
limits or other-hardware support. Device/checkpoint qualification remains a
separate check with the pinned artifacts and matching rebuilt CUDA library.
