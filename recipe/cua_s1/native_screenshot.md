# Native Cua-S1 screenshot serving

The `omni-cua-s1-vision` worker serves the Cua-S1 4B 0.2 `multimodal`
adapter with native Rust/CUDA vision and language execution. It accepts one
inline PNG/JPEG screenshot and 1–8 independent `choice` questions. The vision
encoder runs once per request; each question keeps its own language prompt.
See the [model contract](../../src/models/cua_s1/README.md) for pinned revisions
and the [vision Graph recipe](native_vision_graph.md) for device tests and controls.

## Prepare and build

Prepare the pinned base and adapter using the [reference setup](text.md), keeping
the upstream `weights.lock.json` alongside the downloaded artifacts. Use the
reference Python environment only for this one-time language export. The vision
worker uses the original BF16 vision tensors and unmerged FP32 vision LoRA:

```sh
PYTHONPATH=src .venv/bin/python recipe/cua_s1/export_multimodal_language.py \
  --base weights/Qwen3.5-4B \
  --adapter weights/cua-s1-4b-0.2/multimodal \
  --out weights/cua-s1-multimodal-language

src/backends/cuda/qwen3_5/build.sh target/release 89
cargo build --release --locked -p omni-cua-s1-native -p omni-jev
```

The export refuses an existing output directory. Keep the export and original
weights immutable while serving. `89` targets the validated RTX 4090; choose your
device's supported compute capability when building. Rebuild the CUDA library
and every native consumer together when the ABI changes. This branch requires
ABI 9, including the default host-feature path.

## Start the worker and frontend

```sh
CUA_S1_BASE=weights/Qwen3.5-4B \
CUA_S1_VISION_ADAPTER=weights/cua-s1-4b-0.2/multimodal \
CUA_S1_MODEL=weights/cua-s1-multimodal-language \
CUA_S1_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
CUA_S1_HOST=127.0.0.1 CUA_S1_PORT=8000 \
CUA_S1_GRAPH=0 CUA_S1_VISION_GRAPH=0 CUA_S1_DEVICE_FEATURES=0 \
target/release/omni-cua-s1-vision
```

In another terminal, start the existing Rust frontend:

```sh
OMNI_JEV_BIND=127.0.0.1:8080 \
OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 target/release/omni-jev
curl --fail http://127.0.0.1:8080/health
```

Create a request from a local screenshot, then send it through the frontend:

```sh
python3 - screenshot.png > /tmp/cua-screenshot-request.json <<'PY'
import base64, json, pathlib, sys
image = base64.b64encode(pathlib.Path(sys.argv[1]).read_bytes()).decode()
json.dump({
    "model": "cua-s1-4b-0.2",
    "state": {"image": "data:image/png;base64," + image},
    "questions": {"pick": {
        "type": "choice", "instructions": "Keep the files.",
        "criteria": {"delete": "Click Delete", "cancel": "Click Cancel"}
    }}
}, sys.stdout)
PY
curl --fail-with-body http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/cua-screenshot-request.json
```

Use matching `data:image/jpeg;base64,` for a JPEG. Returned option probabilities
and the selected key depend on the screenshot. The response identifies the
multimodal adapter and reports prompt-token usage with zero generated tokens.

## Execution controls and limits

`CUA_S1_GRAPH=1` enables language Graph replay for host or resident features;
`CUA_S1_VISION_GRAPH=1` independently enables vision Graph replay.
`CUA_S1_DEVICE_FEATURES=1` keeps vision features on the GPU and copies image rows
directly into language scratch before capture/replay. It can be paired with
either vision Graph setting. All three controls default off. See the
[resident-feature contract](native_vision_graph.md#gpu-resident-vision-features)
for strict control parsing, synchronization and reproduction commands.

Requests are serialized on the loaded GPU executor. HTTP concurrency includes
preparation and queueing; there is no cross-request GPU batch. Input is limited
to an 8 MiB body, 4 MiB image file, dimensions up to 2048 on each axis
and 1,048,576 pixels, one image, and 1–26 options per question. Animated images,
remote image URLs, multi-image state, `score`, `noul` and native Metal are unsupported.

Recorded RTX 4090 evidence covers native screenshot parity and Graph regressions;
the resident-feature comparison covers host/device output equality on its fixed
real-screenshot corpus. These scopes do not establish general GUI task accuracy
or native equivalence to every reference precision and environment.

## Request-local language prefix reuse

`CUA_S1_MM_PREFIX=shared` opts into sharing the aligned language prefix of all
questions in one screenshot request. Unset or `off` keeps the existing independent
forwards. `fixed` runs complete prompts independently with the same fixed GEMM
algorithms as `shared`, providing the numerical control. The `matched` mode
below instead checks ordinary full-prompt algorithms. Other values fail before
checkpoint or CUDA loading.

```sh
export CUA_S1_MM_PREFIX=shared
export CUA_S1_GRAPH=0 CUA_S1_DEVICE_FEATURES=0
# Start omni-cua-s1-vision using the pinned artifacts above.
```

The planner compares prepared token IDs and all three position axes, rounds the
common prefix down to 64 tokens, and leaves a nonempty suffix for each question.
It captures attention K/V, GDN recurrent state and convolution history once, then
continues each suffix independently in request order. The engine retains one
snapshot allocation, growing it when a larger shared prefix needs capacity and
freeing it with the engine. Every shared request captures fresh contents before
reading them; a repeated or changed screenshot is always recomputed. One question,
or less than 64 shared tokens, uses complete fixed passes and never reads a
previous snapshot. Retained capacity is additional device memory, separate from
the vision-cache budget.

The `fixed` and `shared` modes require host features and language Graph off.
`matched` supports host or resident features and either language Graph setting;
vision Graph remains independently selectable. They keep the existing prompt, CPU letter head,
answer order and usage accounting. Fixed GEMMs can round differently from default
execution: compare `shared` with `fixed` for continuation parity, and separately
report probability drift/decision changes against `off`. No automatic mode or
cross-request content cache is included. Fixed algorithms still use reference
M=64; the experimental M=256 profile is not part of this implementation.

On one RTX 4090 (SM8.9), a matched eight-payload synthetic HTTP comparison with
vision cache 8 entries/1024 MiB measured shared M64 p50/p95 149.213/223.530 ms
before allocation reuse and 138.404/214.262 ms after it. All response bytes
matched; sampled process maxima were 10282/10288 MiB. Each side excluded an
8-request growth pass, an 8-request all-shape stabilization pass and 5 warmups,
then measured 3 rounds of 8 requests, concurrency 1. The clock includes full
HTTP response parsing and schema validation. The ordinary off control was
118.804/275.748 ms, so this result does not justify enabling shared mode by default.
Source pins, raw samples, exclusions and reproduction inputs are in the
[follow-up qualification](https://github.com/Levius-Fubuki/system1-omni/releases/download/cua-mm-qualification-20261025/cua-mm-followup-qualification-20261025.tar.gz).
These historical measurements apply to the fixed/shared implementation at
`d4908d95c`, before the matched mode below. Other GPUs and general GUI task
success remain unverified.

CPU control/planner tests run in the ordinary Cua suite. On a reserved GPU with
the four artifact variables above, run the exact model and recovery checks:

```sh
cargo test --release --locked -p omni-cua-s1-native --lib prefix_tests \
  -- --ignored --test-threads=1 --nocapture
```

The screenshot corpus is a numerical and execution regression; it does not
establish general GUI task accuracy or a speedup for every request.

### Full-prompt-matched algorithms

`CUA_S1_MM_PREFIX=matched` shares the same request-local aligned prefix using
algorithms selected at the first question's complete prompt length, with the
ordinary BF16 reduction policy. Before any prefix execution, every layer's
projection must have identical algorithm descriptors for every complete question;
the selected algorithms must also support every prefix and suffix window within
the workspace limit. A mismatch or unsupported window falls back to ordinary
independent full passes. A single question or no aligned prefix takes that path
directly. Allocation/setup and execution errors remain errors.

```sh
export CUA_S1_MM_PREFIX=matched
export CUA_S1_GRAPH=0 CUA_S1_DEVICE_FEATURES=0
# Rebuild the CUDA library with src/backends/cuda/qwen3_5/build.sh.
```

This mode requires the optional additive ABI 9 symbols `cs1_gemm_create_matched`
and `cs1_gemm_compatible`. Both are checked during the single-question startup
warmup, before the worker binds its health endpoint; an incomplete library fails
with a rebuild error. Older libraries still serve the existing modes.
The engine retains at most 16 matched GEMM workspaces, 32 MiB each. On a new
reference length beyond that cap, it drains the language stream and releases the
previous generation before allocating another. This is separate from retained
prefix-state capacity and vision-cache limits. No screenshot contents survive
as reusable results between requests. Errors and panics drain submitted work
and restore the ordinary/fixed-M64 handle selection before returning/unwinding.

Descriptor equality and algorithm support do not establish arithmetic invariance
across row counts on every GPU. Qualify the exact GPU/cuBLAS/model combination
before relying on ordinary-output parity. Keep `off` as the ordinary numerical
control and `fixed`/`shared` as the separate M64 controls. No automatic policy, changed prompt or new precision profile is enabled.
With `CUA_S1_GRAPH=1`, admitted matched prefixes and suffixes use bounded window
Graphs; with `CUA_S1_DEVICE_FEATURES=1`, current image rows are copied directly
from the borrowed vision output into owned language scratch.

```sh
cargo test --release --locked -p omni-qwen3-5-native --test kernels \
  matched_gemm_rows_equal_complete_ordinary_projection \
  -- --ignored --test-threads=1 --nocapture
# Also run the prefix_tests command above with vision Graph 0 and 1.
```

At runtime source `02d75b3b4`, on RTX 4090/SM8.9, driver 595.71.05,
CUDA 13.0/cuBLASLt 13.1.1.3, a comparison against `d4908d95c` used concurrency 1
and a direct native-worker HTTP
POST through full JSON/schema validation. Each configuration ran in forward and
reverse order, four timed traversals per visit, after excluding growth,
all-shape stabilization and five warmups. The frozen real suite uses 12 screenshots
(desktop/mobile/web, four each), four correlated click-choice permutations each.

| Workload / mode | timed requests | mean ms | p50 ms | p95 ms | sampled max MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| synthetic / off | 64 | 133.30 | 119.31 | 277.13 | 10138 |
| synthetic / shared M64 | 64 | 134.73 | 139.35 | 215.54 | 10288 |
| synthetic / matched | 64 | 107.17 | 103.24 | 173.69 | 10448 |
| real / off | 96 | 256.36 | 265.33 | 271.98 | 10210 |
| real / shared M64 | 96 | 205.53 | 209.87 | 215.85 | 10318 |
| real / matched | 96 | 152.83 | 152.94 | 160.08 | 10606 |

On this real suite, matched reduces mean latency 25.6% versus shared M64 and
40.4% versus off; p95 reductions are 25.8% and 41.1%. The paired-workload-round
bootstrap 95% intervals for mean reduction are 25.1–26.1% and 39.9–40.7%; these
describe repeated timings, not general task performance. Synthetic mean
reductions are 20.5% and 19.6%. One small two-question synthetic case regresses
4.8% versus off. There is no claim of a speedup for every shape. Real-suite sampled
maximum increases 288 MiB versus M64 and 396 MiB versus off. Memory is sampled
at about 100 ms, not an exact allocator peak. Vision-cache controls are 8 entries/
1024 MiB for synthetic and 16 entries/2048 MiB for real, not new defaults.

Five profiles each completed 80 real/stress layouts twice and 32 single-question
extractions (960 calls total), plus malformed-input/recovery probes. Matched and
candidate off have exactly the ordinary-control FP32 logit bits, complete
responses, token IDs and all three position axes; candidate shared retains the
original M64 results. All single-question extractions match their parent answers.
The 320 projection-window comparisons and prefix model regressions with vision
Graph 0/1 passed on this GPU. Existing default-versus-reference limits remain:
the adapted 48-screenshot/192-correlated-choice set scores 119 for off/matched
and 120 for M64. This is not official ScreenSpot accuracy or evidence of improved
reference quality. Defaults remain off; other GPUs, libraries and concurrency
need qualification.

Raw timings, exclusions, logits, model/source/binary hashes, reproduction inputs
and acquisition scripts, failures and final-source mapping are in the
[2026-10-26 qualification](https://github.com/Levius-Fubuki/system1-omni/releases/download/cua-mm-qualification-20261026/cua-mm-matched-qualification-20261026.tar.gz). Original real screenshot pixels are omitted.
These measurements precede the startup-symbol guard. That guard has host-only
missing-symbol regression coverage; no new GPU numerical or timing run is claimed.

### Resident matched prefixes with window Graphs

```sh
export CUA_S1_MM_PREFIX=matched CUA_S1_DEVICE_FEATURES=1
export CUA_S1_GRAPH=1 CUA_S1_VISION_GRAPH=1
# Keep prefix=off as the ordinary complete-request numerical control.
```

The matched implementation is shared with PR184; this extension combines it
with the resident language-Graph path. Prefix reuse removes repeated computation
across questions; any combined speedup must be attributed to that work as well
as Graph submission savings. A single question or incompatible GEMM plans still
uses ordinary independent execution, preserving the selected Graph/device mode.

Host and device windows use the same token/T-H-W prefix planner and CPU letter
head. Checked device row slices retain the vision producer borrow. The consumer
copies current rows into its own scratch before capture; no vision pointer is
recorded. Device window exits drain the language stream, including errors.

Window Graph keys contain absolute begin/end rows, both source and destination
prefix allocation identities, and the selected GEMM handle. Entries retain both
prefix allocations, including the simultaneous continue-and-capture case. This
makes producer release and prefix-storage replacement safe. At most 128 window
Graphs are retained; eviction drains first. Scratch growth and matched-workspace
retirement clear affected captures after synchronization. Recording failure keeps
the warmed eager result, disables all language Graph modes and retires their
caches. Vision Graph is independent. Retained snapshot allocations, Graph objects
and up to 16 matched 32 MiB workspaces add GPU memory beyond the vision budget.

Warm the largest request and the complete shape working set, then check traces
for new captures or failures in the measured region. Cold capture and cache
churn must be reported separately; all switches remain opt-in.

```sh
CUA_S1_GRAPH=1 CUA_S1_DEVICE_FEATURES=1 CUA_S1_MM_PREFIX=matched \
  cargo test --release --locked -p omni-cua-s1-native --lib prefix_tests \
  -- --ignored --test-threads=1 --nocapture
# Repeat with vision Graph0 and1.
CUA_S1_TEST_MODEL="$CUA_S1_MODEL" cargo test --release --locked \
  -p omni-qwen3-5-native --lib model::graph_tests \
  -- --ignored --test-threads=1 --nocapture
```

### Recorded resident matched qualification

On RTX 4090 / SM8.9 / CUDA 13.0, the 2026-10-26 frozen complete-request
comparison measured the following. O is original PR183 at `911faa0d`, E is
resident matched prefix eager, and G is resident matched prefix with window
Graphs. All use vision Graph and the same pinned artifacts. Each mode has 64
synthetic requests and 96 requests from 12 fixed real screenshots, concurrency 1.

| Workload / mode | Mean ms | p50 ms | p95 ms | Sampled process maximum MiB |
|---|---:|---:|---:|---:|
| Synthetic / O | 130.257 | 117.968 | 269.950 | 10202 |
| Synthetic / E | 105.979 | 102.867 | 169.251 | 10448 |
| Synthetic / G | 104.902 | 101.780 | 172.540 | 10538 |
| Real / O | 248.819 | 258.092 | 263.407 | 10256 |
| Real / E | 150.949 | 151.120 | 157.391 | 10606 |
| Real / G | 147.868 | 148.416 | 154.275 | 10694 |

G reduces the real mean by 40.6% and p95 by 41.4% versus O, with 438 MiB higher
sampled process maximum. Relative to E, Graph reduces the real mean by 2.0%; the
large combined gain therefore includes prefix computation reuse. Synthetic mean
falls 19.5% versus O, while the smallest two-question case regresses 4.8%; G's
synthetic p95 is 1.9% higher than E. These are scoped measurements, not a default-on
recommendation or a universal speedup.

Two balanced O/E/G/G/E/O visits per suite each exclude a growth traversal, a
stabilization traversal and five warmups, then time four shuffled traversals.
The 480 timed HTTP samples include response parsing/schema validation; 300
excluded and 135 feasibility samples remain separate. No timed new capture or
capture failure occurred. Five profiles each produced 192 regular numerical
records: raw logits, full responses and token/T-H-W inputs match original O
exactly, including separate single-question extraction and malformed-input
recovery. Cold/growth timings and startup are retained separately. Real quality
remains 119/192 correlated choice permutations, with existing native-vs-FP32
reference limitations; this is not official ScreenSpot accuracy.

The [raw qualification and portable audit](https://github.com/Levius-Fubuki/system1-omni/releases/tag/cua183-resident-20261026)
bind source/build/model pins, corpus reconstruction, cold and measured samples,
per-case regressions, memory, admission traces and GPU lifecycle tests. Original
screenshot pixels are omitted. Other hardware, concurrent serving and general
GUI task quality require separate qualification.
