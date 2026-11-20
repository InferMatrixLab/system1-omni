# OmniJev-4B v1.1 native worker

The native worker ([`src/models/omnijev/native/`](../../src/models/omnijev/native/))
serves OmniJev-4B v1.1 ([#114](https://github.com/ThinkFlowLab/system1-omni/issues/114))
on `/v1/systemone`: one PNG or JPEG image and any mix of Choice, Noul and Score
questions, with the vision tower and the Qwen3.5-4B language model on the CUDA kernels
in [`src/backends/cuda/qwen3_5/`](../../src/backends/cuda/qwen3_5/) and no Python or
PyTorch. It needs an NVIDIA GPU with compute capability 8.0 or newer and the export from
[the preparation recipe](README.md). [The model README](../../src/models/omnijev/README.md)
describes the contract and what the worker computes.

Run the commands from the repository root. Pass your GPU's compute capability to
`build.sh`; the worker finds `libqwen3_5_cuda.so` next to its executable, or at
`OMNIJEV_CUDA_LIB`:

```sh
src/backends/cuda/qwen3_5/build.sh target/release 89   # needs nvcc and cuBLASLt
cargo build --release --locked -p omni-omnijev-native
OMNIJEV_MODEL=$PWD/omnijev-export target/release/omni-omnijev-native
```

## Explicit BF16 reference vision

`OMNIJEV_VISION_NUMERICS=reference` selects the unadapted OmniJev-4B vision
arithmetic. The default, `legacy`, retains the existing worker output. Unknown
values fail startup. The reference selection needs the optional cuDNN 9 build
and additive OmniJev CUDA symbols; requesting it with an old library fails with
a rebuild instruction. Build and launch it explicitly:

```sh
CUDNN_INCLUDE_DIR=/absolute/cudnn/include CUDNN_LIB_DIR=/absolute/cudnn/lib \
    src/backends/cuda/qwen3_5/build.sh target/release 89
OMNIJEV_VISION_NUMERICS=reference OMNIJEV_MODEL=$PWD/omnijev-export \
    target/release/omni-omnijev-native
```

This path uses a cuDNN BF16 patch convolution, vectorized Welford LayerNorm,
the existing framework GEMM policy with Torch's 1 MiB heuristic budget
(including its reduction schemes), FP32
rotary angles/trigonometry, and head-64 reference attention. It retains BF16
operator outputs and the separate patch-bias addition. It selects vision only:
the language weights still contain the merged LoRA, and language/readout and
calibration arithmetic are unchanged. The official unmerged BF16 model and the
original probability/decision gates remain the acceptance reference. Vision
alignment alone does not establish whole-model numerical acceptance.

The profile uses the existing four-grid FIFO, rejects vision adapters and
explicit bounded-cache policies, and must be selected before the first forward.
`CUA_S1_VISION_GRAPH=1` can capture the vision path after warmup; descriptors,
GEMM plans and workspace are prepared before capture. Validation is specific to
the tested GPU/library stack. See the [stage and operator regressions](../../tests/qwen3_5/README.md#omnijev-4b-reference-vision).

On one L20 (sm_89), Rust 1.100/CUDA 12.8 with cuDNN 9.24 and the documented
reference environment, the original `video.png/noul_region` case matches all
40 vision boundaries exactly, including the 24 blocks and merger. The complete
original 28-request/280-answer comparison still fails the unchanged acceptance
gate: 177 answers are within 0.01, maximum returned-field drift is 0.1758,
and three protected decisions differ (`board.jpg/q8`, `arcade.jpg/long_start`,
`web.jpg/one_token`). All 140 PNG decisions agree, but 20 PNG answers still
exceed 0.01. Unrounded values retain the same three protected failures. The
JPEG failure set changes from the legacy path; do not infer a uniform
whole-model improvement. JPEG preparation and language/LM/unmerged-LoRA
differences still need separate controlled validation. This profile stays
opt-in and does not complete #114. No performance result is claimed for it.

`CUA_S1_GRAPH=1` also enables language window Graphs in this worker, through the
shared `Model::readout_window` implementation. It defaults off. Request and
question prefixes and option continuations can capture and replay; current token,
position and image inputs are staged before replay. Graph entries retain their
prefix source and capture destination allocations. The shared 128-entry FIFO and
scratch growth drain dependent work before retiring affected Graphs; recording
failure disables language Graphs and preserves the eager
result. This switch does not enable vision Graphs or Cua's matched-prefix mode.
The Cua-S1 latency measurements do not establish an OmniJev speedup.

`OMNIJEV_HOST` and `OMNIJEV_PORT` default to `127.0.0.1` and `8000`. At start the
worker checks every file of the export against its manifest, loads the weights and
answers one request with all three question types before it listens, so `/health`
answers only once a request has run. Put the frontend in front of it with
`OMNI_JEV_BACKEND_URL` (see [the frontend README](../../src/frontend/README.md)), or
send requests to the worker directly:

```sh
IMAGE=$(base64 < screenshot.png | tr -d '\n')
curl -s http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' -d @- <<EOF
{"model": "tinnel123/OmniJev",
 "state": {"images": ["data:image/png;base64,$IMAGE"]},
 "questions": {
   "error": {"type": "noul", "instructions": "This screen shows an error dialog."},
   "next": {"type": "choice", "instructions": "Which operation comes next?",
            "criteria": {"click": null, "type text": null, "scroll": null}},
   "risk": {"type": "score", "instructions": "How risky is acting on this screen?",
            "levels": ["safe", "check first", "dangerous"]}}}
EOF
```

The response is `{"model": "tinnel123/OmniJev", "answers": {id: answer},
"usage": {"input_tokens": n, "output_tokens": 0}}`: each answer has the reference's
fields, listed in [the model README](../../src/models/omnijev/README.md#request-and-response),
and `input_tokens` is counted as the reference counts it.

Malformed JSON gets 400, a body over 12 MiB 413, and a request the contract refuses
422, each with a `detail` message. The worker prepares two requests at a time and holds
at most 16, preparing, waiting or running; beyond that it answers 503 with
`Retry-After: 1`.

## Explicit unmerged one-shot profile

Prepare the [separate unmerged artifact](README.md#separate-unmerged-reference-artifact),
and build the CUDA library with the reference vision requirements above. The
library must also provide the existing reference language/FP32 GDN operators.
From the repository root:

```sh
cargo build --release --locked -p omni-omnijev-native -p omni-jev
OMNIJEV_PROFILE=unmerged-reference \
OMNIJEV_MODEL=$PWD/omnijev-unmerged-reference \
OMNIJEV_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
CUA_S1_GRAPH=0 CUA_S1_VISION_GRAPH=0 \
    target/release/omni-omnijev-native
```

`OMNIJEV_PROFILE` accepts `merged-fixed` (default) or `unmerged-reference`;
unknown values fail startup. The latter requires the matching unmerged export,
selects reference vision and language arithmetic, and supports complete
Choice/Noul/Score requests through the existing direct-worker and Rust frontend
interfaces. Its all-type warmup finishes before listening or reporting ready.
Use the frontend launch and request examples above unchanged. Language Graphs
and prefix reuse are rejected. This experimental profile has no speed claim;
full numerical acceptance and JPEG fidelity remain separate gates.

The backbone runs normal independent full rows. LM projections retain the
official default grouping: chunks of 64 complete-request candidate rows,
option-token rows in order, then a separate zq/first-token group. Score rows
remain in the projection layout. Complete-request targets, including those
unused Score targets, are limited to 16,384 and rejected during preparation
with HTTP 422 above that bound. This is stricter than the merged profile's
Score-exempt LM budget. All logits use the exact 248,079-entry denominator.
Readout/window guards remain; this mode uses no continuation state.

On actual original-request hidden, native grouped BF16 logits match the
official 112-row and 31-row outputs exactly. The retained CPU FP64 reduction
differs from official FP32 picked log-probabilities by at most 9.54e-7;
the six features differ by at most 1.53e-5 and same-hidden heads by 2.39e-7.
Those controls isolate the LM boundary; they do not replace the original
branch's complete-answer 0.01 and protected-decision gate.

On one L20 with the pinned BF16 unmerged weights and runtime, the complete
original 28-request/280-answer replay remains **failed** under that unchanged
gate. Raw values round back to the production results on every answer:

| Input population | Answers within 0.01 (raw and returned) | Protected disagreements | Maximum raw field drift |
| --- | ---: | ---: | ---: |
| PNG | 125/140 | 0 | 0.02967435 |
| JPEG | 54/140 | 3 | 0.18771529 |
| Complete corpus | 179/280 | 3 | 0.18771529 |

All 140 PNG decisions agree; fifteen PNG answers still exceed 0.01. The largest
is the original complete-context `video.png/q6`. Protected JPEG disagreements
remain `board.jpg/q8`, `arcade.jpg/long_start`, and `web.jpg/one_token`.
The largest returned-field drift is 0.1878. This is a fresh unmerged candidate
run, separate from the earlier merged/reference-vision result above; do not
combine their metrics or infer a uniform improvement.

For that q6 case, native zq/u equal every BF16 value in a fresh official
plain-full-row control, while differing from the normative cached branch
(maximum hidden delta 0.0625). Official LM/head execution on the identical
native hidden differs from the native head by only 5.96e-8. This locates a
remaining boundary before LM feature consumption; the first differing internal
backbone operator has not been traced. A separate branch-topology stage is
needed, with the original oracle and gates retained.

The real worker completed all-type warmup before readiness, served the original
complete request as its first request, and matched both the direct core replay
and Rust frontend results exactly except timing. Malformed JSON (400), invalid
questions/model and the reference LM budget (422), and body limit (413) matched
through both HTTP paths; a valid request after errors retained the same result.
These serving checks do not pass the numerical gate. Other GPUs, Metal,
reference reuse/Graphs, and performance remain unverified.

## Tests

The CPU tests need only the committed fixtures; the GPU tests need the export and the
CUDA library, as absolute paths:

```sh
cargo test --locked -p omni-omnijev-native
OMNIJEV_EXPORT=$PWD/omnijev-export OMNIJEV_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
    cargo test --release --locked -p omni-omnijev-native --test worker -- --ignored --test-threads=1 --skip reference
QWEN3_5_CHECKPOINT=$PWD/omnijev-export CUA_S1_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
    cargo test --release --locked -p omni-qwen3-5-native --test readout -- --ignored --test-threads=1 --skip reference
```

These commands select the merged worker's tests. The separate unmerged one-shot
tests require their own artifact and Graph-off settings; see the
[preparation recipe](README.md#separate-unmerged-reference-artifact).

The direct OmniJev window-Graph regression selects eager and Graph models without
changing process-wide environment variables. It compares raw FP32 head bits and
answers (excluding timing fields), requires actual capture and warm replay, and
covers same-layout new pixels, changed questions/options, geometry and scratch
growth, returning to an earlier request and malformed-input recovery:

```sh
OMNIJEV_EXPORT=$PWD/omnijev-export OMNIJEV_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
    cargo test --release --locked -p omni-omnijev-native --lib window_graph_tests \
    -- --ignored --test-threads=1 --nocapture
```

Also repeat the `--test worker` command above with `CUA_S1_GRAPH=0` and `1` to
exercise both modes through the production loader. Qualification applies to the
tested model, GPU and CUDA library combination; recheck parity on another stack.

## Comparison with the reference

[`validate.py`](validate.py) runs the reference and the worker on the same images and
questions (`bench/speed_bench.py`'s twelve by default) and compares every answer. The
reference step needs the reference environment from [the preparation
recipe](README.md#reference-environment) and a GPU; the other two need only Python:

```sh
.venv-omnijev/bin/python recipe/omnijev/validate.py reference --reference OmniJev \
    --checkpoint omnijev-v1.1 --base qwen3.5-4b --images images --out reference.json
python3 recipe/omnijev/validate.py native --url http://127.0.0.1:8000 --images images \
    --against reference.json --out native.json
python3 recipe/omnijev/validate.py compare reference.json native.json
```

[`gate_questions.json`](gate_questions.json) adds eight harder questions for `--questions`:
options that differ in one token or share a long start, a Choice where no option fits, a
32-option Choice, a 10-level Score, and plain and region Noul.

The historical comparison and latency figures below use the legacy vision path.
On one RTX 6000 Ada (sm_89, CUDA 13.2), with the reference's seven v1.1 demo stills and
both question sets (140 answers), the input-token counts equal the reference's. The
largest probability difference per answer, against the reference in float32 (`--fp32`)
as the common baseline:

| Run | Median | p90 | Max | Decisions unlike the float32 reference |
| --- | ---: | ---: | ---: | ---: |
| Reference, BF16 | 0.019 | 0.074 | 0.188 | 2 |
| Worker, the stills as PNG | 0.014 | 0.073 | 0.209 | 2 |
| Worker, the stills as JPEG | 0.015 | 0.071 | 0.208 | 3 |

The PNG files carry the RGB values Pillow decodes from the JPEG files, so the reference
sees the same pixels in both runs. Merging the LoRA into BF16 (`--merge`) moves the
reference by at most 0.029, and running its rows one at a time by at most 0.019, on the
first set. Most of the rest comes from the vision tower: in BF16, the worker's and the
reference's image features both differ from float32 by about 14% (relative L2), and from
each other by 6%; given the reference's image features, the worker's question and option
states are within 1.1% of the reference's. Of the worker's decisions that differ, one (`web`,
`one_token`, float32 margin 0.19) differs for the BF16 reference too; the others have a
float32 margin under 0.03, except `video`'s speed_bench question 9 in the PNG run (0.18),
which also changes between the worker's PNG and JPEG runs.

Latency on the same GPU, with a 1600×900 still: the worker with one client, and the
reference with one stream (`bench/speed_bench.py`'s first 1, 3, 6 and 12 questions, and
Choices of 32 and 255 options). Both run the prefix once; the reference also batches a
question's options, which the worker runs one after another:

| Workload | Worker, median | Worker, requests per second | Reference, median |
| --- | ---: | ---: | ---: |
| 1 question | 0.16 s | 6.4 | 0.17 s |
| 3 questions | 0.21 s | 4.9 | 0.18 s |
| 6 questions | 0.35 s | 2.9 | 0.22 s |
| 12 questions | 0.66 s | 1.5 | 0.36 s |
| 32 options | 0.70 s | 1.4 | 0.25 s |
| 255 options | 5.1 s | 0.20 | 1.10 s |

With 8 or 16 clients the worker's throughput stays about the same and the median latency
grows with the queue: 5.0 s and 10.1 s for 12 questions, and 40.6 s for 255 options with
8 clients.
