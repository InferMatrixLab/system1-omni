# Valen-Decision-1.5 reference worker

This recipe serves a released Valen-Decision-1.5 checkpoint through the Python
reference worker and the existing Rust frontend. The model-side implementation
is under [`src/models/valen_decision_1_5/`](../../src/models/valen_decision_1_5/),
and the worker is
[`src/frontend/valen_decision_1_5.py`](../../src/frontend/valen_decision_1_5.py).
It is independent of the [Valen-Preview-0923 recipe](../valen/reference.md).

Run the commands from the repository root on Linux with an NVIDIA GPU and
Python 3.12. The validation records below were collected on one NVIDIA H200
for the 0.8B and 2B sizes. Other sizes and hardware are not covered by these
measurements.

## Scope

The validated slice is:

- model ids `valen-decision-1.5-0.8b` and `valen-decision-1.5-2b`;
- the Python reference worker behind the Rust frontend;
- text state (a non-empty string; objects and arrays are serialized to JSON
  text) or one inline PNG or JPEG image in `state.image`;
- textual instructions and criteria for `choice`, `noul`, and `score`
  questions, freely mixed in one request against one shared state;
- BF16 CUDA inference, with recorded reference parity against the checkpoint's
  own `model.predict` for all three question types on text and image state.
  The in-process comparison was also run in FP32 on CUDA.

The worker also loads the pinned 4B, 9B and 27B checkpoints
(`valen-decision-1.5-4b`, `-9b`, `-27b`). No serving or parity run is recorded
for them here.

The worker does not implement `bbox` questions, video, multiple images, remote
image URLs, or a native Rust/CUDA executor. `bbox` is rejected because the
[frontend contract](../../src/frontend/README.md#http-interface) defines
`choice`, `score` and `noul` decisions only; see the
[worker contract](../../src/models/valen_decision_1_5/README.md#unsupported-bbox).

The protocol limits requests to an 8 MiB JSON body, one PNG/JPEG image of at
most 4 MiB, 2048 pixels per side, 1,048,576 pixels total, and a 200:1 maximum
aspect ratio. Text state is limited to 16,384 characters, the same limit as
combined instruction and criterion text. A request may contain at most 8
questions, 255 named candidates per `choice` question, and an ordered list of
2 to 10 level descriptions per `score` question. The whole request compiles to
one sequence of at most 16,384 tokens.

## Architecture and ownership

~~~text
Rust frontend
  -> Python worker
     -> prepare: validate, materialize the image, compile one shared sequence
     -> execute: Qwen3.5 backbone with block-structured attention, decision head
     -> finish: softmax, typed answers, confidence, usage, response
~~~

The ownership boundary follows
[`docs/architecture.md`](../../docs/architecture.md):

- `pinned.py` owns the release table and checkpoint verification.
- `protocol.py` owns wire validation and image limits.
- `preprocess.py` owns media materialization, the SystemOne record and the
  checkpoint's compiler.
- `engine.py` owns weights, forward, the learned head, device state, and warmup.
- `postprocess.py` owns probability normalization, typed answers, confidence,
  usage, and response reconstruction.
- `frontend.valen_decision_1_5` owns HTTP lifecycle and orchestration.
- The Rust frontend owns transport and forwarding, not model state.

Unlike the Preview worker, this worker needs no Valen source checkout and no
separate Qwen base: the compiler and model code are files of the checkpoint,
and the weights are complete.

## Pinned artifacts

| Size | Checkpoint | Revision | Weights |
| --- | --- | --- | --- |
| 0.8B | [Valen-Team/Valen-Decision-1.5-0.8B](https://huggingface.co/Valen-Team/Valen-Decision-1.5-0.8B) | `e079395fe9b8e0763b384af403e636e91c1a1559` | 1.7 GB |
| 2B | [Valen-Team/Valen-Decision-1.5-2B](https://huggingface.co/Valen-Team/Valen-Decision-1.5-2B) | `1b890606578ba0f40bb70ed46891ab65f778e76d` | 4.5 GB |
| 4B | [Valen-Team/Valen-Decision-1.5-4B](https://huggingface.co/Valen-Team/Valen-Decision-1.5-4B) | `9b0cacc976d7fb72f252ee1adb1564f5f15e462c` | 9.1 GB |
| 9B | [Valen-Team/Valen-Decision-1.5-9B](https://huggingface.co/Valen-Team/Valen-Decision-1.5-9B) | `5a512117ebc0f2b69f311becfcf47cdef8abf568` | 16.8 GB |
| 27B | [Valen-Team/Valen-Decision-1.5-27B](https://huggingface.co/Valen-Team/Valen-Decision-1.5-27B) | `9f1b90bcb6841d9a4a44acfd84c6ecbd4c18fec6` | 52.2 GB |

The file digests for these revisions are in
[`pinned.py`](../../src/models/valen_decision_1_5/pinned.py). The repositories
report Apache-2.0 license metadata. Re-check the metadata and update the
digest table when changing a revision.

## Install and prepare the artifacts

PyTorch wheels must match the host CUDA runtime. The validated environment
used the CUDA 12.6 build:

~~~sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.14.1 torchvision
.venv/bin/python -m pip install -r recipe/valen_decision_1_5/requirements.txt
~~~

Download one pinned checkpoint. The examples use 0.8B:

~~~sh
export VALEN_CKPT="$PWD/weights/Valen-Decision-1.5-0.8B"
.venv/bin/hf download Valen-Team/Valen-Decision-1.5-0.8B \
  --revision e079395fe9b8e0763b384af403e636e91c1a1559 --local-dir "$VALEN_CKPT"
~~~

The worker verifies the directory against the pinned digests each time it
starts. To run the same check alone:

~~~sh
PYTHONPATH=src .venv/bin/python -c '
import sys
from models.valen_decision_1_5.pinned import verify_checkpoint
release = verify_checkpoint(sys.argv[1]).release
print("checkpoint verification: PASS", release.repo, release.revision)
' "$VALEN_CKPT"
~~~

## Start the worker

The worker verifies the checkpoint, loads it, and performs a real image and
text warmup with all three question types before binding port 8000:

~~~sh
PYTHONPATH=src .venv/bin/python -m frontend.valen_decision_1_5 \
  --checkpoint "$VALEN_CKPT" --device cuda --dtype bf16 --port 8000
~~~

`--dtype bf16` is the precision the checkpoints are stored in and the one the
released Transformers quickstart loads with `dtype="auto"`; the decision head
runs in FP32 either way. `--no-verify-weights` skips hashing the weight shards
at startup.

~~~sh
curl -sS -i http://127.0.0.1:8000/health
~~~

Validated response:

~~~json
{"status": "ready", "model": "valen-decision-1.5-0.8b", "modality": "multimodal"}
~~~

## Example requests

The fixtures name the 0.8B id. For another size, change `model` to that
size's id.

| Fixture | State | Questions |
| --- | --- | --- |
| [`example-request.json`](example-request.json) | 64×64 inline PNG | one `choice` |
| [`example-text-request.json`](example-text-request.json) | text | one `choice` |
| [`example-noul-request.json`](example-noul-request.json) | text | two `noul`, one with a rubric |
| [`example-score-request.json`](example-score-request.json) | text | one `score` |
| [`example-mixed-request.json`](example-mixed-request.json) | 64×64 inline PNG | `choice`, `noul` and `score` |

~~~sh
curl -sS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @recipe/valen_decision_1_5/example-mixed-request.json
~~~

Recorded response on the H200 host, Valen-Decision-1.5-0.8B, BF16:

~~~json
{
  "model": "valen-decision-1.5-0.8b",
  "answers": {
    "color": {
      "type": "choice",
      "probabilities": {
        "red": 1.6140388726920612e-05,
        "blue": 0.9999795718728959,
        "green": 4.287738377119385e-06
      },
      "choice": "blue",
      "confidence": 0.9999693578093438
    },
    "square": {
      "type": "noul",
      "noul": 0.9577225335602313
    },
    "shapes": {
      "type": "score",
      "probabilities": {
        "0": 9.677036698434234e-05,
        "1": 0.031645553014193084,
        "2": 0.9682576766188227
      },
      "score": 1.9681609062518384,
      "legend": {
        "0": "There are no shapes.",
        "1": "There is exactly one shape.",
        "2": "There are two or more shapes."
      },
      "confidence": 0.9522413593777573
    }
  },
  "usage": {
    "input_tokens": 233,
    "output_tokens": 0
  },
  "internal_usage": {
    "compute_tokens": 233
  }
}
~~~

These are serving smoke tests, not accuracy results. The numbers follow the
[worker contract](../../src/models/valen_decision_1_5/README.md).

## Start and verify the Rust frontend

~~~sh
cargo build --release --locked
OMNI_JEV_BIND=127.0.0.1:8080 \
OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 \
  target/release/omni-jev
~~~

Send the same request to both ports and compare the bodies:

~~~sh
for name in request text-request noul-request score-request mixed-request; do
  curl -fsS http://127.0.0.1:8000/v1/systemone \
    -H 'Content-Type: application/json' \
    --data-binary @recipe/valen_decision_1_5/example-$name.json > /tmp/valen15-worker.json
  curl -fsS http://127.0.0.1:8080/v1/systemone \
    -H 'Content-Type: application/json' \
    --data-binary @recipe/valen_decision_1_5/example-$name.json > /tmp/valen15-frontend.json
  cmp /tmp/valen15-worker.json /tmp/valen15-frontend.json
done
python recipe/compare_with_backend.py --model valen-decision-1.5-0.8b
~~~

Recorded on the H200 host for the 0.8B and 2B sizes: all five fixtures returned
`200 application/json` from both ports with byte-identical bodies, and
`compare_with_backend.py` reported `PASS` for its three cases (`urgency`,
`refund`, `combined`).

## Error behavior

The request, response and error contract — status codes, `detail` bodies,
confidence, and usage counting — is the
[worker contract](../../src/models/valen_decision_1_5/README.md).

Worker responses are forwarded unchanged. Frontend-generated 502/504 responses
are frontend errors, not model responses.

Recorded rejections through the frontend (0.8B; the 2B run returned the same
status codes):

| Request | Status | Body |
| --- | --- | --- |
| a `bbox` question | 422 | `{"detail": "question 'where': type must be one of choice, noul, score"}` |
| `model: valen-preview-0923` | 422 | `{"detail": "model must be valen-decision-1.5-0.8b"}` |
| a `noul` rubric with only `true` | 422 | `{"detail": "question 'q': noul criteria must define true and false"}` |

## Validation

~~~sh
PYTHONPATH=src .venv/bin/python -m pytest tests/valen_decision_1_5 -q
PYTHONPATH=src .venv/bin/python -m compileall -q \
  src/models/valen_decision_1_5 src/frontend/valen_decision_1_5.py \
  tests/valen_decision_1_5 recipe/valen_decision_1_5
cargo fmt --all --check
cargo test -p omni-jev --test frontend --locked
cargo build --release --locked
~~~

Recorded results: `pytest` 78 passed (Python 3.12 on the GPU host and Python
3.11 on macOS), `compileall` clean, `cargo fmt --all --check` clean,
`cargo test -p omni-jev --test frontend --locked` 12 passed, and
`cargo build --release --locked -p omni-jev` succeeded (rustc 1.99.1, Linux
x86_64).

## Reference parity

[`compare_reference.py`](compare_reference.py) runs each fixture through the
worker pipeline and through the checkpoint's own `model.predict`, the entry
point of the released Transformers quickstart. The reference request carries
the fixture's questions verbatim, with text state as a plain string and the
image as a local file. The declared criteria, fixed before the runs, are:

- identical answers outside their numeric fields: question order, field order,
  candidate order, the chosen key, the score legend;
- equal `input_tokens`, `output_tokens` and `compute_tokens`;
- an absolute difference of at most `1e-6` on every probability (including
  `noul` P(true)), on the `score` expected level, and on confidences;
- a non-finite value on either side fails the case.

There are two modes.

In-process: both paths share one loaded model, so the comparison covers
request construction, the readout and probability normalization, not
checkpoint loading.

~~~sh
PYTHONPATH=src .venv/bin/python recipe/valen_decision_1_5/compare_reference.py \
  --checkpoint "$VALEN_CKPT" --device cuda --dtype bf16
~~~

Served: the worker side is an HTTP request to a running worker or frontend,
and the comparing process loads its own copy of the checkpoint for the
reference. This covers the complete serving path against an independently
loaded model; both copies must fit on the device.

~~~sh
PYTHONPATH=src .venv/bin/python recipe/valen_decision_1_5/compare_reference.py \
  --checkpoint "$VALEN_CKPT" --device cuda --dtype bf16 \
  --url http://127.0.0.1:8080
~~~

`--case` selects fixtures, `--body NAME=PATH` adds a request file, `--save`
writes every worker and reference response to a JSON file, and `--tolerance`
overrides the default.

Recorded on one NVIDIA H200 (140 GiB), CUDA 12.6, torch 2.14.1+cu126,
torchvision 0.29.1+cu126, transformers 5.20.2, on 13 November 2026. Each run
covers the five fixtures (`image`, `text`, `noul`, `score`, `mixed`); the
figures are the maxima over those cases. Every run reported decision
agreement and matching token accounting on every case.

| Size | Path | dtype | Max probability delta | Max answer delta | Max confidence delta |
| --- | --- | --- | ---: | ---: | ---: |
| 0.8B | served through the Rust frontend | BF16 | 1.11e-16 | 0 | 2.22e-16 |
| 0.8B | in-process | BF16 | 1.11e-16 | 0 | 2.22e-16 |
| 0.8B | in-process | FP32 | 2.22e-16 | 4.44e-16 | 3.33e-16 |
| 2B | served through the Rust frontend | BF16 | 1.11e-16 | 2.22e-16 | 0 |
| 2B | in-process | BF16 | 1.11e-16 | 2.22e-16 | 0 |
| 2B | in-process | FP32 | 1.11e-16 | 0 | 3.33e-16 |

The worker and the reference run the same forward implementation from the
checkpoint, so these deltas measure request construction, the readout and
response assembly, not an independent reimplementation of the model.

GPU memory with only the BF16 worker loaded: 2,595 MiB for 0.8B and 5,193 MiB
for 2B.

Verbatim `compare_reference.py` output for the `mixed` case, 0.8B, served
through the Rust frontend:

~~~text
case: mixed
reference model: Valen
worker model: valen-decision-1.5-0.8b
input_tokens: 233 == 233
output_tokens: 0 == 0
compute_tokens: 233 == 233
max probability delta: 1.1102230246e-16
max answer delta: 0.0000000000e+00
confidence delta: 0.0000000000e+00
decision agreement: PASS
probability tolerance <= 1e-06: PASS
answer tolerance <= 1e-06: PASS
confidence tolerance <= 1e-06: PASS
token accounting: PASS

compare_reference: ok
~~~

## Troubleshooting and limits

- The checkpoint directory must match one pinned revision file for file;
  a changed or missing runtime, tokenizer or weight file stops the worker
  before any checkpoint code is imported.
- One PNG or JPEG data URL is accepted. The image is materialized into a
  request-scoped temporary directory because the checkpoint's compiler reads
  media from a local path.
- All questions of a request share one sequence and one forward pass, so an
  answer can depend on the other questions sent with it.
- Without the optional `causal_conv1d` and `flash-linear-attention` packages
  Transformers runs its PyTorch reference implementation of the
  linear-attention layers. The records below were collected that way.
- `/health` is exposed only after model loading and real warmup complete.
- The worker answers one request at a time and returns `503` while busy.
- The smoke responses and memory observations are host-specific; they are not
  general accuracy or performance claims.
- `bbox`, video and native execution remain outside this recipe.
