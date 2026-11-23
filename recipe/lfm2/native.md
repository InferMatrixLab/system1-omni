# LFM2.5-350M Rust preparation

The **omni-lfm2-native** CLI validates and prepares the Choice contract,
interprets supplied sequence scores, and inspects an F32 weight package. It runs
on CPU. The [Transformers recipe](README.md) supplies the runnable worker.
See the [model contract](../../src/models/lfm2/README.md) for API semantics and
the [executor boundary](../../src/models/lfm2/README.md#rust-preparation-and-executor-boundary)
for device state and readout ownership.

## Checkpoint and CPU export

Run from the repository root. Use an existing local snapshot of
[LiquidAI/LFM2.5-350M at the pinned revision](https://huggingface.co/LiquidAI/LFM2.5-350M/tree/9e6c6ccf47cd318696e137d381a7ded8fe4df09f)
under the upstream LFM Open License. The recipe requires its config, tokenizer,
tokenizer config, chat template, license and single BF16 safetensors file. The
[profile](../../src/models/lfm2/native/profile.json) pins each source file's SHA-256.
Use Python 3.12, NumPy 2.4.6, safetensors 0.8.0 and PyTorch 2.14.0 as in the
[reference lock](../../src/models/lfm2/requirements.lock.txt). The recorded CPU
export used the existing Q12 environment with GPU visibility empty. Allow about
1.5 GB of free output space and 3 GiB of available memory for export plus
verification. The output parent must already exist.

~~~sh
export CHECKPOINT=/absolute/path/to/pinned/snapshot
export ARTIFACT=/absolute/path/to/new/lfm2-f32
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python recipe/lfm2/export_f32.py "$CHECKPOINT" "$ARTIFACT"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python recipe/lfm2/export_f32.py "$CHECKPOINT" "$ARTIFACT" --verify
cargo build --locked --release -p omni-lfm2-native
./target/release/omni-lfm2-native inspect "$ARTIFACT"
~~~

Export widens BF16 bits into little-endian F32. An independent PyTorch CPU check
compares every F32 bit and checks finiteness before publishing the directory.
The manifest records the source profile and the byte size and SHA-256 of every
artifact file. The original config and tokenizer files are copied byte for byte.
The config's BF16 dtype describes the source snapshot. The manifest and
safetensors header describe the F32 package. The LM head aliases the input
embedding tensor, with one physical copy. An interrupted export leaves a staging
directory for inspection. Choose a new output path for a retry.

| Artifact inventory | Verified value |
| --- | --- |
| Manifest schema | omni-lfm2-f32-v1 |
| Weight file | weights.f32.safetensors |
| Unique tensors / elements | 148 / 354,483,968 |
| Weight file size | 1,417,952,352 bytes |
| Weight SHA-256 | 4757f9b07ce4eb8540130f40cd1607f0c953ef8d8f53768ddfb6b782db420fb8 |
| Manifest SHA-256 | 591a567ab5af3bb4d0d4e037599ab1fb2af6f3bd54018984325cfb2ffcc10cd4 |

Rust inspection checks the profile, file hashes and tensor inventory including
dtype, geometry and offsets. Keep weight packages outside the source tree.

## Prepare a request and replay captured scores

Use a public Choice request from the [frozen fixture](../../tests/lfm2/fixtures/q12_replay.json)
or supply your own. This example extracts a real request and its original
uncached F32 sequence totals. The finish step takes scores in the question and
candidate order emitted by prepare.

~~~sh
export BIN=./target/release/omni-lfm2-native
python - <<'PY'
import json
from pathlib import Path

with open("tests/lfm2/fixtures/q12_replay.json", encoding="utf-8") as handle:
    case = json.load(handle)["public"][0]
Path("request.json").write_text(json.dumps(case["request"]), encoding="utf-8")
scores = [question["plan"]["fields"][0]["scores"] for question in case["questions"]]
Path("scores.json").write_text(json.dumps(scores), encoding="utf-8")
PY
"$BIN" validate request.json
"$BIN" render request.json
"$BIN" prepare "$ARTIFACT/tokenizer.json" request.json
"$BIN" finish "$ARTIFACT/tokenizer.json" request.json scores.json
~~~

Render emits the literal prompt, field suffix and JSON candidate plus newline.
Prepare emits their independently encoded IDs in the same order, along with
unpadded usage, candidate targets and half-open continuation and full-sequence
logit intervals. A single request or score input can use standard input via the
path "-". Validation errors include the request status and detail on stderr.
Malformed-input text comes from the Rust JSON parser, which accepts Unicode
scalar text and bounds JSON nesting at 512 levels.

## Verify the preparation contract

~~~sh
cargo test --locked -p omni-lfm2-native -- --list
cargo test --locked -p omni-lfm2-native
LFM2_TOKENIZER_JSON="$ARTIFACT/tokenizer.json" LFM2_ARTIFACT_DIR="$ARTIFACT" \
  cargo test --release --locked -p omni-lfm2-native -- --ignored
~~~

The default suite replays frozen token fragments and F32 scores for five public
requests, six questions and 266 candidate branches, plus the original 17
internal schema cases. It verifies exact text hashes, order, readout intervals,
choice and usage. Same-score probabilities and confidence use the Python
verifier's 1e-6 absolute tolerance. Opt-in tests use the actual tokenizer and
inspect the full artifact. The
[fixture extractor](../../tests/lfm2/freeze_native_contract.py) consumes the
frozen captures and pinned tokenizer without running a forward. From the
repository root with the Q12 environment, reproduce it with
`python tests/lfm2/freeze_native_contract.py "$CAPTURE_ROOT" "$CHECKPOINT" replay.json`,
where `CAPTURE_ROOT` is the verified Q12 archive extraction. The
[profile file](../../src/models/lfm2/native/profile.json) identifies the
approved original public capture and its Python and RLCD source revisions.
The historical FP16 runs compare their respective profiles separately.

## Explicit Q12 Transformers reference

The approved numerical reference used an NVIDIA L20 (sm_89, driver 580.178.04)
with CPython 3.12.13, Torch 2.14.0+cu130, Transformers 5.17.0 and cuDNN 9.24.0.43.
It loaded the same BF16 checkpoint directly as CUDA FP32 and used eager
attention and PyTorch convolution. Run the original verifier from the repo
root in that environment with one allocated GPU visible, the pinned RLCD code
available locally and the pinned checkpoint already in the HF cache. The
standard worker still defaults to FP16.

~~~sh
HF_HUB_OFFLINE=1 python - .local/lfm/reference .local/lfm/q12.verify.json <<'PY'
import runpy
import sys
from importlib.util import find_spec

import torch

torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.allow_tf32 = False
assert torch.cuda.device_count() == 1
assert not torch.backends.cuda.matmul.allow_tf32
assert torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
assert torch.backends.cudnn.enabled
assert not torch.backends.cudnn.benchmark and not torch.backends.cudnn.deterministic
assert not torch.are_deterministic_algorithms_enabled()
assert not torch.is_autocast_enabled("cuda")
assert find_spec("causal_conv1d") is None and find_spec("fla") is None
reference, output = sys.argv[1:]
sys.path.insert(0, "recipe/lfm2")
sys.argv = ["recipe/lfm2/verify.py", "--reference-dir", reference,
            "--device", "cuda", "--dtype", "float32",
            "--batch-sizes", "1,8,16,32,64,all", "--output", output]
runpy.run_path(sys.argv[0], run_name="__main__")
PY
~~~

L20 executor acceptance requires exact choices, per-candidate absolute score
error below 0.15 against the independent, uncached F32 reference and the original
cache checks (atol 0.015, rtol 0.08, argmax and isolation). Actual native
execution, HTTP readiness and device limits are pending for #159.
