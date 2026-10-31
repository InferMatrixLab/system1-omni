# Vela 2.0 0.8B native library

The native implementation for
[vllm-sr/Vela-2.0-0.8B](https://huggingface.co/vllm-sr/Vela-2.0-0.8B):
the exported-bundle gate, the wire contract, the token-level
processing, the F32 heads and scoring, and backbone execution on CUDA
through the shared qwen3_5 crate — all mirrored from the pinned
runtime and checked against frozen fixtures. This is a library; the
HTTP worker is the [reference recipe](reference.md)'s Python worker.

Run the commands from the repository root with the same `.venv` and
pinned package as the reference recipe. The export needs about 6 GB of
free disk next to the package; the parity checks need no GPU except
the executor check.

## Export the native bundle

~~~sh
PYTHONPATH=src .venv/bin/python recipe/vela/export_native.py \
  --package weights/vela-2.0-0.8b --out weights/vela-2.0-0.8b-native \
  --bf16-out weights/vela-2.0-0.8b-native/bf16
~~~

The backbone that executes on CUDA is a uniform-BF16 re-save (every
backbone tensor rounds once at export); the reference's own GPU setting
runs the backbone under bf16 autocast but keeps norm weights and GDN
gate math in FP32, so the uniform export is a slightly lower-precision
profile than either the CPU-FP32 recording or the reference GPU path.
The Vela heads stay F32 on the faithful tensors.

The export re-runs the reference worker's package gate first
(provenance), then writes `weights/vela-2.0-0.8b-native/`: the backbone
re-keyed `backbone.` → `model.` with **per-tensor dtypes preserved**
(204 BF16 Linear/conv1d, 116 F32 embed/norms/A_log/dt_bias — the
reference's own mixed layout), the 37 F32 heads in `heads.safetensors`
(including the side-loaded broad span head), `backbone_config` at the
config root (the shared qwen3_5 `Config::load` shape), calibration and
tokenizer, and `vela_native_manifest.json` pinning the source
revision, the gate's file table, the tensor inventory and every file's
SHA-256. `--bf16-out` additionally writes the uniform-BF16 backbone
(`model.safetensors`) that the shared loader consumes. Both sets are
staged in sibling directories and moved into place with one rename, so
a run killed mid-write can no longer strand a partial bundle under
`--out` or `--bf16-out`; a non-empty target directory is refused and
the staging remnant is cleared by the next run.

## Freeze the fixtures (opt-in, needs the package)

~~~sh
PYTHONPATH=src .venv/bin/python recipe/vela/export_token_plan.py \
  --model weights/vela-2.0-0.8b --out tests/vela/fixtures/
PYTHONPATH=src .venv/bin/python recipe/vela/export_head_inputs.py \
  --model weights/vela-2.0-0.8b --out tests/vela/fixtures/
~~~

The token plans record what the pinned runtime feeds the backbone (the
predict mapping, the per-row decomposition, the rendered `p_ids`,
blocks and readout positions, truncation, a windowed long-span case);
the head inputs record the hidden states its F32 heads consume, the
heads' logits and the calibrated predict answers. The example-request
token plan totals 1,041 tokens, matching the oracle fixture's
`usage.input_tokens`.

## Checks

~~~sh
cargo test -p omni-vela-native
cargo fmt --all --check
cargo clippy -p omni-vela-native --all-targets -- -D warnings
VELA_NATIVE_BUNDLE="$PWD/weights/vela-2.0-0.8b-native" \
  cargo test -p omni-vela-native --test vela_native_processing \
    --test vela_native_heads --test vela_native_response -- --ignored --nocapture
VELA_NATIVE_BUNDLE="$PWD/weights/vela-2.0-0.8b-native" \
VELA_CUDA_LIB=/path/to/libqwen3_5_cuda.so \
  cargo test -p omni-vela-native --test vela_native_executor -- --ignored --nocapture
~~~

27 tests run everywhere with no weights and no GPU: the wire contract
against the pinned runtime's 422/400 bodies, the bundle manifest gate,
and the predict-mapping/plan parity on every frozen fixture. The four
opt-in checks (behind `-- --ignored`) replay the fixtures through the
Rust implementation:

- render parity — the token plans match field-for-field (5/5);
- head parity — head-logit drift ≤ 3.8e-5, predict-answer drift
  ≤ 7.3e-7, discrete picks and span structures equal (5/5);
- response parity — the finished SystemOne envelopes match the frozen
  oracles within 1.5e-6 (4/4);
- executor parity — the backbone on CUDA replays the recorded
  reference logits with a worst drift of 0.285 (the uniform-BF16
  backbone against the CPU-FP32 reference; gate 0.35, tighten-only via
  `VELA_GPU_LOGIT_GATE`: a wider value clamps back down), 4/4 small
  cases.

For a field-level diff of one token-plan case:

~~~sh
VELA_NATIVE_BUNDLE="$PWD/weights/vela-2.0-0.8b-native" \
  cargo run -p omni-vela-native --example parity -- example-request
~~~

## Recorded hardware evidence

The executor check above is the first recorded execution of the shared
qwen3_5 CUDA kernels on sm_120 (RTX 5060 Laptop, CUDA 13.1; build with
`src/backends/cuda/qwen3_5/build.sh <out> 120`). The bf16 backbone
takes about 3 GB of GPU memory; the executor check runs the four small
frozen cases in about three minutes on that device.

## Scope and limits

- Implemented: bundle export + manifest gate, wire admission,
  token-level processing, F32 heads, scoring and span decoding, the
  SystemOne response finishing, and backbone execution through the
  shared crate (full-sequence per readout position, packed batches).
- Not implemented: the HTTP worker around the library, long-span GPU
  parity on the windowed case, and measured serving latency. No
  performance claims are made.
