# Vela 2.0 0.8B native library

The native implementation for
[vllm-sr/Vela-2.0-0.8B](https://huggingface.co/vllm-sr/Vela-2.0-0.8B):
the exported-bundle gate and the wire contract, mirrored from the
pinned runtime. This is a library; serving runs through the
[reference recipe](reference.md)'s Python worker.

Run the commands from the repository root with the same `.venv` and
pinned package as the reference recipe. The export needs about 6 GB of
free disk next to the package; the checks need no GPU.

## Export the native bundle

~~~sh
PYTHONPATH=src .venv/bin/python recipe/vela/export_native.py \
  --package weights/vela-2.0-0.8b --out weights/vela-2.0-0.8b-native
~~~

The export re-runs the reference worker's package gate first
(provenance), then writes `weights/vela-2.0-0.8b-native/`: the backbone
re-keyed `backbone.` → `model.` with **per-tensor dtypes preserved**
(204 BF16 Linear/conv1d, 116 F32 embed/norms/A_log/dt_bias — the
reference's own mixed layout), the 37 F32 heads in `heads.safetensors`
(including the side-loaded broad span head), `backbone_config` at the
config root (the shared qwen3_5 `Config::load` shape), calibration and
tokenizer, and `vela_native_manifest.json` pinning the source
revision, the gate's file table, the tensor inventory and every file's
SHA-256.

## Checks

~~~sh
cargo test -p omni-vela-native
cargo fmt --all --check
cargo clippy -p omni-vela-native --all-targets -- -D warnings
~~~

15 tests, no weights, no GPU: the wire contract against the pinned
runtime's 422/400 bodies and the bundle manifest gate against a
synthetic bundle.

## Scope and limits

- Implemented: the bundle export + manifest gate and the wire
  admission.
- Not implemented: token-level processing, the F32 heads, scoring,
  backbone execution, and an HTTP worker. No performance claims are
  made.
