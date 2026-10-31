# Vela 2.0 0.8B native library

The native implementation for
[vllm-sr/Vela-2.0-0.8B](https://huggingface.co/vllm-sr/Vela-2.0-0.8B):
the exported-bundle gate, the wire contract, and the token-level
processing — all mirrored from the pinned runtime and checked against
frozen fixtures. This is a library; serving runs through the
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
SHA-256. The set is staged in a sibling directory and moved into place
with one rename, so a run killed mid-write can no longer strand a
partial bundle under `--out`; a non-empty target directory is refused
and the staging remnant is cleared by the next run.

## Freeze the token-plan fixtures (opt-in, needs the package)

~~~sh
PYTHONPATH=src .venv/bin/python recipe/vela/export_token_plan.py \
  --model weights/vela-2.0-0.8b --out tests/vela/fixtures/
~~~

The token plans record what the pinned runtime feeds the backbone (the
predict mapping, the per-row decomposition, the rendered `p_ids`,
blocks and readout positions, truncation, a windowed long-span case).
The example-request token plan totals 1,041 tokens, matching the
oracle fixture's `usage.input_tokens`.

## Checks

~~~sh
cargo test -p omni-vela-native
cargo fmt --all --check
cargo clippy -p omni-vela-native --all-targets -- -D warnings
VELA_NATIVE_BUNDLE="$PWD/weights/vela-2.0-0.8b-native" \
  cargo test -p omni-vela-native --test vela_native_processing -- --ignored --nocapture
~~~

27 tests run everywhere with no weights and no GPU: the wire contract
against the pinned runtime's 422/400 bodies, the bundle manifest gate,
and the predict-mapping/plan parity on every frozen fixture. The
`#[ignore]`d render check (run it with `-- --ignored`, as above)
replays the five token plans through the Rust processing
field-for-field. For a field-level diff of one case:

~~~sh
VELA_NATIVE_BUNDLE="$PWD/weights/vela-2.0-0.8b-native" \
  cargo run -p omni-vela-native --example parity -- example-request
~~~

## Scope and limits

- Implemented: the bundle export + manifest gate, the wire admission,
  and the token-level processing (state mapping, predict mapping with
  the validation triples, word units, span-head dispatch, the DEV2.0
  rendering, the truncation budget, window rows).
- Not implemented: the F32 heads, scoring and span decoding, backbone
  execution, and an HTTP worker. No performance claims are made.
