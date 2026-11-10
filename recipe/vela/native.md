# Vela 2.0 decoder native library

The native implementation for the Vela 2.0 decoder members
([0.8B](https://huggingface.co/vllm-sr/Vela-2.0-0.8B), 4B, 9B — one
program, member-parameterized): the exported-bundle gate, the wire
contract, the token-level processing, the F32 heads (widths derived
from the tensors' shapes) and scoring, and backbone execution on CUDA
through the shared qwen3_5 crate — all mirrored from the pinned
runtime and checked against frozen fixtures. This is a library; the
HTTP worker is the [reference recipe](reference.md)'s Python worker.
The recorded fixtures and parity evidence below are the 0.8B's (CPU
FP32 recordings). Historical 4B/9B runs on an RTX 4090 48 GB (sm_89,
CUDA 13.2 toolkit) recorded exact-match plans and native executor
replays with worst drifts 0.2271 and 0.2281 respectively, but their
head/response checks used the former span-structure exemption. They are
not current full-parity evidence.

Those 4B/9B head/response measurements used the former exemption for
mismatching span structures and do not establish full parity. The
9B CJK address merge reported in [PR #195](https://github.com/InferMatrixLab/system1-omni/pull/195#issuecomment-6516691352)
was a decoding error: a rejected valid word must end the current span.
The tests now retain all spans and require their counts, labels and
offsets to match; logit comparisons also reject non-finite values and
dimension mismatches. The numeric ceilings are unchanged. Full 4B/9B
head/response replay under these assertions remains unverified until
rerun with the corresponding exported bundles.

The current 0.8B SM89 replay uses one `forward_fixed` call per readout;
this keeps the GEMM rounding independent of readout grouping. On the
RTX 4090 48 GB profile below, all four strict executor cases pass the
unchanged 0.35 gate: example-request 0.1487, set-threshold 0.0152,
score 0.0158 and CJK 0.3297.

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
(`model.safetensors`) that the shared loader consumes, into the
bundle's own `bf16/` subdirectory. The whole set — bf16 included — is
staged in a sibling directory and moved into place with one rename, so
a run killed mid-write can no longer strand a partial bundle under
`--out`; a non-empty target directory is refused and the staging
remnant is cleared by the next run.

The export ends by printing the manifest's own SHA-256. The load-side
gate requires it as an anchor (`Bundle::load`/`Processor::load` take it
as `pinned_manifest_sha256`; the opt-in checks read
`VELA_NATIVE_MANIFEST_SHA256`), so the manifest's file table is checked
against a digest fixed outside the bundle directory — rewriting the
manifest after swapping a file no longer passes the gate. `files_sha256`
keys must also be plain relative names inside the bundle and must not
cross a symlink.

## Freeze the fixtures (opt-in, needs the package)

~~~sh
PYTHONPATH=src .venv/bin/python recipe/vela/export_token_plan.py \
  --model weights/vela-2.0-0.8b --out tests/vela/fixtures/
PYTHONPATH=src .venv/bin/python recipe/vela/export_head_inputs.py \
  --model weights/vela-2.0-0.8b --out tests/vela/fixtures/
~~~

Both scripts pass `--model` through the worker's pin gate before any
package code is imported, and stamp the recorded request with the
member's served name. Only the 0.8B binaries are committed; the 4B/9B
`.safetensors` are re-exported on demand (they are gitignored) —
point `--model` at the pinned `Vela-2.0-4B`/`Vela-2.0-9B` package and
re-run `export_head_inputs.py` (add `--device cuda` to record the
bf16-autocast numbers the member-calibrated gates expect) before the
opt-in head/response/executor replays below.

The token plans record what the pinned runtime feeds the backbone (the
predict mapping, the per-row decomposition, the rendered `p_ids`,
blocks and readout positions, truncation, a windowed long-span case).
The example-request token plan totals 1,041 tokens, matching the
oracle fixture's `usage.input_tokens`. The head inputs record the
hidden states the F32 heads consume, the heads' logits and the
calibrated predict answers. Both recorders run the worker's own
`verify_package()` gate before the package is imported, so the frozen
fixtures always come from the pinned package. Both exporters import
and initialize the runtime with bytecode writing off, so the verified
package stays reusable: a later `verify_files()` gate run or worker
startup is not poisoned by a `__pycache__` an export wrote.

## Checks

~~~sh
cargo test -p omni-vela-native
cargo fmt --all --check
cargo clippy -p omni-vela-native --all-targets -- -D warnings
VELA_NATIVE_BUNDLE="$PWD/weights/vela-2.0-0.8b-native" \
  VELA_NATIVE_MANIFEST_SHA256=<the digest the exporter prints> \
  cargo test -p omni-vela-native --test vela_native_processing \
    --test vela_native_heads --test vela_native_response -- --ignored --nocapture
VELA_NATIVE_BUNDLE="$PWD/weights/vela-2.0-0.8b-native" \
  VELA_NATIVE_MANIFEST_SHA256=<the digest the exporter prints> \
  VELA_CUDA_LIB=/path/to/libqwen3_5_cuda.so \
  cargo test -p omni-vela-native --test vela_native_executor -- --ignored --nocapture
~~~

56 tests run everywhere with no weights and no GPU: the wire
contract against the pinned runtime's 422/400 bodies (one known
residual: a non-finite JSON number answers the generic 400 message,
where the Python worker names the value), the bundle manifest gate
(including the repository pin for the three verbatim runtime files),
the truncation budget, and the predict-mapping/plan parity on every
frozen fixture of every member. The four
opt-in checks (behind `-- --ignored`) replay the fixtures through the
Rust implementation:

- render parity — the token plans match field-for-field (5/5);
- head parity — head-logit drift ≤ 3.8e-5, predict-answer drift
  ≤ 7.3e-7, discrete picks and span structures equal (5/5; the
  windowed long-span case's reference answer has no spans, so its
  window-aggregation arithmetic is pinned only through the per-window
  head logits, not the aggregated answer);
- response parity — the finished SystemOne envelopes match the frozen
  oracles within 1.5e-6 (4/4);
- executor parity — the backbone on CUDA replays the recorded
  reference logits with a worst drift of 0.285 (the uniform-BF16
  backbone against the CPU-FP32 reference; gate 0.35, tighten-only
  via `VELA_GPU_LOGIT_GATE` — an unparseable value is an error and a
  wider one clamps back down), 4/4 small cases.

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
frozen cases in under a minute on that device, worst logit drift
0.2845 (CJK 0.2845, the threshold and score cases 0.0066).

Hardware evidence now covers one exact profile: RTX 4090 48 GB,
49,140 MiB, driver 610.57.04, CUDA toolkit 13.2, sm_89, with the
fixed-GEMM executor path. The four cases above pass and the largest
drift is 0.3297, below the unchanged 0.35 ceiling. An older independent
review profile (RTX 4090 24 GB, CUDA 13.0.88) failed the pre-fix path at
0.3912 on CJK while selected kernel checks passed; that exact 24 GB /
CUDA 13.0 profile has not been rerun after fixed-GEMM and remains
unqualified.

## Scope and limits

- Implemented: bundle export + manifest gate, wire admission,
  token-level processing, F32 heads, scoring and span decoding, the
  SystemOne response finishing, and backbone execution through the
  shared crate (full-sequence per readout position, fixed-GEMM replay).
- Not implemented: the HTTP worker around the library, long-span GPU
  parity on the windowed case, and measured serving latency. No
  performance claims are made.
