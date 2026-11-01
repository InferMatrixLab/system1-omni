# Native multimodal language Graph regressions

These private-item tests live under root `tests/` and are wired into the
`omni-qwen3-5-native` library with `#[cfg(test)]`. Cargo compiles and discovers
all three cases without a generated source copy, a preparation script or manifest
changes. The wiring is excluded from release builds.

Ordinary `cargo test --workspace --locked` leaves the three cases ignored:
standard CI has no CUDA device or model weights. It still compiles their bodies.
Actual execution requires a real NVIDIA CUDA device, a supported BF16 language
checkpoint with `image_token_id` and at least 2048 positions, and the Qwen native
CUDA library. The base checkpoint's F32 tensors need the existing native export;
see [the multimodal language exporter](../../recipe/cua_s1/export_multimodal_language.py)
and [native build instructions](../../recipe/cua_s1/native.md).

```sh
export CUA_S1_TEST_MODEL=/absolute/path/to/merged-bf16-language
export CUA_S1_CUDA_LIB=/absolute/path/to/libqwen3_5_cuda.so
python3 tests/qwen3_5/run_graph_regressions.py \
  --log /tmp/native-mm-graph-gpu.log --result /tmp/native-mm-graph-result.json
```

The runner uses locked, offline, release Cargo execution. Fetch dependencies
beforehand (`cargo fetch --locked`) when a populated Cargo cache/vendor directory
is unavailable. Tests run serially in one process, with `CUA_S1_GRAPH_TRACE`
unset, to avoid shared CUDA device/API state interference. It requires exactly
three passing cases and one unconditional capture-failure diagnostic per mode.
It returns nonzero on missing hardware/weights, Cargo failure or a failed gate;
none of those are treated as skips.

Coverage:

- Capture misses and replay agree exactly with finite eager hidden states;
  changed tokens, BF16 image rows and distinct T/H/W positions are uploaded;
  text/multimodal interleave and scratch growth permit fresh captures. Text
  singleton `[4]` and packed `[1, 3]` / `[3, 1]` keep distinct ordered cache
  keys despite equal total tokens, and replay with changed token IDs.
- 65 lengths exercise the separate 64-entry multimodal FIFO, unchanged scratch
  addresses, hits without insertion-order refresh, eviction/recapture and a
  surviving text replay.
- A real CUDA recording closure returns an injected error. The production
  capture-result handler preserves the completed eager result, clears both
  caches, disables Graph, emits diagnostics and allows repeated public eager
  forwards in both modes. This does not simulate every CUDA failure class.

These are executor regressions, not a screenshot-encoder or agent-accuracy
benchmark. No latency, capture-cost or memory claim is made by these tests.

## Prefix-bidirectional full-row baseline

The optional `Model::forward_prompt_prefix_bidirectional` entry consumes one
complete state+question row. State rows read only state keys; question rows
read that state and their own question. The semantic split must be inside
the row and is not rounded. GDN remains causal; Graph, LoRA, reference numerics,
media, prefix continuation, branch batching and an HTTP worker are outside this
entry's supported scope. It is a correctness baseline, not a speed optimization.

Build the backend for the actual reserved device (`80` for the A800 used in
qualification), then run the kernel checks:

```sh
src/backends/cuda/qwen3_5/build.sh /absolute/backend-output 80
export CUA_S1_CUDA_LIB=/absolute/backend-output/libqwen3_5_cuda.so
cargo test --release --locked -p omni-qwen3-5-native --test kernels \
  prefix_bidirectional -- --ignored --test-threads=1 --nocapture
```

The dense FP64 oracle covers unaligned state lengths, GQA, strided V and
nonzero gates. The fixed gate is the existing attention test's 0.016 maximum
row-normalized absolute error, including the existing BF16 gating policy.
A two-layer test ensures changed question contents do not affect state bits,
future state tokens influence earlier state outputs, and repeated rows agree.
Invalid dimensions use valid allocations; missing buffers are checked separately.

For a small full-layer-loop oracle, use Python with `torch==2.12.1+cu130`,
`transformers==5.17.0` and their dependencies, on the same GPU. The generator
initializes a deterministic, all-BF16, three-layer Qwen model with full/linear/
full attention and uses an independent dense boolean mask. It saves roughly
a few MB of generated weights and reference arrays outside the checkout; no
pretrained model download is needed. The native replay fixes the normalized
hidden-state maximum error gate at 0.03, repeats an earlier row and checks
that a subsequent causal call is bitwise unchanged.

```sh
python tests/qwen3_5/prefix_bidir_reference.py /absolute/new-oracle-directory
export QWEN_PREFIX_BIDIR_FIXTURE=/absolute/new-oracle-directory/reference.json
export QWEN_PREFIX_BIDIR_OUTPUT=/absolute/native-replay.json
cargo test --release --locked -p omni-qwen3-5-native --test prefix_bidir \
  -- --ignored --test-threads=1 --nocapture
```

The output variable is optional and preserves raw hidden states when set.
The synthetic numerical gate is not a D3 decision/probability gate. An external
full-checkpoint replay needs its own frozen inputs, artifact identity and model
acceptance. Standard workspace checks compile these tests but leave GPU cases
ignored; `graph_loading` checks missing capability, unsupported graph/adapter
profiles and input rejection without a GPU.
