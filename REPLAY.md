# Reproduce the main experiments

Keep this evidence checkout separate from the production checkout. Set absolute
paths for the following shell variables. Run every GPU command under your
host's reservation mechanism; on the measured host use
`gpu run --gpu-ids 2 --nonblock --note 'PR196 replay' --timeout 2h -- bash replay.sh`.
Do not set CUDA_VISIBLE_DEVICES independently of the reservation.

```bash
EVIDENCE=/absolute/path/to/evidence-checkout
S1=/absolute/path/to/pr196-checkout
VJ=/absolute/path/to/vllm-jev
MODELS=/absolute/path/to/models
BASE_BIN=/absolute/path/to/original-native-bin
CAND_BIN=/absolute/path/to/optimized-native-bin
OUT=/absolute/path/to/new-results
PY="$VJ/.venv/bin/python"
```

Use original native `22b7717c9710de5ac5c6ccba51a6d6686dae90d9`, optimized source
`43ccbbcff99a94d5c6f744183b82e37853ed873c` (tree-identical to measured local
`95c0e70f2cdf2a72df763d0247ba7f20f8a9327d`), and vllm-jev
`487ff7765a47e48b20fc2d48c14314fb28feaa51`. The PR's later test-only revision
has the same production sources. Use Rust 1.99.1 / CUDA 13.0; Python 3.12.13,
vLLM 0.29.0, PyTorch 2.13.0, Transformers 5.10.4 and `httpx` in the vllm uv
virtual environment. The NVIDIA driver was 570.133.20. Record actual installed
versions when repeating; a different environment is a new measurement.

Follow `recipe/open_jev/native.md` from the source checkout to download and
merge the pinned base `c202236235762e1c871ad0ccb60c8ee5ba337b9a` and Open-Jev
checkpoint `47e966881e489511c0c7f5633a9e1960a676a551`. Expected model layout:

```text
$MODELS/system1-9b/              # pinned BF16 native export, capacity 4096
$MODELS/system1-9b-cap8192/      # identical weights/tokenizer/head; capacity 8192
$MODELS/vllm-workspace/checkpoint/ZefanCai/Open-Jev-9B/
```

The 8192-capacity export changes the `max_length` field in
`open_jev_export.json` only; model weight files can be symlinks. Inputs remain
unchanged and no candidate is truncated. The wider capacity avoids the
reference backend's separately diagnosed 4096-capacity boundary stall.
See `validation/085/validation.json` for exact native export file hashes.

Build each native revision in its own checkout and Cargo target directory:

```bash
# Repeat with checkout/bin/target for original and optimized respectively.
cd "$S1"
CARGO_TARGET_DIR=/absolute/path/to/candidate-target \
  cargo build --release --locked -p omni-open-jev-native -p omni-jev
mkdir -p "$CAND_BIN"
cp /absolute/path/to/candidate-target/release/{omni-open-jev-native,omni-jev} "$CAND_BIN/"
src/backends/cuda/qwen3_5/build.sh "$CAND_BIN" 90
```

Configure NVCC/CPATH/LIBRARY_PATH for the installed CUDA SDK. On the measured
host nvcc was `/usr/local/cuda/bin/nvcc`, CPATH was the uv environment's
`site-packages/nvidia/cu13/include`, and LIBRARY_PATH pointed to the installed
cuBLAS libraries. Use the corresponding library with each revision; do not mix
original and optimized CUDA ABI implementations. Preserve build logs and hashes.

## Mixed packing and full-output correctness

Use the PR's `tests/open_jev/packing_parity.md` to independently regenerate the
original oracle and run both eager and graph tests. There must be no production
source modification in the original worktree; only the copied test harness,
fixtures and Cargo test registration are added. Both probes use the same GPU
and the same model export.

The following additional commands reproduce the typed HTTP parity prechecks
against original native. All run inside a GPU reservation:

```bash
"$PY" "$EVIDENCE/tools/replay.py" --kind parity --s1-repo "$S1" \
  --vllm-repo "$VJ" --model-root "$MODELS" --baseline-bin "$BASE_BIN" -- \
  --candidate "$CAND_BIN" --out "$OUT/public-parity" \
  --manifest "$EVIDENCE/workloads/public231.jsonl" \
  --exact-prefix 1 --candidate-graph 1
"$PY" "$EVIDENCE/tools/replay.py" --kind parity --s1-repo "$S1" \
  --vllm-repo "$VJ" --model-root "$MODELS" --baseline-bin "$BASE_BIN" -- \
  --candidate "$CAND_BIN" --out "$OUT/shapes-parity" \
  --manifest "$EVIDENCE/workloads/shapes.jsonl" \
  --candidate-model "$MODELS/system1-9b-cap8192" \
  --exact-prefix 1 --candidate-graph 1
```

## Public231 performance

Three rounds at each C1/C8/C16/C32, warmup five per round, full fixed manifest;
HTTP clock includes response body decoding/schema validation. Each command
launches fresh native/vllm processes sequentially on the reserved physical GPU.

```bash
"$PY" "$EVIDENCE/tools/replay.py" --kind public --s1-repo "$S1" \
  --vllm-repo "$VJ" --model-root "$MODELS" --baseline-bin "$BASE_BIN" -- \
  --backend both --bin-dir "$CAND_BIN" --out "$OUT/public" \
  --concurrency 1,8,16,32 --exact-prefix 1 --vllm-prefix 1 --native-graph 1
```

For a new controlled unoptimized-native comparison, set `--s1-repo` to the
original checkout (so recorded revision is correct), `--bin-dir "$BASE_BIN"`,
`--backend system1 --exact-prefix 0 --native-graph 0`, and a new output directory.
Keep the reserved GPU and every other workload setting identical. This new
run does not retroactively turn historical 005→068 into a same-device pair.

## Synthetic matrix performance

Two fresh launches × four lengths × four concurrencies; 48 warmups per length,
at least 30 seconds to the next complete balanced cycle, then drain. Both
backends reserve capacity 8192 and receive unchanged inputs up to 4096 tokens.

```bash
"$PY" "$EVIDENCE/tools/replay.py" --kind shapes --s1-repo "$S1" \
  --vllm-repo "$VJ" --model-root "$MODELS" --baseline-bin "$BASE_BIN" -- \
  --backend both --bin-dir "$CAND_BIN" --out "$OUT/shapes" \
  --native-revision 43ccbbcff99a94d5c6f744183b82e37853ed873c \
  --native-model "$MODELS/system1-9b-cap8192" --max-model-len 8192 \
  --rounds 2 --concurrency 1,8,16,32 --exact-prefix 1 --vllm-prefix 1 --native-graph 1
```

Use the actual source SHA when rebuilding a later test-only head. Never report
publication or binary provenance as a different measured revision. Retain all
failed/incomplete runs; do not count their partial throughput as successful.
The wrapper changes only install paths in archived scripts and preserves the
original workload/protocol code. `audit.py` independently verifies historical
results without launching a GPU. Full benchmark reruns are hardware/time
consuming and were not repeated solely for this test-only follow-up.
