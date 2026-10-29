# PR #196: independently inspectable experiment evidence

This is an **evidence-only archive branch**, not a proposed source-code landing.
It records original-native vs optimized Open-Jev-9B correctness and historical
performance measurements for https://github.com/InferMatrixLab/system1-omni/pull/196.
The follow-up PR source is `ecea1a31e287890fee5a595504203b53734ffe2a`
(tree `f36ac83d2bc6bf217be4dd8a24c1a5973a0d60ca`).
Production code was unchanged by the 085 follow-up. The PR remains a draft.

## Start here

- [New mixed-packing protocol and replay commands](packing/packing_parity.md)
- [085 GPU results and model/binary hashes](validation/085/validation.json)
- [Original GPU output](validation/085/original-gpu.log),
  [optimized eager](validation/085/candidate-graph-0.log),
  [optimized graph](validation/085/candidate-graph-1.log)
- [Existing eager regressions](validation/085/prefill-graph-0.log),
  [existing graph regressions](validation/085/prefill-graph-1.log)
- [Detected build-cache harness failure](validation/085/harness-failure-wrong-build.log)
- [Historical 068 GPU validation](validation/068/gpu-tests.log)
- [Public231 frozen manifest](workloads/public231.jsonl),
  [shape frozen manifest](workloads/shapes.jsonl)
- [Full replay instructions](REPLAY.md), [CPU audit program](audit.py)

085 uses 18 fixed token cases, 94 candidate logits per mode; all 188 optimized
logits match the original native FP32 bits. The published portable replay tool
was also executed on 231 public + 192 synthetic HTTP cases: zero decision flips
and zero returned probability/Score drift against freshly launched original
native. These parity runs are not new performance measurements. Mixed lengths and permutations,
changed-token replay, 4096-token and 16-candidate boundaries, question routing,
and a long control are covered. Seven existing GPU regressions also pass in each
mode, including a direct graph replay-count assertion. The synthetic-token test
checks execution numerics, not semantic task accuracy. Reference scope: pinned
BF16 9B export, NVIDIA L20X / measured SM90, CUDA 13.0. Other architectures are
not established by this oracle.

The first candidate probe accidentally reused an original-checkout Cargo test
binary. It failed the packing-enabled assertion before comparison. That attempt
is retained; corrected probes were rebuilt, copied separately, hashed, and run
successfully. Replay instructions require distinct Cargo target directories.

## Data layout and integrity

`SHA256.json` hashes every published file except itself. `runs.json` lists 322
completed historical measurement cells/rounds, totaling 116,106 requests,
including campaigns 005, 015, 021, 068, 069 and repeatability diagnostics 081–084.
The archive does not imply that these campaigns share identical protocols.

Each `measurements/<run>/samples.jsonl` retains every request's original ID,
latency, HTTP status, error, completion state and dispatch ordinal (where present).
`answers_ref` resolves through `answers.json` to the complete original parsed
answer object, without rounding. This dictionary removes repeated identical
answers; no requests or latency samples are sampled out. `summary.json` retains
measurement wall time, so throughput can be recalculated independently.
`config.json`, campaign plans and metadata record runtime revisions, device,
launch commands, binary hashes and settings. Source response-file hashes and
projection details are in `source.json` and `source-records.json`.

These are **lossless projections of the timing/status/parsed-answer fields**,
not full HTTP wire dumps. Redundant base64 bodies, per-response server metadata
and raw profiler traces remain in the local archive. Independent performance
and parsed-answer parity checks need only the published fields. Full original
and optimized response bodies from the two 068 parity prechecks are included
under `parity/`. GPU validation stdout/stderr is included unchanged. No model
weights or executable binaries are included.

`tools/original/` contains retained campaign programs; `tools/replay.py` changes
only their machine-specific path prefixes. `tools/bench.py` preserves the actual
HTTP runner (`f567448c...`), also present in the measured source tree.
Historical absolute paths in logs are provenance, not required install paths.
The archive includes metadata/summaries for 084; it does **not** claim that full
profiler traces or every diagnostic overlay are published or independently
replayable from this bundle. The main correctness/performance campaigns and
085 have the replay commands below.

## Audit without a GPU

```bash
python3 audit.py > audit-result.json
```

This checks every published checksum, recounts every request, recomputes each
cell's P50/P95/QPS, pools rounds from raw samples, and compares all **40,884**
formal optimized-native answers (068 + 069) with original-native references.
It does not assert numerical equality with vllm-jev. The original backends
already differed on some decisions.

## Interpretation limits

005→068 is a historical comparison across different physical GPUs of the same
reported type, not a controlled code-only speedup estimate. 068 vs 015 and
069 vs 021 retain their recorded paired-device setup and workload definitions.
Historical Public C32 ranking did not repeat in 081/082. This evidence does not
establish universal wins, statistical significance, or unchanged behavior on
unmeasured models/hardware. 085 supplies missing correctness-path coverage;
it introduces no new performance-speedup claim.
