# StartLux Instant 4B latency on L20X

This is the measurement to run on one L20X (Ada, sm_89) after the native
worker is serving. No latency number is recorded in this change. Do not
compare the result with the H800 or H200 rows in
[the Public231 leaderboard](../../benchmarks/latency-leaderboard.md): those
used different hardware, clocks, and a different client clock.

Label the hardware **L20X** (Ada, compute capability **sm_89**). Do not
label it H200.

## Operator checklist

From the repository root, on the L20X:

```sh
src/backends/cuda/qwen3_5/build.sh target/release 89
cargo build --release --locked -p omni-startlux-native -p omni-jev
```

Serve Instant with thinking off (omit `think_mode`, or send `never`). Leave
`STARTLUX_NONE_CHECK` unset and do not set `STARTLUX_ALLOW_NONE`.

```sh
STARTLUX_MODEL=/path/to/StartLux-Decision-v1.1-4B-Instant \
STARTLUX_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
STARTLUX_HOST=127.0.0.1 \
STARTLUX_PORT=8000 \
  ./target/release/omni-startlux-native
```

Bench endpoint: `http://127.0.0.1:8000/v1/systemone`. Model alias:
`StartLux-Decision-v1.1-4B-Instant`. The three measured `bench.py` invocations
are in [Native arm](#native-arm). The reference arm listens on
`http://127.0.0.1:8090/v1/systemone`.

## What is measured

Use `benchmarks/bench.py`. Its clock includes the HTTP body, JSON decoding,
and schema checks. It is the same runner as the Open-Jev-9B C1 path in the
new-runner notes, not the historical H800 body-receipt clock.

Closed loop, concurrency 1. Five warmup requests, then three full
traversals of the same manifest. Each traversal gets its own output
directory. Warmups and process startup stay out of the pooled samples.
Report per-round and pooled p50/p95 from `bench.py summary`, with
planned/attempted/successful/failed counts.

Both arms:

- the same Instant 4B folder, revision
  `416088f77cfce09633906a3675c7cd2e5dabcf10`
- BF16
- thinking off (`think_mode` omitted or `never`)
- `none_check` on and `allow_none` off (native defaults; the upstream
  `serve` default is the same)
- one reserved L20X
- no retries

`--score-decimals 4` (the bench default) accepts StartLux full-precision
scores when the scalar equals the expectation of the returned
probabilities.

## Manifest

Public231 is not in this repository. The tasks are JevBench `7ce310c7`
(`https://github.com/fstandhartinger/jevbench/tree/7ce310c7262ed49cc85853339a8a42459298e3f3`).
The five-model campaign records canonical dataset SHA-256
`dc3995d8ae1e2fc8e81ce38431add509eb8bb39b85aadfd0c7c32079382dde51`.
The exported public231 JSONL used by the two-model run is SHA-256
`00d1f10ca1142499a598b5f1c6602d329cce1cef04c961c46fc0318daa2c67d3`
([issue #130](https://github.com/ThinkFlowLab/system1-omni/issues/130)).
Those hashes are different representations. Record which file you replayed.

```sh
python -m pip install -r benchmarks/requirements.txt
python benchmarks/bench.py validate /path/to/public231.jsonl
```

## Metadata

Write `run-metadata.json` before measuring. Required fields are `gpu`,
`gpu_ids`, `driver`, `cuda`, `precision`, `model_revision`,
`runtime_revision`, `cache_policy`, `cuda_evidence`, and `reservation`.
Set `gpu` to `L20X` and record sm_89 in the notes. `model_revision` is
`416088f77cfce09633906a3675c7cd2e5dabcf10`. `runtime_revision` is the worker
git revision. On the reference arm also record StartLux-Decision
`a99977d9332decc9fb27a2488b4a83b533b6cd5f` and
`python -c 'import vllm; print(vllm.__version__)'`. Upstream documents vLLM
0.19.1 and 0.31. `cache_policy` for both arms is no prefix cache.
`reservation` is the bench metadata field for this run's device hold.

## Native arm

Build and serve with [README.md](README.md). Leave `STARTLUX_NONE_CHECK`
unset and do not set `STARTLUX_ALLOW_NONE`. Then, from the repository root,
three traversals:

```sh
python benchmarks/bench.py run /path/to/public231.jsonl \
  --endpoint http://127.0.0.1:8000/v1/systemone \
  --model StartLux-Decision-v1.1-4B-Instant \
  --metadata run-metadata.json \
  --phase feasibility \
  --concurrency 1 --warmup 5 --score-decimals 4 \
  --output /tmp/startlux-l20x-feasibility

python benchmarks/bench.py run /path/to/public231.jsonl \
  --endpoint http://127.0.0.1:8000/v1/systemone \
  --model StartLux-Decision-v1.1-4B-Instant \
  --metadata run-metadata.json \
  --phase measured \
  --concurrency 1 --warmup 5 --score-decimals 4 \
  --output /tmp/startlux-l20x-c1-run1

python benchmarks/bench.py run /path/to/public231.jsonl \
  --endpoint http://127.0.0.1:8000/v1/systemone \
  --model StartLux-Decision-v1.1-4B-Instant \
  --metadata run-metadata.json \
  --phase measured \
  --concurrency 1 --warmup 5 --score-decimals 4 \
  --output /tmp/startlux-l20x-c1-run2

python benchmarks/bench.py run /path/to/public231.jsonl \
  --endpoint http://127.0.0.1:8000/v1/systemone \
  --model StartLux-Decision-v1.1-4B-Instant \
  --metadata run-metadata.json \
  --phase measured \
  --concurrency 1 --warmup 5 --score-decimals 4 \
  --output /tmp/startlux-l20x-c1-run3
```

Feasibility is not a performance sample. Reuse the same server for the
three measured traversals. Existing output directories are not overwritten.

```sh
python benchmarks/bench.py summary /tmp/startlux-l20x-c1-run1
```

Repeat for run2 and run3. Pool the three traversals the way
`benchmarks/README.md` describes; do not average the three p95 figures.

## Reference arm

The baseline is the upstream Instant server on the **same** GPU and the
**same** checkpoint, with the bidirectional plugin enabled. A causal vLLM
process without that plugin is not Instant.

From a checkout of StartLux-Decision `a99977d9332decc9fb27a2488b4a83b533b6cd5f`:

```sh
startlux-decide serve \
  --model /path/to/StartLux-Decision-v1.1-4B-Instant \
  --launch-vllm \
  --gpus 0
```

`--launch-vllm` starts vLLM with the folder's bidirectional plugin. `--gpus`
selects the device (`0` is the process default; point it at the reserved
L20X). `serve` listens on `127.0.0.1:8090` unless `--port` is set. The vLLM
process is the engine behind that server, not the bench endpoint. Keep
none-check on and allow-none off (the `serve` defaults; do not pass
`--no-none-check`).

Run the same three measured commands against
`http://127.0.0.1:8090/v1/systemone`, with a metadata file whose
`runtime_revision` names the vLLM and StartLux pins. Use output directories
`/tmp/startlux-l20x-vllm-c1-run1` through `run3`. Compare native and
reference only inside this L20X campaign.

## What not to claim

This file is the recipe, not a result. Until both arms finish on the
L20X, p50/p95 are unknown. A feasibility run, a single traversal, or a
different GPU does not fill them in.

## Unmeasured costs

Code review, not a profile. No millisecond claim.

The worker shares one tokenizer. `Tokenizer::clone` would copy the BPE
tables on the bench clock (`prepare` and the knockout encoder each used to
clone). `encode` takes `&self`, so those handles are `Arc` clones. Hypothesis:
that copy was large next to a short L20X prefill. Measured delta: none.

Device scratch grows to the next multiple of 1024 tokens on the first prompt
that exceeds the current cap, with a stream sync inside the request. Process
warmup is one short choice, and `bench.py` warms five manifest rows, so a
later longer Public231 prompt can allocate during a measured traversal.
A static count of the Instant 4B scratch layout before the contiguous V
buffer (the other byte offsets in `Scratch::new`, including
`cs1_gdn_workspace_floats`) is about 240 MiB at 1024 tokens and about 15 GiB
at 65536 tokens, before the weight arena. The V buffer is
`kv_heads * head_dim` bfloat16 values per token (2048 bytes on Instant 4B):
2 MiB at 1024 tokens and 128 MiB at 65536 tokens.
That 65536-token buffer is not reserved. L20X free memory was not measured.
Hypothesis: the in-request realloc can move p95. Measured delta: none.
