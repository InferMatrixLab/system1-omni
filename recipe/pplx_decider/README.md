# pplx-decider reference worker

Serves [perplexity-ai/pplx-decider-v1.1-27b](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b)
at revision `14c9d90137962d300f3de95633f7b6535e80c5ce` through
[`src/frontend/pplx_decider.py`](../../src/frontend/pplx_decider.py) and the
existing Rust frontend. The contract is
[`src/models/pplx_decider/README.md`](../../src/models/pplx_decider/README.md).

Run the commands from the repository root. `release-manifest.json` records
the 11 weight shards plus `readout.safetensors` as 52,173,413,928 bytes
(about 48.6 GiB). The card asks for roughly 49 GiB of weights plus working
memory. Plan for that much free disk. This recipe does not claim that a
48 GB device can load the resident set.

## CPU checks

Python 3.12 and Pillow. No weights.

```sh
python3 -m pip install pytest pillow
PYTHONPATH=src python3 -m pytest tests/pplx_decider -q
PYTHONPATH=src python3 -m compileall -q src/models/pplx_decider src/frontend/pplx_decider.py tests/pplx_decider
```

After a full download, this checks the port against the checkpoint's own
prompt and answer functions. It still does not load the backbone:

```sh
CHECKPOINT=weights/pplx-decider-v1.1-27b
PYTHONPATH=src python3 recipe/pplx_decider/check_contract.py "$CHECKPOINT"
```

## Download and start

Install the checkpoint's pinned `requirements.txt` into the environment
that will run the worker (torch 2.14.0, transformers 5.17.0, and the other
pinned lines). Do not point `--device cpu` at a latency run: the checkpoint
then loads the backbone in float32.

```sh
python3 -m pip install huggingface_hub
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "perplexity-ai/pplx-decider-v1.1-27b",
    revision="14c9d90137962d300f3de95633f7b6535e80c5ce",
    local_dir="weights/pplx-decider-v1.1-27b",
)
PY
python3 -m pip install -r weights/pplx-decider-v1.1-27b/requirements.txt
PYTHONPATH=src python3 -m frontend.pplx_decider \
  --model weights/pplx-decider-v1.1-27b \
  --device cuda \
  --host 127.0.0.1 \
  --port 8000
```

`/health` is ready only after the hash gate and the weight load. It reports
`model`, `max_input_tokens` (8192), `modalities` and `attention_mode`. It
does not report a filesystem path.

```sh
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @recipe/pplx_decider/example-request.json
```

The Rust frontend, from the same root, only forwards. Its inference timeout
is 60 seconds and is separate from the worker.

```sh
cargo build --release --locked -p omni-jev
OMNI_JEV_BIND=127.0.0.1:8080 \
OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 \
  ./target/release/omni-jev
```

`--device cuda` is the evaluated BF16 path (`noncausal_full_attention`, last
token, temperature `1.0087417621345625`). Image inputs use the checkpoint
default cap of 2048×2048 pixels, which is wider than the 512×512 training
cap. Pass `--max-image-pixels 262144` only when reproducing that training
cap, and record it. Text Public231 does not send images.

## Public231 concurrency 1 (sm_89)

No CUDA device was available while this recipe was added, so there is no
p50 or p95. Do not copy H800 or H200 numbers into this group, and do not
rename the device. A load failure is a startup failure, not a latency.

The manifest is the Public231 JSONL whose bytes hash to
`00d1f10ca1142499a598b5f1c6602d329cce1cef04c961c46fc0318daa2c67d3`
([latency leaderboard](../../benchmarks/latency-leaderboard.md)). That hash
is the two-model export, not the canonical task-record hash. If `validate`
prints a different digest, stop.

Clock, quantiles and the three-traversal budget are the existing runner in
[benchmarks/README.md](../../benchmarks/README.md): client latency includes
body read, JSON and schema checks; p50 is the median; p95 is nearest rank
`x[ceil(0.95 n) - 1]`; warmups stay out of the percentiles. The historical
H800 board uses a different clock and does not receive this run.

1. Record `nvidia-smi` name, compute capability, driver, CUDA version and
   memory before load. Keep the device index. Omit host names, accounts,
   home directories and scheduler names.
2. Copy `recipe/pplx_decider/c1-metadata.example.json` and replace every
   `FILL` value. A file that still contains `FILL` is not evidence.
   `precision` stays `bf16`. `runtime_revision` is `git rev-parse HEAD`.
3. Start the worker with `--device cuda` and wait until `/health` returns
   `"status": "ready"`. Record memory after load. If the process dies or
   CUDA reports out of memory, stop and publish that startup failure with
   the memory figures. Do not quantize, offload, or switch devices to
   obtain a number.
4. Install `benchmarks/requirements.txt`. From the repository root, with
   the worker still up and the manifest path in `PUBLIC231_JSONL`:

```sh
python3 benchmarks/bench.py validate "$PUBLIC231_JSONL"
python3 benchmarks/bench.py run "$PUBLIC231_JSONL" \
  --endpoint http://127.0.0.1:8000/v1/systemone \
  --model jev-latest \
  --metadata "$RUN_DIR/c1-metadata.json" \
  --phase feasibility \
  --concurrency 1 --warmup 5 --timeout 60 \
  --output "$RUN_DIR/feasibility"
```

5. Leave the same process running. Three measured traversals, each in a
   new output directory:

```sh
python3 benchmarks/bench.py run "$PUBLIC231_JSONL" \
  --endpoint http://127.0.0.1:8000/v1/systemone \
  --model jev-latest \
  --metadata "$RUN_DIR/c1-metadata.json" \
  --phase measured \
  --concurrency 1 --warmup 5 --timeout 60 \
  --output "$RUN_DIR/c1-round-1"
```

Repeat for `c1-round-2` and `c1-round-3`. Then
`python3 benchmarks/bench.py summary` on each directory. Pool latency only
across successful completed requests from the three measured rounds, 693
samples when every request succeeds. Report planned, attempted, successful,
failed and incomplete counts. Keep failure latencies out of p50 and p95.
Requests that hit the 60 second client timeout are failures. Do not retry
them and do not raise the timeout after seeing the times.

6. A second arm through `http://127.0.0.1:8080/v1/systemone` is frontend
   time only. Keep it in different output directories. The Rust frontend's
   own 60 second cap still applies. Do not average it into the worker p50.

`benchmarks/smoke.jsonl` checks the client, not this model and not Public231.

## Optimization log

| Date | Hypothesis | Measurement | p50 / p95 delta |
| --- | --- | --- | --- |
| 2026-10-24 | None applied. The BF16 resident set may exceed a 48 GB sm_89 card; that is untested. | No CUDA device in the implementation environment. Samples: 0. | none |

## Limits

Images and the CUDA forward are unverified. The shared Qwen CUDA backend is
not used. PR #140 is a separate Rust port of this same contract; it does not
serve, and its JSON parser rewrites integers outside i64/u64. This worker
keeps those integers because it uses Python's parser and the checkpoint's
`json.dumps`.
