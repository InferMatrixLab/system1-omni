# pplx-decider reference worker

Serves [perplexity-ai/pplx-decider-v1.1-27b](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b)
at revision `14c9d90137962d300f3de95633f7b6535e80c5ce` through
[`src/frontend/pplx_decider.py`](../../src/frontend/pplx_decider.py) and the
existing Rust frontend. The contract is
[`src/models/pplx_decider/README.md`](../../src/models/pplx_decider/README.md).

Run the commands from the repository root. `release-manifest.json` records
the 11 weight shards plus `readout.safetensors` as 52,173,413,928 bytes
(about 48.6 GiB). The card asks for roughly 49 GiB of weights plus working
memory. Plan for that much free disk. A reported load on 1× lab L20X
(~140 GiB, sm_89) was BF16 resident ~50333 MiB with no OOM. The measured
device is that L20X.

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

`/health` is ready only after the hash gate, the weight load, and the
unpadded full-attention install. It reports `model`, `max_input_tokens`
(8192), `modalities`, `attention_mode`, and
`unpadded_full_attention_layers`. The pinned `config.json` lists 16
`full_attention` layers, so that field is 16 after a normal load. It does
not report a filesystem path.

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

Head `1ba07d15b11a8c0796657e4a10b7ba3e62354421` was measured on 1× lab L20X
(sm_89, ~140 GiB). Keep this group on that device. A load failure is a
startup failure, not a latency. H800 board numbers use a different clock
and stay on that board.

Reported load: BF16 resident ~50333 MiB, no OOM. Warmup 5, concurrency 1,
clock `successful_latency` / `post_to_body_json_and_schema_validation`.
Rows that still contain `thinking` return HTTP 422
`unknown request fields: ['thinking']`. The measured workload is the
stripped no-thinking twin of the same 231 rows. The worker still rejects
`thinking`.

| Round | Successful | p50 | p95 |
| --- | --- | --- | --- |
| run1, primary | 231/231 | 98.777 ms | 341.202 ms |
| run2 | 231/231 | 97.921 ms | 340.804 ms |
| run3 | 231/231 | 98.177 ms | 343.953 ms |

The two-model Public231 export hashes to
`00d1f10ca1142499a598b5f1c6602d329cce1cef04c961c46fc0318daa2c67d3`
([latency leaderboard](../../benchmarks/latency-leaderboard.md)). That hash
is not the canonical task-record hash. The measured file is the stripped
twin: the same 231 rows with the `thinking` field removed. Its SHA-256 was
not in the latency report. Point `PUBLIC231_JSONL` at that twin, record the
digest `validate` prints, and stop if the row count is not 231 or if any
row still has `thinking`. The export that still contains `thinking` returns
422 and is not this measurement.

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
7. Remeasure the unpadded full-attention head on the same stripped twin,
   the same clock, warmup 5, concurrency 1, and three rounds. Compare
   `responses.jsonl` answers with the `1ba07d1` rounds and report how many
   decisions differ. A latency table without that count is an unfinished
   delta. Samples for this head: 0 until that run exists.

`benchmarks/smoke.jsonl` checks the client, not this model and not Public231.

## Optimization log

| Date | Hypothesis | Measurement | p50 / p95 delta |
| --- | --- | --- | --- |
| 2026-10-24 | No kernel change. Asked whether the BF16 resident set fits a 48 GB sm_89 card. | No CUDA device in the implementation environment. Samples: 0. | none |
| 2026-10-25 | Baseline at `1ba07d15b11a8c0796657e4a10b7ba3e62354421`. The checkpoint supplies an all-ones full-attention mask, so SDPA stays on the masked math kernel. | 1× lab L20X, sm_89, ~140 GiB. BF16 resident ~50333 MiB, no OOM. Warmup 5, concurrency 1, clock `successful_latency` / `post_to_body_json_and_schema_validation`. Stripped no-thinking twin. run1 231/231, p50 98.777 ms, p95 341.202 ms (primary). run2 231/231, p50 97.921 ms, p95 340.804 ms. run3 231/231, p50 98.177 ms, p95 343.953 ms. | baseline |
| 2026-10-25 | Omit the explicit full-attention mask when every padding bit is true, and set full-attention `is_causal` false, so Transformers 5.17 can take the grouped-query SDPA kernel (`head_dim` 256). Padded batches keep the bool mask. Linear attention is unchanged. Flash and math are not bit-exact. | No CUDA device in this environment. Samples: 0. | not measured |

## Limits

Image parity is unverified. The CUDA forward of head `1ba07d1` has the
Public231 numbers in the optimization log; decision agreement of the
unpadded-mask head against that head is not measured. The shared Qwen CUDA
backend is not used. PR #140 is a separate Rust port of this same contract;
it does not serve, and its JSON parser rewrites integers outside i64/u64.
This worker keeps those integers because it uses Python's parser and the
checkpoint's `json.dumps`.
