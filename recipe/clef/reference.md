# Clef reference worker

Serve Cloudflare Clef-flash or Clef on `POST /v1/systemone` by loading the
pinned snapshot's joint schema model. The contract, hashes and the
unsupported image, video and native-CUDA paths are in the
[model README](../../src/models/clef/README.md). Commands below run from the
repository root.

Clef-flash is the bring-up variant: about 19.1 GB of safetensors versus
about 55.0 GB for Clef. Both use the same text contract. The recorded
Public231 concurrency-1 result is later in this recipe.

## Pins

| Variant | Snapshot | Revision |
| --- | --- | --- |
| `clef-flash` (default) | `Cloudflare/clef-flash` | `f0b7d3cd65296abfe459e6c228b618f5d8b26d2f` |
| `clef` | `Cloudflare/clef` | `a30162b99d68a481d8bd75d38df069a0cf83df40` |

The card names torch 2.11 and transformers 5.10.2. Install
[requirements.txt](requirements.txt) in a Python 3.12 environment. A CUDA
build of that torch is required for the worker process. CPU inference is
not a supported path. Storage for the flash snapshot is about 19.1 GB plus
the download cache; the 27B snapshot is about 55.0 GB. Device memory has
to hold the bf16 weights and one prefill. No specific GPU size was measured
here.

## Start

```sh
python -m pip install -r recipe/clef/requirements.txt
```

The requirements pin `causal-conv1d>=1.5`, `flash-linear-attention==0.5.2`,
and `flashinfer-python==0.6.16.post3`. The worker imports `causal_conv1d`,
then `fla.ops.gated_delta_rule`, then tries `flashinfer.gdn_prefill`, and
only then imports `transformers.models.qwen3_5.modeling_qwen3_5`. That
order binds `FusedRMSNormGated` and `causal_conv1d_fn`. When the reported
compute capability major is 9, 10, or 12 and FlashInfer loaded, startup
replaces the modeling module's `chunk_gated_delta_rule` with the FlashInfer
GDN prefill wrapper before `load_release_model` builds each
`Qwen3_5GatedDeltaNet`, because the layer copies that global at init.

`causal-conv1d` and `flash-linear-attention` are required. Startup raises
when either is missing, when flash-linear-attention is not exactly `0.5.2`,
when causal-conv1d is older than 1.5, or when `causal_conv1d_fn` and a
chunk kernel did not bind. A missing or different FlashInfer install does
not, by itself, refuse ready. `/health` `linear_attention` is the bind that
started: `flashinfer+fla+causal_conv1d` or `fla+causal_conv1d`. The PyTorch
`torch_chunk_gated_delta_rule` and `F.conv1d` paths are not served.

FlashInfer prefill calls `flashinfer.gdn_prefill.chunk_gated_delta_rule`
with `use_cp=False`. Q and K L2 normalization uses FLA `l2norm_fwd`, and
FlashInfer's `use_qk_l2norm_in_kernel` flag stays false. Log-space `g` is
exponentiated. The call is one sequence and does not continue a recurrent
state. FlashInfer 0.6.16 does not dispatch that kernel when the device
reports capability 8.9, so those processes keep the FLA chunk. vLLM's
separate CuteDSL prefill backend, CUDA graphs, and `torch.compile` are
not used.

The lab host is the machine `nvidia-smi` names `NVIDIA L20X`, including
when it reports compute capability 8.9. Startup does not require the name
`NVIDIA H200` and does not refuse ready because the capability is not 9.0.
No Public231 C1 of this tip has been measured. Answer parity against
`torch_chunk` is not measured.

```sh
python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(
    "Cloudflare/clef-flash",
    revision="f0b7d3cd65296abfe459e6c228b618f5d8b26d2f",
))
PY
```

Use the directory that command prints as `<snapshot>`. Do not point the
worker at a different revision; the hash gate refuses it.

```sh
PYTHONPATH=src python -m frontend.clef \
  --model <snapshot> \
  --variant clef-flash \
  --device cuda \
  --host 127.0.0.1 \
  --port 8000
```

Readiness is the process log line after `load_release_model` returns.
`GET http://127.0.0.1:8000/health` then returns `status: ready`,
`variant: clef-flash`, `prompt_mode: non-thinking`, `modalities: ["text"]`
and `max_input_tokens: 16384`. The first `POST` is the first forward. That
forward was not run in this change.

For the 27B checkpoint, download `Cloudflare/clef` at
`a30162b99d68a481d8bd75d38df069a0cf83df40` and pass `--variant clef`.

## Example

```sh
curl --fail-with-body http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @recipe/clef/example-request.json
```

The Rust frontend can forward the same body. From the repository root,
after `cargo build --release --locked`:

```sh
OMNI_JEV_BIND=127.0.0.1:8080 \
OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 \
  ./target/release/omni-jev
```

```sh
curl --fail-with-body http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @recipe/clef/example-request.json
```

A non-empty `images` or `videos` field returns HTTP 422. A prompt longer
than 16384 tokens returns HTTP 422 and the state is not shortened. The
worker echoes the request's `model` string; it does not have to equal the
loaded repo id.

## Checks that do not load weights

```sh
PYTHONPATH=src python -m pip install pytest
PYTHONPATH=src python -m pytest tests/clef -q
PYTHONPATH=src python -m compileall -q src/models/clef src/frontend/clef.py tests/clef
```

To confirm the checked-in token fixture against a downloaded loader and
tokenizer, without weights:

```sh
PYTHONPATH=src python recipe/clef/export_prompt_fixture.py \
  --loader <snapshot>/joint_schema_model.py \
  --tokenizer <snapshot>/tokenizer.json \
  --output tests/clef/fixtures/text_contract.json \
  --check
```

## Public231 C1 on lab L20X (sm_89)

Operator-reported, not re-run in the environment that edited this recipe.
Clef-flash, worker HEAD `903006530fb7772bb668bbf192ee4278e273e583`, NVIDIA
L20X at compute capability 8.9, warmup 5, concurrency 1. Stock Public231
includes a `thinking` field. That HEAD rejected it with HTTP 422
`Unknown request fields`. The measured manifest was
`public231-no-thinking.jsonl`, with the `thinking` key stripped. The
stripped file's SHA-256 was not reported. Run1 is the baseline for the
tip `594528a9` comparison below. Do not average the percentiles.

| Round | p50 (ms) | p95 (ms) | Successful |
| --- | --- | --- | --- |
| run1 (primary) | 247.518 | 413.025 | 231/231 |
| run2 | 247.404 | 426.231 | 231/231 |
| run3 | 242.421 | 410.369 | 231/231 |

The worker now admits `thinking` when it is an object with `enabled` set to
the JSON boolean `false`, then drops the field before encoding. That
admission has no separate measured latency delta on these `9030065` rounds.
`enabled: true` stays a 422.

The operator named the metric `successful_latency` and the clock
`post_to_body_json_and_schema_validation`. These three rounds are that
measurement. They are not a new campaign.

Those three rounds are the baseline at HEAD `9030065` on the lab host
`nvidia-smi` names `NVIDIA L20X`.

## Public231 C1 of tip `594528a9`

Operator-reported on that same L20X-labeled lab host. Not re-run in the
environment that edited this recipe. Worker
`594528a9f9c1ca38ce0dbb58a549191dfbde5bf3`. `/health` `linear_attention`
was `fla+causal_conv1d`. Successful requests 231/231. p50 164.969 ms. p95
216.755 ms. Against baseline `9030065` run1 (p50 247.518 ms, p95 413.025 ms):
Δp50 −33.35% (−82.5 ms), Δp95 −47.52% (−196.3 ms). This run's bind is the
FLA chunk, not FlashInfer. Answer parity was not part of the paste.
Warmup count, clock name, manifest hash, driver, and CUDA version were
not restated in the paste, so they are not copied from the `9030065`
campaign.

This code tip stays. A later prefill change is a new tip and needs its own
Public231 C1 paste before another delta is claimed.

## Next code tip on the L20X-labeled host

That host is the measurement machine, including when `nvidia-smi` reports
compute capability 8.9. Do not require a card whose name is `NVIDIA H200`.
On a reported 8.9 the ready bind is `fla+causal_conv1d`. When the same
process sees reported major 9, 10, or 12 and FlashInfer loaded, the ready
bind is `flashinfer+fla+causal_conv1d`. Record the `/health` string the
process returns. The standalone GDN timings in
[benchmarks/gdn/README.md](../../benchmarks/gdn/README.md) are a different
experiment and are not a Clef HTTP result.

This is the operator checklist for the next concurrency-1 latency
measurement of the flash worker. It uses the existing
[benchmark runner](../../benchmarks/README.md). The client does not reserve
a GPU and does not prove that CUDA ran. This repository does not contain
the Public231 JSONL.

Record results outside the repository. Do not put hostnames, account names,
home directories, or the name of a reservation system in the metadata or
in a commit.

1. Confirm the GPU the process will use. The lab host is the one
   `nvidia-smi` names `NVIDIA L20X`. Record that name and the compute
   capability it prints, including 8.9. Do not stop because the name is
   not `NVIDIA H200`, and do not stop because the capability is not 9.0.
   Do not report the run as H800 or as the Public231 leaderboard.
2. Start `frontend.clef` as above, on loopback port 8000, variant
   `clef-flash`, device `cuda`. Leave that process up for every traversal.
   `GET /health` must show `status` `ready`. `linear_attention` is
   `fla+causal_conv1d` or `flashinfer+fla+causal_conv1d`, whichever the
   process bound. Record that string. A process that exits at startup did
   not bind `causal_conv1d_fn` and a chunk kernel. Do not measure a
   process that is still on the PyTorch chunk or `F.conv1d` path.
3. Obtain the frozen Public231 request manifest whose file SHA-256 is
   `00d1f10ca1142499a598b5f1c6602d329cce1cef04c961c46fc0318daa2c67d3`.
   That is the JSONL byte hash cited for the JevBench `7ce310c7` export in
   [the latency leaderboard](../../benchmarks/latency-leaderboard.md).
   The canonical task hash
   `dc3995d8ae1e2fc8e81ce38431add509eb8bb39b85aadfd0c7c32079382dde51`
   is a different encoding and must not be substituted. Check the file
   with `sha256sum` before `validate`. The recorded campaign above did not
   use that byte string: the `thinking` key had been removed, and the
   resulting file hash was not reported. Record `sha256sum` of whichever
   file the next run actually sends. Keep `thinking` when its `enabled`
   value is false; stripping it is no longer required for admission.
4. Install `benchmarks/requirements.txt` (`httpx==0.28.1`) into the client
   environment. The client and the worker may use different environments.
5. Write a metadata JSON file. Required keys are `gpu`, `gpu_ids`,
   `driver`, `cuda`, `precision`, `model_revision`, `runtime_revision`,
   `cache_policy`, `cuda_evidence` and `reservation`. Use these values
   unless a measured fact differs:

   ```json
   {
     "gpu": "NVIDIA L20X",
     "gpu_ids": ["0"],
     "driver": "<driver version from nvidia-smi>",
     "cuda": "<CUDA version reported by the torch build>",
     "precision": "bfloat16",
     "model_revision": "Cloudflare/clef-flash@f0b7d3cd65296abfe459e6c228b618f5d8b26d2f",
     "runtime_revision": "<git rev-parse HEAD of this worker>",
     "cache_policy": "one request at a time; no prefix cache; no CUDA graph; transformers cache left at its default",
     "cuda_evidence": "nvidia-smi name NVIDIA L20X, compute capability as printed (8.9 on the lab host), while the worker was running with --device cuda. Copy /health linear_attention.",
     "reservation": "<allocation id>"
   }
   ```

6. Predeclare the budget before the first measured request: concurrency 1,
   five warmups, three separate measured traversals, score decimals 4,
   and one timeout. The runner default timeout is 60 seconds and may be
   too small for a long prompt. Pick the timeout from a feasibility run
   and do not raise it after a measured traversal fails. A 422 from the
   16384-token limit is a failed request. A traversal with failures is not
   a latency result.
7. Run, with a new output directory each time:

   ```sh
   python benchmarks/bench.py validate <public231.jsonl>
   python benchmarks/bench.py run <public231.jsonl> \
     --endpoint http://127.0.0.1:8000/v1/systemone \
     --model clef-flash \
     --metadata <metadata.json> \
     --phase feasibility \
     --concurrency 1 --warmup 5 --timeout <seconds> \
     --output <fresh-directory>/feasibility
   ```

   Repeat `run` three times with `--phase measured` and the same manifest,
   metadata, timeout, warmup and concurrency. Exclude feasibility from the
   pooled latency.
8. For each measured directory, recompute
   `python benchmarks/bench.py summary <directory>`. Pool only the three
   rounds if each summary has `evidence_complete` true and zero failures.
   Concatenate successful `latency_ms` values. p50 is the median. p95 is
   the nearest rank `x[ceil(0.95 * n) - 1]` on the sorted samples. Report
   per-round and pooled counts. Keep failure latencies separate.
9. The clock is the runner's
   `post_to_body_json_and_schema_validation` clock. It is not the H800
   leaderboard clock. Record the `nvidia-smi` name, the printed capability,
   and `/health` `linear_attention` with the p50 and p95. Compare a new
   code tip with the recorded tip `594528a9` only after that new tip has
   its own paste. The `9030065` run1 figures remain the baseline of the
   recorded comparison above.

## What remains unverified

Image and video inputs, native CUDA, and answer parity. The `9030065`
rounds use metric `successful_latency` and clock
`post_to_body_json_and_schema_validation`. The tip `594528a9` p50 and p95
are the operator paste in the section above; this environment has no GPU
and did not repeat either campaign. H1, one encode per request, is already
inside the `9030065` baseline, so its isolated effect is unknown. The
recorded win is the FLA chunk plus `causal_conv1d` on the reported-8.9
host. FlashInfer on a reported major 9, 10, or 12 has no C1. A later code
change is unmeasured until its own paste.
