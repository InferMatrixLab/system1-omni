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
then `fla.ops.gated_delta_rule`, then `flashinfer.gdn_prefill`, and only
then `transformers.models.qwen3_5.modeling_qwen3_5`. That order binds
`FusedRMSNormGated` and `causal_conv1d_fn`. Startup then replaces the
modeling module's `chunk_gated_delta_rule` with the FlashInfer GDN prefill
wrapper before `load_release_model` builds each `Qwen3_5GatedDeltaNet`,
because the layer copies that global at init.

Startup raises when any of the three distributions is missing, when
flash-linear-attention is not exactly `0.5.2`, when flashinfer-python is
not exactly `0.6.16.post3`, when causal-conv1d is older than 1.5, when an
import fails, when `causal_conv1d_fn` did not bind, or when the chunk
global is not the FlashInfer wrapper. A ready process has `/health`
`linear_attention` equal to `flashinfer+fla+causal_conv1d`. The PyTorch
`torch_chunk_gated_delta_rule` and `F.conv1d` paths are not served. An FLA
chunk binding without the FlashInfer wrapper is not ready.

Prefill calls `flashinfer.gdn_prefill.chunk_gated_delta_rule` with
`use_cp=False`, so Hopper uses the fused SM90 kernel. Q and K L2
normalization uses FLA `l2norm_fwd`, and FlashInfer's
`use_qk_l2norm_in_kernel` flag stays false. Log-space `g` is
exponentiated. The call is one sequence and does not continue a recurrent
state. vLLM's separate CuteDSL prefill backend, CUDA graphs, and
`torch.compile` are not used. The device must be CUDA compute capability
9.0. A device named `NVIDIA L20X` at capability 8.9 is the wrong machine.
The same name at capability 9.0 is the known driver mislabel of an H200
and passes the capability check. No H200 Public231 C1 latency, and no
answer parity against `torch_chunk` or the FLA chunk, has been measured.

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
stripped file's SHA-256 was not reported. Run1 is the primary L20X round.
Do not average the percentiles. Run1 is not a baseline for an H200 delta.

| Round | p50 (ms) | p95 (ms) | Successful |
| --- | --- | --- | --- |
| run1 (primary) | 247.518 | 413.025 | 231/231 |
| run2 | 247.404 | 426.231 | 231/231 |
| run3 | 242.421 | 410.369 | 231/231 |

The worker now admits `thinking` when it is an object with `enabled` set to
the JSON boolean `false`, then drops the field before encoding. The next
H200 campaign should send stock Public231 in that shape. That admission
has no measured latency delta on these L20X rounds. `enabled: true` stays
a 422.

The operator named the metric `successful_latency` and the clock
`post_to_body_json_and_schema_validation`. These three rounds are that
measurement. They are not a new campaign.

Those three rounds are the L20X report at HEAD `9030065`. They are not an
H200 baseline. Do not relabel them as H200, and do not treat a later H200
p50 or p95 as a same-device speedup against 247.518 / 413.025.

## Next measurement: H200 (sm_90)

The forward operator path is FlashInfer GDN prefill plus `causal_conv1d`
and FLA, on compute capability 9.0. No H200 Public231 C1 has been recorded.
Answer parity of that prefill against `torch_chunk` and against the FLA
chunk is not measured. The standalone GDN timings in
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

1. Confirm the GPU the process will use. `nvidia-smi` must show compute
   capability 9.0. Record the device name it prints. Capability 9.0 is the
   H200 target, including when the name is the known `NVIDIA L20X`
   mislabel. Capability 8.9 is Ada; stop, and do not report that machine
   as this H200 run. Do not report the run as H800 or as the Public231
   leaderboard.
2. Start `frontend.clef` as above, on loopback port 8000, variant
   `clef-flash`, device `cuda`. Leave that process up for every traversal.
   `GET /health` must show `status` `ready` and `linear_attention`
   `flashinfer+fla+causal_conv1d`. A process that exits at startup did not
   bind FlashInfer GDN prefill and `causal_conv1d_fn`. Do not measure a
   process that is still on the PyTorch chunk, `F.conv1d`, or an FLA-only
   chunk path.
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
     "gpu": "NVIDIA H200",
     "gpu_ids": ["0"],
     "driver": "<driver version from nvidia-smi>",
     "cuda": "<CUDA version reported by the torch build>",
     "precision": "bfloat16",
     "model_revision": "Cloudflare/clef-flash@f0b7d3cd65296abfe459e6c228b618f5d8b26d2f",
     "runtime_revision": "<git rev-parse HEAD of this worker>",
     "cache_policy": "one request at a time; no prefix cache; no CUDA graph; transformers cache left at its default",
     "cuda_evidence": "nvidia-smi showed compute capability 9.0 while the worker was running with --device cuda. Record the printed device name. If that name is NVIDIA L20X and the capability is 9.0, say so; that pair is the known H200 mislabel.",
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
   leaderboard clock. Report this campaign as the H200 C1. Do not relabel
   the L20X table above as this result, and do not compute a same-device
   speedup against 247.518 / 413.025.

## What remains unverified

Image and video inputs, native CUDA, answer parity, and any H200 latency.
The operator-reported L20X numbers above stay on the clock
`post_to_body_json_and_schema_validation`, metric `successful_latency`, at
HEAD `903006530fb7772bb668bbf192ee4278e273e583`. H1, one encode per request,
is already how that HEAD calls the pinned model, so run1 includes it and
its isolated effect is unknown. The H200 path wires FlashInfer GDN prefill,
FLA, and causal-conv1d. That path has no measured p50, p95, or
answer-parity delta. This environment has no GPU and did not repeat the
campaign.
