# Clef reference worker

Serve Cloudflare Clef-flash or Clef on `POST /v1/systemone`. The HTTP server
is the Rust binary `omni-clef`. It admits one request at a time and forwards
the body to one Python executor process. That process loads the pinned
snapshot's joint schema model and scores every request on its main thread.
The contract, hashes, and the unsupported image, video, and native joint-head
paths are in the [model README](../../src/models/clef/README.md). Commands
below run from the repository root.

Clef-flash is the bring-up variant: about 19.1 GB of safetensors versus
about 55.0 GB for Clef. Both use the same text contract. The [controlled migration acceptance](#controlled-migration-acceptance)
records the flash text service; 27B has not been run.

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
to hold the bf16 weights and one prefill. The controlled flash acceptance records device-memory samples in its
local evidence bundle; it does not establish a minimum supported GPU size.

## Start

```sh
python -m pip install -r recipe/clef/requirements.txt
```

The default requirements pin `causal-conv1d==1.7.0` and
`flash-linear-attention==0.5.2`. They also include Accelerate for the
loader's `device_map` and torchvision for the shared processor, even for text.
FlashInfer is optional and was absent from the FLA acceptance environment;
install `flashinfer-python==0.6.16.post3` separately to try its unvalidated
opt-in path. The worker imports `causal_conv1d`,
then `fla.ops.gated_delta_rule`, then tries `flashinfer.gdn_prefill`, and
only then imports `transformers.models.qwen3_5.modeling_qwen3_5`. That
order binds `FusedRMSNormGated` and `causal_conv1d_fn`. The default
prefill is the FLA chunk. `--gdn-prefill flashinfer` is opt-in: startup
replaces the modeling module's `chunk_gated_delta_rule` with the FlashInfer
GDN prefill wrapper before `load_release_model` builds each
`Qwen3_5GatedDeltaNet` only when that flag is set, the package imports,
and the reported capability major is 9, 10, or 12. The layer copies that
global at init.

`causal-conv1d` and `flash-linear-attention` are required. Startup raises
when either is missing, when flash-linear-attention is not exactly `0.5.2`,
when causal-conv1d is older than 1.5, or when `causal_conv1d_fn` and a
chunk kernel did not bind. A missing or different FlashInfer install does
not, by itself, refuse the default FLA path. Requesting
`--gdn-prefill flashinfer` without that package, or on a device whose
reported major is not 9, 10, or 12, refuses startup. `/health`
`linear_attention` is the bind that started:
`fla+causal_conv1d` by default, or `flashinfer+fla+causal_conv1d` when the
opt-in flag bound. The PyTorch `torch_chunk_gated_delta_rule` and
`F.conv1d` paths are not served.

FlashInfer prefill calls `flashinfer.gdn_prefill.chunk_gated_delta_rule`
with `use_cp=False`. Q and K L2 normalization uses FLA `l2norm_fwd`, and
FlashInfer's `use_qk_l2norm_in_kernel` flag stays false. Log-space `g` is
exponentiated. The call is one sequence and does not continue a recurrent
state. FlashInfer 0.6.16 does not dispatch that kernel when the device
reports capability 8.9. vLLM's separate CuteDSL prefill backend, CUDA
graphs, and `torch.compile` are not used.

The lab host is the machine `nvidia-smi` names `NVIDIA L20X`, including
when it reports compute capability 8.9. Startup does not require the name
`NVIDIA H200` and does not refuse the default FLA path because the
capability is not 9.0. The measured primary is that FLA path. FlashInfer
GDN prefill has no measured forward and no answer parity against
`fla+causal_conv1d`. That parity run is the named follow-up blocker
before `--gdn-prefill flashinfer` can become the default.

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

`omni-clef` starts the executor with `CLEF_PYTHON` when that variable is
set, and with `python3` otherwise. The interpreter has to be the recipe
environment from the install above: the one that has `causal-conv1d`.
The default system `python3` does not, and the executor dies during
startup before the port opens. Export the venv that installed this
recipe, for example:

```sh
export CLEF_PYTHON="$HOME/envs/clef-ref/bin/python"
export PYTHONDONTWRITEBYTECODE=1
```

Use that venv's `bin/python`, or another interpreter that imports
`causal_conv1d`, if the environment lives somewhere else. The same
export applies when the process is started through `python -m frontend.clef`.

```sh
cargo build --release --locked -p omni-clef
./target/release/omni-clef \
  --model <snapshot> \
  --variant clef-flash \
  --device cuda \
  --gdn-prefill fla \
  --host 127.0.0.1 \
  --port 8000
```

`PYTHONPATH=src python -m frontend.clef` with the same flags execs that
binary. It does not bind a Python HTTP server. If `omni-clef` is not built,
the launcher exits before opening a port. `CLEF_PYTHON` still selects the
executor interpreter after the exec.

The executor process runs one text forward, then `omni-clef` binds. `GET
http://127.0.0.1:8000/health` is not served until that forward returns.
It then returns `status: ready`, `variant: clef-flash`,
`prompt_mode: non-thinking`, `modalities: ["text"]`,
`max_input_tokens: 16384`, and `linear_attention: fla+causal_conv1d` on
the default flag. A CUDA init failure exits before the port opens. That
startup forward was not run in this editing environment.

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
cargo test --locked -p omni-clef
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

## Controlled migration acceptance

The 2026-10-27 local issue #162 acceptance compares the original #165 Python
HTTP server at `504b272c46dc1963f981c6bed9ef0c7ebdd40c69` with the Rust service
at `3364dbd7f8b2cb414c24e73f24b74dd785888ecc`, based on main `22b7717` including
#182. Later recipe/documentation edits do not change the measured runtime.
Both services ran sequentially in one GPU allocation: NVIDIA L20X as reported
by the device, PyTorch capability 9.0 (nvidia-smi reports 8.9), driver
570.133.20, Torch CUDA 13.0. Both capability readings are retained. Do not
relabel this device H200/H800 or combine these measurements with the
historical results below.

Controls: the Clef-flash snapshot above, BF16, Torch 2.11.0, torchvision 0.26.0,
Transformers 5.10.2, Accelerate 1.15.0, FLA/fla-core 0.5.2, causal-conv1d
1.7.0, no FlashInfer. Model, prompt and postprocessing sources are identical.
Each server starts with an identical copy of the retained reference Triton
cache; the nine selected autotune configurations remain unchanged after the
run. There is no kernel change, CUDA graph, `torch.compile` or cross-request
batching.

Matching FLA package versions alone is insufficient for a zero-tolerance
comparison. The first attempt used independent empty caches and had 146/231
different responses and one Choice flip per round, despite zero HTTP failures.
A two-request main-thread/new-thread crossover traced the difference to four
different autotune selections. That failed experiment is retained. The corrected
run below freezes the reference's existing choices, with no search for faster
configurations and no relaxed tolerance. This control belongs in subsequent
#173 comparisons too; newly compiled configurations require their own parity check.

Input: all 231 easy/original/hard tasks from JevBench `7ce310c7`, with
`thinking.enabled: false`. Canonical task SHA256:
`dc3995d8ae1e2fc8e81ce38431add509eb8bb39b85aadfd0c7c32079382dde51`.
This export's JSONL SHA256:
`d35da44333e0bed0cea45ee3565147c041e480769c3bcba0279749f62f3fe761`.
The encoding differs from the historical export; both services use these
same bytes. Each service completed one four-case feasibility and three C1
rounds, five warmups before each round, without restarting its server.
No retries. Latency includes POST through body receipt, JSON decoding and
schema validation; loading, warmups and direct model calls are excluded.

| HTTP service | Round | Successful | Failed | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| python | 1 | 231/231 | 0 | 181.643 | 271.345 |
| python | 2 | 231/231 | 0 | 181.229 | 249.311 |
| python | 3 | 231/231 | 0 | 181.815 | 251.320 |
| rust | 1 | 231/231 | 0 | 177.692 | 241.648 |
| rust | 2 | 231/231 | 0 | 53.209 | 96.859 |
| rust | 3 | 231/231 | 0 | 52.484 | 96.036 |

All 693 measured response pairs match exactly, including JSON ordering, model
echo and token usage. Probability/score drift and decision flips are zero.
First/repeated mixed requests and the three invalid-input cases also match.
Real-GPU checks pass for four concurrent requests, two client disconnect
timings followed by correct responses, and health/inference 503 after killing
the idle executor. CPU barriers separately establish the cancellation and
resource lifetime boundaries described [below](#service-lifecycle-acceptance).

This is a warmed compiler-cache comparison. Retain round 1 as well as rounds
2 and 3; do not substitute the historical baseline or direct-call timings.
The changed path includes persistent model-thread ownership and HTTP/1.1
connection reuse. The experiment does not assign every saved millisecond to
CUDA thread initialization. Baseline-then-candidate order and shared host
activity limit causal precision. 27B, media, FlashInfer, other GPUs, a native
joint head and sustained-load behavior remain unverified.

The local delivery includes `validation-report.md`, `protocol.md`, per-round
raw responses/metadata, the failure and crossover evidence, the frozen cache,
and checksummed reproduction scripts in `issue-162-evidence.tar.gz`. It has
not been published to GitHub. `handoff.md` identifies the service commit and
exact local launch command for #173. The selected-config manifest SHA256 is
`bd8c86c615878419578d96593ede668ff257e52ef082b3405b4df3a4c852cf3f`.

For a matched comparison, copy the retained reference cache into a fresh
directory for each service, export `TRITON_CACHE_DIR` to that directory before
starting it, and preserve it through all three rounds. Compare the selected
`*.autotune.json` configurations before/after. The cache is tied to the pinned
software/device stack; a cache miss on another stack is not the same control.
The [measurement procedure](#next-code-tip-on-the-l20x-labeled-host) below
records the HTTP samples. Use `bench.py compare` on each corresponding round
with `--max-probability-drift 0 --max-score-drift 0 --max-flips 0`, and check
the full response JSON/order/usage separately.

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

Operator-reported remeasure on that same L20X-labeled lab host. Not re-run
in the environment that edited this recipe. This was the primary within that historical experiment.

Worker `594528a9f9c1ca38ce0dbb58a549191dfbde5bf3`. The paste states warmup
5, concurrency 1, stock Public231 with `thinking.enabled` false, and metric
`successful_latency`. `/health` `linear_attention` was `fla+causal_conv1d`.
Successful requests 231/231. p50 164.969 ms. p95 216.755 ms. Against
baseline `9030065` run1 (p50 247.518 ms, p95 413.025 ms, 231/231): Δp50
−82.549 ms (−33.35%), Δp95 −196.270 ms (−47.52%). This run's bind is the
FLA chunk, not FlashInfer.

The paste does not restate the clock name, the manifest SHA-256, the
driver, the CUDA version, or answer parity. Those fields are not copied
from the `9030065` rounds. Those rounds used `public231-no-thinking.jsonl`
because that HEAD returned HTTP 422 on `thinking`. This remeasure names
stock Public231 with `thinking.enabled` false. Neither paste includes a
`sha256sum` of the file that was sent.

These numbers describe that historical comparison only. Use a matched
reference/candidate run for a new prefill change; do not reuse them as a
measurement of the current Rust service.

## Public231 C1 of `omni-clef` tip `f9127a0`

Operator-reported on the same L20X-labeled lab host. Not re-run in the
environment that edited this recipe. Worker
`f9127a03b732aaf6e88c99a0212e02acfdb01880`. Smoke 4/4. `/health`
`linear_attention` was `fla+causal_conv1d`. Three Public231 concurrency-1
rounds versus the primary pair 164.969 / 216.755 ms:

| Round | p50 (ms) | p95 (ms) | Successful |
| --- | --- | --- | --- |
| R1 | 169.605 | 220.003 | 231/231 |
| R2 | 53.580 | 97.031 | pass count not restated in this handoff |
| R3 | 52.950 | 97.608 | pass count not restated in this handoff |

The operator reports quality `choice_acc=0.8417`, `noul_acc=0.7973`,
`score_mae=0.2488`, matching the `594528a9` baseline. This paste is a
serve-path measurement, not a latency KEEP. R1 does not pass the dual
gate against 164.969 / 216.755. R2 and R3 are steady-state rounds and
are not a KEEP and not a new primary. The primary remains tip
`594528a9`.

## Next code tip on the L20X-labeled host

That host is the measurement machine, including when `nvidia-smi` reports
compute capability 8.9. Do not require a card whose name is `NVIDIA H200`.
On a reported 8.9 the ready bind is `fla+causal_conv1d`. That is also the
default on reported majors 9, 10, and 12. `flashinfer+fla+causal_conv1d`
is ready only when the process was started with `--gdn-prefill flashinfer`
and that kernel could bind. Record the `/health` string the process
returns. The standalone GDN timings in
[benchmarks/gdn/README.md](../../benchmarks/gdn/README.md) are a different
experiment and are not a Clef HTTP result.

Use the frozen service/configuration from the controlled migration acceptance
as the starting point for #173, then measure baseline and candidate under
matched controls. Historical `594528a9` numbers are not a fresh baseline.

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
2. Export `CLEF_PYTHON` to the recipe venv, as in [Start](#start), then
   start `omni-clef` on loopback port 8000, variant `clef-flash`, device
   `cuda`. The default system `python3` lacks `causal-conv1d` and the
   executor dies before the port opens. Leave that process up for every
   traversal. The recorded `9030065` and `594528a9` pastes used the
   previous Python HTTP server. Tip `f9127a0` is an `omni-clef` paste and
   is not a KEEP; those historical figures remain attributed to that experiment.
   A later code tip needs its own matched comparison.
   `GET /health` is absent until the startup forward returns, then must
   show `status` `ready`. The default `linear_attention` is
   `fla+causal_conv1d`. `flashinfer+fla+causal_conv1d` means the process
   was started with `--gdn-prefill flashinfer`. Record that string. A
   process that exits at startup did not bind `causal_conv1d_fn` and a
   chunk kernel. Do not measure a process that is still on the PyTorch
   chunk or `F.conv1d` path.
3. Freeze the exact Public231 request manifest sent to both services. The
   controlled migration export from JevBench `7ce310c7` has file SHA-256
   `d35da44333e0bed0cea45ee3565147c041e480769c3bcba0279749f62f3fe761`.
   The canonical task hash is
   `dc3995d8ae1e2fc8e81ce38431add509eb8bb39b85aadfd0c7c32079382dde51`
   and is not a JSONL byte hash. Check the file with `sha256sum` before
   `validate`. The older leaderboard export used byte hash `00d1f10c...`;
   do not substitute it for this export. Record the bytes actually sent.
   Keep `thinking` when its `enabled` value is false; stripping it is no
   longer required for admission. The local acceptance bundle includes the
   export script and manifest.
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
     "cache_policy": "one request at a time; identical copies of the frozen reference Triton cache per service; record selected-config hash; persistent server/cache through all rounds; no prefix cache or CUDA graph",
     "cuda_evidence": "record both nvidia-smi and PyTorch name/capability, device UUID, and /health linear_attention while the worker runs with --device cuda",
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
   python benchmarks/bench.py run benchmarks/smoke.jsonl \
     --endpoint http://127.0.0.1:8000/v1/systemone \
     --model clef-flash \
     --metadata <metadata.json> \
     --phase feasibility \
     --concurrency 1 --warmup 5 --timeout <seconds> \
     --output <fresh-directory>/feasibility
   ```

   For the three measured traversals, replace `benchmarks/smoke.jsonl` with
   the frozen `<public231.jsonl>` and use `--phase measured`, distinct output
   directories, and the same metadata, timeout, warmup and concurrency. Leave
   the service alive between traversals. Exclude feasibility from pooled latency.
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
   code tip with its newly measured reference under the same fixed operator
   configuration. Keep the historical `9030065`/`594528a9` comparison separate.

## What remains unverified

Clef 27B, images/video, the native joint head, FlashInfer and other GPUs have
not been run in this acceptance. The default remains FLA; making another
prefill path the default needs its own forward, output-parity and HTTP evidence.
The earlier `9030065`, `594528a9` and `f9127a0` operator reports retain their
original provenance and limitations; they were not reproduced or relabelled
as results for this service. No vLLM performance comparison is established.

## Service lifecycle acceptance

Use a writable recipe environment and keep the pinned snapshot immutable. Set
`PYTHONDONTWRITEBYTECODE=1` for both the Rust launcher and direct Python
checks: the snapshot verifier rejects a `__pycache__` directory in the snapshot.
Start the server through the execution host's GPU reservation mechanism.

Readiness requires the executor's actual startup forward to finish before the
HTTP listener opens. The first `/v1/systemone` request after readiness and a
repeat of that request should return the same answers and token usage. The
Python process owns the model; its main thread loads, warms and scores it.

The server reads each complete HTTP body before FIFO admission. Unlike the
previous Python handler's `503 worker busy`, concurrent requests wait for the
single executor. A client that disconnects while waiting is removed from that
queue. A disconnect after dispatch does not interrupt the model: the blocking
IPC exchange retains the executor and scheduler permit until the response is
fully read. The next request cannot consume the cancelled request's response.

Health checks also inspect whether the child process has exited. An idle exit
returns `503 {"detail": "worker unavailable"}` without requiring a probe
inference. An IPC failure during execution returns `500 {"detail": "inference
failed"}` to a still-connected caller, marks the executor unavailable before
releasing admission, and makes later health checks and requests return 503.
This failure publication also occurs when the original client has disconnected.
Requests already past the initial liveness check, including queued requests,
can return 500 if the executor fails before their exchange. New requests after
failure is detected return 503.
Restart the server to recover from a dead or invalid IPC stream. A valid
model-generated error response retains its original status/body and does not by
itself poison the stream.

The queue is not bounded and a live executor that never answers has no inference
deadline. Client cancellation does not promise to stop submitted CUDA work.

The CPU tests under `tests/clef/serve.rs` cover real TCP disconnects, queued
cancellation, retained resources after dispatched cancellation, idle child exit,
and protocol failure after the caller has gone away. Run those with
`cargo test --locked -p omni-clef`; these fake-worker tests do not prove GPU
output parity or latency. The shared scheduler's resource tests are in
`cargo test --locked -p omni-runtime`.
