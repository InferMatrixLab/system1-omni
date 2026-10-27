# Clef reference worker

Serve Cloudflare Clef-flash or Clef on `POST /v1/systemone`. The HTTP server
is the Rust binary `omni-clef`. It admits one request at a time and forwards
the body to one Python executor process. That process loads the pinned
snapshot's joint schema model and scores every request on its main thread.
The contract, hashes, and the unsupported image, video, and native joint-head
paths are in the [model README](../../src/models/clef/README.md). Commands
below run from the repository root.

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
opt-in flag bound. `/health` `backbone` is `eager` and `backbone_graphs`
is 0 unless the process was started with `--backbone piecewise` and
inductor actually captured. The PyTorch `torch_chunk_gated_delta_rule` and
`F.conv1d` paths are not served.

FlashInfer prefill calls `flashinfer.gdn_prefill.chunk_gated_delta_rule`
with `use_cp=False`. Q and K L2 normalization uses FLA `l2norm_fwd`, and
FlashInfer's `use_qk_l2norm_in_kernel` flag stays false. Log-space `g` is
exponentiated. The call is one sequence and does not continue a recurrent
state. FlashInfer 0.6.16 does not dispatch that kernel when the device
reports capability 8.9. vLLM's separate CuteDSL prefill backend is not
used. CUDA graphs and `torch.compile` stay off on the default
`--backbone eager` path. `--backbone piecewise` is a separate opt-in
described below. It does not select FA3 and it does not change the FLA
prefill.

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
  --backbone eager \
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
`max_input_tokens: 16384`, `linear_attention: fla+causal_conv1d`,
`backbone: eager`, and `backbone_graphs: 0` on the default flags. A CUDA
init failure exits before the port opens. That startup forward was not
run in this editing environment.

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
PYTHONPATH=src python -m compileall -q src/models/clef src/frontend/clef.py tests/clef recipe/clef/answer_parity.py
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
in the environment that edited this recipe. This paste is the primary a
later tip has to beat.

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

This code tip stays. A later prefill change is a new tip and needs its own
Public231 C1 paste before another delta is claimed. Compare that paste
with p50 164.969 ms and p95 216.755 ms.

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

The primary to beat is the `594528a9` remeasure: p50 164.969 ms, p95
216.755 ms, 231/231, `/health` `fla+causal_conv1d`. A new tip needs its
own paste against those two figures.

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
   is not a KEEP; the primary remains `594528a9`. A later code tip still
   needs its own paste.
   `GET /health` is absent until the startup forward returns, then must
   show `status` `ready`. The default `linear_attention` is
   `fla+causal_conv1d` and the default `backbone` is `eager` with
   `backbone_graphs` 0. `flashinfer+fla+causal_conv1d` means the process
   was started with `--gdn-prefill flashinfer`. `piecewise+inductor` means
   `--backbone piecewise` captured. Record both strings. A
   process that exits at startup did not bind `causal_conv1d_fn` and a
   chunk kernel. Do not measure a process that is still on the PyTorch
   chunk or `F.conv1d` path.
3. Obtain the frozen Public231 request manifest whose file SHA-256 is
   `00d1f10ca1142499a598b5f1c6602d329cce1cef04c961c46fc0318daa2c67d3`.
   That is the JSONL byte hash cited for the JevBench `7ce310c7` export in
   [the latency leaderboard](../../benchmarks/latency-leaderboard.md).
   The canonical task hash
   `dc3995d8ae1e2fc8e81ce38431add509eb8bb39b85aadfd0c7c32079382dde51`
   is a different encoding and must not be substituted. Check the file
   with `sha256sum` before `validate`. The `9030065` rounds did not use
   that byte string: the `thinking` key had been removed, and the resulting
   file hash was not reported. The `594528a9` remeasure names stock
   Public231 with `thinking.enabled` false and also does not report a file
   hash. Record `sha256sum` of whichever file the next run actually sends.
   Keep `thinking` when its `enabled` value is false; stripping it is no
   longer required for admission.
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
     "cache_policy": "one request at a time on the executor main thread; Rust FIFO admission; no prefix cache; backbone eager (no CUDA graph); transformers cache left at its default",
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
   code tip with the recorded primary, tip `594528a9` at p50 164.969 ms
   and p95 216.755 ms, only after that new tip has its own paste. The
   `9030065` run1 figures remain the baseline of that recorded comparison.

## Opt-in piecewise backbone

This is a draft tip. It does not change the default and it does not move
the `594528a9` primary. FlashInfer GDN prefill and FA3 stay deferred.
`torch.compile` of `ClefModel` is refused: an operator negative control of
that full-model compile was +773 ms, attributed to `.item()` in
`JointSchemaHead.forward`, and was not remeasured here.

`--backbone piecewise` compiles the token mixer and the MLP inside each
`Qwen3_5DecoderLayer` with inductor `mode=reduce-overhead`,
`fullgraph=False`, and `dynamic=False`. The layer forward itself stays
eager. Each residual is `hidden_states.clone()` outside `torch.compile`,
once before the mixer and once before the MLP, so the next CUDAGraph
replay cannot overwrite that output. Capture splits, and only these:

- FLA `ChunkGatedDeltaRuleFunction`, via `torch.compiler.disable` on each
  `Qwen3_5GatedDeltaNet.chunk_gated_delta_rule`. The chunk runs eager. A
  warmup that sees the chunk inside CUDA capture, or inside Dynamo
  tracing, exits before `/health`.
- `JointSchemaHead`. Its forward is disabled and is not compiled.
  `ClefModel.forward` stays eager, so the head's `.item()` is outside
  every graph.

Full attention stays the checkpoint kernel. The flag does not select FA3
and does not pass `--gdn-prefill flashinfer`. Start it with the same
`CLEF_PYTHON` recipe interpreter as the eager server:

```sh
PYTHONPATH=src python -m frontend.clef \
  --model <snapshot> \
  --variant clef-flash \
  --device cuda \
  --gdn-prefill fla \
  --backbone piecewise \
  --host 127.0.0.1 \
  --port 8000
```

`/health` `backbone` is `piecewise+inductor` and `backbone_graphs` is a
positive integer only after the three startup forwards captured at least
one inductor `CUDAGraphNode` and the chunk split ran outside that
capture. Any other outcome exits. Do not treat a process that is still
`eager`, or that died at startup, as this bind. `linear_attention` stays
`fla+causal_conv1d`.

Each new sequence length specializes a static graph. The first request of
a length pays capture. Later requests of that length replay. This editing
environment did not capture.

Answer parity is the gate against eager FLA. Tolerances are fixed at
probability drift 0.002, score drift 0.002, and zero decision flips. Do
not widen them. From the repository root, after two bench runs and the
two health documents:

```sh
PYTHONPATH=src python recipe/clef/answer_parity.py \
  --eager-run <eager-bench-dir> \
  --piecewise-run <piecewise-bench-dir> \
  --eager-health <eager-health.json> \
  --piecewise-health <piecewise-health.json> \
  --device "<nvidia-smi name>" \
  --compute-capability "<printed capability>" \
  --report <fresh-report.json>
```

Exit 1 fails the tip. The report records `latency_keep: false`.

GPU work for the parent, not run here. Record `device` and
`compute capability` on every paste. Clock
`post_to_body_json_and_schema_validation`. The operator-supplied
comparison target for one measured round is omni p50 55.412 ms and p95
129.859 ms, against the cited vLLM pair 36.407 / 96.107. Those figures
were not produced in this checkout and are not a KEEP.

1. Capture-boundary smoke. `benchmarks/smoke.jsonl` must validate, and
   `/health` must show `status: ready`, `linear_attention:
   fla+causal_conv1d`, `backbone: piecewise+inductor`, and
   `backbone_graphs` of at least 1. A startup failure is not a latency
   result.
2. Answer parity, command above, on the same manifest for `--backbone
   eager` and `--backbone piecewise`. Both sides stay on
   `--gdn-prefill fla`.
3. One Public231 measured round after warmup 5 and concurrency 1, only
   if the parity gate passed. Compare that round with 55.412 / 129.859.
   One round does not replace tip `594528a9` and does not authorize a
   merge.

## What remains unverified

Image and video inputs, and a native joint head. The `9030065`
rounds use metric `successful_latency` and clock
`post_to_body_json_and_schema_validation`. The tip `594528a9` remeasure
states metric `successful_latency`, warmup 5, concurrency 1, and stock
Public231 with `thinking.enabled` false. It does not restate the clock
name, manifest hash, driver, CUDA version, or answer parity. This
environment has no GPU and did not repeat either campaign, and it did
not repeat the `f9127a0` `omni-clef` paste above. That paste is not a
KEEP, and it does not replace tip `594528a9` as the primary. H1, one encode
per request, is already inside the `9030065` baseline, so its isolated
effect is unknown. The recorded win, and the primary a later tip has to
beat, is the FLA chunk plus `causal_conv1d` on the reported-8.9 host:
p50 164.969 ms, p95 216.755 ms. The default prefill remains that FLA
chunk, including on reported majors 9, 10, and 12. `--gdn-prefill
flashinfer` is opt-in. It has no measured forward and no answer parity
against `fla+causal_conv1d`; that parity run is the named follow-up
blocker before the flag becomes the default. `--backbone piecewise` is
also opt-in and unmeasured. Its capture, answer parity, and the
single-round comparison with the operator-supplied 55.412 / 129.859
pair were not run here. The startup forward that
runs before `/health` is separate from the client warmup in the C1
protocol and has no latency claim. A later code change is unmeasured
until its own paste.
