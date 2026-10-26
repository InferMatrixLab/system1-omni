# GPU serving benchmark

First harness for #39: replay the same labelled requests against CUDA-backed
`laya-serve`, vllm-jev, and System1-Omni. Python 3.11+ and `httpx==0.28.1` are
required (`python -m pip install -r benchmarks/requirements.txt`). Reuse an
existing benchmark environment when available.

The [latency leaderboard](latency-leaderboard.md) records the five-model H800
comparison and its timing rules. Its historical body-receipt clock is kept
separate from this runner's body/JSON/schema-validation clock.

The runner is an HTTP client: it neither launches models nor reserves or accesses
a GPU. Start each server separately through the host's verified GPU scheduler,
using the same exact reserved device and pinned Laya checkpoint. Verify CUDA
execution from the worker's configuration/logs and scheduler evidence. An HTTP
response alone cannot prove the backend used CUDA.

## Freeze requests and controls

The JSONL format is one object per request:

```json
{"id":"sample-1","request":{"state":"Refund my duplicate charge.","questions":{"refund":{"type":"noul","instructions":"Does the customer request a refund?"}}},"expected":{"refund":true}}
```

Use a criteria map and a label target for Choice; an ordered criteria list and a
numeric level target for Score. Every question needs a target. With the default
`--protocol systemone`, the runner adds only the backend model alias; all other
request fields are replayed unchanged.
Keep source dataset revision, selected record IDs, sampling seed, transformation
code revision, token lengths and truncation decisions with the manifest. Freeze
short/near-limit and option-count slices as separate manifests.

`smoke.jsonl` contains four synthetic contract checks, including a mixed request.
Its scores are illustrative labels, not a general accuracy benchmark or a useful
throughput workload. Prepare a larger held-out manifest for measurements, such as
a fixed 500-question MMLU test sample. Dataset acquisition and conversion are not
implemented by this first runner.

```sh
python benchmarks/bench.py validate benchmarks/smoke.jsonl
```

Before measuring, save run metadata as JSON. Required fields are `gpu`, `gpu_ids`,
`driver`, `cuda`, `precision`, `model_revision`, `runtime_revision`, `cache_policy`,
`cuda_evidence`, and `reservation`. These are operator-supplied evidence, not
automatically verified hardware measurements. Use exact revisions and include
working-tree patches, dependency versions, checkpoint configuration, tokenizer,
temperatures, affinity, launch commands and preparation/readiness timings.
Record peak GPU memory and its measurement method externally under the task's
reservation; the client does not sample GPU memory.

Fix precision, context/truncation policy, inputs and cache conditions across
backends. Record necessary differences. Keep cold-cache experiments separate;
this runner always performs readiness inference and warmup and thus measures
warmed serving. It does not reset caches. Never drop shared host caches.

## Run against a ready GPU server

```sh
python benchmarks/bench.py run requests.jsonl \
  --endpoint http://127.0.0.1:8000/v1/systemone --model english \
  --metadata run-metadata.json --phase feasibility \
  --concurrency 1 --output /tmp/laya-feasibility

python benchmarks/bench.py run requests.jsonl \
  --endpoint http://127.0.0.1:8000/v1/systemone --model english \
  --metadata run-metadata.json --phase measured \
  --concurrency 1 --warmup 5 --output /tmp/laya-c1-run1
```

Use the checkpoint's served alias for vllm-jev/native implementations. For the
frontend-overhead comparison, reuse the same external CUDA worker and change
only the endpoint. Optional authentication comes from `OMNI_JEV_TEST_TOKEN`;
the token is not saved. Redirects, proxy environment variables and retries are
disabled. `--timeout` is a total deadline for each HTTP exchange, including body
receipt; response validation is included in recorded client latency.
Each answer must be an object whose `type` matches the requested question type.
Missing or incorrect System One answer types are `invalid_response`; the runner
keeps the original body and does not supply missing fields.
For JEV-VL, use `--protocol jev-vl` with a manifest containing one Choice question
per request. The adapter checks the returned options, index, label and maximal
probability, then supplies the requested `type` in the normalized answer envelope.
Both online collection and offline replay use this same adapter. An image-only
state object is sent as a media-parts list; other state objects remain text data.
Use `--score-decimals 2`, `3` or `4` to match the model's scalar Score precision
(default `4`); this is separate from the GPU dtype in metadata. Probabilities
remain four-decimal wire values. Score validation bounds their independent
rounding together with the scalar's rounding, and accepts Decider's all-zero
Score distribution only with its complete isolated zero-fit metadata.
New runs preserve HTTPX `response.content` bytes and decode them as strict UTF-8
before JSON/schema validation, both online and offline. Invalid UTF-8 is an
`invalid_response`, even in an otherwise valid answer's unused field. HTTP charset
headers do not override this JSON contract. No BOM is stripped or missing field
supplied; an initial U+FEFF remains invalid. These bytes are HTTPX's body after
HTTP content decoding, not compressed wire bytes or HTTP framing.

For Decider, pass `--score-decimals 2` in both feasibility and measured runs;
its scalar Score is rounded to two decimals while probabilities use four.
The default four-decimal Score contract can reject valid Decider responses,
including during readiness and warmup. For example, with a ready native worker:

```sh
python benchmarks/bench.py run requests.jsonl \
  --endpoint http://127.0.0.1:8000/v1/systemone --model decider \
  --metadata run-metadata.json --phase feasibility --score-decimals 2 \
  --concurrency 1 --output /tmp/decider-feasibility
```

Use the same flag for each measured run. See [Score wire precision](#score-wire-precision)
for the validation rules and recorded precision metadata.

Start with the weight-free preflight, then one feasibility run per configuration.
Exclude feasibility from performance comparisons. Use two measured runs at each
of concurrency 1, 8 and 16, with the same request ordering. Reuse the server;
report variation rather than silently extending an inconclusive run budget.
Use separate output directories; existing results are never overwritten.

For the initial frozen 231-task text campaign, predeclare three complete measured
traversals at concurrency 1, with five warmup requests per traversal and the same
running server. Invoke `run` separately for each output directory. Repeating the
manifest three times still represents 231 independent tasks, not 693 tasks.

A Public231 tip that already has a full-C1 primary can be dropped before those
three traversals. Validate `benchmarks/smoke.jsonl` first. `--fast-screen` then
samples 60 requests from a manifest whose cases include integer `prompt_tokens`
(not sent on the wire): quotas 10, 10, 8, 8, 8, and 16 for buckets 0–127,
128–255, 256–511, 512–1023, 1024–2047, and 2048+. Seed 231, one measured round,
concurrency 1, warmup 5. The screen baseline is an earlier `--fast-screen` of
the primary SHA on those same 60 IDs (`--baseline-screen`). Do not pass
`--baseline-p50-ms` or `--baseline-p95-ms`; full-C1 pooled numbers, including
StartLux Instant 7.338/51.816, are refused. The StartLux Instant screen
primary is `TBD — measure on primary SHA same 60 IDs`. An Instant
`usage.input_tokens` index was refused: bucket 1024–2047 has 6 prompts and
the quota is 8. Do not fill the slot until that bucket has at least 8, or
InferRSI approves a quota change.
`fast_screen_decision` is `screen_primary`, `drop` (only when both p50 and p95
are worse by ≥ +2% or ≥ +0.5 ms), `full_c1`, or `inconclusive`. A `drop` writes
`dropped_at` in `fast_screen_drop.json`. The field is never a KEEP. The gate is
`.agents/skills/public231-c1/SKILL.md`.

```sh
python benchmarks/bench.py run requests.jsonl \
  --endpoint http://127.0.0.1:8000/v1/systemone --model english \
  --metadata run-metadata.json --phase measured \
  --concurrency 1 --warmup 5 \
  --fast-screen --baseline-screen /tmp/public231-screen-primary \
  --output /tmp/public231-fast-screen
```

By default warmup cycles through the measured manifest. Use
`--warmup-manifest warmup-requests.jsonl` to supply independent labelled requests
in the same format. They are saved with their checksum and never enter measured
counts. The first entry in `warmup.json` is the first request observed by this
client invocation. It is not a measurement of server process startup, model
loading or service-internal warmup; save those boundaries externally. An
independent warmup manifest does not establish image-path or cache coverage.

Concurrency is a closed-loop count of in-flight HTTP requests. Fixed workers
consume the manifest in order; completion order is naturally variable. Each
worker sends its next request only when the previous one completes. There is no
arrival-rate simulation or client queue-delay measurement.

Outputs:

- `config.json`: manifest/runner SHA256, runtime metadata and client settings.
- `requests.jsonl`: exact manifest snapshot.
- `warmup-requests.jsonl`: independent warmup snapshot, when supplied.
- `warmup.json`: first-inference readiness check and warmup responses, excluded
  from load timing. Warmup failure stops the run and saves completion/summary
  evidence that all measured requests are unattempted.
- `responses.jsonl`: flushed after each request finishes, in completion order
  during replay, then atomically reordered to manifest input order for a complete
  traversal. Interrupted runs retain their actual saved completion order,
  with the frozen request ID, lossless `response_bytes_base64`, its SHA256, HTTP status,
  validation result, client latency and any error. Join records to the manifest
  by ID; `compare` accepts different completion orders. The `response` text is a
  display copy decoded with replacement, never the new format's validation input.
  No HTTP body is represented by null; an empty HTTP body has an empty base64 string.
- `completion.json`: atomically replaced before each measured request starts and
  after it finishes. Contains attempted/completed/active IDs, input/config/response
  checksums, measured start, terminal stop reason and real measured wall time.
  A failed/interrupted warmup has no measured start/finish/wall or responses file;
  `stop_reason` is `warmup_failed` or `warmup_interrupted`, with no attempted IDs.
- `summary.json`: successful requests/s and decisions/s, successful-request
  p50/p95 latency (nearest-rank p95), failure counts by kind, and quality metrics
  with denominators, failure latency p50/p95 and planned/attempted/unattempted/
  incomplete request and decision counts. A decision is a question, not an
  internal candidate branch.

Any request error makes the run exit nonzero after saving results. Malformed
responses count as failures. Quality and latency summaries cover successful
requests only; completed failure latencies have a separate summary. Always publish
failure counts alongside those summaries. Different failure rates invalidate a
simple speedup claim.

The request clock spans POST through complete body receipt, JSON parsing and wire
schema validation (`latency_basis=post_to_body_json_and_schema_validation`). It
excludes result hashing and persistence. Measured wall includes per-request
hashing, start/completion saves and closed-loop client bookkeeping; throughput is
successful requests or decisions divided by this saved wall. The real elapsed
time includes the complete traversal's response reorder/save/checksum; final
terminal-state and summary writes occur after that boundary. Use the same
collector version and persistence policy across configurations. Do not derive speedup ratios against
historical body-only client timing.

A soft interruption saves finished samples and records cancelled requests as
`incomplete`, with `elapsed_until_abort_ms` and no completed `latency_ms`. If a
complete body was received before cancellation, its bytes remain available for
independent offline scoring. Abort durations do not enter latency percentiles.
Cancellation of the client does not prove the server stopped processing.

Request counts obey `planned = attempted + unattempted` and
`attempted = successful + failed + incomplete`; decision counts follow the same
partition. `failed` refers to completed failures. The retained `requests` field
counts persisted response records, including explicit incomplete records.
`evidence_complete` means a terminal, complete traversal, including any failures;
it does not certify model quality or a performance comparison. Conditional quality
is this tool's simple metric and must not replace a dataset's pinned scorer.

## Recompute a saved summary

```sh
python benchmarks/bench.py summary /tmp/laya-c1-run1
```

This command makes no requests and prints the recomputed summary without changing
saved files. It checks frozen IDs, duplicates and unknown entries, saved manifest,
config, warmup and response checksums, and agreement between saved response bodies and
normalized answers/validation. It requires the new `completion.json` evidence;
older runs remain usable by `compare` but cannot acquire unrecorded timing or
interruption boundaries through this command.
Runs recorded before `warmup_sha256` was added must use their original collector
for offline summaries. Do not add a checksum to historical evidence to make it
look verified at collection time.

New runs save `protocol`, `score_decimals` and
`validation_protocol=probabilities-4dp-model-score-v1` in the checksummed config.
Offline replay validates these settings before interpreting response records.
Unknown protocol/version values and unsupported or non-integer precisions are
rejected, even if the config checksum matches. Missing protocol and precision
selectors default to `systemone` and `4`.

The independent `response_format=httpx-content-base64-utf8-v1` selector binds the
new byte evidence and strict decoder to the config checksum. Offline replay
checks base64, the original byte SHA256 and the display text's consistency before
revalidating answers. Unknown response formats are rejected. Supported archives
without this selector, or with `httpx-text-v1`, retain their original saved-text
parsing and text SHA256 semantics, including HTTPX replacement decoding. Such
archives cannot establish the validity of the original UTF-8 bytes. Do not add
byte fields or rewrite old configs/hashes to make historical evidence look verified.

For supported archives without `validation_protocol`, the reader uses
`systemone-4dp-score-abs1e-4-v1`: the original typed System One validation and
`math.isclose(..., abs_tol=1e-4)` Score rule. This legacy contract only permits
System One and four-decimal Score. It does not reinterpret old failures using
the newer independent-rounding allowance. These defaults do not make pre-hash
archives readable: runs without the completion/warmup checksums still require
their producing collector. Do not rewrite historical configs or hashes.

The Score rules reuse [#93](https://github.com/InferMatrixLab/system1-omni/pull/93),
and the wire adapters reuse [#141](https://github.com/InferMatrixLab/system1-omni/pull/141).
The weekly consumer must require `evidence_complete` to be exactly true alongside
zero failures before aggregating measured passes. That consumer change belongs
to #141; retain its companion guard when combining the branches.
CPU regression coverage includes both protocols, supported precisions,
interrupted/torn snapshots, raw-response hashes and online/offline/CLI agreement,
alongside the retained deep-JSON and HTTP 600/999 cases.

For pooled percentiles, first validate each of the three measured rounds with its
original collector's `summary`; pool only rounds with the same deployment,
manifest and timing/collection settings. Concatenate `latency_ms` from their
`responses.jsonl` records where `state` is `completed` and `error` is `null`,
keeping every repetition and excluding warmups and unsuccessful records. For
sorted samples `x` and `n = len(x)`, p50 is `statistics.median(x)` and p95 is
`x[math.ceil(0.95 * n) - 1]` (zero-based nearest rank; both null if empty).
Recompute from these samples, never average round percentiles; report per-round
and pooled request counts alongside them.

After a hard kill, a valid start snapshot can distinguish unattempted IDs from
attempted IDs without completed records. Without a saved terminal boundary,
`wall_seconds` and both throughput fields are null and `evidence_complete` is
false. A response flush and its completion snapshot are separate writes. A kill
between them can leave an extra response with a stale checksum or completed-ID
list. Such runs, partial files and missing claimed records are rejected; inspect
the retained originals instead of repairing timing or
declaring a complete baseline. Hashes detect inconsistency, not authenticity
against someone rewriting all evidence and hashes. Interrupted runs provide
partial observations with explicit coverage, not complete traversals.
Flushing records and atomically replacing state preserve consistent snapshots
when both writes finish. They do not promise machine power-loss
durability; this tool does not call `fsync`.

## Compare output fidelity

Predeclare tolerances; these flags are illustrative, not established merge gates:

```sh
python benchmarks/bench.py compare /tmp/reference /tmp/candidate \
  --max-probability-drift 0.002 --max-score-drift 0.002 --max-flips 0
```

Comparison requires identical manifests and covers all request IDs. It fails on
request errors or exceeded tolerances, reporting probability/Score drift and
Choice/Noul flips separately from ground-truth quality. This is an output-fidelity
check, not automatic proof of equivalent hardware or a performance winner.

## Self-review and A/B evidence

Use the repository's [self-review skill](../.agents/skills/self-review/SKILL.md)
with its contribution guide and PR template. Decide whether measurements are
needed before starting GPU work:

- Performance or accuracy improvement claims require reproducible comparison
  evidence against a relevant reference.
- For scheduler, batching, cache, kernel, precision, model or serving changes,
  identify a concrete performance/numerical risk and choose a focused comparison
  when it can resolve that risk. Output-affecting changes need reference-output
  checks and applicable labelled quality; equal accuracy alone is not parity.
- Documentation, unrelated changes and benchmark tooling without performance
  claims do not automatically need GPU experiments. Run the applicable checks
  and state why an A/B test is unnecessary.

Freeze revisions, the isolated variable, controls, success criteria and stop
condition before execution. Follow the run budget above: one feasibility run,
then two measured repetitions per configuration unless a different budget was
predeclared. Preserve every result and report variation. Reuse prior evidence
only if its revisions, inputs and controls cover the reviewed change.

Review errors and metric denominators alongside latency and throughput. Keep
SDK batch timing separate from HTTP latency and frontend overhead separate from
native acceleration. Report missing measurements as unverified, and remove or
qualify unsupported claims. Stop at the declared error/run limit; do not silently
relax tolerances, discard failures or add runs to obtain a preferred outcome.

### Known validator limitation

The initial GPU run at `9ae70ce` rejected two rounded probability distributions
summing to `0.9998999999999999`: floating-point roundoff put their deviation just
outside `abs_tol=1e-4`. See the [run report on PR #40](https://github.com/ThinkFlowLab/system1-omni/pull/40#issuecomment-5894999739).
That run stopped before completing the A/B comparison. Check for a fix and
regression tests before relying on this runner for full measurements. A validator
change needs a recorded revised protocol; partial results are not a validated
performance baseline.

## Remaining work in #39

Completed GPU comparisons, a checked-in frozen MMLU manifest, automated memory collection, NLL/Brier
metrics, aggregated repeated-run reports and a separate offline SDK comparison
remain to be delivered. Reuse the upstream [Laya batch benchmark] and
[evaluation harness]. SDK batch time must not be mixed with HTTP latency, and
batch time divided by size is not request latency. A frontend/external-worker
result is not a native-engine speedup. Metal and vision benchmarks follow CUDA.

[Laya batch benchmark]: https://github.com/NandhaKishorM/laya/blob/main/benchmarks/bench_predict_batch.py
[evaluation harness]: https://github.com/NandhaKishorM/laya/blob/main/docs/evals.md

## Tests (no model or GPU)

```sh
python -m unittest discover -s tests/benchmarks -p 'test_*.py' -v
```

## Frozen serving matrix

See [the matrix client guide](serving_matrix.md) and [example plan](serving_matrix.example.json)
for exact response oracles, workload variants, concurrency rounds, pinned runtime
metadata and environment-based bearer authentication.

## Laya on Apple Silicon

[`laya_mps/`](laya_mps/README.md) holds the scripts behind the numbers of the
[Apple Silicon recipe](../recipe/laya/apple-silicon.md): in-process and HTTP latency, paired comparisons,
profiling and the report with its output-parity section.

## Score wire precision

`bench.py run --score-decimals {2,3,4}` declares the model's scalar Score
rounding contract (default: four decimals; Decider: two). Probability precision
remains four decimals. The validator bounds independent scalar and weighted
probability rounding, requires the exact level range, and preserves wire values.
The same optional `score_decimals` argument is accepted by `read_answers`,
`request_one` and `replay`; pass it by keyword when calling the collector, since
`replay` retains its positional `output` argument. Run configuration records it
for reproducibility.
Decider's isolated all-zero Score distribution requires complete consistent
zero-fit metadata; other non-normalized distributions remain invalid.
