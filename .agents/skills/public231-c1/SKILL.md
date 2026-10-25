---
name: public231-c1
description: Measure Public231 C1 request latency for a model or pull request. Use when a campaign needs warmup 5, concurrency 1, three measured rounds, SHA, pass rate, and p50/p95 on the bench.py clock. Every paste records measured device and compute capability. The documented production target is H200 / SM90; a different measured identity stays a valid run and pairs only with that same recorded identity. Includes how to treat a thinking field and what a room post may contain.
---

# Public231 C1

Measure one deployment of the frozen 231-task Public231 text workload at
concurrency 1. Read [the benchmark protocol](../../../benchmarks/README.md) and
[the latency leaderboard](../../../benchmarks/latency-leaderboard.md) first.
`benchmarks/smoke.jsonl` is a contract smoke file, not this campaign.

This skill records a measurement. It does not authorize a merge, a hardware
relabel, or a room post that contains operator metadata.

## Measured hardware

The production GPU, when a document is naming that intended target, is
**H200 / SM90**. That is the preferred label in examples. It is not a gate.
A run is not INCONCLUSIVE because `device` is not H200, not L20X, or not
another SKU, and not because `compute capability` is not SM90 or not sm_89.

Every paste of a run records both measured fields:

- `device`: the device-name string the host reported. `nvidia-smi` is the
  source. The runner metadata field for that string is `gpu`.
- `compute capability`: the capability from the worker or device log.

Preferred production paste:

```text
device: H200
compute capability: SM90
```

If `nvidia-smi` prints a different name, paste that string as `device` and
paste the capability that was actually reported. Do not rewrite either field
to H200 / SM90, and do not rewrite them to L20X / sm_89. Archived notes in
[the GDN note](../../../benchmarks/gdn/README.md),
[the prefill-batching note](../../../benchmarks/prefill_batching/README.md), and
[the Open-Jev 27B validation](../../../recipe/open_jev/validation.md) disagree
about a driver string versus a corrected label for those campaigns. This
protocol keeps the string this host printed. If either field cannot be read,
leave it missing. Do not invent it.

Paired runs, including a baseline and a later tip, compare only when both
sides use this same measurement protocol and the same named hardware: the
same recorded `device` string and the same recorded `compute capability`.
A different recorded identity is a separate group. It does not by itself make
either run inconclusive.

## Freeze the run

Use one frozen Public231 text manifest of 231 tasks. Record
`manifest_sha256` from the run's `config.json`. The leaderboard's canonical
dataset hash and the issue #130 JSONL hash are different representations of
related inputs; do not treat them as the same file checksum, and do not mix
this runner's clock with the historical H800 body-receipt clock.

The runner replays request fields unchanged aside from the served model alias
(`--protocol systemone`, the default). Keep precision, request order, cache
policy, and the same server process across the three measured rounds. Exclude
process startup, readiness, and warmup from the latency samples. A feasibility
run, if you take one, stays out of the comparison.

Set runner metadata as
[the protocol requires](../../../benchmarks/README.md#run-against-a-ready-gpu-server):
`gpu`, `gpu_ids`, `driver`, `cuda`, `precision`, `model_revision`,
`runtime_revision`, `cache_policy`, `cuda_evidence`, and `reservation`.
`runtime_revision` is the full git SHA that was actually measured. Fill
`reservation` under the execution host's own rules. Do not copy that field,
or any hostname, SSH target, username, filesystem path, or reservation-tool
name, into a room post.

Pass the model wire contract the server implements. Decider's scalar Score is
two decimals, so that deployment needs `--score-decimals 2` on feasibility and
measured runs or warmup will reject valid responses. The default is 4.

## Thinking field

C1 is a no-thinking measurement unless the campaign text freezes thinking on.
Document which of these two treatments the manifest uses, and use only one
inside a campaign:

1. **Accept `enabled: false`.** Leave a `thinking` object in the request and
   set `enabled` to `false`. Keep the key when the frozen manifest or the
   server branches on the object being present.
2. **Strip the field.** Delete `thinking` from the no-thinking twin so the
   body matches workers whose no-thinking path is the absence of the field.
   On the JEV-VL single-question contract the field is optional; the
   implemented string values are `default` and `off`, and `auto` / `on` are
   rejected.

Do not send thinking-on content and call the run C1. Do not flip Cua-S1's chat
template with `enable_thinking=False`; that changes the prompt suffix and is a
different contract (`src/models/cua_s1/README.md`). The `jev-vl` bench adapter
rebuilds a one-choice body and drops other fields, so it is not the Public231
text campaign.

## Measure

`bench.py` defaults `--warmup` to 3. This protocol passes **5** on every
measured round. The historical H800 board warmed up once for all three
traversals and used a different clock; do not copy that shape. Invoke `run`
once per fresh output directory. Three traversals are still 231 tasks, not 693.

```sh
python benchmarks/bench.py validate <public231.jsonl>

python benchmarks/bench.py run <public231.jsonl> \
  --endpoint <ready /v1/systemone URL> --model <served-alias> \
  --metadata <run-metadata.json> --phase measured \
  --concurrency 1 --warmup 5 \
  --output <fresh-directory-round-1>
```

Repeat the measured command for rounds 2 and 3 on the same server, each with
its own new directory. Then recompute each round with no extra requests:

```sh
python benchmarks/bench.py summary <fresh-directory-round-N>
```

## What to report

Pool only rounds that share deployment, manifest, and timing settings.
Concatenate `latency_ms` from `responses.jsonl` where `state` is `completed`
and `error` is null. Exclude warmups and unsuccessful records. For sorted
samples `x` of length `n`, p50 is the statistical median and p95 is the
zero-based nearest rank `x[ceil(0.95 * n) - 1]`. Recompute both from the
pooled samples. Do not average the three round percentiles.

The request clock is
`latency_basis=post_to_body_json_and_schema_validation`: POST through complete
body receipt, JSON parsing, and wire schema validation. Hashing and
persistence are outside that clock. Historical leaderboard rows that stop at
body receipt and use linear-interpolation p95 are a different group. Do not
publish a speedup across those clocks.

Publish:

- full `runtime_revision` SHA and `manifest_sha256`
- `device` and `compute capability` (both, on every paste)
- thinking treatment (`enabled: false` or stripped)
- per-round and pooled p50/p95 in milliseconds
- serving pass rate: `successful_requests / planned_requests`, plus failed,
  incomplete, and unattempted counts
- `evidence_complete` from each `summary.json`

Labelled choice/noul accuracy and score MAE in
`quality_on_successful_requests` are conditional on successes. Report them
separately. They are not the serving pass rate, and a fast success subset is
not a ranked result.

A round is not a C1 baseline when warmup failed, `evidence_complete` is false,
any measured request failed or was incomplete, or `device` or
`compute capability` was not recorded. Call that run **INCONCLUSIVE** and do
not repair hashes or counts to make it look complete. Failing to match H200,
SM90, L20X, or sm_89 is not a reason to mark the run INCONCLUSIVE.

## Room posts

Post metrics only: model or PR, SHA, `device`, `compute capability`, warmup 5,
concurrency 1, three rounds, clock basis, thinking treatment, pass rate, and
p50/p95, plus INCONCLUSIVE when that is the result. A production-target
example is `device: H200` and `compute capability: SM90`; if the host string
differs, post the recorded pair instead.
Omit hostnames, SSH targets, usernames, filesystem paths, and reservation-tool
names. Local output directories stay local.
