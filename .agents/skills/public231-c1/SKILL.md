---
name: public231-c1
description: Measure Public231 C1 request latency for a model or pull request. Use when a campaign needs the contract smoke, a fast-screen (quotas 10/10/8/8/6/16, 58 requests when every bucket is full) that can drop a clear regression, then warmup 5, concurrency 1, three measured rounds, SHA, pass rate, and p50/p95 on the bench.py clock. A fast-screen is not a KEEP and is not a gain. A short 2048+ bucket is inconclusive and cannot early-DROP. Every paste records measured device and compute capability. The documented production target is H200 / SM90; a different measured identity stays a valid run and pairs only with that same recorded identity. Includes how to treat a thinking field and what a room post may contain.
---

# Public231 C1

Measure one deployment of the frozen 231-task Public231 text workload at
concurrency 1. Read [the benchmark protocol](../../../benchmarks/README.md) and
[the latency leaderboard](../../../benchmarks/latency-leaderboard.md) first.
`benchmarks/smoke.jsonl` is a contract smoke file, not this campaign.

This skill records a measurement. It does not authorize a merge, a hardware
relabel, or a room post that contains operator metadata.

## Order of operations

1. **Smoke.** Unchanged. `benchmarks/smoke.jsonl` is the contract smoke, not
   this campaign. `python benchmarks/bench.py validate benchmarks/smoke.jsonl`
   must succeed before any latency screen. A model recipe that already runs a
   server contract smoke still runs that smoke. A failed smoke is not a
   latency result and is not a DROP.
2. **Fast-screen.** One round after smoke. Quotas are 10/10/8/8/6/16 (58
   requests when every bucket is full). The screen baseline is the primary
   SHA served on the same machine and measured on these exact request IDs.
   It is not the full-C1 pooled p50/p95. DROP the tip when both screen
   percentiles are clearly worse than that same-ID baseline. Otherwise run
   full C1. The screen never KEEPs a tip. If 2048+ has fewer than 16
   prompts, the screen is inconclusive and cannot early-DROP.
3. **Full Public231 C1.** Unchanged measurement: warmup 5, concurrency 1,
   three measured rounds. KEEP only on the full dual gate below.

## Fast-screen

Use this on the frozen Public231 manifest after smoke, before spending a
full C1. The manifest cases carry a nonnegative integer `prompt_tokens` next
to `id` / `request` / `expected`. That field is the model's prompt-token
count. `wire_request` does not send it. A file without it cannot be
screened. Do not invent lengths.

Sample with `benchmarks/bench.py run --fast-screen`. The draw is fixed:

- seed **231**
- rank is SHA-256 of the UTF-8 string `{seed}\n{bucket}\n{id}`
- each bucket keeps its quota of lowest ranks
- selected requests are replayed in original manifest order
- one invocation, one fresh output directory, one measured round
- concurrency 1, warmup 5 (warmup stays out of the percentiles)
- clock `post_to_body_json_and_schema_validation`
- same recorded `device`, `compute capability`, precision, cache policy,
  and thinking treatment as the primary

Bucket quotas (inclusive lower bound, exclusive upper bound):

| Bucket | Quota |
|---|---:|
| 0–127 | 10 |
| 128–255 | 10 |
| 256–511 | 8 |
| 512–1023 | 8 |
| 1024–2047 | 6 |
| 2048+ | 16 |

That is 58 requests when every bucket is full. 2048+ is the overweight.

Cap-to-available: if a bucket has fewer prompts than its quota, take every
prompt in that bucket and record `quota_shortfall` as `{name, quota, found}`.
Do not fill the gap from another bucket. Do not substitute another model's
lengths. Do not run a second screen round and pool it. If **2048+ has fewer
than 16**, the whole screen is `inconclusive`. It cannot early-DROP, and it
cannot be a screen primary.

Compare the tip screen's successful p50 and p95 only to a screen primary:
the same primary SHA, same `device` / `gpu`, same manifest, same seed 231,
same bucket quotas, and the same request IDs. Pass that saved run as
`--baseline-screen`. Do not pass `--baseline-p50-ms` or `--baseline-p95-ms`.
The runner rejects those flags. A full-C1 output directory is also rejected.

Full-C1 pooled numbers are not this baseline. StartLux Instant's full-C1
pooled primary remains runtime revision
`51f80ad681c8f7b1be9b22b5e700bef34e33534c` (`51f80ad`), pooled p50 7.338 ms,
pooled p95 51.816 ms, and the runner refuses that pair as a screen baseline.
Clef and pplx-decider keep their full-C1 primaries in their recipes and in
[the latency leaderboard](../../../benchmarks/latency-leaderboard.md). Do not
paste any model's pooled C1 pair into the screen.

Screen primary, same request IDs. Fill this only from a completed
`--fast-screen` whose decision is `screen_primary`. Do not invent the pair,
and do not copy the full-C1 pooled pair into it.

StartLux Instant screen primary, measured on hk2 with quotas 10/10/8/8/6/16.
58/58 success, `evidence_complete` true, decision `screen_primary`,
`quota_shortfall` `{}`:

```text
screen_primary_sha: 51f80ad681c8f7b1be9b22b5e700bef34e33534c
screen_primary_p50_ms: 19.112976267933846
screen_primary_p95_ms: 73.46072793006897
n: 58
bucket_counts: 0-127:10, 128-255:10, 256-511:8, 512-1023:8, 1024-2047:6, 2048+:16
device: L20X
compute capability: sm_89
```

The measuring client was `923c8c739bcaf3cc2b098f3cf373c2b547d2666c` plus a
local-only patch that changed the 1024–2047 quota from 8 to 6. The remote
tip may still have been on quota 8 when this was measured. The committed
runner samples these quotas (n=58), so the next screen does not need that
local patch. Artifacts on hk2:
`~/experiments/2026-10-26-pr181-screen-primary/screen-primary-n58/`
(`SCREEN_PRIMARY.json`, `fast-screen/`, `screen_ids.json`,
`local-quota-patch.diff`).

Pass that saved run as `--baseline-screen`. Do not type
19.112976267933846 / 73.46072793006897, and do not type the full-C1 pooled
7.338/51.816. Other models do not inherit this pair. An earlier hk2 screen
of Instant was refused under the previous 1024–2047 quota of 8; that
refusal is not this screen primary.

Record a screen primary with `--fast-screen` and no `--baseline-screen`.
That run's decision is `screen_primary`. It is not a tip verdict and not a
KEEP. Later tips pass that directory.

```sh
python benchmarks/bench.py run <public231.jsonl> \
  --endpoint <ready /v1/systemone URL> --model <served-alias> \
  --metadata <run-metadata.json> --phase measured \
  --concurrency 1 --warmup 5 \
  --fast-screen \
  --output <primary-screen-directory>

python benchmarks/bench.py run <public231.jsonl> \
  --endpoint <ready /v1/systemone URL> --model <served-alias> \
  --metadata <run-metadata.json> --phase measured \
  --concurrency 1 --warmup 5 \
  --fast-screen --baseline-screen <primary-screen-directory> \
  --output <tip-screen-directory>
```

Delta is tip minus baseline, in milliseconds. A percentile is **worse** when
its delta is ≥ +2% of that baseline value or ≥ +0.5 ms. The runner evaluates
that boundary as `tip >= baseline * 1.02` or `tip >= baseline + 0.5`, so an
exact threshold stays worse. Early **DROP** only when **both** p50 and p95
are worse. Mixed results and results inside that band proceed to full C1.
The saved summary field `fast_screen_decision` is `screen_primary`, `drop`,
`full_c1`, or `inconclusive`. It is never `keep`.

`drop` means do not run full C1. The run writes `fast_screen_drop.json` and
copies `dropped_at` onto the summary. `dropped_at` is UTC
`YYYY-MM-DDTHH:MM:SSZ`, written once. A DROP without `dropped_at` is not an
auditable DROP. `inconclusive` means any measured request failed, was
incomplete, or was unattempted, warmup failed, `evidence_complete` is not
true, or bucket 2048+ has fewer than 16 prompts. A short 2048+ tail cannot
early-DROP. An inconclusive screen is not a DROP and not permission to
continue. A negative screen delta is not a gain, not a new primary, and
not a KEEP. Do not claim a speedup from fast-screen numbers. Percentiles
use the same median and nearest-rank p95 as full C1, on this one capped
round (58 requests when every bucket is full), not an average of rounds.

hk2 contract smoke of client
`d2fab1e08f231fe2e23540f3b25b2650cd416d09` against StartLux Instant served
at `51f80ad681c8f7b1be9b22b5e700bef34e33534c` passed: 4/4,
`evidence_complete` true. That client has no StartLux or CUDA diff versus
`51f80ad`. Offline `tests/benchmarks` on that client: 109 OK. An offline
`httpx.MockTransport` unit test is not a substitute for that smoke and is
not a review PASS. Smoke artifacts are under
`~/experiments/2026-10-26-pr181-screen-primary/primary/smoke/` on hk2.
The 4/4 smoke is not a screen primary, not a fast-screen
latency pair, and not a KEEP. The Instant screen primary is the n=58 block
above. Room posts still omit hostnames; the hk2 note is PR evidence.

## Full C1 KEEP

The measurement commands in [Measure](#measure) stay the full gate: warmup
5, concurrency 1, three rounds, then recompute pooled p50 and p95 from the
concatenated successful samples. KEEP requires all of:

- pooled p50 strictly lower than the primary
- pooled p95 strictly lower than the primary
- pooled **2048+** p95 strictly lower than the primary's pooled 2048+ p95
- no quality collapse: serving pass rate not lower, and the quality and
  contract checks in
  [optimize-remeasure-gate](../optimize-remeasure-gate/SKILL.md) stay green

When every case has `prompt_tokens`, each round's summary includes
`latency_by_prompt_bucket`. Pool 2048+ the same way as the campaign p95:
concatenate successful `latency_ms` whose `prompt_tokens` is ≥ 2048 across
the three rounds, then take nearest-rank p95. Do not average the three
bucket percentiles. The fast-screen's 16 long requests are not this figure.
A manifest without `prompt_tokens` omits the bucket report; do not KEEP
until the same lengths are joined offline and that pooled p95 is shown.
Equal latency is not "down" and is not a KEEP. The +2% / +0.5 ms band is
only the early DROP rule. It is not a KEEP tolerance.

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

This is the full Public231 C1, run only after smoke and only when the
fast-screen decision is `full_c1` (or when establishing a model's first
primary, which has no screen baseline yet). A fast-screen `drop` stops
here.

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

A fast-screen paste is a separate record. Label it fast-screen, and include
the decision (`screen_primary`, `drop`, `full_c1`, or `inconclusive`), seed
231, the request IDs, any `quota_shortfall`, the primary SHA of the screen baseline when the run
is a tip, both deltas when they exist, `dropped_at` when the decision is
`drop`, and the same `device` and `compute capability`. Do not publish that
paste as a C1 baseline or as a speedup. Do not fill screen-primary p50/p95
from the full-C1 pooled pair.

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
names. Local output directories stay local. A fast-screen room post uses the
same omission rule and the fast-screen label above. It does not announce a
KEEP.
