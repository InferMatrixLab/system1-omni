---
name: optimize-remeasure-gate
description: Gate continuous optimization after a measured Public231 C1 baseline, including Cloud Agent and coding-agent loops. Use when a new tip must not land until smoke passes and a fast-screen either drops a clear regression or the tip is remeasured with full C1 on the same recorded device and compute capability. Every paste includes those two fields. The documented production target is H200 / SM90. Each tip names its RSI phase and the primary change versus alternatives considered, or it is not merge-clean. Incomplete evidence is INCONCLUSIVE. A fast-screen cannot KEEP a tip. The pull request stays draft until a separate owner yes.
---

# Optimize only against a remeasured tip

Use this after a Public231 C1 baseline measured with
[public231-c1](../public231-c1/SKILL.md). It is the loop for later latency
work on that deployment, including a Cloud Agent or other coding agent that
would otherwise keep committing. Read
[the benchmark self-review rules](../../../benchmarks/README.md#self-review-and-ab-evidence)
before changing inference or serving.

The gate controls the next commit. It does not authorize a merge.

## Paste the baseline before editing

Put the measured primary numbers in the task before the first optimization
edit, and again before every later tip. Every paste in this loop includes
`device` and `compute capability`: the baseline paste, the tip remeasure,
and the delta fed back. A paste that omits either field is incomplete.

- baseline full git SHA (`runtime_revision`)
- pooled p50 and p95 in milliseconds
- serving pass rate (`successful_requests / planned_requests`)
- clock basis `post_to_body_json_and_schema_validation`
- `device` (the nvidia-smi name string; runner metadata `gpu`)
- `compute capability`

The documented production target is H200 / SM90. A paste of that machine
reads `device: H200` and `compute capability: SM90`. If the host prints a
different name, paste that string and the capability that was measured.
Those are the primary figures. Per-round values stay attached so a pooled
number can be checked. A recalled or rounded figure, a single round, or a
percentile averaged across rounds is not a baseline.

## One tip, then screen, then full C1

A tip is one optimization commit on the draft branch. Do not add another
commit on top of it until that tip has either been dropped by the fast-screen
or has its own full Public231 C1 remeasure. The full gate requires the same
measurement protocol and the same named hardware as the baseline. It does not
require the SKU to be H200, L20X, SM90, or sm_89.

Same measurement protocol for that full remeasure means warmup 5, concurrency
1, three measured rounds, the same manifest, and the same clock
(`post_to_body_json_and_schema_validation`). Same named hardware means the
same recorded `device` string and the same recorded `compute capability`.
The remeasure SHA must be the tip SHA. A measurement of an older commit does
not clear the tip.

Keep controls matched to the baseline (manifest, precision, request order,
cache policy, thinking treatment, server shape). State any necessary
difference. Do not reset the comparison by changing the clock, the quantile
method, or the named hardware.

## Fast-screen before full C1

Follow [public231-c1](../public231-c1/SKILL.md). Smoke must pass before the
screen. Run one fast-screen: seed 231, concurrency 1, warmup 5, bucket
quotas 0–127:10, 128–255:10, 256–511:8, 512–1023:8, 1024–2047:6, 2048+:16
(58 requests when every bucket is full). If a bucket has fewer prompts than
its quota, take every prompt in that bucket and record `quota_shortfall`.
Do not fill from another bucket and do not substitute another model's
lengths. If 2048+ has fewer than 16, the screen is `inconclusive` and cannot
early-DROP. Compare the tip's screen p50 and p95 to the primary SHA's own
screen of those same request IDs on the same device, compute capability, and
clock. Pass that directory with `--baseline-screen`. Do not pass full-C1
pooled percentiles. `--baseline-p50-ms` and `--baseline-p95-ms` are refused.

StartLux Instant's full-C1 pooled primary remains
`51f80ad681c8f7b1be9b22b5e700bef34e33534c` (`51f80ad`), p50 7.338 ms, p95
51.816 ms. That pair is not the screen baseline. Clef and pplx-decider keep
the full-C1 primaries in their recipes and the latency leaderboard. For
every model the screen-primary slot is:

```text
screen_primary_p50_ms: TBD — measure on primary SHA same request IDs
screen_primary_p95_ms: TBD — measure on primary SHA same request IDs
```

Fill it only from a completed `--fast-screen` whose decision is
`screen_primary` on the primary SHA, same machine, same request IDs. A full-C1
KEEP moves the full-C1 primary; it does not by itself fill this slot.
StartLux Instant is still TBD until measured with quotas 10/10/8/8/6/16.
An earlier hk2 screen of the Instant `usage.input_tokens` index was refused
under the previous 1024–2047 quota of 8 (that bucket has 6 prompts). InferRSI
set the quota to 6 on 2026-10-26. Do not write 7.338/51.816 as that screen
primary. Remeasure with the new quotas before filling the slot.

Early DROP only when both screen Δp50 and Δp95 are worse than that same-ID
baseline. A metric is worse when its delta (tip minus baseline) is ≥ +2%
of that baseline or ≥ +0.5 ms. Otherwise, including a mixed result, run
full C1. The summary records `screen_primary`, `drop`, `full_c1`, or
`inconclusive`. It does not record a KEEP.

A `drop` is a regression. The artifact includes `dropped_at` in
`fast_screen_drop.json` and on the summary. Do not run full C1. Do not
stack the next commit on that tip. Do not adopt the screen p50/p95 as the
next baseline. A negative screen delta is not a gain and is not a KEEP. An
`inconclusive` screen stops the loop.

KEEP, merge, and ready-for-review require the full C1 dual gate: pooled p50
and pooled p95 both strictly lower, pooled 2048+ p95 strictly lower, and no
quality collapse under the rules in the next section. The +2% / +0.5 ms
band does not make a full-C1 tie a KEEP. hk2 contract smoke of client
`d2fab1e08f231fe2e23540f3b25b2650cd416d09` against Instant served at
`51f80ad681c8f7b1be9b22b5e700bef34e33534c` passed 4/4 with
`evidence_complete` true. Artifacts are under
`~/experiments/2026-10-26-pr181-screen-primary/primary/smoke/` on hk2.
That client has no StartLux or CUDA diff versus `51f80ad`, and offline
`tests/benchmarks` there were 109 OK. That smoke is not the screen primary.
Offline MockTransport tests are not a review PASS. The Instant screen
primary stays TBD until measured with quotas 10/10/8/8/6/16.

## INCONCLUSIVE stops the loop

Mark the tip **INCONCLUSIVE** and do not stack the next commit when any of
these are true:

- the baseline p50/p95 were not recorded before the edit
- smoke did not pass before the latency screen
- the fast-screen was skipped on a tip that already has a full-C1 primary
- the screen baseline is a full-C1 pooled p50/p95, a typed percentile, or
  different request IDs than `--fast-screen` just sampled
- a short bucket was filled from another bucket, or 2048+ has fewer than 16
  and the screen was treated as a DROP or as a screen primary
- the fast-screen decision is `inconclusive`
- a DROP has no `dropped_at`
- a KEEP, a gain, or a new primary is claimed from fast-screen numbers
- ready or review PASS is claimed from offline MockTransport output without
  the hk2 contract smoke, or a screen-primary p50/p95 is filled from the
  full-C1 pooled pair, or from a screen that has not been measured with
  quotas 10/10/8/8/6/16
- the fast-screen decision was `full_c1` but fewer than three measured
  rounds finished
- a completed `drop` is treated as a KEEP or as the next baseline
- `evidence_complete` is not true, or warmup failed
- any measured request failed, was incomplete, or was unattempted
- `device` or `compute capability` is missing on the baseline or the tip
- the tip's `device` string or `compute capability` differs from the
  baseline, so the two sides are not the same named hardware
- the tip does not use the same measurement protocol as the baseline
- the clock basis is missing or is not
  `post_to_body_json_and_schema_validation`
- the measured SHA is not the tip SHA
- pooled p50/p95 were averaged from round percentiles instead of recomputed
  from the pooled samples

INCONCLUSIVE is not a win and not a regression. Do not drop failures, relax a
tolerance, or add undeclared rounds to produce a preferred delta. A recorded
`device` other than H200, or a `compute capability` other than SM90, is not
by itself INCONCLUSIVE. A fast-screen `drop` is a regression. It is not
INCONCLUSIVE, and it is not a reason to relax the rule that produced it.

## Quality and contract stay green

A latency delta is acceptable only when the tip's quality and contract checks
are green:

- the [validation for the files the tip touched](../../../CONTRIBUTING.md#choose-validation-by-changed-area)
- the model request/response contract tests that cover the changed path
- `python benchmarks/bench.py compare` against the baseline outputs when the
  change can affect decisions or probabilities, with tolerances declared
  before the run

A faster tip that fails a contract test, flips decisions beyond the declared
tolerance, or lowers the serving pass rate is not an accepted tip. Do not
stack past it.

## Feed the delta back

Before the agent proposes the next change, give it this record:

- baseline SHA, pooled p50, pooled p95, pass rate
- tip SHA; pooled p50, pooled p95, and pass rate when full C1 ran
- full-C1 p50 and p95 delta (tip minus baseline), in milliseconds, when
  full C1 ran
- clock basis, `device`, and `compute capability` for both sides
- RSI phase, alternatives considered, and which change is primary
- fast-screen decision (`screen_primary`, `drop`, `full_c1`, or
  `inconclusive`), the request IDs and any `quota_shortfall`, `dropped_at` when the decision is
  `drop`, and the screen p50/p95 deltas labeled as screen-only
- verdict: improved, regressed, or INCONCLUSIVE
- quality/contract result when the path reached full C1 or a KEEP decision

Improved is only a full-C1 KEEP. A `drop` verdict is regressed, and the
screen numbers stay out of the next baseline. The agent uses a full-C1
record as the next baseline only when the verdict is improved or regressed
and the evidence is complete. An INCONCLUSIVE tip stays the head until it
is fixed and remeasured, or until the tip is reverted. A dropped tip is
reverted or left unstacked; the previous primary stays the baseline.

## RSI phase, or the tip is not merge-clean

Every tip states which RSI phase it belongs to, one of:

- bring-up/scheduling
- parallelism/comms
- kernel
- launch/post

It also states which alternatives were considered and which of them is the
primary change in the tip. A tip that omits the phase, or omits alternatives
versus the primary, is not merge-clean. Put that statement in the tip record
and in the delta fed back to the agent. A conclusive latency result does not
fill in a missing phase.

## Draft, and a separate yes to merge

Leave the pull request in draft through the loop. Do not mark it ready for
review and do not merge it because a remeasure improved p50 or p95, or
because a fast-screen was not a drop. Merge and ready-for-review each need
a separate explicit yes from the owner, and only after a full-C1 KEEP.
Passing CI is not that yes. A tip that is not merge-clean on the RSI phase
stays unmerged even when the owner has not yet been asked. Room posts of a
delta follow the metrics-only rule in [public231-c1](../public231-c1/SKILL.md)
and include `device` and `compute capability`.
