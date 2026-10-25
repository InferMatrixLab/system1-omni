---
name: optimize-remeasure-gate
description: Gate continuous optimization after a measured Public231 C1 baseline, including Cloud Agent and coding-agent loops. Use when a new tip must not land until that tip is remeasured with the same protocol on the same recorded device and compute capability. Every paste includes those two fields. The documented production target is H200 / SM90. Each tip names its RSI phase and the primary change versus alternatives considered, or it is not merge-clean. Incomplete evidence is INCONCLUSIVE. The pull request stays draft until a separate owner yes.
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

## One tip, then remeasure

A tip is one optimization commit on the draft branch. Do not add another
commit on top of it until that tip has its own Public231 C1 remeasure. The
gate requires the same measurement protocol and the same named hardware as
the baseline. It does not require the SKU to be H200, L20X, SM90, or sm_89.

Same measurement protocol means warmup 5, concurrency 1, three measured
rounds, the same manifest, and the same clock
(`post_to_body_json_and_schema_validation`). Same named hardware means the
same recorded `device` string and the same recorded `compute capability`.
The remeasure SHA must be the tip SHA. A measurement of an older commit does
not clear the tip.

Keep controls matched to the baseline (manifest, precision, request order,
cache policy, thinking treatment, server shape). State any necessary
difference. Do not reset the comparison by changing the clock, the quantile
method, or the named hardware.

## INCONCLUSIVE stops the loop

Mark the tip **INCONCLUSIVE** and do not stack the next commit when any of
these are true:

- the baseline p50/p95 were not recorded before the edit
- fewer than three measured rounds finished
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
by itself INCONCLUSIVE.

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
- tip SHA, pooled p50, pooled p95, pass rate
- p50 and p95 delta (tip minus baseline), in milliseconds
- clock basis, `device`, and `compute capability` for both sides
- RSI phase, alternatives considered, and which change is primary
- verdict: improved, regressed, or INCONCLUSIVE
- quality/contract result

The agent uses that record as the next baseline only when the verdict is
improved or regressed and the evidence is complete. An INCONCLUSIVE tip stays
the head until it is fixed and remeasured, or until the tip is reverted.

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
review and do not merge it because a remeasure improved p50 or p95. Merge
and ready-for-review each need a separate explicit yes from the owner.
Passing CI is not that yes. A tip that is not merge-clean on the RSI phase
stays unmerged even when the owner has not yet been asked. Room posts of a
delta follow the metrics-only rule in [public231-c1](../public231-c1/SKILL.md)
and include `device` and `compute capability`.
