---
name: optimize-remeasure-gate
description: Gate continuous optimization after a measured Public231 C1 baseline, including Cloud Agent and coding-agent loops. Use when a new tip must not land on the previous tip until that tip has an L20X remeasure, evidence is complete or explicitly INCONCLUSIVE, quality and contract checks are green, and the pull request stays draft until a separate owner yes.
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
edit, and again before every later tip:

- baseline full git SHA (`runtime_revision`)
- pooled p50 and p95 in milliseconds
- serving pass rate (`successful_requests / planned_requests`)
- clock basis `post_to_body_json_and_schema_validation`
- hardware label L20X / sm_89

Those are the primary figures. Per-round values stay attached so a pooled
number can be checked. A recalled or rounded figure, a single round, or a
percentile averaged across rounds is not a baseline.

## One tip, then remeasure

A tip is one optimization commit on the draft branch. Do not add another
commit on top of it until that tip has its own L20X Public231 C1 remeasure:
warmup 5, concurrency 1, three measured rounds, same manifest and clock as
the baseline. The remeasure SHA must be the tip SHA. A measurement of an
older commit does not clear the tip.

Keep controls matched to the baseline (manifest, precision, request order,
cache policy, thinking treatment, server shape). State any necessary
difference. Do not reset the comparison by changing the clock, the quantile
method, or the hardware label.

## INCONCLUSIVE stops the loop

Mark the tip **INCONCLUSIVE** and do not stack the next commit when any of
these are true:

- the baseline p50/p95 were not recorded before the edit
- fewer than three measured rounds finished
- `evidence_complete` is not true, or warmup failed
- any measured request failed, was incomplete, or was unattempted
- the device is not confirmed L20X / sm_89
- the clock basis is missing or is not
  `post_to_body_json_and_schema_validation`
- the measured SHA is not the tip SHA
- pooled p50/p95 were averaged from round percentiles instead of recomputed
  from the pooled samples

INCONCLUSIVE is not a win and not a regression. Do not drop failures, relax a
tolerance, or add undeclared rounds to produce a preferred delta.

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
- clock basis and L20X / sm_89
- verdict: improved, regressed, or INCONCLUSIVE
- quality/contract result

The agent uses that record as the next baseline only when the verdict is
improved or regressed and the evidence is complete. An INCONCLUSIVE tip stays
the head until it is fixed and remeasured, or until the tip is reverted.

## Draft, and a separate yes to merge

Leave the pull request in draft through the loop. Do not mark it ready for
review and do not merge it because a remeasure improved p50 or p95. Merge
and ready-for-review each need a separate explicit yes from the owner.
Passing CI is not that yes. Room posts of a delta follow the metrics-only
rule in [public231-c1](../public231-c1/SKILL.md).
