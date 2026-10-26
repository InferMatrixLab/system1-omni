---
name: new-model-integration
description: Bring a new open model into System1-Omni on its own GitHub issue and draft pull request, then a recipe and operator checklist, a Public231 C1 measurement, and continuous optimization on that same branch. Use when adding model support. Keep serving-chain work on separate issues from the new-model RFC.
---

# Integrate a new open model

Follow this order for a model that is not already supported. Implementation
placement, contracts, tests, and docs stay in
[add-new-model](../add-new-model/SKILL.md). This skill is the sequence around
that work: issue, draft PR, recipe, Public231 C1, then measured optimization.
Complete the [contributor-intent gate](../self-review/SKILL.md#human-contributor-intent-gate)
before editing and before opening the draft.

Do not merge, and do not mark the PR ready, without a separate explicit owner
yes. A green check or a C1 number is not that yes.

## 1. Open an independent issue

Open a new GitHub issue for this model. Do not add the model to another
model's issue or to a serving-chain issue.

Use the `new model` label. Title it `[New Model]:` for an implementation
request, or `[RFC]:` when the issue is still a design proposal, matching
issues such as #136, #114, and #121. Say which checkpoint, modalities, and
question types are in scope. Link the rolling tracker
[#83](https://github.com/InferMatrixLab/system1-omni/issues/83) and the
roadmap [#125](https://github.com/InferMatrixLab/system1-omni/issues/125);
do not use either tracker as a substitute for the model's own issue.

Serving-chain work stays on its own issues. The roadmap separates model
milestones from shared serving features, and it says reference serving, native
execution, and optimized execution are separate milestones. Shared
`/v1/systemone` contract work is #61, observation reuse is #85, and
admission, batching, and scheduling stay with #150 and the feature items on
#125. Link those issues when the model depends on them. Do not implement that
shared work inside the new-model RFC or its PR. A cross-request batching or
scheduler change needs its own follow-up issue.

## 2. Open a draft pull request

Implement on a branch whose PR targets the model's issue and stays **draft**.
[CONTRIBUTING](../../../CONTRIBUTING.md#self-review-before-requesting-review)
keeps draft PRs in draft until self-review is complete. Use
[precheck-pr](../precheck-pr/SKILL.md) before asking to mark it ready, and
wait for the owner's separate yes before ready or merge.

Keep the diff to this model's processors, executor, recipe, tests, and the
docs `add-new-model` lists. Reuse a shared backend only when the model
actually needs it. A new model is not a reason to redesign the frontend,
scheduler, or batching policy.

## 3. Recipe and operator checklist

Land the operator-facing recipe in the same draft, under `recipe/<model>/`,
before calling the integration runnable. `add-new-model` lists the required
pages (model README, `recipe/README.md`, root README supported-model status,
`mkdocs.yml`, and `docs/architecture.md` when a shared boundary changes).

The recipe must let an operator do all of the following without private
host notes:

- pin the checkpoint, tokenizer, adapters, and dependency versions
- build and launch the worker and, when used, the frontend
- check health and send one real example request
- run the model's validation commands and read the stated limits
- see which hardware was actually tested, named as that hardware

GPU claims name the measured `device` and `compute capability`. When the
recipe is documenting the production target, the preferred label is
**H200 / SM90**. If the host reported a different nvidia-smi string, the
recipe records that string and the measured capability, and does not rewrite
them. Do not describe untested hardware as validated. Do not put hostnames,
SSH targets, usernames, or reservation-tool names in the recipe.

## 4. Public231 C1

When the draft serves the model, measure it with
[public231-c1](../public231-c1/SKILL.md). Every paste records the draft SHA,
`device`, `compute capability`, the serving pass rate, and pooled p50/p95.
The documented production target is H200 / SM90. The protocol does not treat
a different measured identity as INCONCLUSIVE; paired runs still have to share
the recorded identity. A recipe that does not yet serve stays explicit about
that: do not invent a C1 result. CPU contract tests are not a latency baseline.

## 5. Continuous optimization on the same branch

Further latency work stays on this model's draft branch. Run it under
[optimize-remeasure-gate](../optimize-remeasure-gate/SKILL.md): paste the C1
p50/p95, `device`, and `compute capability` first. After that primary
exists, smoke and the fast-screen run before another full C1. The screen
compares the same 60 request IDs against the primary SHA's own screen, not
against the full-C1 pooled pair. That screen-primary pair stays
`TBD — measure on primary SHA same 60 IDs` until it is measured. A clear
regression is dropped there, with `dropped_at`, and is not a KEEP. Do not
stack the next tip commit until that tip has been dropped by the screen or
remeasured with full C1 on the same named hardware. Incomplete evidence is
INCONCLUSIVE.
Quality and contract checks stay green. Each tip states its RSI phase
(bring-up/scheduling,
parallelism/comms, kernel, or launch/post) and which alternatives were
considered versus the primary; without that it is not merge-clean. Feed each
delta back to the agent before the next edit.

Optimized execution remains a separate milestone from reference serving and
from native bring-up. Promote a number into the supported-model matrix only
when the recipe and the measured SHA agree with it.
