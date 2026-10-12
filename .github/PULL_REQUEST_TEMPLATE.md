## Required human contributor answers

Answer every question before implementation and PR submission, including drafts.
Reuse explicit, still-applicable human answers with a traceable source reference;
ask the human for anything missing and wait. An agent must not invent answers.
Blank fields, placeholders, or an agent-written rationale do not satisfy this gate.
Share only information and source links appropriate for this PR's audience.

1. **What outcome are you trying to achieve?**

   Answer:

2. **What is the highest-leverage variable for that outcome?**

   Answer:

3. **How will this work move that variable rather than create busywork?**

   Answer:

4. **Who is the customer or user, and what do they actually need?**

   Answer:

**After answering the four questions: is a PR needed at all to achieve this outcome?**

Human decision and rationale (or explicit source reference):

**For a bug fix: link actual observed failure evidence.**
Include the observed result of a minimal reproduction, failing regression test,
or relevant trace. Label untested diagnoses as unverified. For a non-bugfix,
state why this is not applicable.

Evidence:

<!-- PLEASE FILL IN THE PR DESCRIPTION HERE. -->

## Purpose

## Test Plan

**System1-Omni Version / Commit:**

## Test Result

## Demo / evidence

<!-- Follow .agents/skills/self-review/SKILL.md: exact baseline/head, model, hardware,
precision, workload/concurrency, reproduction commands, and relevant raw repeated-run
latency/throughput/memory/quality or parity evidence. Separate cold startup and warm
runs, and kernel/native/HTTP results. Include useful demo/figure links, source
revisions, limitations, asset rights and redaction. Label illustrations separately
from actual runs. Video is optional; use N/A with a reason where evidence does not
apply, and unverified for relevant measurements that could not be made. -->

## Self-review

Before marking this PR ready for review or requesting maintainer review, complete
the [self-review checklist](https://github.com/ThinkFlowLab/system1-omni/blob/main/CONTRIBUTING.md#self-review-before-requesting-review).
Keep the PR in draft while this work is incomplete.
For agent assistance, use the optional [precheck-pr skill](https://github.com/ThinkFlowLab/system1-omni/blob/main/.agents/skills/precheck-pr/SKILL.md).

- [ ] I have reviewed the full diff and addressed the issues I found.
- [ ] I have checked that the change follows the project's architecture and stays focused on the stated purpose.
- [ ] I have run the checks appropriate to this change and reported commands, results, and anything I could not verify above.
- [ ] I have checked that the PR description, documentation, and any accuracy or performance claims match the implementation and available evidence.
