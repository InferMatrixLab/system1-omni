# Development loop and handoff

This is a contributor-operated workflow using existing GitHub issues, PR bodies,
comments and check links. It defines no installed watcher, background runner,
webhook, hook or new permission. Automation deployment and its owner, storage,
concurrency controls and delivery verification remain a separate decision.

Start from [CONTRIBUTING](../CONTRIBUTING.md); agents also read
[AGENTS.md](../AGENTS.md). Consult [lessons](../lessons.md) before changing or
reviewing an area. Architecture and validation stay in their
[canonical contracts](architecture.md) and
[change-to-validation matrix](../CONTRIBUTING.md#choose-validation-by-changed-area).

## One change, one evidence trail

1. Complete the existing human intent gate, define the bounded change and select
   validation from the matrix. Record affected components and prerequisites.
2. Implement with tests and contract/documentation updates. Capture revision,
   platform/tool versions, exact command, start/end time, result and evidence URL.
3. Self-review the complete diff and fill the existing PR template. A green CPU
   suite does not validate GPU behavior; author evidence does not replace
   maintainer review. Keep blocked validation and its handoff owner visible.
4. Before pausing or handing off, update a single **Handoff checkpoint** in the
   authorized PR body or issue. Read it back to verify persistence. For work
   without a remote item, retain the same fields in the contributor's private
   task workspace until publication is authorized; do not commit run state.
5. On resumption, read the checkpoint and current remote state before choosing the
   next action. Preserve past results as history, clearly separated from evidence
   applicable to the current head. A closed issue or merged PR does not complete
   every dependency automatically.

## Handoff checkpoint fields

Use the existing item as the durable record; do not open a duplicate tracker.
The following fields can be plain text in a PR body/comment or issue:

- Repository identity and item number/URL; observation time and sources.
- Open/closed/merged state; actual PR base/head SHA. Issues have no inherent head;
  link their relevant PRs and revisions instead.
- Each relevant check's name, run/attempt, checked SHA, status/conclusion,
  platform, observation time and evidence link. Keep unknown/stale separate.
- Review/requirement status and validation owners, independently of author test
  results. Unknown branch-protection or merge authority remains unknown.
- Dependencies, blockers, next action and responsible person, or explicitly
  unassigned handoff. Include what evidence will unblock it.
- For an authorized external action: a stable transition/target key, purpose,
  planned/in-flight/confirmed/uncertain state and returned external identifier.
  Preserve enough history to avoid replaying it after a restart.

An action key includes repository/item, action type and meaningful evidence
version (for example head + check run/attempt + conclusion), not only the head.
Read before editing a shared record; coordinate concurrent writers. If a write's
result is uncertain, inspect the destination before retrying. A deduplication
record is not a guarantee of exactly-once delivery or authorization to act.

## Reconcile changes and demonstrate the protocol

This synthetic walkthrough exercises the rules, not a deployed tracker:

| Observation | Checkpoint / next action |
| --- | --- |
| Head A, run 10 pending | Save A/run 10 and the pending validation handoff; read it back after a pause. |
| Same A/run 10 failed | Refresh CI evidence despite unchanged head; record a new failure transition and diagnosis handoff. |
| Identical failure observed again | Retain the existing transition key; do not duplicate an action or notification. |
| Same A, rerun 11 succeeds | Record the new attempt and supersede current CI status; do not erase history or infer human approval. |
| Head B replaces A | A's check evidence is historical; select and obtain B's checks. Reassess review applicability under actual repository policy. |
| Base changes while head stays B | Reassess merge-result validation; refresh reviews, requirements and mergeability independently. |

Before a separately authorized merge or other consequential action, refresh the
actual head and relevant gates. Check runs and legacy commit statuses are separate
sources: an empty status list is not a failing check. If a source cannot be read,
record the last known result and its age without calling it current.

## Bounded execution and useful measurements

Choose a time/attempt budget before expensive work. Retry only safe transient
failures within that budget; a retry cannot expand authorization. Diagnose
deterministic failures instead of repeatedly running them. Stop on access denial,
exhausted budget, scope change or missing human decision, and checkpoint the
blocker. Do not run parallel jobs against conflicting mutable resources.

Use current CI timeouts and the [benchmark protocol](../benchmarks/README.md)
where applicable; GPU reservations, downloads and paid runs require their own
scope and resources. Record failures and skipped/ignored tests honestly.

For an iteration improvement claim, define the observation window and metric
before collecting it. Examples are elapsed time from first submitted revision to
first actionable validation result, or PR open-to-merge time. Record timestamps,
data sources, sample count, exclusions and unfinished items. Count independent
incidents using [lessons](../lessons.md), not alerts. No baseline or speedup is
claimed by this initialization; one successful PR cannot establish causality.
