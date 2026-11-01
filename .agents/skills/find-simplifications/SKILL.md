---
name: find-simplifications
description: Find evidence-backed simplification candidates in System1-Omni and, when requested, turn them into focused proposals or code changes. Use for audits of dead, duplicated, speculative, over-generalized, unnecessarily defensive, or hand-rolled code. Use system1-omni-review for ordinary correctness review and optimize-remeasure-gate for performance-first optimization on a measured baseline.
---

# Find System1-Omni Simplifications

Find a small number of well-proven ways to reduce code, state, APIs, or maintenance cost without weakening supported behavior. Treat this as an architecture and ownership audit, not a dead-code search or style cleanup.

The default deliverable is a read-only report. Do not add TODOs, design documents, issues, or code unless the user asks for those changes.

## Establish the Contract

Before judging a candidate:

1. Read [`CONTRIBUTING.md`](../../../CONTRIBUTING.md), [`README.md`](../../../README.md), and applicable `AGENTS.md` instructions from the checkout.
2. Read the [architecture contracts](../../../docs/architecture.md).
3. Select only the relevant contract sections and component READMEs, for example the [runtime contract](../../../src/runtime/README.md) and the [frontend contract](../../../src/frontend/README.md). Architecture sections marked planned are target design, not current contracts.
4. Identify the comparison base with the PR's actual base branch or `git merge-base HEAD upstream/main`; do not assume the current branch represents the intended baseline.
5. If tests or benchmark coverage may change, follow the test routing in [`CONTRIBUTING.md`](../../../CONTRIBUTING.md) and the benchmark self-review rules in [`benchmarks/README.md`](../../../benchmarks/README.md).

Separate an implementation accident from an intentional boundary. In particular:

- System1-Omni adapts pinned upstream model reference code and shares device kernels across models. A similar upstream or shared implementation is evidence for reuse only after comparing serialization, tokenization, scoring, and lifetime semantics.
- Model configuration, recipes, worker selection, cargo features, and optional dependencies create dynamic consumers that a static reference count can miss.
- Hardware backends, model families, execution modes, and worker implementations may be intentionally parallel. Do not propose collapsing one only because it is inactive in the local environment.
- Frontend, admission, dispatch, executor, cache, and backend state can look redundant while representing different owners. Prove that two fields or barriers encode the same fact before merging them.
- A shorter hot path is not a simplification if it adds synchronization, host-device copies, host materialization, graph recapture, or a measurable quality/latency regression.

## Strong Candidates

A strong candidate removes a meaningful surface and has a clear replacement or deletion boundary. Examples include:

- A public method, configuration option, registry entry, route, message field, or event has no production consumer and no compatibility obligation.
- Tests or documentation are the only consumers of behavior that is not part of a supported contract.
- Multiple layers mirror the same request, lifecycle, cache-validity, readiness, or topology state.
- Model-specific, backend, or platform implementations duplicate a stable common operation and can share it without hiding performance-critical differences.
- An upstream model reference routine or a shared kernel utility is reimplemented locally without a System1-Omni-specific semantic difference.
- Compatibility or fallback branches protect environments the project no longer supports.
- Serialization, FFI, output formatting, or forwarding code materializes the same payload more than once or converts between equivalent representations.
- A helper, abstraction, module split, or extension point has only one real implementation and adds more indirection than policy.
- Hand-written infrastructure can be replaced by an existing workspace dependency, a Rust standard-library feature, a shared device kernel, or an existing runtime facility with net deletion and equal observability.
- Tests maintain large mocks, snapshots, or expected data solely for an unused API or representation.

Do not elevate typo fixes, formatting, isolated renames, or vague complexity complaints into simplification candidates. Bundle small cleanups only when they support one substantive removal.

## Survey by Ownership Boundary

For broad audits, cover the largest or riskiest production deltas first. Useful domains are:

- `src/runtime/`: admission, dispatch, scheduling, execution units, cancellation, and permit/resource lifetimes.
- `src/frontend/`: envelope and header handling, backend routing, timeout and status mapping, health forwarding, cancellation, and shutdown.
- `src/models/` and model workers: serialization, prompt/template and tokenization, one-pass prefill scoring, response materialization, weight loading, and lifecycle.
- `src/backends/` and the native FFI boundary: CUDA/Metal parallelism, buffer and graph lifetimes, stream and device ownership, and error propagation across the Rust/C ABI.
- Platform backends: genuine vendor differences versus copied control flow that has drifted.
- Configuration, model recipes, and benchmarks: duplicate sources of truth and options that never reach a runtime consumer.
- Tests and CI: redundant fixtures, mocks of obsolete contracts, and expensive coverage that can move to a cheaper level without losing the guarded behavior.

When the user explicitly requests a repository-wide or many-candidate audit and parallel agent work is available and authorized, divide these domains among agents and require the same evidence format from each. Otherwise survey them sequentially.

## Prove or Reject Each Candidate

Use `rg` first, then read every relevant caller and implementation. Search exact Rust symbols plus serialized names, route paths, configuration keys, recipe values, cargo feature names, and CLI spellings.

Classify evidence:

- **Production:** `src/`, runtime configuration, serving entrypoints, model workers, recipes used for supported deployments, and executable loader paths.
- **Validation/support:** `tests/`, `benchmarks/`, recipes, docs, CI, and developer tooling.
- **External/dynamic:** upstream model reference implementations, model repositories and checkpoints, optional backends and features, and external workers. State what was checked and what remains unknown.

For each candidate, answer:

1. What exact surface would disappear or merge?
2. Who owns the state or behavior today?
3. Which production consumers exist, including dynamic consumers?
4. What supported behavior, compatibility, performance, memory, or observability could change?
5. What is the smallest validation that would catch a bad simplification?
6. Is the result net deletion after tests, adapters, migration glue, and documentation are counted?

Reject or downgrade the candidate when a live consumer exists and removal is actually a feature decision; when an active design contract explains the separation; when the change only moves complexity; or when hardware/model evidence is unavailable for a performance-sensitive claim.

## Lifecycle and Data-Movement Audits

For asynchronous or device-bound code, map the ownership graph before recommending deletion:

```text
frontend transport -> runtime admission -> dispatch -> model executor
  -> device backend or model worker -> response materialization -> terminal cleanup
```

Associate every lock, channel, task, permit, readiness flag, timeout, abort path, and cache marker with an owner and transition. Coalesce mechanisms only when they represent the same state and have the same failure boundary. Preserve separate mechanisms when they protect publication versus rollback, local versus remote ownership, first-terminal-outcome arbitration, or cleanup after partial failure.

For tensors and media, also trace device, process, and ownership transitions. Count host-device copies, FFI buffer transfers, serialization, tokenization/decode steps, and buffer/graph release responsibility. Prefer eliminating a transfer or representation over merely hiding it behind a helper.

## Dependency Replacements

Prefer facilities already available in the repository before adding a dependency. For a proposed replacement:

- Name the exact local behavior it covers and any semantics that remain in glue code.
- Check maintenance, license, feature boundaries, platform support, and transitive footprint.
- Compare net deletion, test burden, build/startup cost, and runtime behavior.
- Benchmark when the replaced code is on a model, scheduling, serialization, FFI, or serving hot path.

A wrapper that retains the same state machine or conversion logic is not a simplification.

## Deliver the Result

Report only candidates supported by concrete evidence. For each one include:

- **Surface:** files and symbols involved.
- **Evidence:** production consumers found, dynamic paths checked, and relevant design contract.
- **Simplification:** exactly what to remove, merge, inline, or reuse.
- **Benefit:** deleted API/state/code, reduced data movement, or reduced maintenance/test burden.
- **Risk and validation:** behavior that could change and the narrowest useful tests or measurements.
- **Confidence:** high or medium, with the unresolved evidence named for medium-confidence items.

Also list representative ideas that were rejected when that prevents repeated investigation. Prefer three high-confidence candidates over a long speculative inventory.

If the user asks for a durable design proposal, place it under `docs/rfc/` following the conventions of the existing RFC documents; keep implementation details proportional to the decision. If the user asks for implementation, make the smallest coherent change, update affected tests/docs, and preserve public compatibility unless the user explicitly accepts a breaking change.

## Folding Work from Another Branch

Diff the source branch against its own merge base with `upstream/main`, not against the current feature branch. Port only independently justified candidates. Consolidate overlaps into the candidate with stronger evidence, and do not preserve a candidate count by carrying weaker duplicates.

## Validation

Match validation to the outgoing diff:

- Documentation or skill changes: run the repository skill validator when available, check links/paths, search for stale source-specific terminology, and run `git diff --check`.
- Rust changes: run the narrowest relevant `cargo test` coverage with `--locked` and any required features, then apply `cargo fmt` and `cargo clippy` to the changed code.
- Model, backend, hardware, performance, or accuracy changes: run available CPU/static checks and state the exact GPU/Metal/model gap; never present simulated evidence as device validation.
- Before submission, use [`precheck-pr`](../precheck-pr/SKILL.md) for the branch-level self-review.

When summarizing, distinguish checks that passed from checks that were not runnable, and state the snapshot or branch against which the audit was performed.
