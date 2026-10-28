# Native execution runtime

`omni-runtime` provides `SerialScheduler` for native workers and `BatchScheduler`
for Open-Jev-9B default execution. Cua-S1, Open-Jev, Laya, and Decider assemble a
processor, one scheduler for the loaded executor, and the executor. HTTP
handlers still coordinate `prepare` → `execute` → `finish`. Clef uses the
same scheduler. Its admitted closure is the frame RPC to its Python
executor and does not call CUDA.

The serial scheduler owns FIFO admission and blocking dispatch. A caller waits
asynchronously for one execution permit before entering Tokio's blocking pool.
Cloning a scheduler shares that admission queue; separate loaded executors have
separate queues. The model mutex remains a guard for Rust's mutable executor
state. Waiting requests no longer occupy blocking threads waiting for that mutex.

| Worker | Admitted unit | Work outside admission |
| --- | --- | --- |
| Cua-S1 | One question's unpadded Qwen forward. Release before admitting its next question. | Request preparation, CPU letter projection and response finishing. |
| Open-Jev | 9B default: original candidate batches or one exclusive exact-prefix request through `BatchScheduler`; other modes: one complete request through `SerialScheduler`. | Request preparation and calibrated response finishing. |
| Laya | One complete padded request, including GPU scorer/action head and copyback. | Request preparation/padding and calibrated response finishing. |
| Decider | One complete request's independent rows, including within-request packing and LM-head projection. | Request preparation and calibrated response finishing. An optional request reservation covers both; token reservation begins after preparation. |
| Clef | One complete text request. The closure writes one frame and waits for that same executor process. | HTTP body reads and the cached health response. CUDA stays on the executor process main thread, including the warmup that runs before the port opens. |

Models retain weights, heads, device state, scratch buffers and graph caches.
Serial scheduling takes an owned closure; batch scheduling takes owned items and
one executor callback. Neither scheduler knows model identifiers,
question/candidate mappings, tensor layouts or backend APIs.
The existing numerical, ordering, whole-request failure and readiness contracts
remain in the processors, executors and worker handlers.

Cancellation while waiting removes the caller from admission without dispatch.
After dispatch, the blocking closure retains the permit and its captured
resources until completion, even if the caller disconnects or stops waiting.
Executors must finish queued device work before returning; the current Qwen
forward synchronizes the final hidden-state download. Errors and panics release
admission when the blocking task finishes. Model mutex poisoning retains its
existing failure behavior.

## Optional request capacity and execution control

Two admission layers have different lifetimes. `SerialScheduler` owns the FIFO
execution queue and permits one execution unit per loaded executor. It waits
instead of refusing work and does not itself bound the number of waiting
requests. A worker can additionally use the shared `admission::BoundedAdmission`
to refuse requests when an outstanding-request or caller-defined cost budget is
full. `try_admit(cost)` returns an owned `Permit` or refuses without reserving;
costs above the entire budget are refused, never clamped. Dropping the permit
releases its reservation. Neither layer coordinates separately running workers.

`control::ExecutionContext` carries an optional absolute deadline, shared
cancellation state and retained admission permits. Context clones share each
retained reservation; capacity is released after its last owner drops. The
awaiting caller holds a `CancellationGuard` from `cancel_on_drop()` so abandoning
the request signals cancellation while detached blocking work retains its
context and capacity. `check()` reports `ExecutionStopped::Deadline` or
`ExecutionStopped::Cancelled`; it does not interrupt running work.

`SerialScheduler::run_with_context` checks the context before queueing and again
on the blocking thread immediately before calling the work closure, including
when that closure waited in Tokio's blocking pool. The context and scheduler
permit remain owned through completion. This method does not install an async
timeout or preempt device work; the HTTP adapter enforces the response deadline,
and the executor must synchronize submitted device work before returning.
The existing `SerialScheduler::run` remains available without request control.

The Decider HTTP worker opts into both layers. Its worker adapter reads the
environment, reserves a request slot before upload/preparation, and reserves
processed tokens after preparation. The model processor defines that token cost;
the shared admission type enforces the budgets, and the scheduler still owns
FIFO execution admission. Context clones retain reservations through blocking
preparation and execution even after a timeout or disconnect. Response
reconstruction remains in the model adapter. Internal `Engine::predict`,
including startup warmup, uses the direct inference path without HTTP limits.
See the [Decider recipe](../../recipe/decider/README.md) for configuration,
health fields and overload/deadline responses. Other workers keep their current
admission behavior unless they explicitly opt in.

## Cross-request batching for Open-Jev-9B

`BatchScheduler` owns a FIFO queue and one execution thread per loaded executor.
An idle worker runs immediately. It combines work already in the queue without a
batching delay. Each item declares an item count, a cost and whether it must run
alone; the generic scheduler enforces the configured budgets. An oversized atomic
item runs alone, without truncation. Queue admission itself is not capacity-limited.

The 9B adapter preserves its original batches of at most 16 candidates and 4096
tokens as atomic items. A GPU dispatch may combine these items up to 16 candidates
and 8192 tokens. Input/gate-up projections retain the original batch row counts;
output/down projections retain individual candidate lengths. Compatible GEMMs can
share a call, and incompatible ones keep separate calls. Prefix reuse is an
exclusive item using the existing request-local state path. Other Open-Jev modes
and checkpoints keep serial execution. Complete request outputs are reconstructed
before per-question calibration.

Cancellation before the first item dispatch discards that queued request. Once
any item starts, all remaining items and resources stay alive through completion,
even when the caller disconnects. Errors fail affected complete requests; unrelated
queued requests remain ordered. The model callback synchronizes partially submitted
GPU work on errors or panics, and a panic closes the execution queue. The callback
and its model resources remain owned by the worker until it finishes. Dropping all
scheduler handles closes admission and joins the execution thread, waiting for
dispatched work and callback resources to retire before the owner exits.

Shared processing orchestration and cross-worker coordination remain planned.
Batch layout keys include both candidate lengths and original input projection
groups, so CUDA Graph replay cannot conflate different numerical layouts.

CPU tests cover FIFO serialization, queued cancellation, retention of admission
and resources after dispatched cancellation, errors/panics, independent loaded
executors, bounded request/cost admission, and context checks under blocking-pool
backlog. Decider adds configuration and HTTP upload/preparation deadline checks:

```sh
cargo test --locked -p omni-runtime
cargo test --locked -p omni-decider-native --test runtime
cargo test --locked -p omni-decider-native --lib serve::runtime_tests
```

See the [architecture contracts](../../docs/architecture.md) and
[benchmark protocol](../../benchmarks/README.md) when changing admission policy.
