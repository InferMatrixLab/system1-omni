# Native execution runtime

`omni-runtime` provides `SerialScheduler` for the Cua-S1, Open-Jev, Laya,
Decider, and Clef workers. Cua-S1, Open-Jev, Laya, and Decider assemble a
processor, one scheduler for the loaded executor, and the executor. HTTP
handlers still coordinate `prepare` → `execute` → `finish`. Clef uses the
same scheduler. Its admitted closure is the frame RPC to its Python
executor and does not call CUDA.

The scheduler owns FIFO admission and blocking dispatch. A caller waits
asynchronously for one execution permit before entering Tokio's blocking pool.
Cloning a scheduler shares that admission queue; separate loaded executors have
separate queues. The model mutex remains a guard for Rust's mutable executor
state. Waiting requests no longer occupy blocking threads waiting for that mutex.

| Worker | Admitted unit | Work outside admission |
| --- | --- | --- |
| Cua-S1 | One question's unpadded Qwen forward. Release before admitting its next question. | Request preparation, CPU letter projection and response finishing. |
| Open-Jev | One complete request's independent candidate forwards and CPU scalar heads. | Request preparation and calibrated response finishing. |
| Laya | One complete padded request, including GPU scorer/action head and copyback. | Request preparation/padding and calibrated response finishing. |
| Decider | One complete request's independent rows, including within-request packing and LM-head projection. | Request preparation and calibrated response finishing. An optional request reservation covers both; token reservation begins after preparation. |
| Clef | One complete text request. The closure writes one frame and waits for that same executor process. | HTTP body reads and the cached health response. CUDA stays on the executor process main thread, including the warmup that runs before the port opens. |

Models retain weights, heads, device state, scratch buffers and graph caches.
The scheduler takes an owned closure and returns its result; it does not know
model identifiers, question/candidate mappings, tensor layouts or backend APIs.
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

Shared processing orchestration, compatibility grouping, cross-request dynamic
batching and a shared worker drain lifecycle remain planned. Models may pack
supported inputs within one request; this scheduler does not assemble GPU batches.
The Qwen executor supports single prompts, bounded packed prefill (Open-Jev
candidates or Decider complete rows), and request-local shared prefixes
(Decider opt-in). Graph, fixed and shared execution retain the same complete
request admission unit and context lifetime; the scheduler does not construct
GPU batches or coordinate separately running workers.

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

Clef's cached health payload is returned only while its child process remains
alive. The executor checks child exit independently of admission. IPC failures
publish unavailable state inside the blocking exchange, before its scheduler
permit is released, including when the waiting HTTP caller was cancelled.
