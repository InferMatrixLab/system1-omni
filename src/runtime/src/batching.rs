//! FIFO batching of independent items with request-scoped output routing.

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};

use anyhow::{Context, Result, anyhow, ensure};
use tokio::sync::oneshot;

/// A single oversized item runs alone; items are never truncated or split.
#[derive(Clone, Copy)]
pub struct BatchLimits {
    pub items: usize,
    pub cost: usize,
}

/// Model-provided resource weight of one atomic execution item.
/// Exclusive items cannot share a dispatch, even when their costs fit.
#[derive(Clone, Copy)]
pub struct BatchCost {
    pub items: usize,
    pub cost: usize,
    pub exclusive: bool,
}

impl From<usize> for BatchCost {
    fn from(cost: usize) -> Self {
        Self {
            items: 1,
            cost,
            exclusive: false,
        }
    }
}

struct Request<I, O> {
    pending: VecDeque<I>,
    outputs: Vec<O>,
    response: oneshot::Sender<Result<Vec<O>>>,
    started: bool,
}

/// One owning execution thread per loaded executor. Already queued requests can
/// share a batch. An idle worker dispatches immediately, without a batching timer.
/// Cancellation before dispatch discards the request; after its first item is
/// dispatched, the complete request retains its inputs until execution finishes.
pub struct BatchScheduler<I, O> {
    worker: Arc<Worker<I, O>>,
}

struct Worker<I, O> {
    sender: Option<mpsc::Sender<Request<I, O>>>,
    thread: Option<std::thread::JoinHandle<()>>,
    available: Arc<AtomicBool>,
}

// Also marks unavailable on an unexpected unwind outside the execute callback.
struct WorkerExit(Arc<AtomicBool>);

impl Drop for WorkerExit {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

impl<I, O> Drop for Worker<I, O> {
    fn drop(&mut self) {
        // Close admission before joining, so an idle receiver wakes and any
        // dispatched work and callback-owned device resources retire before exit.
        self.sender.take();
        if let Some(thread) = self.thread.take()
            && thread.thread().id() != std::thread::current().id()
        {
            let _ = thread.join();
        }
    }
}

impl<I, O> Clone for BatchScheduler<I, O> {
    fn clone(&self) -> Self {
        Self {
            worker: self.worker.clone(),
        }
    }
}

impl<I: Send + 'static, O: Send + 'static> BatchScheduler<I, O> {
    /// The callback owns device execution and must synchronize before returning,
    /// including errors. Its outputs must correspond to its inputs in order.
    /// Dropping the last scheduler closes admission and waits for the worker and
    /// callback resources to retire. Callbacks must not depend on that dropping
    /// thread to unblock their execution; async callers can drop off-thread.
    pub fn new(
        limits: BatchLimits,
        cost: impl Fn(&I) -> BatchCost + Send + 'static,
        execute: impl FnMut(&[I]) -> Result<Vec<O>> + Send + 'static,
    ) -> Result<Self> {
        ensure!(limits.items > 0 && limits.cost > 0, "empty batch budget");
        let (sender, receiver) = mpsc::channel();
        let available = Arc::new(AtomicBool::new(true));
        let worker_available = available.clone();
        let thread = std::thread::Builder::new()
            .name("omni-batch".into())
            .spawn(move || {
                let _exit = WorkerExit(worker_available.clone());
                dispatch(receiver, limits, cost, execute, &worker_available);
            })
            .context("start batch execution thread")?;
        Ok(Self {
            worker: Arc::new(Worker {
                sender: Some(sender),
                thread: Some(thread),
                available,
            }),
        })
    }

    /// Whether the worker can still execute requests; shared by all clones.
    /// A busy worker remains available. A retired worker requires replacement.
    pub fn is_available(&self) -> bool {
        self.worker.available.load(Ordering::Acquire)
    }

    pub async fn run(&self, items: Vec<I>) -> Result<Vec<O>> {
        ensure!(self.is_available(), "batch executor is unavailable");
        if items.is_empty() {
            return Ok(Vec::new());
        }
        let (response, receiver) = oneshot::channel();
        self.worker
            .sender
            .as_ref()
            .expect("live scheduler owns its sender")
            .send(Request {
                outputs: Vec::with_capacity(items.len()),
                pending: items.into(),
                response,
                started: false,
            })
            .map_err(|_| anyhow!("batch executor is unavailable"))?;
        receiver.await.context("batch execution stopped")?
    }
}

fn dispatch<I, O>(
    receiver: mpsc::Receiver<Request<I, O>>,
    limits: BatchLimits,
    cost: impl Fn(&I) -> BatchCost,
    mut execute: impl FnMut(&[I]) -> Result<Vec<O>>,
    available: &AtomicBool,
) {
    let mut pending = None;
    loop {
        let mut inputs = Vec::with_capacity(limits.items);
        let mut parts = Vec::new();
        let mut total = 0usize;
        let mut item_total = 0usize;
        let mut exclusive = false;
        loop {
            let next = pending.take().or_else(|| {
                if inputs.is_empty() {
                    receiver.recv().ok()
                } else {
                    receiver.try_recv().ok()
                }
            });
            let Some(mut request) = next else { break };
            if !request.started && request.response.is_closed() {
                continue;
            }
            let start = inputs.len();
            while let Some(item) = request.pending.front() {
                let weight = cost(item);
                if !inputs.is_empty()
                    && (inputs.len() == limits.items
                        || weight.exclusive
                        || exclusive
                        || item_total >= limits.items
                        || weight.items > limits.items.saturating_sub(item_total)
                        || total > limits.cost
                        || weight.cost > limits.cost.saturating_sub(total))
                {
                    break;
                }
                total = total.saturating_add(weight.cost);
                item_total = item_total.saturating_add(weight.items);
                exclusive = weight.exclusive;
                inputs.push(request.pending.pop_front().expect("checked pending item"));
                request.started = true;
            }
            let count = inputs.len() - start;
            if count == 0 {
                pending = Some(request);
                break;
            }
            let incomplete = !request.pending.is_empty();
            parts.push((request, count));
            if incomplete
                || exclusive
                || inputs.len() == limits.items
                || item_total >= limits.items
                || total >= limits.cost
            {
                break;
            }
        }
        if inputs.is_empty() {
            return;
        }
        let result = catch_unwind(AssertUnwindSafe(|| execute(&inputs)));
        let panicked = result.is_err();
        if panicked {
            // Publish failure before notifying affected callers, so their next
            // health probe cannot observe a ready but permanently dead worker.
            available.store(false, Ordering::Release);
        }
        let result = result
            .unwrap_or_else(|_| Err(anyhow!("batch executor panicked")))
            .and_then(|outputs| {
                ensure!(outputs.len() == inputs.len(), "batch output count mismatch");
                Ok(outputs)
            });
        match result {
            Ok(outputs) => {
                let mut outputs = outputs.into_iter();
                for (mut request, count) in parts {
                    request.outputs.extend(outputs.by_ref().take(count));
                    if request.pending.is_empty() {
                        let _ = request.response.send(Ok(request.outputs));
                    } else {
                        debug_assert!(pending.is_none());
                        pending = Some(request);
                    }
                }
            }
            Err(error) => {
                let message = format!("{error:#}");
                for (request, _) in parts {
                    let _ = request.response.send(Err(anyhow!(message.clone())));
                }
            }
        }
        // A panicked device executor may have invalid state. Close its queue;
        // callers receive errors and no new work runs on that executor.
        if panicked {
            return;
        }
    }
}
