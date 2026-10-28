use std::future::{Future, poll_fn};
use std::pin::Pin;
use std::sync::{Arc, Mutex, mpsc};
use std::task::Poll;
use std::time::Duration;

use anyhow::{Result, bail};
use omni_runtime::batching::{BatchCost, BatchLimits, BatchScheduler};
use tokio::sync::oneshot;

async fn done<T>(future: impl Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(5), future)
        .await
        .expect("batch scheduler stopped making progress")
}

async fn queued<T>(mut future: Pin<&mut impl Future<Output = T>>) {
    poll_fn(|cx| {
        assert!(future.as_mut().poll(cx).is_pending());
        Poll::Ready(())
    })
    .await;
}

#[tokio::test]
async fn batches_ready_requests_and_restores_each_request_in_fifo_order() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let observed = calls.clone();
    let (started, ready) = oneshot::channel();
    let mut started = Some(started);
    let (release, wait) = mpsc::channel();
    let scheduler = BatchScheduler::new(
        BatchLimits { items: 4, cost: 6 },
        |&item| BatchCost::from(if item == 10 { 99 } else { 1 }),
        move |items: &[usize]| {
            observed.lock().unwrap().push(items.to_vec());
            if items == [0] {
                started.take().unwrap().send(()).unwrap();
                wait.recv().unwrap();
            }
            Ok(items.iter().map(|x| x * 100).collect())
        },
    )
    .unwrap();
    let mut first = Box::pin(scheduler.run(vec![0]));
    queued(first.as_mut()).await;
    done(ready).await.unwrap();
    let mut a = Box::pin(scheduler.run(vec![1, 2, 3]));
    let mut b = Box::pin(scheduler.run(vec![4, 5]));
    let mut large = Box::pin(scheduler.run(vec![10]));
    let mut last = Box::pin(scheduler.run(vec![11, 12]));
    queued(a.as_mut()).await;
    queued(b.as_mut()).await;
    queued(large.as_mut()).await;
    queued(last.as_mut()).await;
    release.send(()).unwrap();
    assert_eq!(done(first).await.unwrap(), [0]);
    assert_eq!(done(a).await.unwrap(), [100, 200, 300]);
    assert_eq!(done(b).await.unwrap(), [400, 500]);
    assert_eq!(done(large).await.unwrap(), [1000]);
    assert_eq!(done(last).await.unwrap(), [1100, 1200]);
    assert_eq!(
        *calls.lock().unwrap(),
        vec![vec![0], vec![1, 2, 3, 4], vec![5], vec![10], vec![11, 12]]
    );
}

#[tokio::test]
async fn cancellation_before_dispatch_skips_the_request() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let observed = calls.clone();
    let (started, ready) = oneshot::channel();
    let mut started = Some(started);
    let (release, wait) = mpsc::channel();
    let scheduler = BatchScheduler::new(
        BatchLimits { items: 4, cost: 4 },
        |_| BatchCost::from(1),
        move |items: &[usize]| {
            observed.lock().unwrap().extend_from_slice(items);
            if items == [0] {
                started.take().unwrap().send(()).unwrap();
                wait.recv().unwrap();
            }
            Ok(items.to_vec())
        },
    )
    .unwrap();
    let mut first = Box::pin(scheduler.run(vec![0]));
    queued(first.as_mut()).await;
    done(ready).await.unwrap();
    let mut cancelled = Box::pin(scheduler.run(vec![1, 2]));
    queued(cancelled.as_mut()).await;
    drop(cancelled);
    let mut next = Box::pin(scheduler.run(vec![3]));
    queued(next.as_mut()).await;
    release.send(()).unwrap();
    done(first).await.unwrap();
    assert_eq!(done(next).await.unwrap(), [3]);
    assert_eq!(*calls.lock().unwrap(), [0, 3]);
}

#[tokio::test]
async fn cancellation_after_dispatch_finishes_all_parts_before_later_requests() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let observed = calls.clone();
    let (started, ready) = oneshot::channel();
    let mut started = Some(started);
    let (release, wait) = mpsc::channel();
    let scheduler = BatchScheduler::new(
        BatchLimits { items: 1, cost: 1 },
        |_| BatchCost::from(1),
        move |items: &[usize]| {
            observed.lock().unwrap().extend_from_slice(items);
            if items == [1] {
                started.take().unwrap().send(()).unwrap();
                wait.recv().unwrap();
            }
            Ok(items.to_vec())
        },
    )
    .unwrap();
    let mut cancelled = Box::pin(scheduler.run(vec![1, 2]));
    queued(cancelled.as_mut()).await;
    done(ready).await.unwrap();
    drop(cancelled);
    let mut next = Box::pin(scheduler.run(vec![3]));
    queued(next.as_mut()).await;
    release.send(()).unwrap();
    assert_eq!(done(next).await.unwrap(), [3]);
    assert_eq!(*calls.lock().unwrap(), [1, 2, 3]);
}

#[tokio::test]
async fn errors_discard_partial_results_and_leave_later_requests_usable() {
    for malformed in [false, true] {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let observed = calls.clone();
        let scheduler = BatchScheduler::new(
            BatchLimits { items: 1, cost: 1 },
            |_| BatchCost::from(1),
            move |items: &[usize]| {
                observed.lock().unwrap().extend_from_slice(items);
                if items == [2] {
                    if malformed {
                        return Ok(Vec::new());
                    }
                    bail!("device error");
                }
                Ok(items.to_vec())
            },
        )
        .unwrap();
        assert!(done(scheduler.run(vec![1, 2, 3])).await.is_err());
        assert_eq!(done(scheduler.run(vec![4])).await.unwrap(), [4]);
        assert_eq!(*calls.lock().unwrap(), [1, 2, 4]);
    }
}

#[tokio::test]
async fn a_panicked_executor_closes_admission_instead_of_reusing_bad_state() {
    let scheduler: BatchScheduler<usize, usize> = BatchScheduler::new(
        BatchLimits { items: 1, cost: 1 },
        |_| BatchCost::from(1),
        |_| -> Result<Vec<usize>> { panic!("device executor failed") },
    )
    .unwrap();
    assert!(
        done(scheduler.run(vec![1]))
            .await
            .unwrap_err()
            .to_string()
            .contains("panicked")
    );
    assert!(done(scheduler.run(vec![2])).await.is_err());
}

#[tokio::test]
async fn oversized_cost_cannot_overflow_the_batch_budget() {
    let scheduler = BatchScheduler::new(
        BatchLimits {
            items: 3,
            cost: usize::MAX - 1,
        },
        |&item| BatchCost::from(item),
        |items: &[usize]| {
            assert!(items.len() == 1 || items.iter().sum::<usize>() < usize::MAX);
            Ok(items.to_vec())
        },
    )
    .unwrap();
    assert_eq!(
        done(scheduler.run(vec![usize::MAX, 1, 2])).await.unwrap(),
        [usize::MAX, 1, 2]
    );
    assert!(done(scheduler.run(Vec::new())).await.unwrap().is_empty());
}

#[tokio::test]
async fn dropping_callers_and_schedulers_retains_executing_resources() {
    struct Resource(Option<oneshot::Sender<()>>);
    impl Drop for Resource {
        fn drop(&mut self) {
            let _ = self.0.take().unwrap().send(());
        }
    }
    let (dropped, mut retired) = oneshot::channel();
    let resource = Resource(Some(dropped));
    let (started, ready) = oneshot::channel();
    let mut started = Some(started);
    let (release, wait) = mpsc::channel();
    let scheduler = BatchScheduler::new(
        BatchLimits { items: 1, cost: 1 },
        |_| BatchCost::from(1),
        move |items: &[usize]| {
            let _keep = &resource;
            started.take().unwrap().send(()).unwrap();
            wait.recv().unwrap();
            Ok(items.to_vec())
        },
    )
    .unwrap();
    let mut request = Box::pin(scheduler.run(vec![1]));
    queued(request.as_mut()).await;
    done(ready).await.unwrap();
    drop(request);
    // Dropping the last owner now waits for shutdown; do it off the async
    // thread which supplies this test callback's release signal.
    let shutdown = tokio::task::spawn_blocking(move || drop(scheduler));
    assert!(matches!(
        retired.try_recv(),
        Err(oneshot::error::TryRecvError::Empty)
    ));
    release.send(()).unwrap();
    done(retired).await.unwrap();
    done(shutdown).await.unwrap();
}

#[tokio::test]
async fn weighted_atomic_items_and_exclusive_work_keep_fifo_boundaries() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let seen = calls.clone();
    let (started, ready) = oneshot::channel();
    let mut started = Some(started);
    let (release, wait) = mpsc::channel();
    let scheduler = BatchScheduler::new(
        BatchLimits {
            items: 4,
            cost: 100,
        },
        |&id| BatchCost {
            items: match id {
                1 => 3,
                2 | 3 => 2,
                _ => 1,
            },
            cost: 1,
            exclusive: id == 9,
        },
        move |ids: &[usize]| {
            seen.lock().unwrap().push(ids.to_vec());
            if ids == [0] {
                started.take().unwrap().send(()).unwrap();
                wait.recv().unwrap();
            }
            Ok(ids.iter().map(|id| id * 10).collect())
        },
    )
    .unwrap();
    let mut first = Box::pin(scheduler.run(vec![0]));
    queued(first.as_mut()).await;
    done(ready).await.unwrap();
    let mut a = Box::pin(scheduler.run(vec![1, 2, 3]));
    let mut b = Box::pin(scheduler.run(vec![9, 4, 5]));
    queued(a.as_mut()).await;
    queued(b.as_mut()).await;
    release.send(()).unwrap();
    assert_eq!(done(first).await.unwrap(), [0]);
    assert_eq!(done(a).await.unwrap(), [10, 20, 30]);
    assert_eq!(done(b).await.unwrap(), [90, 40, 50]);
    assert_eq!(
        *calls.lock().unwrap(),
        [vec![0], vec![1], vec![2, 3], vec![9], vec![4, 5]]
    );
}

#[test]
fn last_scheduler_drop_joins_callback_resource_retirement() {
    use std::sync::atomic::{AtomicBool, Ordering};
    struct Retired(Arc<AtomicBool>);
    impl Drop for Retired {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    let retired = Arc::new(AtomicBool::new(false));
    let resource = Retired(retired.clone());
    let scheduler = BatchScheduler::new(
        BatchLimits { items: 1, cost: 1 },
        |_| BatchCost::from(1),
        move |items: &[usize]| {
            let _keep_alive = &resource;
            Ok(items.to_vec())
        },
    )
    .unwrap();
    let last = scheduler.clone();
    drop(scheduler);
    assert!(!retired.load(Ordering::SeqCst));
    drop(last);
    assert!(
        retired.load(Ordering::SeqCst),
        "callback resources must retire before owner exit"
    );
}
