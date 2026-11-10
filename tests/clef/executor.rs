//! Inject a Rust panic while the live worker's pipes are locked. Worker input
//! cannot currently trigger this; the boundary must also survive future panics.

use std::{
    process::Command,
    sync::{Arc, mpsc},
    time::Duration,
};

use omni_runtime::SerialScheduler;

use super::{Executor, frame};

async fn panic_during_exchange(cancel_caller: bool) {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../..");
    let mut command = Command::new("python3");
    command
        .arg(root.join("tests/clef/fake_worker.py"))
        .env("PYTHONPATH", root.join("src"));
    let executor = Arc::new(Executor::spawn(command).expect("fake worker"));
    let scheduler = SerialScheduler::default();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let task_executor = Arc::clone(&executor);
    let task_scheduler = scheduler.clone();
    let caller = tokio::spawn(async move {
        task_scheduler
            .run(move || {
                task_executor.with_pipes(|pipes| {
                    frame::write_request(&mut pipes.stdin, frame::OPCODE_DECIDE, b"{}")?;
                    entered_tx.send(()).unwrap();
                    release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                    panic!("injected panic with the pipes locked")
                })
            })
            .await
    });
    entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(executor.ready(), "the child must be alive before injection");
    if cancel_caller {
        caller.abort();
    }
    release_tx.send(()).unwrap();
    match caller.await {
        Err(error) if cancel_caller => assert!(error.is_cancelled()),
        Ok(Err(error)) if !cancel_caller => {
            assert!(error.to_string().contains("execution task failed"));
        }
        _ => panic!("expected a cancelled caller or a failed execution task"),
    }

    // Run at the very next admission. No inference is allowed to repair the
    // state first: this is the same ready() check used by GET /health.
    tokio::time::timeout(
        Duration::from_secs(5),
        scheduler.run(move || {
            assert!(executor.pipes.is_poisoned(), "the injected panic ran");
            assert!(
                !executor.ready(),
                "a panic must publish failure before releasing admission"
            );
            assert!(
                executor.child.lock().unwrap().try_wait().unwrap().is_some(),
                "the failed child must already be reaped"
            );
            assert!(
                executor
                    .exchange(b"{}")
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("unavailable"),
                "later requests must not reuse the poisoned stream"
            );
            Ok(())
        }),
    )
    .await
    .expect("admission released after panic")
    .expect("failure was published inside the blocking task");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn exchange_panic_marks_unavailable_before_next_admission() {
    panic_during_exchange(false).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancelled_exchange_panic_still_marks_unavailable() {
    panic_during_exchange(true).await;
}
