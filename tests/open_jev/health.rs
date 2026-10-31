//! HTTP health must follow the worker's actual lifecycle, without needing a GPU.
use super::health_response;
use axum::{body::to_bytes, http::StatusCode};
use omni_runtime::batching::{BatchCost, BatchLimits, BatchScheduler};
use serde_json::Value;

#[tokio::test]
async fn health_fails_after_panic_but_not_after_an_ordinary_error() {
    let scheduler: BatchScheduler<usize, usize> = BatchScheduler::new(
        BatchLimits { items: 1, cost: 1 },
        |_| BatchCost::from(1),
        |items: &[usize]| {
            if items == [1] {
                anyhow::bail!("recoverable request failure");
            }
            if items == [2] {
                panic!("executor state lost");
            }
            Ok(items.to_vec())
        },
    )
    .unwrap();
    assert!(scheduler.run(vec![1]).await.is_err());
    assert_eq!(
        health_response("open-jev-9b", scheduler.is_available()).status(),
        StatusCode::OK
    );
    assert_eq!(scheduler.run(vec![3]).await.unwrap(), [3]);
    assert!(scheduler.run(vec![2]).await.is_err());
    let response = health_response("open-jev-9b", scheduler.is_available());
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = to_bytes(response.into_body(), 1024).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "failed");
    assert_eq!(json["model"], "open-jev-9b");
    assert!(scheduler.run(vec![3]).await.is_err());
}
