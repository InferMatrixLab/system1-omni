use super::*;
use crate::runtime::{Runtime, RuntimeConfig};
use axum::body::Body;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn policy(requests: Option<&str>, timeout: Option<&str>) -> Runtime {
    Runtime::new(RuntimeConfig::from_values(requests, None, timeout).unwrap()).unwrap()
}
#[tokio::test]
async fn overload_is_json_503_with_retry_after_and_never_reads_the_body() {
    let runtime = policy(Some("1"), None);
    let held = runtime.admit(Instant::now()).unwrap();
    let response = within_budget(&runtime, Request::new(Body::empty()), |_, _| async {
        panic!("refused request reached preparation");
        #[allow(unreachable_code)]
        error(StatusCode::OK, "")
    })
    .await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(response.headers()["retry-after"], "1");
    drop(held);
    assert_eq!(runtime.report().unwrap().depth, 0);
}
#[tokio::test]
async fn upload_is_inside_the_same_deadline_and_releases_capacity() {
    let runtime = Arc::new(policy(Some("1"), Some("80")));
    let app = Router::new()
        .route(
            "/",
            post(
                |State(runtime): State<Arc<Runtime>>, request: Request| async move {
                    within_budget(&runtime, request, |_, _| async {
                        error(StatusCode::OK, "done")
                    })
                    .await
                },
            ),
        )
        .with_state(runtime.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let mut socket = tokio::net::TcpStream::connect(address).await.unwrap();
    socket.write_all(b"POST / HTTP/1.1\r\nHost: localhost\r\nContent-Length: 100\r\nConnection: close\r\n\r\nx").await.unwrap();
    let mut response = String::new();
    tokio::time::timeout(Duration::from_secs(3), socket.read_to_string(&mut response))
        .await
        .unwrap()
        .unwrap();
    assert!(response.starts_with("HTTP/1.1 504"), "{response}");
    assert_eq!(runtime.report().unwrap().depth, 0);
    server.abort();
}
#[tokio::test]
async fn deadline_covers_preparation_and_returns_504() {
    let runtime = policy(Some("1"), Some("20"));
    let (release, wait) = std::sync::mpsc::channel();
    let response = within_budget(
        &runtime,
        Request::new(Body::empty()),
        |_, context| async move {
            let context = context.clone();
            let outcome = tokio::task::spawn_blocking(move || {
                wait.recv().unwrap();
                context.check()
            })
            .await
            .unwrap();
            match outcome {
                Ok(()) => error(StatusCode::OK, ""),
                Err(stopped) => runtime_error(stopped.into(), StatusCode::INTERNAL_SERVER_ERROR),
            }
        },
    )
    .await;
    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    assert_eq!(runtime.report().unwrap().depth, 1);
    release.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(2), async {
        while runtime.report().unwrap().depth > 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}
#[tokio::test]
async fn successful_response_and_existing_body_rejections_are_preserved() {
    let runtime = policy(Some("1"), Some("1000"));
    let response = within_budget(
        &runtime,
        Request::new(Body::from("ok")),
        |raw, context| async move {
            assert_eq!(raw.as_ref(), b"ok");
            context.check().unwrap();
            error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "original validation error",
            )
        },
    )
    .await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(runtime.report().unwrap().depth, 0);
    let response = within_budget(
        &runtime,
        Request::new(Body::from(vec![0; 3 * 1024 * 1024])),
        |_, _| async {
            panic!("oversize reached preparation");
            #[allow(unreachable_code)]
            error(StatusCode::OK, "")
        },
    )
    .await;
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(runtime.report().unwrap().depth, 0);
}

#[tokio::test]
async fn token_ceiling_refusal_is_json_422_without_retry_after() {
    let runtime =
        Runtime::new(RuntimeConfig::from_values(Some("2"), Some("100"), None).unwrap()).unwrap();
    // A permanently oversized request is invalid both when idle and under contention.
    for held_tokens in [0, 60] {
        let held = runtime
            .reserve_tokens(runtime.admit(Instant::now()).unwrap(), held_tokens)
            .unwrap();
        let response = within_budget(&runtime, Request::new(Body::empty()), |_, context| async {
            match runtime.reserve_tokens(context, 101) {
                Ok(_) => panic!("oversized request reached execution"),
                Err(error) => runtime_error(error, StatusCode::INTERNAL_SERVER_ERROR),
            }
        })
        .await;
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert!(!response.headers().contains_key("retry-after"));
        assert_eq!(response.headers()["content-type"], "application/json");
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&body).unwrap(),
            json!({"detail": "request exceeds processed-token capacity"})
        );
        assert_eq!(runtime.report().unwrap().depth, 1);
        drop(held);
        assert_eq!(runtime.report().unwrap().depth, 0);
    }
    assert_eq!(runtime.report().unwrap().rejected, 2);
    let whole = runtime
        .reserve_tokens(runtime.admit(Instant::now()).unwrap(), 100)
        .unwrap();
    drop(whole);
}

#[tokio::test]
async fn temporary_token_contention_is_json_503_with_retry_after_and_recovers() {
    let runtime =
        Runtime::new(RuntimeConfig::from_values(Some("2"), Some("100"), None).unwrap()).unwrap();
    let held = runtime
        .reserve_tokens(runtime.admit(Instant::now()).unwrap(), 60)
        .unwrap();
    let response = within_budget(&runtime, Request::new(Body::empty()), |_, context| async {
        match runtime.reserve_tokens(context, 41) {
            Ok(_) => panic!("overloaded request reached execution"),
            Err(error) => runtime_error(error, StatusCode::INTERNAL_SERVER_ERROR),
        }
    })
    .await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(response.headers()["retry-after"], "1");
    assert_eq!(response.headers()["content-type"], "application/json");
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&body).unwrap(),
        json!({"detail": "inference capacity exhausted"})
    );
    assert_eq!(runtime.report().unwrap().depth, 1);
    assert_eq!(runtime.report().unwrap().rejected, 1);
    drop(held);
    let retried = runtime
        .reserve_tokens(runtime.admit(Instant::now()).unwrap(), 41)
        .unwrap();
    drop(retried);
    assert_eq!(runtime.report().unwrap().depth, 0);
}
