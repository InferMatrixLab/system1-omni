//! HTTP transport only; parsing, processing and answer semantics stay in Kev modules.
use crate::{contract, engine::Engine};
use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, State, rejection::BytesRejection},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use omni_runtime::{admission::BoundedAdmission, control::ExecutionContext};
use serde_json::json;
use std::sync::Arc;

#[derive(Clone)]
struct Shared {
    engine: Arc<Engine>,
    admission: BoundedAdmission,
}

pub fn error(status: StatusCode, detail: impl ToString) -> Response {
    (status, Json(json!({"detail":detail.to_string()}))).into_response()
}

async fn decide(
    State(shared): State<Shared>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    if !headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .eq_ignore_ascii_case("application/json")
    {
        return error(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "Content-Type must be application/json",
        );
    }
    let raw = match body {
        Ok(b) => b,
        Err(e) => return error(e.status(), e.body_text()),
    };
    let Some(permit) = shared.admission.try_admit(raw.len()) else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            [(header::RETRY_AFTER, "1")],
            Json(json!({"detail":"worker is busy"})),
        )
            .into_response();
    };
    let context = ExecutionContext::default().retain(permit);
    let _cancel = context.cancel_on_drop();
    let processor = shared.engine.processor.clone();
    let preparing = context.clone();
    let prepared = tokio::task::spawn_blocking(move || {
        let _hold = preparing;
        let request =
            omni_qwen3_5_native::json::parse(&raw).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
        processor
            .prepare(&request)
            .map_err(|e| (StatusCode::UNPROCESSABLE_ENTITY, e.to_string()))
    })
    .await;
    let prepared = match prepared {
        Ok(Ok(p)) => p,
        Ok(Err((status, detail))) => return error(status, detail),
        Err(e) => {
            eprintln!("Kev preparation failed: {e:#}");
            return error(StatusCode::INTERNAL_SERVER_ERROR, "preparation failed");
        }
    };
    match shared.engine.infer(prepared, context).await {
        Ok(body) => Json(body).into_response(),
        Err(e) => {
            eprintln!("Kev inference failed: {e:#}");
            error(StatusCode::INTERNAL_SERVER_ERROR, "model inference failed")
        }
    }
}

/// Call only after Engine::warmup succeeds; the binary binds its listener afterward.
pub fn router(engine: Arc<Engine>) -> Router {
    let shared = Shared {
        engine,
        admission: BoundedAdmission::new(16, Some(16 * contract::MAX_BODY)),
    };
    Router::new()
        .route(
            "/health",
            get(|| async {
                Json(json!({
                    "status":"ready", "model":contract::MODEL,
                    "execution":"eager", "prefix_reuse":false
                }))
            }),
        )
        .route("/v1/systemone", post(decide))
        .layer(DefaultBodyLimit::max(contract::MAX_BODY))
        .with_state(shared)
}
