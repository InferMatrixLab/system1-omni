use std::{net::SocketAddr, path::PathBuf, sync::Arc};

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, State, rejection::BytesRejection},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use omni_intern_native::{
    engine::Engine,
    finishing::{BACKEND, MODEL_NAME},
    processing::RequestError,
};
use omni_qwen3_5_native::cuda;
use serde_json::json;

const WARMUP: &[u8] = br#"{"state":"The light is red.","questions":{"color":{"type":"choice","instructions":"Which color?","criteria":{"red":"Red","blue":"Blue"}}}}"#;
const MAX_BODY: usize = 2 << 20;

fn error(status: StatusCode, message: &str) -> Response {
    (status, Json(json!({"error": message}))).into_response()
}

async fn systemone(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let content_type = headers
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .split(';')
        .next()
        .unwrap_or_default()
        .trim();
    if !content_type.eq_ignore_ascii_case("application/json") {
        return error(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "Content-Type must be application/json",
        );
    }
    let body = match body {
        Ok(body) => body,
        Err(err) => return error(err.status(), &err.body_text()),
    };
    match engine.decide(&body).await {
        Ok(result) => Json(result).into_response(),
        Err(err) => {
            if let Some(request) = err.downcast_ref::<RequestError>() {
                return error(
                    StatusCode::from_u16(request.status).unwrap(),
                    &request.to_string(),
                );
            }
            eprintln!("Intern inference failed: {err:#}");
            error(StatusCode::INTERNAL_SERVER_ERROR, "model inference failed")
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let checkpoint = std::env::var_os("INTERN_CHECKPOINT")
        .map(PathBuf::from)
        .context("set INTERN_CHECKPOINT to the pinned Intern-Decision-4B snapshot")?;
    let library = std::env::var_os("INTERN_CUDA_LIB")
        .map(PathBuf::from)
        .map_or_else(cuda::default_library, Ok)?;
    let address: SocketAddr = std::env::var("INTERN_LISTEN")
        .unwrap_or_else(|_| "127.0.0.1:8000".into())
        .parse()
        .context("invalid INTERN_LISTEN")?;
    let engine = Arc::new(Engine::load(&checkpoint, &library).await?);
    engine
        .decide(WARMUP)
        .await
        .context("Intern warmup failed")?;
    let app = Router::new()
        .route(
            "/health",
            get(|| async {
                Json(json!({"status": "ready", "model": MODEL_NAME, "backend": BACKEND}))
            }),
        )
        .route("/v1/systemone", post(systemone))
        .layer(DefaultBodyLimit::max(MAX_BODY))
        .with_state(engine);
    let listener = tokio::net::TcpListener::bind(address).await?;
    eprintln!("Intern listening on {address}");
    axum::serve(listener, app).await?;
    Ok(())
}
