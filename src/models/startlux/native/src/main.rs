//! STARTLUX_MODEL=<Instant 4B folder> omni-startlux-native
//!
//! Weights are CC BY-NC 4.0 and are not in this repository.

use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, State, rejection::BytesRejection},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use omni_qwen3_5_native::cuda;
use omni_startlux_native::contract::RequestError;
use omni_startlux_native::engine::{self, Engine};
use serde_json::json;

const WARMUP: &[u8] = br#"{"state":"Dialog: service is up.","questions":{"q":{"type":"choice","instructions":"Continue.","criteria":{"ok":"OK","wait":"Wait"}}}}"#;
const BODY_LIMIT: usize = 32 << 20;

fn contract_error(err: &RequestError) -> Response {
    let status = StatusCode::from_u16(err.status).unwrap_or(StatusCode::UNPROCESSABLE_ENTITY);
    (status, Json(err.body())).into_response()
}

fn failure(err: anyhow::Error) -> Response {
    if let Some(request) = err.downcast_ref::<RequestError>() {
        return contract_error(request);
    }
    eprintln!("inference failed: {err:#}");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({"error": "model inference failed"})),
    )
        .into_response()
}

async fn systemone(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let content_type = headers
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .split(';')
        .next()
        .unwrap_or("")
        .trim();
    if !content_type.eq_ignore_ascii_case("application/json") {
        return (
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            Json(json!({"error": "Content-Type must be application/json"})),
        )
            .into_response();
    }
    let raw = match body {
        Ok(raw) => raw,
        Err(err) => {
            return (err.status(), Json(json!({"error": err.body_text()}))).into_response();
        }
    };
    match engine.decide(&raw).await {
        Ok(body) => Json(body).into_response(),
        Err(err) => failure(err),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let model = std::env::var_os("STARTLUX_MODEL")
        .context("set STARTLUX_MODEL to the Instant 4B folder")?;
    let library = std::env::var_os("STARTLUX_CUDA_LIB")
        .map(Into::into)
        .map_or_else(cuda::default_library, Ok)?;
    let settings = engine::settings_from_env()?;
    eprintln!(
        "StartLux-Decision Instant 4B weights are {} ({}); this binary does not distribute them",
        omni_startlux_native::contract::LICENSE,
        omni_startlux_native::contract::WEIGHTS_REPO
    );
    let engine = Arc::new(Engine::load(model.as_ref(), &library, settings).await?);
    ensure!(
        systemone_status(&engine, WARMUP).await == StatusCode::OK,
        "warmup failed"
    );
    let host = std::env::var("STARTLUX_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port: u16 = std::env::var("STARTLUX_PORT")
        .ok()
        .map(|value| value.parse())
        .transpose()
        .context("STARTLUX_PORT")?
        .unwrap_or(8000);
    let app = Router::new()
        .route("/health", get(|| async { Json(engine::health()) }))
        .route("/v1/systemone", post(systemone))
        .layer(DefaultBodyLimit::max(BODY_LIMIT))
        .with_state(engine);
    let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await?;
    println!("listening on {host}:{port}");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn systemone_status(engine: &Engine, raw: &[u8]) -> StatusCode {
    match engine.decide(raw).await {
        Ok(_) => StatusCode::OK,
        Err(err) => {
            eprintln!("warmup failed: {err:#}");
            StatusCode::INTERNAL_SERVER_ERROR
        }
    }
}
