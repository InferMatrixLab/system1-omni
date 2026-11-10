//! `/v1/systemone` and `/health` for Clef.
//!
//! Status codes and JSON bodies match the previous Python handler. The
//! generic in-process frontend maps every refusal onto `{"error": ...}`
//! and replaces this health document, so Clef keeps its own transport and
//! forwards executor bytes unchanged.

use std::{sync::Arc, time::Duration};

use anyhow::Result;
use axum::{
    Router,
    body::Body,
    extract::{DefaultBodyLimit, Request, State},
    http::{HeaderMap, Method, StatusCode, header},
    response::Response,
};
use http_body_util::{BodyExt, Limited};
use omni_runtime::SerialScheduler;
use tokio::net::TcpListener;

use crate::executor::Executor;

pub const MAX_BODY: usize = 8 * 1024 * 1024;
const BODY_TIMEOUT: Duration = Duration::from_secs(15);

pub struct Engine {
    executor: Arc<Executor>,
    scheduler: SerialScheduler,
    pub body_timeout: Duration,
}

impl Engine {
    pub fn spawn(command: std::process::Command) -> Result<Self> {
        Ok(Self {
            executor: Arc::new(Executor::spawn(command)?),
            scheduler: SerialScheduler::default(),
            body_timeout: BODY_TIMEOUT,
        })
    }

    pub fn ready(&self) -> bool {
        self.executor.ready()
    }

    pub async fn decide(&self, body: Vec<u8>) -> Response {
        if !self.executor.ready() {
            return detail(StatusCode::SERVICE_UNAVAILABLE, "worker unavailable");
        }
        let executor = Arc::clone(&self.executor);
        match self.scheduler.run(move || executor.exchange(&body)).await {
            Ok(response) => match StatusCode::from_u16(response.status) {
                Ok(status) => json_response(status, response.body),
                Err(_) => detail(StatusCode::INTERNAL_SERVER_ERROR, "inference failed"),
            },
            Err(error) => {
                eprintln!("clef executor request failed: {error:#}");
                detail(StatusCode::INTERNAL_SERVER_ERROR, "inference failed")
            }
        }
    }

    fn health_response(&self) -> Response {
        if !self.executor.ready() {
            return detail(StatusCode::SERVICE_UNAVAILABLE, "worker unavailable");
        }
        json_response(StatusCode::OK, self.executor.health().to_vec())
    }
}

pub fn router(engine: Arc<Engine>) -> Router {
    Router::new()
        .fallback(dispatch)
        .layer(DefaultBodyLimit::disable())
        .with_state(engine)
}

pub async fn serve_on(listener: TcpListener, engine: Arc<Engine>) -> Result<()> {
    axum::serve(listener, router(engine)).await?;
    Ok(())
}

async fn dispatch(State(engine): State<Arc<Engine>>, request: Request) -> Response {
    match (request.method(), request.uri().path()) {
        (&Method::GET, "/health") => engine.health_response(),
        (&Method::POST, "/v1/systemone") => systemone(&engine, request).await,
        _ => detail(StatusCode::NOT_FOUND, "unknown route"),
    }
}

async fn systemone(engine: &Engine, request: Request) -> Response {
    let (parts, body) = request.into_parts();
    if parts.headers.contains_key(header::TRANSFER_ENCODING) {
        return detail(
            StatusCode::LENGTH_REQUIRED,
            "Content-Length is required; chunked requests are unsupported",
        );
    }
    let lengths = content_lengths(&parts.headers);
    if lengths.len() != 1 {
        return detail(
            StatusCode::LENGTH_REQUIRED,
            "one Content-Length is required",
        );
    }
    let length = match parse_content_length(&lengths[0]) {
        Ok(length) => length,
        Err(LengthError::TooLarge) => {
            return detail(
                StatusCode::PAYLOAD_TOO_LARGE,
                "request exceeds the 8 MiB body limit",
            );
        }
        Err(LengthError::Invalid) => {
            return detail(StatusCode::BAD_REQUEST, "invalid Content-Length");
        }
    };
    if !json_content_type(&parts.headers) {
        return detail(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "Content-Type must be application/json",
        );
    }
    let raw = match take_body(body, length, engine.body_timeout).await {
        Ok(raw) => raw,
        Err(BodyError::Timeout) => {
            return detail(StatusCode::REQUEST_TIMEOUT, "request body timed out");
        }
        Err(BodyError::Incomplete) => {
            return detail(StatusCode::BAD_REQUEST, "incomplete body");
        }
    };
    engine.decide(raw).await
}

enum LengthError {
    Invalid,
    TooLarge,
}

fn content_lengths(headers: &HeaderMap) -> Vec<String> {
    let mut values = Vec::new();
    for value in headers.get_all(header::CONTENT_LENGTH) {
        let Ok(text) = value.to_str() else {
            values.push(String::new());
            continue;
        };
        if text.is_empty() {
            values.push(String::new());
            continue;
        }
        for part in text.split(',') {
            values.push(part.trim().to_string());
        }
    }
    values
}

fn parse_content_length(text: &str) -> Result<usize, LengthError> {
    if text.is_empty() {
        return Err(LengthError::Invalid);
    }
    match text.parse::<i128>() {
        Ok(value) if value < 0 || value > MAX_BODY as i128 => Err(LengthError::TooLarge),
        Ok(value) => Ok(usize::try_from(value).map_err(|_| LengthError::TooLarge)?),
        Err(_) if text.bytes().all(|byte| byte.is_ascii_digit()) => Err(LengthError::TooLarge),
        Err(_) => Err(LengthError::Invalid),
    }
}

fn json_content_type(headers: &HeaderMap) -> bool {
    headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("application/json"))
}

enum BodyError {
    Timeout,
    Incomplete,
}

async fn take_body(body: Body, length: usize, budget: Duration) -> Result<Vec<u8>, BodyError> {
    let read = async move {
        Limited::new(body, length)
            .collect()
            .await
            .map(|collected| collected.to_bytes().to_vec())
            .map_err(|_| BodyError::Incomplete)
    };
    match tokio::time::timeout(budget, read).await {
        Ok(Ok(bytes)) if bytes.len() == length => Ok(bytes),
        Ok(Ok(_)) => Err(BodyError::Incomplete),
        Ok(Err(error)) => Err(error),
        Err(_) => Err(BodyError::Timeout),
    }
}

fn detail(status: StatusCode, message: &str) -> Response {
    debug_assert!(!message.contains('"') && !message.contains('\\'));
    json_response(status, format!(r#"{{"detail": "{message}"}}"#))
}

fn json_response(status: StatusCode, body: impl Into<Vec<u8>>) -> Response {
    let body = body.into();
    let length = body.len();
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::CONTENT_LENGTH, length)
        .body(Body::from(body))
        .expect("response headers are valid")
}
