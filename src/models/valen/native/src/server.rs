//! The native `/v1/systemone` HTTP surface (spec 03, N4), contract-identical
//! to the reference worker `frontend/valen.py`: the same route set, the same
//! header/body gate order (411 chunked, 411 single Content-Length, 400 invalid
//! length, 413 size limit, 415 media type, 503 busy, 400 incomplete body, 408
//! slow body) and the same `{"detail": ...}` error bodies, ready only after a
//! real warmup inference, serial admission over one request at a time.
//!
//! The transport is generic over [`Inference`]: the model-owned
//! [`crate::engine::Engine`] in the worker binary, or a stub in the CPU
//! contract tests (the reference suite's `FakeProcessor`/`FakeExecutor`
//! pattern).

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::{StatusCode, header};
use axum::response::Response;
use axum::routing::post;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::engine::{Engine, Outcome};
use crate::wire::RequestError;

/// The reference's body limit and socket timeout, verbatim.
pub const MAX_BODY: usize = 8 * 1024 * 1024;
pub const BODY_TIMEOUT: Duration = Duration::from_secs(15);

/// The ready health body, `json.dumps` byte-identical.
const HEALTH: &str =
    "{\"status\": \"ready\", \"model\": \"valen-preview-0923\", \"modality\": \"multimodal\"}";
const UNKNOWN_ROUTE: &str = "{\"detail\": \"unknown route\"}";

/// One decision request: raw bytes in, the serialized response body out; the
/// error carries the reference status and `detail` text.
pub trait Inference: Send + Sync {
    fn decide(&self, raw: &[u8]) -> Result<Vec<u8>, RequestError>;
}

/// The blocking decision step behind the admission mutex: the engine's full
/// request path. A trait separate from [`Inference`] so the CPU contract
/// tests can put a flaky stub behind the same `Mutex` recovery without CUDA.
pub trait Decide: Send {
    fn decide(&mut self, raw: &[u8]) -> Result<Outcome, RequestError>;
}

impl Decide for Engine {
    fn decide(&mut self, raw: &[u8]) -> Result<Outcome, RequestError> {
        Engine::decide(self, raw)
    }
}

/// The mutex-guarded model engine as the serving [`Inference`]: one request at
/// a time (the HTTP admission serializes callers before this is reached). A
/// panicking decision poisons the mutex; the poison is recovered — the
/// reference's `with`-block lock simply releases on the way out, so the next
/// request proceeds instead of failing with permanent 500s.
impl<T: Decide> Inference for Mutex<T> {
    fn decide(&self, raw: &[u8]) -> Result<Vec<u8>, RequestError> {
        let mut engine = self.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let Outcome {
            answers,
            logical_tokens,
            compute_tokens,
        } = engine.decide(raw)?;
        crate::finish::build_response(&answers, logical_tokens, compute_tokens)
            .map_err(|_| crate::wire::error(500, "inference failed"))
    }
}

/// The shared worker state: the inference and the single-request admission.
pub struct Worker {
    inference: Arc<dyn Inference>,
    /// Nonblocking single admission, as the reference's `inference_lock`.
    admission: Arc<Semaphore>,
    /// The reference's fixed 15 s body timeout; overridable in tests only.
    body_timeout: Duration,
}

impl Worker {
    pub fn new(inference: Arc<dyn Inference>) -> Self {
        Self {
            inference,
            admission: Arc::new(Semaphore::new(1)),
            body_timeout: BODY_TIMEOUT,
        }
    }

    /// The model-owned worker: the loaded engine behind its mutex.
    pub fn with_engine(engine: Engine) -> Self {
        Self::new(Arc::new(Mutex::new(engine)))
    }

    /// Test affordance: the reference deliberately fixes the body timeout;
    /// nothing outside the CPU contract suite sets this.
    pub fn with_body_timeout(mut self, timeout: Duration) -> Self {
        self.body_timeout = timeout;
        self
    }
}

fn json_response(status: StatusCode, body: impl Into<Bytes>) -> Response {
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.into()))
        .expect("a static response head")
}

fn detail(status: u16, message: &str) -> Response {
    let body = format!(
        "{{\"detail\": {}}}",
        crate::finish::json_string(message)
    );
    json_response(StatusCode::from_u16(status).expect("a static status"), body)
}

/// The admitted decision pipeline: the permit is held across the whole
/// blocking inference, exactly as the reference holds its lock through
/// `decide`.
async fn decide(inference: Arc<dyn Inference>, raw: Bytes, permit: OwnedSemaphorePermit) -> Response {
    let result = tokio::task::spawn_blocking(move || {
        let _guard = permit;
        inference.decide(&raw)
    })
    .await;
    match result {
        Ok(Ok(body)) => json_response(StatusCode::OK, body),
        Ok(Err(error)) => detail(error.status, &error.message),
        Err(_) => detail(500, "inference failed"),
    }
}

/// `do_POST`'s gate order, byte-identical rejections.
async fn systemone(State(worker): State<Arc<Worker>>, request: Request) -> Response {
    let headers = request.headers().clone();
    if headers.contains_key(header::TRANSFER_ENCODING) {
        return detail(
            411,
            "Content-Length is required; chunked requests are unsupported",
        );
    }
    let lengths = headers.get_all(header::CONTENT_LENGTH);
    if lengths.iter().count() != 1 {
        return detail(411, "one Content-Length is required");
    }
    let length = lengths
        .iter()
        .next()
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<i64>().ok());
    let Some(length) = length else {
        return detail(400, "invalid Content-Length");
    };
    if !(0..=(MAX_BODY as i64)).contains(&length) {
        return detail(413, "request exceeds body limit");
    }
    // `email.message.get_content_type()`: the first Content-Type's media type,
    // parameters stripped, lowercased; absent means the text/plain default.
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.split(';').next().unwrap_or("").trim().to_ascii_lowercase())
        .unwrap_or_else(|| "text/plain".to_owned());
    if content_type != "application/json" {
        return detail(415, "Content-Type must be application/json");
    }
    // Busy before the body, as the reference orders it; the permit is held
    // through the response.
    let Ok(permit) = worker.admission.clone().try_acquire_owned() else {
        return detail(503, "worker busy");
    };
    let body = match tokio::time::timeout(
        worker.body_timeout,
        axum::body::to_bytes(request.into_body(), MAX_BODY),
    )
    .await
    {
        Ok(Ok(bytes)) => bytes,
        Ok(Err(_)) => return detail(400, "incomplete body"),
        Err(_) => return detail(408, "request body timed out"),
    };
    if body.len() != length as usize {
        return detail(400, "incomplete body");
    }
    decide(worker.inference.clone(), body, permit).await
}

async fn health() -> Response {
    json_response(StatusCode::OK, HEALTH)
}

async fn unknown() -> Response {
    json_response(StatusCode::NOT_FOUND, UNKNOWN_ROUTE)
}

/// The worker's router: `/health` and `/v1/systemone` served, every other
/// route (or method on them) the reference's 404 body.
pub fn router(worker: Arc<Worker>) -> Router {
    Router::new()
        .route("/health", axum::routing::get(health).post(unknown))
        .route("/v1/systemone", post(systemone).get(unknown))
        .fallback(unknown)
        .with_state(worker)
}

/// The serving loop: never returns under normal operation.
pub async fn serve(worker: Arc<Worker>, host: &str, port: u16) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind((host, port)).await?;
    eprintln!("Valen worker ready on {host}:{port}");
    axum::serve(listener, router(worker)).await?;
    Ok(())
}
