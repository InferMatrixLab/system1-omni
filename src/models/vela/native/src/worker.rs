//! HTTP transport for the pinned Vela 2.0 0.8B CUDA bundle.

use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Duration,
};

use anyhow::{Context, Result, ensure};
use axum::{
    Json, Router,
    body::to_bytes,
    extract::{Request, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use omni_runtime::SerialScheduler;
use serde_json::{Value, json};

use crate::{
    bundle::{Bundle, MEMBERS},
    contract::{self, RequestError},
    executor::Executor,
    processing::Processor,
    scoring::Calibration,
};

pub struct Engine {
    model_name: String,
    processor: Arc<Mutex<Processor>>,
    executor: Arc<Mutex<Executor>>,
    calibration: Arc<Calibration>,
    scheduler: SerialScheduler,
}

impl Engine {
    /// Asset verification and actual inference complete before the router is served.
    pub async fn load(bundle: PathBuf, library: PathBuf, anchor: String) -> Result<Self> {
        tokio::task::spawn_blocking(move || Self::load_and_warm(&bundle, &library, &anchor)).await?
    }

    fn load_and_warm(bundle: &Path, library: &Path, anchor: &str) -> Result<Self> {
        let manifest = Bundle::load(bundle, anchor)?;
        ensure!(
            manifest.member.model_id == MEMBERS[0].model_id,
            "HTTP worker requires the Vela 2.0 0.8B bundle"
        );
        let model_name = manifest.model_name().to_owned();
        let processor = Processor::load(bundle, anchor)?;
        let calibration = Calibration::load(bundle, anchor)?;
        let mut executor = Executor::load(bundle, library, anchor)?;
        let body = json!({
            "model": model_name,
            "state": {"request": "Contact alice@example.com about the delivery."},
            "questions": {
                "delivery": {"type": "noul", "instructions": "Does this mention a delivery?"},
                "pii": {"type": "span", "instructions": "Find email addresses.",
                    "criteria": {"EMAIL_ADDRESS": "email address"}, "over": "request"},
                "event": {"type": "span", "instructions": "Find delivery events.",
                    "criteria": {"delivery": "delivery event"}, "over": "request"}
            }
        });
        let planned = processor
            .plan(body.as_object().context("warmup request")?)
            .map_err(|error| anyhow::anyhow!("warmup planning failed: {}", error.body))?;
        executor.execute(&model_name, &planned, &calibration, |_, _| {})?;
        Ok(Self {
            model_name,
            processor: Arc::new(Mutex::new(processor)),
            executor: Arc::new(Mutex::new(executor)),
            calibration: Arc::new(calibration),
            scheduler: SerialScheduler::default(),
        })
    }

    pub async fn decide(&self, raw: &[u8]) -> std::result::Result<Value, RequestError> {
        let body = contract::compile(raw, &self.model_name)?;
        let processor = Arc::clone(&self.processor);
        let planned = tokio::task::spawn_blocking(move || {
            let processor = processor.lock().map_err(report_failure)?;
            let planned = processor.plan(&body);
            processor.clear_request_cache();
            planned
        })
        .await
        .map_err(report_failure)??;
        let executor = Arc::clone(&self.executor);
        let calibration = Arc::clone(&self.calibration);
        let name = self.model_name.clone();
        self.scheduler
            .run(move || {
                // Keep the sole mutable CUDA owner and scheduler permit through
                // every row; forward_fixed reselects device 0 and synchronizes.
                let mut executor = executor
                    .lock()
                    .map_err(|_| anyhow::anyhow!("executor lock poisoned"))?;
                executor.execute(&name, &planned, &calibration, |_, _| {})
            })
            .await
            .map_err(report_failure)
    }
}

fn report_failure(error: impl std::fmt::Display) -> RequestError {
    eprintln!("Vela inference failed: {error:#}");
    RequestError::internal()
}

pub fn router(engine: Arc<Engine>) -> Router {
    Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/ready", get(|| async { Json(json!({"ready": true})) }))
        .route("/v1/systemone", post(system_one))
        .fallback(|| async { rejection(RequestError::detail(404, "unknown route")) })
        .with_state(engine)
}

async fn system_one(State(engine): State<Arc<Engine>>, request: Request) -> Response {
    let Some(content_type) = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
    else {
        return rejection(RequestError::detail(
            415,
            "Content-Type must be application/json",
        ));
    };
    if !content_type
        .split(';')
        .next()
        .is_some_and(|v| v.trim().eq_ignore_ascii_case("application/json"))
    {
        return rejection(RequestError::detail(
            415,
            "Content-Type must be application/json",
        ));
    }
    if request
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .is_some_and(|n| n > contract::MAX_BODY as u64)
    {
        return rejection(RequestError::detail(
            413,
            "request exceeds the 8 MiB body limit",
        ));
    }
    let raw = match tokio::time::timeout(
        Duration::from_secs(15),
        to_bytes(request.into_body(), contract::MAX_BODY),
    )
    .await
    {
        Ok(Ok(raw)) => raw,
        Ok(Err(error))
            if std::error::Error::source(&error)
                .is_some_and(|source| source.is::<http_body_util::LengthLimitError>()) =>
        {
            return rejection(RequestError::detail(
                413,
                "request exceeds the 8 MiB body limit",
            ));
        }
        Ok(Err(_)) => return rejection(RequestError::detail(400, "incomplete body")),
        Err(_) => return rejection(RequestError::detail(408, "request body timed out")),
    };
    match engine.decide(&raw).await {
        Ok(value) => Json(value).into_response(),
        Err(error) => rejection(error),
    }
}

fn rejection(error: RequestError) -> Response {
    (
        StatusCode::from_u16(error.status).expect("valid Vela status"),
        Json(error.body),
    )
        .into_response()
}
