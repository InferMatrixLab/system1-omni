//! Valen-Preview-0923 native `/v1/systemone` worker (spec 03, N4).
//!
//!     VALEN_MODEL=<spec-02 export> VALEN_BASE=<Qwen3.5-2B base> omni-valen-native
//!
//! `VALEN_CUDA_LIB` (default: next to this executable), `VALEN_HOST` and
//! `VALEN_PORT` are optional; see recipe/valen/native.md. The worker binds
//! its listener only after one real image and one real text warmup inference,
//! so `/health`'s ready means what it says — as the reference worker does.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use omni_valen_native::engine::Engine;
use omni_valen_native::server::{self, Worker};

/// The reference worker's warmup requests, verbatim (`frontend/valen.py`):
/// one 2×2 white PNG choice and one text choice, image first.
const IMAGE_WARMUP: &str = concat!(
    r#"{"model": "valen-preview-0923", "state": {"image": "data:image/png;base64,"#,
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFklEQVR4nGP8//8/AwMDEwMDAwMDAwAkBgMB/DXemwAAAABJRU5ErkJggg==",
    r#""}, "questions": {"_warmup": {"type": "choice", "instructions": "Choose the only candidate.", "criteria": {"ok": "OK"}}}}"#
);
const TEXT_WARMUP: &str = r#"{"model": "valen-preview-0923", "state": "Warmup: choose the only candidate.", "questions": {"_warmup": {"type": "choice", "instructions": "Choose the only candidate.", "criteria": {"ok": "OK"}}}}"#;

fn main() -> Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("the worker needs Tokio I/O")?;
    runtime.block_on(run())
}

async fn run() -> Result<()> {
    let export = PathBuf::from(
        std::env::var_os("VALEN_MODEL").context("set VALEN_MODEL to the spec-02 export directory")?,
    );
    let base = PathBuf::from(
        std::env::var_os("VALEN_BASE")
            .context("set VALEN_BASE to the pinned Qwen3.5-2B base directory")?,
    );
    let library = match std::env::var_os("VALEN_CUDA_LIB") {
        Some(path) => path.into(),
        None => omni_qwen3_5_native::cuda::default_library()?,
    };
    ensure!(
        std::env::args_os().count() == 1,
        "usage: omni-valen-native (configure using VALEN_MODEL, VALEN_BASE, VALEN_CUDA_LIB, \
         VALEN_HOST and VALEN_PORT)"
    );
    let (export, base, library) = (export, base, library);
    let mut engine = tokio::task::spawn_blocking(move || Engine::load(&export, &base, &library))
        .await
        .context("engine load task failed")??;
    // One real inference per modality before the listener exists.
    for warmup in [IMAGE_WARMUP, TEXT_WARMUP] {
        engine.decide(warmup.as_bytes()).map_err(|error| {
            anyhow::anyhow!("warmup failed: {} {}", error.status, error.message)
        })?;
    }
    let worker = Arc::new(Worker::with_engine(engine));
    let host = std::env::var("VALEN_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port: u16 = std::env::var("VALEN_PORT")
        .map_or(Ok(8000), |v| v.parse())
        .context("VALEN_PORT")?;
    server::serve(worker, &host, port).await
}
