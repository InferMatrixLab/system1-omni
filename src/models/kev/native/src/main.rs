//! KEV_MODEL=<verified export> omni-kev-native
use anyhow::{Context, Result};
use omni_kev_native::engine::Engine;
use omni_qwen3_5_native::cuda;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    let dir =
        std::env::var_os("KEV_MODEL").context("set KEV_MODEL to the pinned export directory")?;
    let library = std::env::var_os("KEV_CUDA_LIB")
        .map(Into::into)
        .map_or_else(cuda::default_library, Ok)?;
    let engine = Arc::new(Engine::load(dir.as_ref(), &library).await?);
    engine.warmup().await.context("Kev real warmup failed")?;
    let host = std::env::var("KEV_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port: u16 = std::env::var("KEV_PORT")
        .map_or(Ok(8000), |v| v.parse())
        .context("KEV_PORT")?;
    let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await?;
    println!("Kev ready on {host}:{port}");
    axum::serve(listener, omni_kev_native::http::router(engine)).await?;
    Ok(())
}
