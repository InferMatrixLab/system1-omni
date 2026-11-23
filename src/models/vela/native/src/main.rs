use std::{net::SocketAddr, sync::Arc};

use anyhow::{Context, Result};
use omni_vela_native::worker::{Engine, router};

#[tokio::main]
async fn main() -> Result<()> {
    let bundle = std::env::var_os("VELA_NATIVE_BUNDLE").context("set VELA_NATIVE_BUNDLE")?;
    let library = std::env::var_os("VELA_CUDA_LIB").context("set VELA_CUDA_LIB")?;
    let anchor =
        std::env::var("VELA_NATIVE_MANIFEST_SHA256").context("set VELA_NATIVE_MANIFEST_SHA256")?;
    let bind: SocketAddr = std::env::var("VELA_NATIVE_BIND")
        .unwrap_or_else(|_| "127.0.0.1:8000".into())
        .parse()
        .context("invalid VELA_NATIVE_BIND socket address")?;
    let engine = Engine::load(bundle.into(), library.into(), anchor).await?;
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .context("bind Vela worker")?;
    eprintln!(
        "Vela 2.0 0.8B warmup complete, ready on {}",
        listener.local_addr()?
    );
    axum::serve(listener, router(Arc::new(engine))).await?;
    Ok(())
}
