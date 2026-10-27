#[tokio::main]
async fn main() -> anyhow::Result<()> {
    omni_clef::serve_from_env().await
}
