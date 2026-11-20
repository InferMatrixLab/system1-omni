use anyhow::{Context, Result, ensure};
use omni_modernbert::structure::ModernBertStructure;
use omni_modernbert::{ModernBertConfigAdapter, SourceMeta};
use serde_json::Value as Json;
use std::path::Path;

/// Reads the pinned bundle layout: one `config.json` whose root is the
/// `vela2-unified` model description with the ModernBERT fields under
/// `encoder_config`. The root's recipe fields (`max_length`, markers,
/// calibration pointers) are worker-side and never structure input.
pub fn read_structure(dir: &Path) -> Result<(ModernBertStructure, SourceMeta)> {
    let path = dir.join("config.json");
    let bytes = std::fs::read(&path).with_context(|| format!("{}", path.display()))?;
    let root: Json =
        serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
    ensure!(
        root["model_type"].as_str() == Some("vela2-unified"),
        "model_type {:?} is not the vela2-unified 0.3B bundle",
        root["model_type"]
    );
    let encoder = root
        .get("encoder_config")
        .context("encoder_config is missing")?;
    ensure!(
        encoder["model_type"].as_str() == Some("modernbert"),
        "encoder_config model_type {:?} is not modernbert",
        encoder["model_type"]
    );
    ModernBertConfigAdapter::from_value(encoder, "config.json#encoder_config")
}
