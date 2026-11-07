//! Shared fixtures for the vela-native test targets. The root config is
//! the pinned vllm-sr/Vela-2.0-0.3B@d6f03aa9 config, structure-relevant
//! members only; recipe fields stay worker-side. Regenerate it together
//! with the census and yarn references when the pinned revision moves.

use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use tempfile::{TempDir, tempdir};

pub fn pinned_config() -> Value {
    serde_json::from_str(include_str!("../data/vela-config.json")).unwrap()
}

/// Writes the pinned root config into a temp bundle dir. Keep the
/// returned `TempDir` alive for as long as the path is read.
pub fn bundle_dir() -> (TempDir, PathBuf) {
    let dir = tempdir().unwrap();
    fs::write(dir.path().join("config.json"), pinned_config().to_string()).unwrap();
    let path = dir.path().to_path_buf();
    (dir, path)
}
