//! Verify the complete preparation contract before loading any CUDA library.
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs::File, io::Read, path::Path};

pub const PINS: &str = include_str!("../../../../../recipe/kev/pins.json");
const SHARDS: [&str; 2] = [
    "model.safetensors-00001-of-00002.safetensors",
    "model.safetensors-00002-of-00002.safetensors",
];
const FILES: [&str; 7] = [
    "config.json",
    "head.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    SHARDS[0],
    SHARDS[1],
];

pub fn validate_manifest(manifest: &Value) -> Result<()> {
    let pins: Value = serde_json::from_str(PINS)?;
    for (key, value) in pins.as_object().unwrap() {
        ensure!(
            manifest.get(key) == Some(value),
            "unsupported Kev export field: {key}"
        );
    }
    let files = manifest["files"]
        .as_object()
        .context("missing export files")?;
    ensure!(
        files.keys().map(String::as_str).collect::<BTreeSet<_>>() == FILES.into_iter().collect(),
        "unexpected or missing export files"
    );
    for (name, digest) in files {
        let digest = digest.as_str().context("export hash must be a string")?;
        ensure!(
            digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit()),
            "invalid hash: {name}"
        );
    }
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
        ensure!(
            files[name] == pins["inputs"]["base"][name],
            "expected pinned base {name}"
        );
    }
    ensure!(
        files["head.safetensors"]
            == "65698f2568fc1b6ebb60a1c2a24c7bb6622dcc23e1c6d9efe8d3274ad918a9a7",
        "expected the pinned FP32 head"
    );
    Ok(())
}

pub fn sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("reading {}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

pub fn verify(dir: &Path) -> Result<()> {
    let manifest: Value = serde_json::from_slice(
        &std::fs::read(dir.join("kev_export.json")).context("run recipe/kev/export.py first")?,
    )?;
    validate_manifest(&manifest)?;
    for (name, digest) in manifest["files"].as_object().unwrap() {
        ensure!(
            sha256(&dir.join(name))? == digest.as_str().unwrap(),
            "export hash mismatch: {name}"
        );
    }
    ensure!(
        !dir.join("model.safetensors").exists(),
        "unlisted model.safetensors would shadow the verified shards"
    );
    let index: Value =
        serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json"))?)?;
    let map = index["weight_map"]
        .as_object()
        .context("weight_map must be an object")?;
    ensure!(!map.is_empty(), "empty weight map");
    for (name, shard) in map {
        ensure!(
            name.starts_with("model.language_model.")
                && shard.as_str().is_some_and(|s| SHARDS.contains(&s)),
            "invalid language weight map entry"
        );
    }
    // Check shapes without accepting a differently sized pointer head.
    let bytes = std::fs::read(dir.join("head.safetensors"))?;
    let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
    for name in ["q.weight", "k.weight"] {
        ensure!(
            tensors.tensor(name)?.shape() == [256, 2560],
            "expected Kev-4B {name}"
        );
    }
    Ok(())
}
