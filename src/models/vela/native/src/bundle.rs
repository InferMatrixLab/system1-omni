//! The exported native bundle's manifest gate.
//!
//! [`recipe/vela/export_native.py`](../../../recipe/vela/export_native.py)
//! produces the bundle from the pinned package; this module loads its
//! manifest, pins the source revision and the member's tensor
//! inventory, and re-hashes every bundle file before the (future)
//! executor touches it — the same provenance-before-import discipline
//! as the reference worker's package gate.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde_json::Value;

pub const FORMAT: &str = "vela2-native-bundle/1";
pub const MODEL_ID: &str = "vllm-sr/Vela-2.0-0.8B";
pub const SOURCE_REVISION: &str = "326b01d81f61fff7fd4ae570e54b87a49119e490";
/// This member's exported inventory, pinned from the pinned package:
/// 320 backbone tensors (204 BF16 Linear/conv1d + 116 F32) and 37 F32
/// head tensors (CandidateHead, set_bias, span2 router, span_broad).
pub const BACKBONE_TENSORS: (usize, usize, usize) = (320, 204, 116);
pub const HEAD_TENSORS: usize = 37;

#[derive(Debug)]
pub struct Bundle {
    pub dir: PathBuf,
    pub manifest: Value,
}

impl Bundle {
    /// Load and fully verify one exported bundle directory.
    pub fn load(dir: &Path) -> Result<Self> {
        let manifest: Value = serde_json::from_slice(
            &std::fs::read(dir.join("vela_native_manifest.json"))
                .context("export the pinned package first; see recipe/vela/export_native.py")?,
        )
        .context("vela_native_manifest.json must be JSON")?;
        ensure!(
            manifest["format"] == FORMAT
                && manifest["model_id"] == MODEL_ID
                && manifest["source"]["revision"] == SOURCE_REVISION,
            "expected a bundle exported from the pinned {MODEL_ID} package"
        );
        let backbone = &manifest["backbone"];
        ensure!(
            backbone["tensors"].as_u64() == Some(BACKBONE_TENSORS.0 as u64)
                && backbone["bf16"].as_u64() == Some(BACKBONE_TENSORS.1 as u64)
                && backbone["fp32"].as_u64() == Some(BACKBONE_TENSORS.2 as u64),
            "unexpected backbone inventory; re-export from the pinned package"
        );
        ensure!(
            manifest["heads"]["tensors"].as_u64() == Some(HEAD_TENSORS as u64),
            "unexpected head inventory; re-export from the pinned package"
        );
        for (name, digest) in manifest["files_sha256"]
            .as_object()
            .context("files_sha256 must be an object")?
        {
            let actual = hash_file(&dir.join(name))?;
            ensure!(
                actual == digest.as_str().unwrap_or(""),
                "{name} does not match the manifest SHA-256"
            );
        }
        ensure!(
            manifest["max_length"].as_u64() == Some(16_384)
                && manifest["model_name"].as_str() == Some(MODEL_ID),
            "unexpected member identity in the manifest"
        );
        Ok(Self {
            dir: dir.to_owned(),
            manifest,
        })
    }

    /// The served model name (upstream: config.json `model_name`).
    pub fn model_name(&self) -> &str {
        self.manifest["model_name"].as_str().unwrap_or(MODEL_ID)
    }

    /// The member's input window.
    pub fn max_length(&self) -> usize {
        self.manifest["max_length"].as_u64().unwrap_or(16_384) as usize
    }
}

fn hash_file(path: &Path) -> Result<String> {
    use anyhow::bail;
    use sha2::{Digest, Sha256};
    use std::io::Read;

    let mut hasher = Sha256::new();
    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(error) => bail!("cannot read {}: {error}", path.display()),
    };
    let mut reader = std::io::BufReader::new(file);
    let mut chunk = vec![0u8; 8 * 1024 * 1024];
    loop {
        let read = reader.read(&mut chunk)?;
        if read == 0 {
            break;
        }
        hasher.update(&chunk[..read]);
    }
    Ok(hex(&hasher.finalize()))
}

/// Lower-hex of a digest; sha2 0.11's output has no `LowerHex` impl.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
