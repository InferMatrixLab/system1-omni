//! The exported native bundle's manifest gate.
//!
//! [`recipe/vela/export_native.py`](../../../recipe/vela/export_native.py)
//! produces the bundle from the pinned package; this module loads its
//! manifest, pins the source revision and the member's tensor
//! inventory, and re-hashes every bundle file before the (future)
//! executor touches it — the same provenance-before-import discipline
//! as the reference worker's package gate.
//!
//! The three verbatim runtime files are additionally cross-checked
//! against the repository's own pin table ([`RUNTIME_FILE_SHA256`],
//! mirrored from the reference worker's package gate), so a bundle
//! exported from a tampered package fails even with a valid anchor.

use std::collections::HashSet;
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
/// The runtime files the export copies verbatim from the pinned
/// package, with the repository's pin digests from
/// `src/models/vela/hybrid.py`'s `PINNED_FILES_SHA256`. The bundle
/// gate cross-checks the manifest's digests for these three against
/// this table, so a self-consistently rewritten bundle fails the same
/// way the reference worker's package gate fails a rewritten package.
pub const RUNTIME_FILE_SHA256: &[(&str, &str)] = &[
    (
        "calibration.json",
        "d96e447962d0dc1d58fa8422dddbae8b99960e21fd75a10ef9b096f700a211f5",
    ),
    (
        "tokenizer.json",
        "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    ),
    (
        "tokenizer_config.json",
        "bee8eba30f0eb4af73c0fe2cd06d0f89b657d7819941c438157ec42f7c80ea87",
    ),
];

#[derive(Debug)]
pub struct Bundle {
    pub dir: PathBuf,
    pub manifest: Value,
}

impl Bundle {
    /// Load and fully verify one exported bundle directory.
    ///
    /// `pinned_manifest_sha256` anchors the whole integrity chain: the
    /// manifest's own bytes must hash to it, so the `files_sha256` table
    /// is checked against a value fixed outside the bundle directory
    /// (the exporter prints it) instead of against itself.
    pub fn load(dir: &Path, pinned_manifest_sha256: &str) -> Result<Self> {
        Self::load_with_runtime_pins(dir, pinned_manifest_sha256, RUNTIME_FILE_SHA256)
    }

    /// Test seam mirroring the reference gate's injectable pin table:
    /// the same anchored verification against caller-supplied digests
    /// for the three verbatim runtime files (production always passes
    /// [`RUNTIME_FILE_SHA256`]).
    pub fn load_with_runtime_pins(
        dir: &Path,
        pinned_manifest_sha256: &str,
        runtime_pins: &[(&str, &str)],
    ) -> Result<Self> {
        let manifest_bytes = std::fs::read(dir.join("vela_native_manifest.json"))
            .context("export the pinned package first; see recipe/vela/export_native.py")?;
        ensure!(
            hash_bytes(&manifest_bytes) == pinned_manifest_sha256,
            "vela_native_manifest.json does not match the pinned SHA-256 anchor; \
             re-export the pinned package and re-pin the anchor it prints"
        );
        let manifest: Value = serde_json::from_slice(&manifest_bytes)
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
        let files = manifest["files_sha256"]
            .as_object()
            .context("files_sha256 must be an object")?;
        ensure!(
            !files.is_empty(),
            "files_sha256 pins no files; the gate must cover the whole bundle"
        );
        for (name, digest) in files {
            let path = pinned_path(dir, name)?;
            let actual = hash_file(&path)?;
            ensure!(
                actual == digest.as_str().unwrap_or(""),
                "{name} does not match the manifest SHA-256"
            );
        }
        // The verbatim runtime files must also match the repository's
        // own pin table, not just the manifest's anchored one.
        for (name, pinned) in runtime_pins {
            ensure!(
                files.get(*name).and_then(Value::as_str) == Some(*pinned),
                "{name} does not match the repository pin for the pinned package; \
                 re-export from the pinned package"
            );
        }
        // The uniform-BF16 backbone the executor runs is pinned too.
        if manifest["files_sha256"]
            .get("bf16/model.safetensors")
            .is_some()
        {
            ensure!(
                dir.join("bf16/model.safetensors").is_file(),
                "the manifest pins bf16/model.safetensors but the file is missing; re-export"
            );
        }
        // Fail closed on the directory, as the reference gate walks the
        // package root: a top-level entry must be a pinned file, this
        // manifest, or a directory holding pinned paths (the export's
        // bf16/ backbone) — anything else is unlisted, and inside an
        // admitted directory every file must be pinned too; a pinned
        // entry must be a regular file, never a symlink.
        let pinned: HashSet<&str> = files
            .keys()
            .map(String::as_str)
            .chain(std::iter::once("vela_native_manifest.json"))
            .collect();
        let mut unpinned: Vec<String> = Vec::new();
        for entry in std::fs::read_dir(dir)
            .with_context(|| format!("cannot list the bundle directory {}", dir.display()))?
        {
            let entry = entry.context("a readable bundle entry")?;
            let name = entry.file_name().to_string_lossy().into_owned();
            let file_type = entry.file_type().context("a typed bundle entry")?;
            ensure!(
                !file_type.is_symlink(),
                "bundle entry {name} is a symlink; the gate walks regular files"
            );
            if pinned.contains(name.as_str()) {
                continue;
            }
            let prefix = format!("{name}/");
            if !files.keys().any(|key| key.starts_with(&prefix)) {
                unpinned.push(name);
                continue;
            }
            for inner in std::fs::read_dir(entry.path())
                .with_context(|| format!("cannot list the bundle subdirectory {name}"))?
            {
                let inner = inner.context("a readable bundle entry")?;
                let inner_name = format!("{name}/{}", inner.file_name().to_string_lossy());
                if !pinned.contains(inner_name.as_str()) {
                    unpinned.push(inner_name);
                }
            }
        }
        unpinned.sort();
        ensure!(
            unpinned.is_empty(),
            "unpinned file(s) in the bundle directory: {unpinned:?}"
        );
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

/// Resolve one `files_sha256` key inside the bundle directory: the key
/// must be a plain relative path (no absolute prefix, no `.`/`..`
/// components) and must not cross a symlink at any level — otherwise a
/// crafted manifest could make the gate hash files outside the bundle.
fn pinned_path(dir: &Path, name: &str) -> Result<PathBuf> {
    ensure!(!name.is_empty(), "files_sha256 pins an empty name");
    let relative = Path::new(name);
    ensure!(
        !relative.is_absolute(),
        "files_sha256 key must be a plain relative name: {name:?}"
    );
    let mut checked = dir.to_path_buf();
    for component in relative.components() {
        let part = match component {
            std::path::Component::Normal(part) => part,
            _ => anyhow::bail!("files_sha256 key must be a plain relative name: {name:?}"),
        };
        checked.push(part);
        let meta = std::fs::symlink_metadata(&checked)
            .with_context(|| format!("cannot stat {}", checked.display()))?;
        ensure!(
            !meta.file_type().is_symlink(),
            "files_sha256 key must not cross a symlink: {name:?}"
        );
    }
    Ok(checked)
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

fn hash_bytes(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex(&Sha256::digest(bytes))
}
