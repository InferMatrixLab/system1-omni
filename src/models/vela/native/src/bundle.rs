//! The exported native bundle's manifest gate.
//!
//! [`recipe/vela/export_native.py`](../../../recipe/vela/export_native.py)
//! produces the bundle from a pinned package; this module loads its
//! manifest, pins the source revision and the member's tensor
//! inventory, and re-hashes every bundle file before the executor
//! touches it — the same provenance-before-import discipline as the
//! reference worker's package gate. One program serves every decoder
//! member; the table below mirrors the Python member registry
//! (`models.vela.hybrid.MEMBERS`), with the inventories counted from
//! each pinned revision's shard map and `broad_head.safetensors`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde_json::Value;

pub const FORMAT: &str = "vela2-native-bundle/1";

/// One pinned decoder member and its exported-inventory pins.
#[derive(Debug)]
pub struct MemberPins {
    pub model_id: &'static str,
    pub revision: &'static str,
    /// (total, bf16, fp32) backbone tensors in the export.
    pub backbone: (u64, u64, u64),
    /// F32 head tensors: the shard map's `head.*` / `span2.*` /
    /// `set_bias` entries plus the side-loaded `span_broad.*` file.
    pub heads: u64,
    pub max_length: u64,
}

/// The pinned members, in family order. 4B and 9B share one layer
/// structure (32 hybrid layers; the widths differ), so their tensor
/// counts are identical; 0.8B is the compact layout.
pub const MEMBERS: &[MemberPins] = &[
    MemberPins {
        model_id: "vllm-sr/Vela-2.0-0.8B",
        revision: "326b01d81f61fff7fd4ae570e54b87a49119e490",
        backbone: (320, 204, 116),
        heads: 37,
        max_length: 16_384,
    },
    MemberPins {
        model_id: "vllm-sr/Vela-2.0-4B",
        revision: "d692394127f6f87fc581d8634c9558fe4776efce",
        backbone: (426, 272, 154),
        heads: 37,
        max_length: 16_384,
    },
    MemberPins {
        model_id: "vllm-sr/Vela-2.0-9B",
        revision: "068425bb145c05884a98b870c219047da22d9aea",
        backbone: (426, 272, 154),
        heads: 37,
        max_length: 16_384,
    },
];

/// The pins for a model id, or None when the id names no member.
pub fn member(model_id: &str) -> Option<&'static MemberPins> {
    MEMBERS.iter().find(|member| member.model_id == model_id)
}

#[derive(Debug)]
pub struct Bundle {
    pub dir: PathBuf,
    pub manifest: Value,
    pub member: &'static MemberPins,
}

impl Bundle {
    /// Load and fully verify one exported bundle directory.
    pub fn load(dir: &Path) -> Result<Self> {
        let manifest: Value = serde_json::from_slice(
            &std::fs::read(dir.join("vela_native_manifest.json"))
                .context("export the pinned package first; see recipe/vela/export_native.py")?,
        )
        .context("vela_native_manifest.json must be JSON")?;
        let model_id = manifest["model_id"]
            .as_str()
            .context("the manifest's model_id must be a string")?;
        let member = member(model_id).with_context(|| {
            format!(
                "expected a bundle exported from a pinned member ({}); got {model_id}",
                MEMBERS
                    .iter()
                    .map(|m| m.model_id)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
        ensure!(
            manifest["format"] == FORMAT
                && manifest["source"]["revision"].as_str() == Some(member.revision),
            "expected a bundle exported from the pinned {} package",
            member.model_id
        );
        let backbone = &manifest["backbone"];
        ensure!(
            backbone["tensors"].as_u64() == Some(member.backbone.0)
                && backbone["bf16"].as_u64() == Some(member.backbone.1)
                && backbone["fp32"].as_u64() == Some(member.backbone.2),
            "unexpected backbone inventory; re-export from the pinned package"
        );
        ensure!(
            manifest["heads"]["tensors"].as_u64() == Some(member.heads),
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
            let actual = hash_file(&dir.join(name))?;
            ensure!(
                actual == digest.as_str().unwrap_or(""),
                "{name} does not match the manifest SHA-256"
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
        // package root: every file under the bundle, at any depth, must
        // be pinned (or be this manifest), and every directory must lie
        // on the path of a pinned file (the export's bf16/ backbone) —
        // anything else is unlisted.
        let pinned: HashSet<&str> = files.keys().map(String::as_str).collect();
        let mut unpinned: Vec<String> = Vec::new();
        walk_unpinned(dir, "", &pinned, &mut unpinned)
            .with_context(|| format!("cannot walk the bundle directory {}", dir.display()))?;
        unpinned.sort();
        ensure!(
            unpinned.is_empty(),
            "unpinned file(s) in the bundle directory: {unpinned:?}"
        );
        ensure!(
            manifest["max_length"].as_u64() == Some(member.max_length)
                && manifest["model_name"].as_str() == Some(member.model_id),
            "unexpected member identity in the manifest"
        );
        Ok(Self {
            dir: dir.to_owned(),
            manifest,
            member,
        })
    }

    /// The served model name (upstream: config.json `model_name`).
    pub fn model_name(&self) -> &str {
        self.manifest["model_name"]
            .as_str()
            .unwrap_or(self.member.model_id)
    }

    /// The member's input window.
    pub fn max_length(&self) -> usize {
        self.manifest["max_length"]
            .as_u64()
            .unwrap_or(self.member.max_length) as usize
    }
}

/// One level of the fail-closed walk: `prefix` is the bundle-relative
/// path of `dir` (with a trailing `/` below the root).
fn walk_unpinned(
    dir: &Path,
    prefix: &str,
    pinned: &HashSet<&str>,
    unpinned: &mut Vec<String>,
) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let rel = format!("{prefix}{name}");
        if entry.path().is_dir() {
            let nested = format!("{rel}/");
            if pinned.iter().any(|key| key.starts_with(&nested)) {
                walk_unpinned(&entry.path(), &nested, pinned, unpinned)?;
            } else {
                unpinned.push(rel);
            }
        } else if rel != "vela_native_manifest.json" && !pinned.contains(rel.as_str()) {
            unpinned.push(rel);
        }
    }
    Ok(())
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
