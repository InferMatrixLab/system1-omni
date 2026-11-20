//! The exported native bundle's manifest gate.
//!
//! [`recipe/vela/export_native.py`](../../../recipe/vela/export_native.py)
//! produces the bundle from the pinned package; this module loads its
//! manifest, pins the source revision and the member's tensor
//! inventory, and re-hashes every bundle file before the executor
//! touches it — the same provenance-before-import discipline as the
//! reference worker's package gate. One program serves every decoder
//! member; the table below mirrors the Python member registry
//! (`models.vela.hybrid.MEMBERS`), with the inventories counted from
//! each pinned revision's shard map and `broad_head.safetensors`.
//!
//! `pinned_manifest_sha256` anchors the whole integrity chain: the
//! manifest's own bytes must hash to it, so the `files_sha256` table
//! is checked against a value fixed outside the bundle directory (the
//! exporter prints it) instead of against itself. The three verbatim
//! runtime files are additionally cross-checked against the
//! repository's per-member pin table, so a bundle exported from a
//! tampered package fails even with a valid anchor.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde_json::Value;

pub const FORMAT: &str = "vela2-native-bundle/1";

/// `tokenizer.json` hashes the same at every pinned revision; the
/// calibration and tokenizer-config digests are member-specific (the
/// reference worker's `MEMBERS` tables are the source of all three).
const TOKENIZER_JSON_SHA: &str = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523";
const RUNTIME_PINS_0_8B: &[(&str, &str)] = &[
    (
        "calibration.json",
        "d96e447962d0dc1d58fa8422dddbae8b99960e21fd75a10ef9b096f700a211f5",
    ),
    ("tokenizer.json", TOKENIZER_JSON_SHA),
    (
        "tokenizer_config.json",
        "bee8eba30f0eb4af73c0fe2cd06d0f89b657d7819941c438157ec42f7c80ea87",
    ),
];
const RUNTIME_PINS_4B: &[(&str, &str)] = &[
    (
        "calibration.json",
        "2a3f3cfa2d1cf6fa133d887960261b71c09420899f817f4156677962ac02460f",
    ),
    ("tokenizer.json", TOKENIZER_JSON_SHA),
    (
        "tokenizer_config.json",
        "353c083805d863719e54973335a38e252a92e1ac64d32b83bb0b333cc1644ef2",
    ),
];
const RUNTIME_PINS_9B: &[(&str, &str)] = &[
    (
        "calibration.json",
        "9fa0f31ce1c7b784220fd73320ee64da9b3b789c403755af91bcbefb5740d766",
    ),
    ("tokenizer.json", TOKENIZER_JSON_SHA),
    (
        "tokenizer_config.json",
        "bee8eba30f0eb4af73c0fe2cd06d0f89b657d7819941c438157ec42f7c80ea87",
    ),
];

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
    /// The repository's pin digests for the three verbatim runtime
    /// files, cross-checked against the manifest's own table so a
    /// self-consistently rewritten bundle fails the same way the
    /// reference worker's package gate fails a rewritten package.
    pub runtime_pins: &'static [(&'static str, &'static str)],
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
        runtime_pins: RUNTIME_PINS_0_8B,
    },
    MemberPins {
        model_id: "vllm-sr/Vela-2.0-4B",
        revision: "d692394127f6f87fc581d8634c9558fe4776efce",
        backbone: (426, 272, 154),
        heads: 37,
        max_length: 16_384,
        runtime_pins: RUNTIME_PINS_4B,
    },
    MemberPins {
        model_id: "vllm-sr/Vela-2.0-9B",
        revision: "068425bb145c05884a98b870c219047da22d9aea",
        backbone: (426, 272, 154),
        heads: 37,
        max_length: 16_384,
        runtime_pins: RUNTIME_PINS_9B,
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
    ///
    /// `pinned_manifest_sha256` anchors the whole integrity chain: the
    /// manifest's own bytes must hash to it, so the `files_sha256` table
    /// is checked against a value fixed outside the bundle directory
    /// (the exporter prints it) instead of against itself.
    pub fn load(dir: &Path, pinned_manifest_sha256: &str) -> Result<Self> {
        Self::load_with_runtime_pins(dir, pinned_manifest_sha256, None)
    }

    /// Test seam mirroring the reference gate's injectable pin table:
    /// the same anchored verification against caller-supplied digests
    /// for the three verbatim runtime files (production always uses the
    /// selected member's own table).
    pub fn load_with_runtime_pins(
        dir: &Path,
        pinned_manifest_sha256: &str,
        runtime_pins: Option<&[(&str, &str)]>,
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
            let path = pinned_path(dir, name)?;
            let actual = hash_file(&path)?;
            ensure!(
                actual == digest.as_str().unwrap_or(""),
                "{name} does not match the manifest SHA-256"
            );
        }
        // The verbatim runtime files must also match the repository's
        // own pin table, not just the manifest's anchored one.
        for (name, pinned) in runtime_pins.unwrap_or(member.runtime_pins) {
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
