//! Manifest-verified vision weights from the Valen checkpoint export.
//!
//! The spec-02 export (`valen-checkpoint-export/2`, `valen_export.json`) is
//! the only weight source: the base's `model.visual.*` tensors with the
//! trained vision-top blocks and merger substituted as plain replacements.
//! Validation follows the export-policy style used across the native workers —
//! the manifest's format, provenance pins and per-file hashes are checked
//! before any tensor is read, and the manifest's tensor records must agree
//! with the configuration-derived inventory (names, shapes, dtypes and which
//! tensors are trained replacements). No vision adapter exists for Valen.
use super::{BASE, Inventory, TensorStore, VisionConfig, VisionModel, inside, is_visual};
use anyhow::{Context, Result, ensure};
use safetensors::tensor::TensorView;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::Read,
    path::Path,
};

const MANIFEST: &str = "valen_export.json";
const FORMAT: &str = "valen-checkpoint-export/2";
const CHECKPOINT_REVISION: &str = "81b9c6396ded3a45a92ae8079af85f988aa40501";
const CHECKPOINT_SHA256: &str = "836622efe78fe757e2627aa6050c223d461c424ea430a1c110c15e6f42f5a012";
const CHECKPOINT_STAGE: &str = "vision_top";
const BASE_REPO: &str = "Qwen/Qwen3.5-2B";
const BASE_REVISION: &str = "15852e8c16360a2fea060d615a32b45270f8a8fc";

/// The trained tensors the vision_top stage replaces: the last four blocks
/// (48 tensors) plus the merger (6), matching `backbone.visual.blocks[-4:]`
/// and `merger` in the pinned training builder.
pub(super) fn trained_tensor_names(config: &VisionConfig) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for block in config.depth - 4..config.depth {
        for module in [
            "norm1",
            "norm2",
            "attn.qkv",
            "attn.proj",
            "mlp.linear_fc1",
            "mlp.linear_fc2",
        ] {
            for kind in ["weight", "bias"] {
                names.insert(format!("{BASE}blocks.{block}.{module}.{kind}"));
            }
        }
    }
    for module in ["norm", "linear_fc1", "linear_fc2"] {
        for kind in ["weight", "bias"] {
            names.insert(format!("{BASE}merger.{module}.{kind}"));
        }
    }
    names
}

/// Validate format, provenance pins and the visual tensor records against the
/// configuration-derived inventory. Pure: no file is read.
pub(super) fn validate_manifest(manifest: &Value, config: &VisionConfig) -> Result<()> {
    ensure!(
        manifest["format"] == FORMAT,
        "unsupported export manifest format: {:?}",
        manifest["format"]
    );
    let sources = &manifest["sources"];
    ensure!(
        sources["checkpoint"]["revision"].as_str() == Some(CHECKPOINT_REVISION)
            && sources["checkpoint"]["sha256"].as_str() == Some(CHECKPOINT_SHA256)
            && sources["checkpoint"]["stage"].as_str() == Some(CHECKPOINT_STAGE),
        "manifest checkpoint provenance does not match the pinned Valen-Preview-0923 export"
    );
    ensure!(
        sources["base"]["repo"].as_str() == Some(BASE_REPO)
            && sources["base"]["revision"].as_str() == Some(BASE_REVISION),
        "manifest base provenance does not match the pinned Qwen3.5-2B revision"
    );
    let tensors = manifest["tensors"]
        .as_array()
        .context("manifest has no tensor records")?;
    let mut visual = BTreeMap::new();
    for record in tensors {
        let name = record["name"]
            .as_str()
            .context("tensor record without a name")?;
        if is_visual(name) {
            ensure!(
                visual.insert(name, record).is_none(),
                "duplicate visual tensor record {name}"
            );
        }
    }
    let inventory: Inventory = config.inventory();
    let trained = trained_tensor_names(config);
    ensure!(
        trained.iter().all(|name| inventory.contains_key(name)),
        "trained replacement outside the visual inventory"
    );
    for (name, shape) in &inventory {
        let record = visual
            .get(name.as_str())
            .with_context(|| format!("manifest is missing the visual tensor {name}"))?;
        ensure!(
            record["file"] == "model.safetensors",
            "{name} must come from model.safetensors"
        );
        ensure!(
            record["dtype"] == "bfloat16",
            "{name} must be bfloat16, not {:?}",
            record["dtype"]
        );
        let recorded: Vec<usize> = record["shape"]
            .as_array()
            .context("tensor record without a shape")?
            .iter()
            .map(|v| {
                v.as_u64()
                    .filter(|n| *n <= usize::MAX as u64)
                    .map(|n| n as usize)
            })
            .collect::<Option<_>>()
            .with_context(|| format!("invalid shape for {name}"))?;
        ensure!(
            recorded == *shape,
            "{name} shape {recorded:?} does not match the configured {shape:?}"
        );
        let expected_component = if name.contains(".merger.") {
            "merger"
        } else if trained.contains(name) {
            "vision-top"
        } else {
            "backbone"
        };
        let expected_source = if trained.contains(name) {
            "checkpoint"
        } else {
            "base"
        };
        ensure!(
            record["component"] == expected_component && record["source"] == expected_source,
            "{name} must be recorded as {expected_component}/{expected_source}, not {}/{}",
            record["component"],
            record["source"]
        );
    }
    for name in visual.keys() {
        ensure!(
            inventory.contains_key(*name),
            "manifest records the unexpected visual tensor {name}"
        );
    }
    Ok(())
}

/// The export's model config must carry the shared Qwen3.5 vision layout with
/// the Valen 2B merger width and a text tower of the same hidden size.
fn export_vision_config(config: &Value) -> Result<VisionConfig> {
    let vision = VisionConfig::from_value(config["vision_config"].clone())?;
    ensure!(
        config["model_type"] == "qwen3_5"
            && config["text_config"]["hidden_size"].as_u64() == Some(vision.out_hidden_size as u64),
        "vision/text config mismatch"
    );
    Ok(vision)
}

/// Immutable mmap storage for the vision tensors of one verified export.
pub struct ValenVisionCheckpoint {
    config: VisionConfig,
    base: TensorStore,
}
impl ValenVisionCheckpoint {
    /// Loads and validates the export on CPU. Keep the files immutable while
    /// mapped. Every listed file is hashed before any tensor is read.
    pub fn load(export: impl AsRef<Path>) -> Result<Self> {
        let dir = fs::canonicalize(export).context("Valen export directory")?;
        let manifest: Value = serde_json::from_slice(&fs::read(inside(&dir, MANIFEST)?)?)
            .context("Valen export manifest")?;
        let files = manifest["files"]
            .as_object()
            .context("Valen export manifest has no file hashes")?;
        for name in ["config.json", "model.safetensors"] {
            ensure!(files.contains_key(name), "export must hash {name}");
        }
        verify_export_files(&dir, files, MANIFEST)?;
        let config: Value = serde_json::from_slice(&fs::read(inside(&dir, "config.json")?)?)
            .context("Valen export config")?;
        let config = export_vision_config(&config)?;
        validate_manifest(&manifest, &config)?;
        let expected = config.inventory();
        let base = TensorStore::load(
            &dir,
            BTreeSet::from(["model.safetensors".into()]),
            &expected,
            safetensors::Dtype::BF16,
            None,
        )?;
        Ok(Self { config, base })
    }
    pub fn vision_config(&self) -> &VisionConfig {
        &self.config
    }
    pub fn base_names(&self) -> impl Iterator<Item = &str> {
        self.base.tensors.keys().map(String::as_str)
    }
    pub fn base_tensor(&self, name: &str) -> Result<TensorView<'_>> {
        self.base.tensor(name)
    }
}

impl VisionModel {
    /// Upload the manifest-verified Valen 2B vision tower: the base layout
    /// with the trained vision-top replacements and no adapter.
    pub fn load_valen(export: impl AsRef<Path>, library: &Path) -> Result<Self> {
        let checkpoint = ValenVisionCheckpoint::load(export)?;
        let base = checkpoint
            .base_names()
            .map(|name| Ok((name.to_owned(), checkpoint.base_tensor(name)?)))
            .collect::<Result<Vec<_>>>()?;
        Self::from_tensors(
            checkpoint.vision_config().clone(),
            base,
            Vec::new(),
            library,
        )
    }
}

/// Verify a manifest's file hashes; every listed file is checked for size and
/// sha256, and no unlisted artifact may sit beside the hashed ones and the
/// manifest itself (`.cache` is ignored).
fn verify_export_files(
    root: &Path,
    files: &serde_json::Map<String, Value>,
    manifest: &str,
) -> Result<()> {
    ensure!(!files.is_empty(), "empty artifact hash manifest");
    for (name, expected) in files {
        ensure!(
            Path::new(name).components().count() == 1
                && matches!(
                    Path::new(name).components().next(),
                    Some(std::path::Component::Normal(_))
                ),
            "invalid artifact name"
        );
        let path = root.join(name);
        ensure!(
            Some(fs::metadata(&path)?.len()) == expected["size"].as_u64(),
            "{} size mismatch",
            path.display()
        );
        ensure!(
            Some(hash_file(&path)?.as_str()) == expected["sha256"].as_str(),
            "{} checksum mismatch",
            path.display()
        );
    }
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_str().context("non-UTF8 artifact")?;
        ensure!(
            name == ".cache" || name == manifest || files.contains_key(name),
            "unlisted artifact: {name}"
        );
    }
    Ok(())
}

fn hash_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

#[cfg(test)]
#[path = "../../../../../../tests/valen/valen_manifest.rs"]
mod valen_manifest_tests;
