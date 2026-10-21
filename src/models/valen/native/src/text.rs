//! The unmerged text-path LoRA application for the Valen five-file export
//! (spec 03, ticket N2).
//!
//! The export (`valen-checkpoint-export/2`, `recipe/valen/export_checkpoint.py`)
//! serves the FP32 language adapters separately from the BF16 base; this module
//! validates that export policy before any CUDA loading and then reuses the
//! shared `#143` machinery (`Model::load_with_lora` plus the reference text
//! numerics: framework-compatible norm/GEMM/attention, FP32 LoRA branches on
//! the activations, FP32 chunked GDN with the pinned Torch cumulative-decay
//! association, per-prompt state reset, prefix continuation rejected).
//!
//! The manifest's application formula is the semantic contract: per targeted
//! linear, `y = bf16(fp32(bf16(x @ W.t())) + scaling * ((x.float() @ A.t()) @
//! B.t()))` — the base GEMM runs in BF16, the delta is computed in FP32 and
//! added to the upcast base output, and the sum is rounded to BF16 once. The
//! Python suite (`tests/valen/test_export.py`) pins it against peft 0.18.1;
//! `tests/qwen3_5/lora_valen.rs` ports those pins onto the shared loader.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use omni_qwen3_5_native::model::{Config, Model};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// The spec-02 export manifest name and format.
const MANIFEST: &str = "valen_export.json";
const FORMAT: &str = "valen-checkpoint-export/2";
/// The pinned adapter profile: peft rank 32, alpha 64 (`scaling = alpha/rank`).
const SCALE: f32 = 2.0;
const RANK: u64 = 32;
const ALPHA: u64 = 64;
/// `PEFT_LANGUAGE_PREFIX` in the export tool; the adapters live under the
/// wrapped `backbone.language_model` with peft's named `default` adapter.
const PEFT_PREFIX: &str = "backbone.language_model.base_model.model.";

/// The validated export facts the text path consumes.
#[derive(Clone, Debug, PartialEq)]
pub struct ExportPolicy {
    /// The manifest-relative adapter file name.
    pub adapter_file: PathBuf,
    /// The pinned `alpha/rank` scaling, applied on the FP32 delta.
    pub scale: f32,
    /// Declared LoRA targets (must equal the configuration-derived census).
    pub targets: usize,
    /// Declared LoRA tensors (two per target).
    pub keys: usize,
}

/// Validate the export policy and the adapter bytes before CUDA loading,
/// mirroring the Cua-S1 `cua-s1-text-unmerged/1` gate: the manifest format,
/// the pinned LoRA profile, the hashed adapter file, and the complete
/// configuration-derived inventory.
pub fn validate_export(dir: &Path) -> Result<ExportPolicy> {
    let manifest: Value = serde_json::from_slice(
        &std::fs::read(dir.join(MANIFEST))
            .with_context(|| format!("{} is missing", dir.join(MANIFEST).display()))?,
    )
    .context("invalid Valen export manifest")?;
    ensure!(
        manifest["format"].as_str() == Some(FORMAT),
        "unsupported export format {:?}; expected {FORMAT} (see recipe/valen/native.md)",
        manifest["format"]
    );
    let lora = &manifest["lora"];
    let policy = || -> Result<ExportPolicy> {
        let scale = lora["scaling"].as_f64().context("lora.scaling")?;
        let alpha = lora["alpha"].as_u64().context("lora.alpha")?;
        let rank = lora["rank"].as_u64().context("lora.rank")?;
        ensure!(scale.is_finite() && scale > 0., "LoRA profile: invalid scaling");
        ensure!(
            (alpha as f64 / rank as f64 - scale).abs() < f64::EPSILON,
            "LoRA profile: scaling must be alpha/rank"
        );
        ensure!(
            rank == RANK && alpha == ALPHA,
            "LoRA profile: the pinned adapter is rank {RANK} / alpha {ALPHA}"
        );
        ensure!(scale as f32 == SCALE, "LoRA profile: scaling must be {SCALE}");
        ensure!(
            lora["dtype"].as_str() == Some("float32"),
            "LoRA profile: adapters must be FP32"
        );
        let adapter = lora["file"].as_str().context("lora.file")?;
        ensure!(
            adapter == "lora.safetensors",
            "LoRA profile: the export serves its adapters as lora.safetensors"
        );
        ensure!(
            Path::new(adapter).components().count() == 1,
            "LoRA profile: adapter file must be a plain name"
        );
        let targets = lora["targets"].as_u64().context("lora.targets")? as usize;
        let keys = lora["keys"].as_u64().context("lora.keys")? as usize;
        ensure!(
            keys == targets.saturating_mul(2),
            "LoRA profile: keys must be two per target"
        );
        Ok(ExportPolicy {
            adapter_file: PathBuf::from(adapter),
            scale: scale as f32,
            targets,
            keys,
        })
    }()?;
    let files = manifest["files"]
        .as_object()
        .context("export manifest has no file hashes")?;
    for name in [
        String::from("config.json"),
        policy.adapter_file.to_string_lossy().into_owned(),
    ] {
        let expected = files
            .get(name.as_str())
            .with_context(|| format!("export must hash {name}"))?;
        let path = dir.join(&name);
        ensure!(
            Some(std::fs::metadata(&path).with_context(|| format!("{name}"))?.len())
                == expected["size"].as_u64(),
            "{name} size mismatch"
        );
        ensure!(
            Some(hash_file(&path)?.as_str()) == expected["sha256"].as_str(),
            "{name} checksum mismatch"
        );
    }
    // The pinned full inventory: the configuration's layer mix derives the
    // exact target census, and the adapter must carry every declared pair
    // (the shared loader also accepts partial adapters; the export does not).
    let cfg = Config::load(dir)?;
    let inventory = inventory(&cfg);
    ensure!(
        inventory.len() == policy.targets,
        "inventory: the configuration has {} LoRA targets, the manifest declares {}",
        inventory.len(),
        policy.targets
    );
    let names = adapter_tensor_names(&dir.join(&policy.adapter_file))?;
    ensure!(
        names.len() == policy.keys,
        "inventory: the adapter carries {} tensors, the manifest declares {}",
        names.len(),
        policy.keys
    );
    for name in &names {
        ensure!(
            inventory.contains(name),
            "inventory: {name} is not a configured LoRA target"
        );
    }
    let mut counts = std::collections::BTreeMap::new();
    for name in &names {
        *counts.entry(name.as_str()).or_insert(0usize) += 1;
    }
    ensure!(
        counts.values().all(|&count| count == 2),
        "inventory: every target needs exactly one A and one B tensor"
    );
    ensure!(
        counts.len() == inventory.len(),
        "inventory: the adapter must carry every configured target exactly once"
    );
    Ok(policy)
}

/// The configuration-derived target census: per GDN layer the five
/// `linear_attn` projections, per full-attention layer the four `self_attn`
/// projections, and every layer's three MLP projections.
fn inventory(cfg: &Config) -> Vec<String> {
    let mut targets = Vec::new();
    for (layer, &full) in cfg.full_attention.iter().enumerate() {
        if full {
            for name in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"] {
                targets.push(format!("layers.{layer}.{name}"));
            }
        } else {
            for name in [
                "linear_attn.in_proj_qkv",
                "linear_attn.in_proj_z",
                "linear_attn.in_proj_b",
                "linear_attn.in_proj_a",
                "linear_attn.out_proj",
            ] {
                targets.push(format!("layers.{layer}.{name}"));
            }
        }
        for name in ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"] {
            targets.push(format!("layers.{layer}.{name}"));
        }
    }
    targets
}

/// The target names an adapter file carries, without loading it onto a device.
fn adapter_tensor_names(path: &Path) -> Result<Vec<String>> {
    let bytes = std::fs::read(path)?;
    let header_len = usize::try_from(u64::from_le_bytes(
        bytes.get(..8).context("adapter header length")?.try_into()?,
    ))?;
    let header: Value =
        serde_json::from_slice(bytes.get(8..8 + header_len).context("adapter header")?)
            .context("invalid adapter header")?;
    let mut names = Vec::new();
    for name in header.as_object().context("adapter header")?.keys() {
        let Some(native) = name.strip_prefix(PEFT_PREFIX) else {
            continue;
        };
        let target = native
            .strip_suffix(".lora_A.default.weight")
            .or_else(|| native.strip_suffix(".lora_B.default.weight"));
        names.push(target.context("unsupported adapter tensor name")?.to_string());
    }
    ensure!(!names.is_empty(), "adapter carries no language tensors");
    Ok(names)
}

fn hash_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        use std::io::Read;
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

/// The Valen unmerged text model: the BF16 backbone with the FP32 adapters
/// applied at every projection, under the reference text numeric profile.
pub struct TextModel {
    model: Model,
}

impl TextModel {
    /// Validate the export policy, then load the base weights plus the
    /// unmerged FP32 adapters and select the reference text numerics.
    /// The optional CUDA symbols (framework language operations, FP32 GDN)
    /// are resolved here and fail before the first forward if absent.
    pub fn load(export: &Path, library: &Path) -> Result<Self> {
        let policy = validate_export(export)?;
        let adapter = export.join(&policy.adapter_file);
        let mut model = Model::load_with_lora(export, library, &adapter, policy.scale)?;
        model.enable_valen_text_numerics()?;
        Ok(Self { model })
    }

    /// The final-norm hidden state at the last position, as FP32 (BF16 values
    /// upcast); per-prompt GDN state resets between calls.
    pub fn forward(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        self.model.forward(ids)
    }

    /// Synchronize the underlying stream before external reuse.
    pub fn synchronize(&self) -> Result<()> {
        self.model.synchronize()
    }

    pub fn config(&self) -> &Config {
        &self.model.cfg
    }
}
