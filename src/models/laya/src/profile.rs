//! The Laya bundle dialect and its one registered family profile.
//!
//! The two-file recipe bundle (`rl_agent_config.json` +
//! `encoder/config.json`) is a Laya checkpoint format, so it is parsed
//! here rather than in the family crate. The dialect reads exactly the
//! keys the native worker has always admitted: dimensions, explicit
//! `layer_types` schedule, `local_attention`, the two-entry
//! `rope_parameters`, `norm_eps`, `vocab_size`. HF-side keys the
//! worker does not consume (`max_position_embeddings`, `head_dim`) are
//! recorded as unrecognized instead of gated — the bundle's rotary
//! tables exercise 512 positions and the head dim is hidden/heads, as
//! before the family split.

use anyhow::{Context, Result, ensure};
use omni_modernbert::gate::{BackendLimits, RegisteredProfile};
use omni_modernbert::structure::{
    Activation, AttentionKind, FieldOrigin, ModernBertStructure, RopeSpec,
};
use serde_json::Value as Json;
use std::collections::BTreeMap;
use std::path::Path;

/// Reads the two-file Laya bundle layout: the encoder half of
/// `rl_agent_config.json` + `encoder/config.json`. Agent fields are
/// task bundle and stay in the Laya crate; this adapter reads
/// `encoder/config.json` only.
pub struct LayaBundleAdapter;

impl LayaBundleAdapter {
    pub fn read(dir: &Path) -> Result<(ModernBertStructure, SourceMeta)> {
        let path = dir.join("encoder/config.json");
        let bytes = std::fs::read(&path).with_context(|| format!("{}", path.display()))?;
        let config: Json =
            serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
        parse_laya_bundle(&config, "encoder/config.json")
    }
}

/// Marker and file evidence for one parsed source, mirroring the family
/// adapter's metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceMeta {
    pub config_files: Vec<String>,
    pub model_type: String,
    /// Top-level keys the dialect does not consume. Recorded, never a
    /// rejection.
    pub unrecognized_keys: Vec<String>,
}

fn uint(config: &Json, key: &str) -> Result<usize> {
    config[key]
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .with_context(|| format!("{key} is missing or not an unsigned integer"))
}

fn float(config: &Json, key: &str) -> Result<f64> {
    config[key]
        .as_f64()
        .with_context(|| format!("{key} is missing or not a number"))
}

fn require_modernbert(config: &Json) -> Result<()> {
    ensure!(
        config["model_type"].as_str() == Some("modernbert"),
        "model_type {:?} is not modernbert",
        config["model_type"]
    );
    ensure!(
        config["hidden_activation"].as_str() == Some("gelu"),
        "hidden_activation {:?} is not gelu",
        config["hidden_activation"]
    );
    Ok(())
}

fn unrecognized(config: &Json, known: &[&str]) -> Vec<String> {
    config
        .as_object()
        .map(|object| {
            object
                .keys()
                .filter(|k| !known.contains(&k.as_str()))
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

fn parse_laya_bundle(config: &Json, source: &str) -> Result<(ModernBertStructure, SourceMeta)> {
    require_modernbert(config)?;
    for key in ["attention_bias", "mlp_bias", "norm_bias"] {
        ensure!(
            config[key] == Json::Bool(false),
            "{key} must be stated and false"
        );
    }
    let layer_types = config["layer_types"]
        .as_array()
        .context("layer_types is missing or not an array")?;
    let schedule = layer_types
        .iter()
        .map(|t| {
            t.as_str()
                .map(AttentionKind::parse)
                .unwrap_or_else(|| Err(anyhow::anyhow!("layer_types entry {t:?} is not a string")))
        })
        .collect::<Result<Vec<_>>>()?;
    let rope = |kind: &str, provenance: &mut BTreeMap<String, FieldOrigin>| {
        let entry = &config["rope_parameters"][kind];
        ensure!(
            entry["rope_type"].as_str() == Some("default"),
            "rope_parameters.{kind} rope_type {:?} is not default",
            entry["rope_type"]
        );
        provenance.insert(
            format!("rope_{kind}"),
            FieldOrigin::Source(format!("{source}#rope_parameters.{kind}")),
        );
        Ok(RopeSpec {
            theta: entry["rope_theta"].as_f64().with_context(|| {
                format!("rope_parameters.{kind}.rope_theta is missing or not a number")
            })?,
            scaling: None,
        })
    };
    let mut provenance = BTreeMap::new();
    let rope_full = rope("full_attention", &mut provenance)?;
    let rope_sliding = rope("sliding_attention", &mut provenance)?;
    let hidden = uint(config, "hidden_size")?;
    let heads = uint(config, "num_attention_heads")?;
    provenance.insert(
        "max_positions".into(),
        FieldOrigin::Default("Laya bundle rotary tables exercise 512 positions"),
    );
    provenance.insert(
        "head_dim".into(),
        FieldOrigin::Derived("hidden_size / num_attention_heads".into()),
    );
    for key in [
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "norm_eps",
        "num_attention_heads",
        "schedule",
        "local_window",
        "vocab_size",
    ] {
        let json_key = match key {
            "schedule" => "layer_types",
            "local_window" => "local_attention",
            _ => key,
        };
        provenance.insert(
            key.into(),
            FieldOrigin::Source(format!("{source}#{json_key}")),
        );
    }
    provenance.insert(
        "type_embedding_rows".into(),
        FieldOrigin::Default("the Laya bundle checkpoint carries a [3, hidden] type embedding"),
    );
    let structure = ModernBertStructure {
        hidden_size: hidden,
        intermediate_size: uint(config, "intermediate_size")?,
        num_hidden_layers: uint(config, "num_hidden_layers")?,
        activation: Activation::Gelu,
        norm_eps: float(config, "norm_eps")? as f32,
        num_attention_heads: heads,
        head_dim: hidden / heads,
        schedule,
        local_window: uint(config, "local_attention")?,
        rope_full,
        rope_sliding,
        vocab_size: uint(config, "vocab_size")?,
        type_embedding_rows: Some(3),
        max_positions: 512,
        provenance,
    };
    structure.validate()?;
    let meta = SourceMeta {
        config_files: vec![source.into()],
        model_type: "modernbert".into(),
        unrecognized_keys: unrecognized(
            config,
            &[
                "model_type",
                "hidden_activation",
                "attention_bias",
                "mlp_bias",
                "norm_bias",
                "hidden_size",
                "intermediate_size",
                "num_attention_heads",
                "num_hidden_layers",
                "vocab_size",
                "norm_eps",
                "local_attention",
                "layer_types",
                "rope_parameters",
            ],
        ),
    };
    Ok((structure, meta))
}

/// The Laya 0.3.20 encoder as shipped, ABI 1.
pub fn laya_0_3_20() -> RegisteredProfile {
    RegisteredProfile {
        name: "laya-0.3.20",
        structure: laya_structure(),
        limits: BackendLimits {
            hidden_sizes: vec![1024],
            batch_max: 16,
            len_min: 16,
            len_max: 512,
            len_step: 16,
            yarn_scaling: false,
        },
        backend: "laya_cuda",
        abi_version: 1,
    }
}

fn laya_structure() -> ModernBertStructure {
    let mut provenance = BTreeMap::new();
    for field in [
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "norm_eps",
        "num_attention_heads",
        "head_dim",
        "schedule",
        "local_window",
        "rope_full",
        "rope_sliding",
        "vocab_size",
        "type_embedding_rows",
        "max_positions",
    ] {
        provenance.insert(field.to_string(), origin_of_profile());
    }
    ModernBertStructure {
        hidden_size: 1024,
        intermediate_size: 2624,
        num_hidden_layers: 28,
        activation: Activation::Gelu,
        norm_eps: 1e-5,
        num_attention_heads: 16,
        head_dim: 64,
        schedule: (0..28)
            .map(|i| {
                if i % 3 == 0 {
                    AttentionKind::Full
                } else {
                    AttentionKind::Sliding
                }
            })
            .collect(),
        local_window: 128,
        rope_full: RopeSpec {
            theta: 160000.0,
            scaling: None,
        },
        rope_sliding: RopeSpec {
            theta: 10000.0,
            scaling: None,
        },
        vocab_size: 50368,
        type_embedding_rows: Some(3),
        max_positions: 512,
        provenance,
    }
}

fn origin_of_profile() -> FieldOrigin {
    FieldOrigin::Source("registered profile laya-0.3.20".into())
}
