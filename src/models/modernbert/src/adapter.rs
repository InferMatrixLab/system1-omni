use crate::structure::{
    Activation, AttentionKind, FieldOrigin, ModernBertStructure, RopeSpec, YarnScaling,
};
use anyhow::{Context, Result, ensure};
use serde_json::Value as Json;
use std::collections::BTreeMap;

/// Marker and file evidence for one parsed source.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceMeta {
    pub config_files: Vec<String>,
    pub model_type: String,
    /// Top-level keys the adapters do not consume. Recorded, never a
    /// rejection: a checkpoint may carry recipe or training fields beside
    /// the structure.
    pub unrecognized_keys: Vec<String>,
}

/// Reads any single-object ModernBERT config: a stock HF
/// `config.json` or a member object such as the Vela bundle's
/// `encoder_config`. Dialect bundle layouts (for example the Laya
/// two-file recipe) are parsed by the model crates that ship them.
pub struct ModernBertConfigAdapter;

impl ModernBertConfigAdapter {
    pub fn from_value(
        config: &Json,
        source_name: &str,
    ) -> Result<(ModernBertStructure, SourceMeta)> {
        parse_hf_config(config, source_name)
    }
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

fn derive_head_dim(
    hidden: usize,
    heads: usize,
    config: &Json,
    provenance: &mut BTreeMap<String, FieldOrigin>,
) -> Result<usize> {
    let head_dim = match config.get("head_dim") {
        Some(value) => value
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .context("head_dim is not an unsigned integer")?,
        None => {
            provenance.insert(
                "head_dim".into(),
                FieldOrigin::Derived("hidden_size / num_attention_heads".into()),
            );
            hidden / heads
        }
    };
    Ok(head_dim)
}

fn parse_hf_config(config: &Json, source: &str) -> Result<(ModernBertStructure, SourceMeta)> {
    require_modernbert(config)?;
    let mut provenance = BTreeMap::new();
    for key in ["attention_bias", "mlp_bias", "norm_bias"] {
        match config.get(key) {
            None => {
                provenance.insert(
                    key.into(),
                    FieldOrigin::Default("HF ModernBERT states no bias by default"),
                );
            }
            Some(value) => ensure!(
                value == &Json::Bool(false),
                "{key} must be false, got {value}"
            ),
        }
    }
    let schedule = if let Some(layer_types) = config["layer_types"].as_array() {
        let parsed = layer_types
            .iter()
            .map(|t| {
                t.as_str().map(AttentionKind::parse).unwrap_or_else(|| {
                    Err(anyhow::anyhow!("layer_types entry {t:?} is not a string"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        provenance.insert(
            "schedule".into(),
            FieldOrigin::Source(format!("{source}#layer_types")),
        );
        parsed
    } else {
        let every = uint(config, "global_attn_every_n_layers")
            .context("neither layer_types nor global_attn_every_n_layers is present")?;
        provenance.insert(
            "schedule".into(),
            FieldOrigin::Derived(format!(
                "global_attn_every_n_layers={every}, full at layer index % {every} == 0"
            )),
        );
        (0..uint(config, "num_hidden_layers")?)
            .map(|i| {
                if i % every == 0 {
                    AttentionKind::Full
                } else {
                    AttentionKind::Sliding
                }
            })
            .collect()
    };
    let theta =
        |field: &str, json_key: &str, provenance: &mut BTreeMap<String, FieldOrigin>| match config
            .get(json_key)
        {
            None => {
                provenance.insert(
                    field.into(),
                    FieldOrigin::Default("HF ModernBERT rope_theta default"),
                );
                Ok(10000.0)
            }
            Some(_) => {
                provenance.insert(
                    field.into(),
                    FieldOrigin::Source(format!("{source}#{json_key}")),
                );
                float(config, json_key)
            }
        };
    let theta_full = theta("rope_full", "global_rope_theta", &mut provenance)?;
    let theta_sliding = theta("rope_sliding", "local_rope_theta", &mut provenance)?;
    let scaling = match config.get("rope_scaling") {
        None => None,
        Some(entry) => {
            ensure!(
                entry["rope_type"].as_str() == Some("yarn"),
                "rope_scaling type {:?} is not yarn",
                entry["rope_type"]
            );
            ensure!(
                entry
                    .get("truncate")
                    .map(|v| v == &Json::Bool(true))
                    .unwrap_or(true),
                "rope_scaling truncate must be true"
            );
            let yarn = YarnScaling {
                factor: float(entry, "factor")?,
                original_max_positions: uint(entry, "original_max_position_embeddings")?,
                beta_fast: match entry.get("beta_fast") {
                    None => 32.0,
                    Some(_) => float(entry, "beta_fast")?,
                },
                beta_slow: match entry.get("beta_slow") {
                    None => 1.0,
                    Some(_) => float(entry, "beta_slow")?,
                },
            };
            provenance.insert(
                "rope_full.scaling".into(),
                FieldOrigin::Source(format!("{source}#rope_scaling")),
            );
            provenance.insert(
                "rope_sliding.scaling".into(),
                FieldOrigin::Source(format!("{source}#rope_scaling")),
            );
            Some(yarn)
        }
    };
    if let Some(position_type) = config
        .get("position_embedding_type")
        .and_then(|v| v.as_str())
    {
        ensure!(
            position_type == "sans_pos" || position_type == "rope",
            "position_embedding_type {position_type:?} is not rotary-only"
        );
    }
    let eps_key = if config.get("norm_eps").is_some() {
        "norm_eps"
    } else if config.get("layer_norm_eps").is_some() {
        "layer_norm_eps"
    } else {
        anyhow::bail!("neither norm_eps nor layer_norm_eps is present")
    };
    let norm_eps = float(config, eps_key)? as f32;
    let hidden = uint(config, "hidden_size")?;
    let heads = uint(config, "num_attention_heads")?;
    let head_dim = derive_head_dim(hidden, heads, config, &mut provenance)?;
    let type_embedding_rows = match config.get("type_vocab_size") {
        None => None,
        Some(_) => {
            let rows = uint(config, "type_vocab_size")?;
            provenance.insert(
                "type_embedding_rows".into(),
                FieldOrigin::Source(format!("{source}#type_vocab_size")),
            );
            (rows > 0).then_some(rows)
        }
    };
    for key in [
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "vocab_size",
        "local_window",
    ] {
        let json_key = match key {
            "local_window" => "local_attention",
            _ => key,
        };
        provenance.insert(
            key.into(),
            FieldOrigin::Source(format!("{source}#{json_key}")),
        );
    }
    provenance.insert(
        "norm_eps".into(),
        FieldOrigin::Source(format!("{source}#{eps_key}")),
    );
    provenance.insert(
        "max_positions".into(),
        FieldOrigin::Source(format!("{source}#max_position_embeddings")),
    );
    let structure = ModernBertStructure {
        hidden_size: hidden,
        intermediate_size: uint(config, "intermediate_size")?,
        num_hidden_layers: uint(config, "num_hidden_layers")?,
        activation: Activation::Gelu,
        norm_eps,
        num_attention_heads: heads,
        head_dim,
        schedule,
        local_window: uint(config, "local_attention")?,
        rope_full: RopeSpec {
            theta: theta_full,
            scaling,
        },
        rope_sliding: RopeSpec {
            theta: theta_sliding,
            scaling,
        },
        vocab_size: uint(config, "vocab_size")?,
        type_embedding_rows,
        max_positions: uint(config, "max_position_embeddings")?,
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
                "layer_norm_eps",
                "local_attention",
                "layer_types",
                "global_attn_every_n_layers",
                "global_rope_theta",
                "local_rope_theta",
                "rope_theta",
                "rope_scaling",
                "max_position_embeddings",
                "head_dim",
                "position_embedding_type",
                "type_vocab_size",
            ],
        ),
    };
    Ok((structure, meta))
}
