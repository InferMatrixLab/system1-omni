//! The Qwen3.5 hybrid family layer: the source adapter that reads a
//! checkpoint config, the structure it produces, the frozen ABI-8
//! capability set the CUDA kernels implement, and the prefix-state
//! layout derived from a structure. Parsing and capability live apart
//! so both are checkable without a device or a checkpoint.

use anyhow::{Context, Result, bail, ensure};
use serde_json::Value as Json;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

pub const SCHEMA_VERSION: usize = 1;
pub const FAMILY: &str = "qwen3_5_hybrid";
pub const BACKEND: &str = "qwen3_5_cuda";
pub const ABI_VERSION: usize = 8;

const BF16: usize = 2;
const F32: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    FullAttention,
    GatedDeltaNet,
}

/// The hybrid structure: family semantics and math geometry with
/// per-field provenance. The multimodal behavior around it (image
/// token, M-RoPE positions) stays opaque and is carried, not
/// parameterized.
#[derive(Debug, Clone, PartialEq)]
pub struct QwenHybridStructure {
    pub hidden: usize,
    pub intermediate: usize,
    pub eps: f32,
    /// Ordered per-layer mixer kinds; index 0 is the first layer.
    pub layer_kinds: Vec<LayerKind>,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Half the number of rotary dims (rotate_half pairs i with i + half).
    pub rotary_half: usize,
    pub rope_theta: f64,
    pub mrope_section: [usize; 3],
    pub max_positions: usize,
    pub image_token_id: Option<u32>,
    pub lin_k_heads: usize,
    pub lin_v_heads: usize,
    pub lin_k_dim: usize,
    pub lin_v_dim: usize,
    /// Source-declared family behavior the kernels pin; the gate
    /// rejects anything but the frozen combination.
    pub hidden_act: Option<String>,
    pub attn_output_gate: bool,
    pub attention_bias: bool,
    pub rope_type: String,
    pub conv_kernel_dim: usize,
    pub mrope_interleaved: bool,
    pub provenance: BTreeMap<String, String>,
}

impl QwenHybridStructure {
    pub fn full_attention_layers(&self) -> usize {
        self.layer_kinds
            .iter()
            .filter(|k| **k == LayerKind::FullAttention)
            .count()
    }

    pub fn gdn_layers(&self) -> usize {
        self.layer_kinds.len() - self.full_attention_layers()
    }

    pub fn key_dim(&self) -> usize {
        self.lin_k_heads * self.lin_k_dim
    }

    pub fn value_dim(&self) -> usize {
        self.lin_v_heads * self.lin_v_dim
    }

    /// Width of the fused Gated DeltaNet input projection (conv q/k/v,
    /// gate/value and beta rows stacked).
    pub fn gdn_in_width(&self) -> usize {
        2 * self.key_dim() + self.value_dim() + self.value_dim() + 2 * self.lin_v_heads
    }

    fn canonical_json(&self) -> Json {
        let num = |v: usize| Json::String(v.to_string());
        let mut object = serde_json::Map::new();
        object.insert("schema_version".into(), num(SCHEMA_VERSION));
        object.insert("family".into(), Json::String(FAMILY.into()));
        object.insert("hidden".into(), num(self.hidden));
        object.insert("intermediate".into(), num(self.intermediate));
        object.insert("eps".into(), Json::String(self.eps.to_string()));
        object.insert(
            "layer_kinds".into(),
            Json::Array(
                self.layer_kinds
                    .iter()
                    .map(|k| {
                        Json::String(
                            match k {
                                LayerKind::FullAttention => "full_attention",
                                LayerKind::GatedDeltaNet => "linear_attention",
                            }
                            .into(),
                        )
                    })
                    .collect(),
            ),
        );
        object.insert("heads".into(), num(self.heads));
        object.insert("kv_heads".into(), num(self.kv_heads));
        object.insert("head_dim".into(), num(self.head_dim));
        object.insert("rotary_half".into(), num(self.rotary_half));
        object.insert(
            "rope_theta".into(),
            Json::String(self.rope_theta.to_string()),
        );
        object.insert(
            "mrope_section".into(),
            Json::Array(
                self.mrope_section
                    .iter()
                    .map(|s| Json::String(s.to_string()))
                    .collect(),
            ),
        );
        object.insert("max_positions".into(), num(self.max_positions));
        object.insert(
            "image_token_id".into(),
            self.image_token_id
                .map(|id| Json::String(id.to_string()))
                .unwrap_or(Json::Null),
        );
        object.insert("lin_k_heads".into(), num(self.lin_k_heads));
        object.insert("lin_v_heads".into(), num(self.lin_v_heads));
        object.insert("lin_k_dim".into(), num(self.lin_k_dim));
        object.insert("lin_v_dim".into(), num(self.lin_v_dim));
        object.insert(
            "hidden_act".into(),
            self.hidden_act
                .clone()
                .map(Json::String)
                .unwrap_or(Json::Null),
        );
        object.insert(
            "attn_output_gate".into(),
            Json::String(self.attn_output_gate.to_string()),
        );
        object.insert(
            "attention_bias".into(),
            Json::String(self.attention_bias.to_string()),
        );
        object.insert("rope_type".into(), Json::String(self.rope_type.clone()));
        object.insert("conv_kernel_dim".into(), num(self.conv_kernel_dim));
        object.insert(
            "mrope_interleaved".into(),
            Json::String(self.mrope_interleaved.to_string()),
        );
        Json::Object(object)
    }
}

/// The source adapter: reads `config.json` from a checkpoint directory,
/// preferring `text_config`, exactly as the loader always has. No
/// capability judgment happens here.
pub fn read_structure(dir: &Path) -> Result<QwenHybridStructure> {
    let path = dir.join("config.json");
    let root: Json = serde_json::from_str(
        &std::fs::read_to_string(&path).with_context(|| format!("{}", path.display()))?,
    )?;
    let c = root.get("text_config").unwrap_or(&root);
    let int = |k: &str| {
        c[k].as_u64()
            .map(|v| v as usize)
            .with_context(|| format!("config.json: `{k}` is missing"))
    };
    let rope = &c["rope_parameters"];
    let partial = rope["partial_rotary_factor"]
        .as_f64()
        .or(c["partial_rotary_factor"].as_f64())
        .unwrap_or(1.0);
    let head_dim = int("head_dim")?;
    let layer_kinds = c["layer_types"]
        .as_array()
        .context("config.json: `layer_types` is missing")?
        .iter()
        .map(|t| match t.as_str() {
            Some("full_attention") => Ok(LayerKind::FullAttention),
            Some("linear_attention") => Ok(LayerKind::GatedDeltaNet),
            other => bail!("unknown layer type {other:?}"),
        })
        .collect::<Result<Vec<_>>>()?;
    let sections = rope
        .get("mrope_section")
        .cloned()
        .unwrap_or(serde_json::json!([11, 11, 10]));
    let sections = sections
        .as_array()
        .context("mrope_section must be an array")?
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .context("mrope_section must contain integers")
        })
        .collect::<Result<Vec<_>>>()?;
    let mrope_section: [usize; 3] = sections
        .try_into()
        .map_err(|_| anyhow::anyhow!("mrope_section must have three entries"))?;
    let image_token_id = root
        .get("image_token_id")
        .map(|v| {
            v.as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .context("image_token_id must be a u32")
        })
        .transpose()?;
    let num_hidden_layers = int("num_hidden_layers")?;
    let mut provenance = BTreeMap::new();
    for field in [
        "hidden",
        "intermediate",
        "eps",
        "layer_kinds",
        "heads",
        "kv_heads",
        "head_dim",
        "rotary_half",
        "rope_theta",
        "mrope_section",
        "max_positions",
        "lin_k_heads",
        "lin_v_heads",
        "lin_k_dim",
        "lin_v_dim",
    ] {
        let json_key = match field {
            "hidden" => "hidden_size",
            "intermediate" => "intermediate_size",
            "eps" => "rms_norm_eps",
            "layer_kinds" => "layer_types",
            "heads" => "num_attention_heads",
            "kv_heads" => "num_key_value_heads",
            "max_positions" => "max_position_embeddings",
            "lin_k_heads" => "linear_num_key_heads",
            "lin_v_heads" => "linear_num_value_heads",
            "lin_k_dim" => "linear_key_head_dim",
            "lin_v_dim" => "linear_value_head_dim",
            "rotary_half" => "rope_parameters.partial_rotary_factor",
            _ => field,
        };
        provenance.insert(
            field.to_string(),
            format!(
                "config.json#{}{json_key}",
                if root.get("text_config").is_some() {
                    "text_config."
                } else {
                    ""
                }
            ),
        );
    }
    let hidden_act = c["hidden_act"].as_str().map(str::to_string);
    let attn_output_gate = c["attn_output_gate"].as_bool().unwrap_or(true);
    let attention_bias = c["attention_bias"].as_bool().unwrap_or(false);
    let rope_type = rope["rope_type"].as_str().unwrap_or("default").to_string();
    let mrope_interleaved = rope["mrope_interleaved"].as_bool().unwrap_or(true);
    // The layer-count check keeps its base position: after every field
    // extraction, before the gate's other checks. linear_conv_kernel_dim
    // is read after it, as in the base loader where the gate extracted
    // it at its own step.
    ensure!(
        layer_kinds.len() == num_hidden_layers,
        "layer_types does not match num_hidden_layers"
    );
    let conv_kernel_dim = int("linear_conv_kernel_dim")?;
    Ok(QwenHybridStructure {
        hidden_act,
        attn_output_gate,
        attention_bias,
        rope_type,
        conv_kernel_dim,
        mrope_interleaved,
        hidden: int("hidden_size")?,
        intermediate: int("intermediate_size")?,
        eps: c["rms_norm_eps"].as_f64().context("rms_norm_eps")? as f32,
        heads: int("num_attention_heads")?,
        kv_heads: int("num_key_value_heads")?,
        head_dim,
        rotary_half: (head_dim as f64 * partial) as usize / 2,
        rope_theta: rope["rope_theta"]
            .as_f64()
            .or(c["rope_theta"].as_f64())
            .context("rope_theta")?,
        mrope_section,
        max_positions: int("max_position_embeddings")?,
        image_token_id,
        lin_k_heads: int("linear_num_key_heads")?,
        lin_v_heads: int("linear_num_value_heads")?,
        lin_k_dim: int("linear_key_head_dim")?,
        lin_v_dim: int("linear_value_head_dim")?,
        layer_kinds,
        provenance,
    })
}

/// The frozen ABI-8 kernel capability set: exactly what the CUDA
/// kernels implement, no more. Widening anything here is a kernel/ABI
/// decision, not a loader change. Every rejection reports the actual
/// value.
pub fn check_abi(s: &QwenHybridStructure) -> Result<()> {
    ensure!(
        s.hidden_act.as_deref() == Some("silu"),
        "hidden_act is not silu"
    );
    ensure!(s.attn_output_gate, "attention without the output gate");
    ensure!(!s.attention_bias, "attention with bias");
    ensure!(s.rope_type == "default", "rope type {:?}", s.rope_type);
    ensure!(s.conv_kernel_dim == 4, "conv kernel is not 4");
    ensure!(
        s.head_dim == 256 && s.lin_k_dim == 128 && s.lin_v_dim == 128,
        "head dims {} / {} / {}",
        s.head_dim,
        s.lin_k_dim,
        s.lin_v_dim
    );
    ensure!(s.rotary_half == 32, "{} rotary dims", 2 * s.rotary_half);
    ensure!(
        s.rope_theta.is_finite() && s.rope_theta > 0.0,
        "invalid rope_theta"
    );
    ensure!(s.mrope_interleaved, "non-interleaved mrope is unsupported");
    ensure!(
        s.mrope_section
            .iter()
            .try_fold(0usize, |sum, &x| sum.checked_add(x))
            == Some(s.rotary_half),
        "mrope sections must sum to rotary_half"
    );
    ensure!(
        s.max_positions > 0 && s.max_positions <= (1 << 24),
        "unsupported max_position_embeddings"
    );
    ensure!(
        s.kv_heads > 0 && s.heads.is_multiple_of(s.kv_heads),
        "attention heads"
    );
    ensure!(
        s.lin_k_heads > 0 && s.lin_v_heads.is_multiple_of(s.lin_k_heads),
        "linear attention heads"
    );
    ensure!(s.hidden.is_multiple_of(8), "hidden size");
    Ok(())
}

/// Per-layer prefix-state byte layout for one token length, derived
/// from the structure: per full-attention layer the post-prep K and
/// raw V rows, per Gated DeltaNet layer the float32 recurrent state
/// and the three pre-conv projection columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixLayout {
    pub len: usize,
    pub full_layers: usize,
    pub gdn_layers: usize,
    /// One full-attention layer's K-plus-V bytes (`2 * len * kv row`).
    pub full_kv_bytes: usize,
    /// One GDN layer's float32 `[H, K, V]` state bytes.
    pub gdn_state_bytes: usize,
    /// One GDN layer's three conv-tail columns.
    pub conv_tail_bytes: usize,
}

impl PrefixLayout {
    pub fn of(s: &QwenHybridStructure, len: usize) -> Self {
        let kvrow = s.kv_heads * s.head_dim * BF16;
        let state_floats = s.lin_v_heads * s.lin_k_dim * s.lin_v_dim;
        Self {
            len,
            full_layers: s.full_attention_layers(),
            gdn_layers: s.gdn_layers(),
            full_kv_bytes: 2 * len * kvrow,
            gdn_state_bytes: state_floats * F32,
            conv_tail_bytes: 3 * s.gdn_in_width() * BF16,
        }
    }

    /// Total device bytes of one prefix of `len` tokens.
    pub fn total_bytes(&self) -> usize {
        self.full_layers * self.full_kv_bytes
            + self.gdn_layers * (self.gdn_state_bytes + self.conv_tail_bytes)
    }
}

/// Stable identity of one structure executing on the ABI-8 backend.
/// Any prefix-cache record must carry it; a change means the state
/// must be re-captured, never reused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenIdentity {
    pub schema_version: usize,
    pub family: &'static str,
    pub backend: &'static str,
    pub abi_version: usize,
    pub digest: String,
}

pub fn structure_identity(s: &QwenHybridStructure) -> QwenIdentity {
    let mut body = serde_json::Map::new();
    body.insert(
        "schema_version".into(),
        Json::String(SCHEMA_VERSION.to_string()),
    );
    body.insert("family".into(), Json::String(FAMILY.into()));
    body.insert("backend".into(), Json::String(BACKEND.into()));
    body.insert("abi_version".into(), Json::String(ABI_VERSION.to_string()));
    body.insert("structure".into(), s.canonical_json());
    let body = Json::Object(body).to_string();
    QwenIdentity {
        schema_version: SCHEMA_VERSION,
        family: FAMILY,
        backend: BACKEND,
        abi_version: ABI_VERSION,
        digest: format!("{:x}", Sha256::digest(body.as_bytes())),
    }
}
