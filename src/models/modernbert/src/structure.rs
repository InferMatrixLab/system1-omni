use anyhow::{Result, ensure};
use std::collections::BTreeMap;

/// Bumped when the canonical serialization below changes meaning.
pub const SCHEMA_VERSION: usize = 1;

/// One encoder layer's attention mixer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionKind {
    Full,
    Sliding,
}

impl AttentionKind {
    /// The name checkpoint configs spell in `layer_types`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full_attention",
            Self::Sliding => "sliding_attention",
        }
    }

    /// The suffix the family CUDA kernels carry per attention kind.
    pub fn kernel_suffix(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Sliding => "local",
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "full_attention" => Ok(Self::Full),
            "sliding_attention" => Ok(Self::Sliding),
            other => anyhow::bail!("unknown attention kind {other:?}"),
        }
    }
}

/// The only activation the family kernels implement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Gelu,
}

/// YaRN frequency scaling applied on top of a base theta.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YarnScaling {
    pub factor: f64,
    pub original_max_positions: usize,
    pub beta_fast: f64,
    pub beta_slow: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeSpec {
    pub theta: f64,
    pub scaling: Option<YarnScaling>,
}

/// Where one structure field's value came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldOrigin {
    Source(String),
    Default(&'static str),
    Derived(String),
}

/// The ModernBERT encoder structure: family semantics and math geometry
/// with per-field provenance. Weight names, kernels, and buffers are not
/// part of the structure; they are derived in [`crate::geometry`] and
/// [`crate::plan`].
#[derive(Debug, Clone, PartialEq)]
pub struct ModernBertStructure {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub activation: Activation,
    pub norm_eps: f32,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub schedule: Vec<AttentionKind>,
    pub local_window: usize,
    pub rope_full: RopeSpec,
    pub rope_sliding: RopeSpec,
    pub vocab_size: usize,
    pub type_embedding_rows: Option<usize>,
    pub max_positions: usize,
    pub provenance: BTreeMap<String, FieldOrigin>,
}

impl ModernBertStructure {
    /// Family invariants. Every failure names the field and its value.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.num_hidden_layers > 0,
            "num_hidden_layers is {}",
            self.num_hidden_layers
        );
        ensure!(
            self.num_attention_heads > 0
                && self.num_attention_heads * self.head_dim == self.hidden_size,
            "num_attention_heads {} with head_dim {} does not form hidden_size {}",
            self.num_attention_heads,
            self.head_dim,
            self.hidden_size
        );
        ensure!(
            self.head_dim.is_multiple_of(2),
            "head_dim {} has no rotary half",
            self.head_dim
        );
        ensure!(
            self.schedule.len() == self.num_hidden_layers,
            "schedule holds {} entries for {} layers",
            self.schedule.len(),
            self.num_hidden_layers
        );
        ensure!(
            self.intermediate_size > 0,
            "intermediate_size is {}",
            self.intermediate_size
        );
        ensure!(
            self.local_window > 0,
            "local_attention is {}",
            self.local_window
        );
        ensure!(self.vocab_size > 0, "vocab_size is {}", self.vocab_size);
        ensure!(
            self.norm_eps > 0.0 && self.norm_eps.is_finite(),
            "norm_eps is {}",
            self.norm_eps
        );
        ensure!(
            self.max_positions > 0,
            "max_position_embeddings is {}",
            self.max_positions
        );
        for (name, rope) in [("full", &self.rope_full), ("sliding", &self.rope_sliding)] {
            ensure!(
                rope.theta.is_finite() && rope.theta > 0.0,
                "{name} rope_theta is {}",
                rope.theta
            );
            if let Some(YarnScaling {
                factor,
                original_max_positions,
                beta_fast,
                beta_slow,
            }) = rope.scaling
            {
                ensure!(factor > 1.0, "yarn factor is {factor}");
                ensure!(
                    original_max_positions > 0,
                    "yarn original_max_position_embeddings is {original_max_positions}"
                );
                ensure!(
                    beta_slow > 0.0 && beta_fast > beta_slow,
                    "yarn beta_fast {beta_fast} / beta_slow {beta_slow}"
                );
            }
        }
        Ok(())
    }

    /// Canonical, sorted-key JSON body (provenance excluded) used for
    /// identity digests. Same structure always serializes identically.
    pub fn canonical_json(&self) -> serde_json::Value {
        let rope = |r: &RopeSpec| {
            let mut value = serde_json::json!({ "theta": r.theta });
            if let Some(y) = &r.scaling {
                value["scaling"] = serde_json::json!({
                    "factor": y.factor,
                    "original_max_positions": y.original_max_positions,
                    "beta_fast": y.beta_fast,
                    "beta_slow": y.beta_slow,
                });
            }
            value
        };
        serde_json::json!({
            "schema_version": SCHEMA_VERSION,
            "family": "modernbert",
            "activation": "gelu",
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "norm_eps": self.norm_eps,
            "num_attention_heads": self.num_attention_heads,
            "head_dim": self.head_dim,
            "schedule": self.schedule.iter().map(|k| k.as_str()).collect::<Vec<_>>(),
            "local_window": self.local_window,
            "rope_full": rope(&self.rope_full),
            "rope_sliding": rope(&self.rope_sliding),
            "vocab_size": self.vocab_size,
            "type_embedding_rows": self.type_embedding_rows,
            "max_positions": self.max_positions,
        })
    }
}
