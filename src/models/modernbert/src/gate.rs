use crate::structure::{ModernBertStructure, SCHEMA_VERSION};
use anyhow::{Result, ensure};
use sha2::{Digest, Sha256};

/// What one CUDA bundle can execute: the geometry its kernels were
/// generated for and the request envelope it admits. A new bundle
/// registers new limits; the gate code does not change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendLimits {
    /// Hidden sizes the bundle's kernels were instantiated for.
    pub hidden_sizes: Vec<usize>,
    /// Batch bound; a request batch must be a power of two up to it.
    pub batch_max: usize,
    pub len_min: usize,
    pub len_max: usize,
    pub len_step: usize,
    /// Whether the bundle carries YaRN-scaled rotary tables.
    pub yarn_scaling: bool,
}

impl BackendLimits {
    pub fn check_request(&self, batch: usize, len: usize) -> Result<()> {
        ensure!(
            batch.is_power_of_two() && batch <= self.batch_max,
            "batch {batch} exceeds the power-of-two bound {}",
            self.batch_max
        );
        ensure!(
            (self.len_min..=self.len_max).contains(&len),
            "packed length {len} is outside {}..={}",
            self.len_min,
            self.len_max
        );
        ensure!(
            len.is_multiple_of(self.len_step),
            "packed length {len} is not a multiple of {}",
            self.len_step
        );
        Ok(())
    }

    pub fn check_structure(&self, s: &ModernBertStructure) -> Result<()> {
        ensure!(
            self.hidden_sizes.contains(&s.hidden_size),
            "hidden_size {} is not instantiated in this bundle ({:?})",
            s.hidden_size,
            self.hidden_sizes
        );
        ensure!(
            self.yarn_scaling
                || (s.rope_full.scaling.is_none() && s.rope_sliding.scaling.is_none()),
            "rope scaling is not available in this bundle"
        );
        Ok(())
    }
}

/// A named, frozen structure-and-limits pair. Admission checks family
/// invariants, backend limits, and equality with the canonical
/// structure, and returns the identity every cache key must carry.
#[derive(Debug, Clone)]
pub struct RegisteredProfile {
    pub name: &'static str,
    pub structure: ModernBertStructure,
    pub limits: BackendLimits,
    pub backend: &'static str,
    pub abi_version: usize,
}

impl RegisteredProfile {
    pub fn admit(&self, candidate: &ModernBertStructure) -> Result<StructureIdentity> {
        candidate.validate()?;
        self.limits.check_structure(candidate)?;
        let expected = &self.structure;
        let field = |name: &str, actual: String, wanted: String| {
            format!("profile {}: {name} {actual}, expected {wanted}", self.name)
        };
        ensure!(
            candidate.hidden_size == expected.hidden_size,
            "{}",
            field(
                "hidden_size",
                candidate.hidden_size.to_string(),
                expected.hidden_size.to_string()
            )
        );
        ensure!(
            candidate.intermediate_size == expected.intermediate_size,
            "{}",
            field(
                "intermediate_size",
                candidate.intermediate_size.to_string(),
                expected.intermediate_size.to_string()
            )
        );
        ensure!(
            candidate.num_hidden_layers == expected.num_hidden_layers,
            "{}",
            field(
                "num_hidden_layers",
                candidate.num_hidden_layers.to_string(),
                expected.num_hidden_layers.to_string()
            )
        );
        ensure!(
            candidate.num_attention_heads == expected.num_attention_heads,
            "{}",
            field(
                "num_attention_heads",
                candidate.num_attention_heads.to_string(),
                expected.num_attention_heads.to_string()
            )
        );
        ensure!(
            candidate.head_dim == expected.head_dim,
            "{}",
            field(
                "head_dim",
                candidate.head_dim.to_string(),
                expected.head_dim.to_string()
            )
        );
        ensure!(
            candidate.schedule == expected.schedule,
            "{}",
            field(
                "schedule",
                render_schedule(candidate),
                render_schedule(expected)
            )
        );
        ensure!(
            candidate.local_window == expected.local_window,
            "{}",
            field(
                "local_window",
                candidate.local_window.to_string(),
                expected.local_window.to_string()
            )
        );
        ensure!(
            candidate.norm_eps.to_bits() == expected.norm_eps.to_bits(),
            "{}",
            field(
                "norm_eps",
                candidate.norm_eps.to_string(),
                expected.norm_eps.to_string()
            )
        );
        ensure!(
            candidate.activation == expected.activation,
            "{}",
            field(
                "activation",
                format!("{:?}", candidate.activation),
                format!("{:?}", expected.activation)
            )
        );
        ensure!(
            candidate.rope_full == expected.rope_full
                && candidate.rope_sliding == expected.rope_sliding,
            "{}",
            field(
                "rope",
                format!("{:?}/{:?}", candidate.rope_full, candidate.rope_sliding),
                format!("{:?}/{:?}", expected.rope_full, expected.rope_sliding)
            )
        );
        ensure!(
            candidate.vocab_size == expected.vocab_size,
            "{}",
            field(
                "vocab_size",
                candidate.vocab_size.to_string(),
                expected.vocab_size.to_string()
            )
        );
        ensure!(
            candidate.type_embedding_rows == expected.type_embedding_rows,
            "{}",
            field(
                "type_embedding_rows",
                format!("{:?}", candidate.type_embedding_rows),
                format!("{:?}", expected.type_embedding_rows)
            )
        );
        ensure!(
            candidate.max_positions == expected.max_positions,
            "{}",
            field(
                "max_positions",
                candidate.max_positions.to_string(),
                expected.max_positions.to_string()
            )
        );
        Ok(structure_identity(
            candidate,
            self.backend,
            self.abi_version,
        ))
    }
}

fn render_schedule(s: &ModernBertStructure) -> String {
    s.schedule
        .iter()
        .map(|k| k.as_str())
        .collect::<Vec<_>>()
        .join(",")
}

/// Stable identity of one structure executing on one backend. Any change
/// to the digest inputs means previously captured runtime state (graphs,
/// caches, prefix state) must not be reused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructureIdentity {
    pub schema_version: usize,
    pub family: &'static str,
    pub backend: String,
    pub abi_version: usize,
    pub digest: String,
}

pub fn structure_identity(
    s: &ModernBertStructure,
    backend: &str,
    abi_version: usize,
) -> StructureIdentity {
    let body = serde_json::json!({
        "schema_version": SCHEMA_VERSION,
        "family": "modernbert",
        "backend": backend,
        "abi_version": abi_version,
        "structure": s.canonical_json(),
    })
    .to_string();
    StructureIdentity {
        schema_version: SCHEMA_VERSION,
        family: "modernbert",
        backend: backend.to_string(),
        abi_version,
        digest: format!("{:x}", Sha256::digest(body.as_bytes())),
    }
}
