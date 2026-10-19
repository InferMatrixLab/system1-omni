use omni_modernbert::gate::{BackendLimits, RegisteredProfile};
use omni_modernbert::structure::{
    Activation, AttentionKind, FieldOrigin, ModernBertStructure, RopeSpec, YarnScaling,
};

/// The registered `vela-2.0-0.3b` profile: the pinned checkpoint's
/// structure (read from `encoder_config`, cross-checked against the
/// recorded tensor census) against the limits of the regenerated CUDA
/// bundle this profile targets — hidden 768 instantiations, YaRN rotary
/// tables, requests packed to 8 192 tokens. The bundle's own ABI is
/// checked when it is loaded; the executor slice lands with it.
pub fn profile() -> RegisteredProfile {
    RegisteredProfile {
        name: "vela-2.0-0.3b",
        structure: canonical_structure(),
        limits: BackendLimits {
            hidden_sizes: vec![768],
            batch_max: 4,
            len_min: 16,
            len_max: 8192,
            len_step: 16,
            yarn_scaling: true,
        },
        backend: "laya_cuda",
        abi_version: 2,
    }
}

fn canonical_structure() -> ModernBertStructure {
    let mut provenance = std::collections::BTreeMap::new();
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
        provenance.insert(
            field.to_string(),
            FieldOrigin::Source("registered profile vela-2.0-0.3b".into()),
        );
    }
    let yarn = YarnScaling {
        factor: 4.0,
        original_max_positions: 8192,
        beta_fast: 32.0,
        beta_slow: 1.0,
    };
    ModernBertStructure {
        hidden_size: 768,
        intermediate_size: 1152,
        num_hidden_layers: 22,
        activation: Activation::Gelu,
        norm_eps: 1e-5,
        num_attention_heads: 12,
        head_dim: 64,
        schedule: (0..22)
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
            scaling: Some(yarn),
        },
        rope_sliding: RopeSpec {
            theta: 160000.0,
            scaling: Some(yarn),
        },
        vocab_size: 256008,
        type_embedding_rows: None,
        max_positions: 32768,
        provenance,
    }
}
