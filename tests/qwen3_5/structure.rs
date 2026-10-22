use omni_qwen3_5_native::model::Config;
use omni_qwen3_5_native::structure::{self, LayerKind, PrefixLayout, QwenHybridStructure};
use serde_json::json;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

/// A minimal gate-passing structure; every test mutates from here.
fn hybrid(
    hidden: usize,
    intermediate: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    lin_k_heads: usize,
    lin_v_heads: usize,
) -> QwenHybridStructure {
    let (_keep, path) = fixture_dir(default_config());
    let mut s = structure::read_structure(&path).unwrap();
    s.hidden = hidden;
    s.intermediate = intermediate;
    s.heads = heads;
    s.kv_heads = kv_heads;
    s.lin_k_heads = lin_k_heads;
    s.lin_v_heads = lin_v_heads;
    s.layer_kinds = (0..layers)
        .map(|i| {
            if i % 4 == 3 {
                LayerKind::FullAttention
            } else {
                LayerKind::GatedDeltaNet
            }
        })
        .collect();
    s
}

fn vela_08b() -> QwenHybridStructure {
    hybrid(1024, 3584, 24, 8, 2, 16, 16)
}

fn vela_4b() -> QwenHybridStructure {
    hybrid(2560, 9216, 32, 16, 4, 16, 32)
}

fn vela_9b() -> QwenHybridStructure {
    hybrid(4096, 12288, 32, 16, 4, 16, 32)
}

fn default_config() -> serde_json::Value {
    json!({
        "model_type": "qwen3_5_text",
        "hidden_act": "silu", "attn_output_gate": true, "attention_bias": false,
        "hidden_size": 5120, "intermediate_size": 17408,
        "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
        "num_hidden_layers": 4,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "rms_norm_eps": 1e-6,
        "linear_num_key_heads": 16, "linear_num_value_heads": 48,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "max_position_embeddings": 262144,
        "rope_parameters": {
            "rope_type": "default", "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25, "mrope_interleaved": true,
            "mrope_section": [11, 11, 10]
        }
    })
}

fn fixture_dir(config: serde_json::Value) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempdir().unwrap();
    fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
    let path = dir.path().to_path_buf();
    (dir, path)
}

#[test]
fn adapter_reads_the_checkpoint_and_records_provenance() {
    let (_keep, path) = fixture_dir(default_config());
    let s = structure::read_structure(&path).unwrap();
    assert_eq!(s.hidden, 5120);
    assert_eq!(s.head_dim, 256);
    assert_eq!(s.rotary_half, 32);
    assert_eq!(s.image_token_id, None);
    assert_eq!(
        s.layer_kinds,
        [
            LayerKind::GatedDeltaNet,
            LayerKind::GatedDeltaNet,
            LayerKind::GatedDeltaNet,
            LayerKind::FullAttention
        ]
    );
    assert_eq!(
        s.provenance.get("hidden"),
        Some(&"config.json#hidden_size".to_string())
    );
    assert_eq!(s.hidden_act.as_deref(), Some("silu"));
    assert_eq!(s.conv_kernel_dim, 4);
    structure::check_abi(&s).unwrap();
}

#[test]
fn adapter_rejects_broken_sources() {
    for (name, config) in [
        ("missing layer_types", {
            let mut c = default_config();
            c.as_object_mut().unwrap().remove("layer_types");
            c
        }),
        ("unknown kind", {
            let mut c = default_config();
            c["layer_types"][0] = json!("sparse_attention");
            c
        }),
        ("length mismatch", {
            let mut c = default_config();
            c["layer_types"] = json!(["linear_attention"]);
            c
        }),
        ("missing conv", {
            let mut c = default_config();
            c.as_object_mut().unwrap().remove("linear_conv_kernel_dim");
            c
        }),
    ] {
        assert!(
            {
                let (_keep, path) = fixture_dir(config);
                structure::read_structure(&path).is_err()
            },
            "accepted {name}"
        );
    }
}

#[test]
fn the_frozen_gate_accepts_the_three_vela_hybrid_backbones() {
    for (name, s) in [("0.8B", vela_08b()), ("4B", vela_4b()), ("9B", vela_9b())] {
        structure::check_abi(&s).unwrap_or_else(|e| panic!("{name} rejected: {e}"));
    }
}

#[test]
fn the_frozen_gate_rejects_every_capability_violation() {
    type Mutation = Box<dyn Fn(&mut QwenHybridStructure)>;
    let rejects: Vec<(&str, Mutation)> = vec![
        (
            "head_dim",
            Box::new(|s: &mut QwenHybridStructure| s.head_dim = 128),
        ),
        ("lin_k_dim", Box::new(|s| s.lin_k_dim = 64)),
        ("lin_v_dim", Box::new(|s| s.lin_v_dim = 64)),
        ("conv", Box::new(|s| s.conv_kernel_dim = 5)),
        (
            "hidden_act",
            Box::new(|s| s.hidden_act = Some("relu".into())),
        ),
        ("output gate", Box::new(|s| s.attn_output_gate = false)),
        ("attention bias", Box::new(|s| s.attention_bias = true)),
        ("rope type", Box::new(|s| s.rope_type = "linear".into())),
        ("mrope", Box::new(|s| s.mrope_interleaved = false)),
        ("mrope sum", Box::new(|s| s.mrope_section = [11, 11, 11])),
        ("rotary", Box::new(|s| s.rotary_half = 16)),
        (
            "max positions upper",
            Box::new(|s| s.max_positions = 1 << 25),
        ),
        ("max positions zero", Box::new(|s| s.max_positions = 0)),
        ("rope theta zero", Box::new(|s| s.rope_theta = 0.0)),
        ("rope theta negative", Box::new(|s| s.rope_theta = -1.0)),
        ("rope theta nan", Box::new(|s| s.rope_theta = f64::NAN)),
        (
            "heads divisibility",
            Box::new(|s: &mut QwenHybridStructure| {
                s.heads = 9;
                s.kv_heads = 2;
            }),
        ),
        (
            "linear heads divisibility",
            Box::new(|s: &mut QwenHybridStructure| {
                s.lin_v_heads = 17;
            }),
        ),
        ("hidden alignment", Box::new(|s| s.hidden = 1025)),
    ];
    assert_eq!(rejects.len(), 19);
    for (name, mutate) in rejects {
        let (_keep, path) = fixture_dir(default_config());
        let mut s = structure::read_structure(&path).unwrap();
        mutate(&mut s);
        assert!(structure::check_abi(&s).is_err(), "gate accepted {name}");
    }
    let (_keep, path) = fixture_dir(default_config());
    let mut s = structure::read_structure(&path).unwrap();
    s.rope_type = "linear".into();
    let message = structure::check_abi(&s).unwrap_err().to_string();
    assert!(message.contains("\"linear\""), "{message}");
}

#[test]
fn config_load_rejects_a_defective_checkpoint_end_to_end() {
    let mut config = default_config();
    config["hidden_act"] = json!("relu");
    let (_keep, path) = fixture_dir(config);
    assert!(Config::load(&path).is_err());
}

#[test]
fn the_open_jev_27b_checkpoint_passes_the_gate_unchanged() {
    let dir = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../../tests/qwen3_5/data"
    ));
    let s = structure::read_structure(dir).unwrap();
    structure::check_abi(&s).unwrap();
    assert_eq!(s.hidden, 5120);
    assert_eq!(s.layer_kinds.len(), 64);
    assert_eq!(s.full_attention_layers(), 16);
    assert_eq!(s.gdn_layers(), 48);
    // The facade projects the same values it always did.
    let cfg = Config::load(dir).unwrap();
    assert_eq!(cfg.hidden, s.hidden);
    assert_eq!(cfg.full_attention.len(), s.layer_kinds.len());
    assert_eq!(
        cfg.full_attention.iter().filter(|&&f| f).count(),
        s.full_attention_layers()
    );
}

#[test]
fn prefix_layout_sizes_derive_from_the_structure() {
    let layout = PrefixLayout::of(&vela_08b(), 64);
    assert_eq!(layout.full_layers, 6);
    assert_eq!(layout.gdn_layers, 18);
    assert_eq!(layout.full_kv_bytes, 2 * 64 * 2 * 256 * 2);
    assert_eq!(layout.gdn_state_bytes, 16 * 128 * 128 * 4);
    assert_eq!(layout.conv_tail_bytes, 3 * 8224 * 2);
    assert_eq!(
        layout.total_bytes(),
        6 * layout.full_kv_bytes + 18 * (layout.gdn_state_bytes + layout.conv_tail_bytes)
    );
    // Grows linearly in tokens through the full-attention rows only.
    let longer = PrefixLayout::of(&vela_08b(), 128);
    assert_eq!(longer.full_kv_bytes, 2 * layout.full_kv_bytes);
    assert_eq!(longer.gdn_state_bytes, layout.gdn_state_bytes);
}

#[test]
fn identity_is_stable_and_sensitive() {
    let a = structure::structure_identity(&vela_08b());
    let b = structure::structure_identity(&vela_08b());
    assert_eq!(a, b);
    assert_eq!(a.abi_version, 9);
    assert_eq!(a.family, "qwen3_5_hybrid");
    assert_ne!(a, structure::structure_identity(&vela_4b()));
    let mut drifted = vela_08b();
    drifted.rope_theta = 5e6;
    assert_ne!(a, structure::structure_identity(&drifted));
    // Provenance is not part of identity.
    let mut reworded = vela_08b();
    reworded.provenance.clear();
    assert_eq!(a, structure::structure_identity(&reworded));
}
