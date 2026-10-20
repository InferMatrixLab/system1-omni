use omni_laya::profile::{LayaBundleAdapter, laya_0_3_20};
use omni_modernbert::structure::{AttentionKind, FieldOrigin};
use serde_json::{Value, json};
use std::fs;
use std::path::PathBuf;
use tempfile::{TempDir, tempdir};

/// The Laya 0.3.20 bundle's `encoder/config.json` as shipped: the keys
/// the native worker consumes. The HF-side members the dialect ignores
/// (`max_position_embeddings`, `head_dim`) are added by name where a
/// test exercises that behavior.
fn laya_encoder_config() -> Value {
    json!({
        "model_type": "modernbert", "hidden_activation": "gelu",
        "attention_bias": false, "mlp_bias": false, "norm_bias": false,
        "hidden_size": 1024, "intermediate_size": 2624,
        "num_attention_heads": 16, "num_hidden_layers": 28,
        "vocab_size": 50368, "norm_eps": 0.00001, "local_attention": 128,
        "layer_types": (["full_attention", "sliding_attention", "sliding_attention"]
            .repeat(10))[..28],
        "rope_parameters": {
            "full_attention": {"rope_type": "default", "rope_theta": 160000.0},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}
        }
    })
}

fn laya_bundle(encoder: &Value) -> (TempDir, PathBuf) {
    let dir = tempdir().unwrap();
    fs::create_dir(dir.path().join("encoder")).unwrap();
    fs::write(dir.path().join("encoder/config.json"), encoder.to_string()).unwrap();
    let path = dir.path().to_path_buf();
    (dir, path)
}

fn parse_laya(encoder: &Value) -> omni_modernbert::structure::ModernBertStructure {
    let (_dir, path) = laya_bundle(encoder);
    LayaBundleAdapter::read(&path).unwrap().0
}

#[test]
fn bundle_adapter_parses_the_profile_with_provenance() {
    let (_dir, path) = laya_bundle(&laya_encoder_config());
    let (structure, meta) = LayaBundleAdapter::read(&path).unwrap();
    assert_eq!(structure.hidden_size, 1024);
    assert_eq!(structure.intermediate_size, 2624);
    assert_eq!(structure.num_hidden_layers, 28);
    assert_eq!(structure.num_attention_heads, 16);
    assert_eq!(structure.head_dim, 64);
    assert_eq!(structure.local_window, 128);
    assert_eq!(structure.vocab_size, 50368);
    assert_eq!(structure.type_embedding_rows, Some(3));
    assert_eq!(structure.rope_full.theta, 160000.0);
    assert_eq!(structure.rope_sliding.theta, 10000.0);
    assert_eq!(structure.max_positions, 512);
    assert_eq!(meta.model_type, "modernbert");
    assert_eq!(meta.config_files, ["encoder/config.json"]);
    assert_eq!(meta.unrecognized_keys, Vec::<String>::new());
    assert_eq!(
        structure.provenance.get("schedule"),
        Some(&FieldOrigin::Source(
            "encoder/config.json#layer_types".into()
        ))
    );
    assert_eq!(
        structure.provenance.get("head_dim"),
        Some(&FieldOrigin::Derived(
            "hidden_size / num_attention_heads".into()
        ))
    );
    assert!(matches!(
        structure.provenance.get("max_positions"),
        Some(FieldOrigin::Default(_))
    ));
}

/// The dialect ignores the HF position and head-dim members the native
/// worker never consumed: a bundle declaring them (as stock HF
/// ModernBERT configs do) still admits with the bundle's own 512
/// exercised positions and derived head dim, and the keys are recorded
/// as unrecognized instead of gated.
#[test]
fn bundle_adapter_ignores_hf_position_and_head_dim_members() {
    let mut config = laya_encoder_config();
    config["max_position_embeddings"] = json!(8192);
    config["head_dim"] = json!(32);
    let structure = parse_laya(&config);
    assert_eq!(structure.max_positions, 512);
    assert_eq!(structure.head_dim, 64);
    let (_dir, path) = laya_bundle(&config);
    let (_, meta) = LayaBundleAdapter::read(&path).unwrap();
    let mut keys = meta.unrecognized_keys.clone();
    keys.sort();
    assert_eq!(keys, ["head_dim", "max_position_embeddings"]);
    laya_0_3_20().admit(&structure).unwrap();
}

#[test]
fn bundle_adapter_rejects_marker_and_family_violations() {
    for (pointer, value) in [
        ("/model_type", json!("bert")),
        ("/hidden_activation", json!("relu")),
        ("/attention_bias", json!(true)),
        ("/mlp_bias", json!(true)),
        ("/norm_bias", json!(true)),
        ("/layer_types/0", json!("sparse_attention")),
        ("/rope_parameters/full_attention/rope_type", json!("linear")),
        ("/rope_parameters/sliding_attention/rope_theta", json!(null)),
    ] {
        let mut bad = laya_encoder_config();
        *bad.pointer_mut(pointer).unwrap() = value;
        let (_dir, path) = laya_bundle(&bad);
        assert!(
            LayaBundleAdapter::read(&path).is_err(),
            "accepted {pointer}"
        );
    }
    let mut missing = laya_encoder_config();
    missing.as_object_mut().unwrap().remove("layer_types");
    let (_dir, path) = laya_bundle(&missing);
    assert!(LayaBundleAdapter::read(&path).is_err());
    let mut short = laya_encoder_config();
    short["layer_types"] = json!(["full_attention"]);
    let (_dir, path) = laya_bundle(&short);
    assert!(LayaBundleAdapter::read(&path).is_err());
}

#[test]
fn the_registered_profile_admits_its_structure_and_rejects_field_drift() {
    let profile = laya_0_3_20();
    let structure = parse_laya(&laya_encoder_config());
    let identity = profile.admit(&structure).unwrap();
    assert_eq!(identity.schema_version, 1);
    assert_eq!(identity.family, "modernbert");
    assert_eq!(identity.backend, "laya_cuda");
    assert_eq!(identity.abi_version, 1);

    for mutate in [
        ("hidden_size", {
            let mut s = parse_laya(&laya_encoder_config());
            s.hidden_size = 768;
            s
        }),
        ("rope", {
            let mut s = parse_laya(&laya_encoder_config());
            s.rope_sliding.theta = 160000.0;
            s
        }),
        ("schedule", {
            let mut s = parse_laya(&laya_encoder_config());
            s.schedule[1] = AttentionKind::Full;
            s
        }),
        ("type_embedding_rows", {
            let mut s = parse_laya(&laya_encoder_config());
            s.type_embedding_rows = None;
            s
        }),
        ("max_positions", {
            let mut s = parse_laya(&laya_encoder_config());
            s.max_positions = 8192;
            s
        }),
    ] {
        assert!(profile.admit(&mutate.1).is_err(), "accepted {}", mutate.0);
    }
}

#[test]
fn the_registered_limits_check_the_request_envelope() {
    let limits = &laya_0_3_20().limits;
    limits.check_request(16, 512).unwrap();
    limits.check_request(1, 16).unwrap();
    for (batch, len) in [(32, 512), (3, 512), (1, 8), (1, 513), (1, 520)] {
        assert!(
            limits.check_request(batch, len).is_err(),
            "accepted batch {batch} len {len}"
        );
    }
}
