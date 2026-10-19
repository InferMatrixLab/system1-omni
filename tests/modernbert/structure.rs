use omni_modernbert::adapter::{LayaBundleAdapter, ModernBertConfigAdapter};
use omni_modernbert::gate::{self, structure_identity};
use omni_modernbert::geometry::{self, EncoderScratch};
use omni_modernbert::structure::{AttentionKind, FieldOrigin};
use serde_json::{Value, json};
use std::fs;
use tempfile::{TempDir, tempdir};

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

/// The pinned vllm-sr/Vela-2.0-0.3B@d6f03aa9 `encoder_config` member.
fn vela_encoder_config() -> Value {
    json!({
        "model_type": "modernbert", "hidden_activation": "gelu",
        "attention_bias": false, "mlp_bias": false, "norm_bias": false,
        "hidden_size": 768, "intermediate_size": 1152,
        "num_attention_heads": 12, "num_hidden_layers": 22,
        "vocab_size": 256008, "norm_eps": 1e-5, "layer_norm_eps": 1e-5,
        "local_attention": 128, "global_attn_every_n_layers": 3,
        "global_rope_theta": 160000, "local_rope_theta": 160000,
        "max_position_embeddings": 32768, "position_embedding_type": "sans_pos",
        "rope_scaling": {
            "rope_type": "yarn", "factor": 4.0,
            "original_max_position_embeddings": 8192,
            "beta_fast": 32.0, "beta_slow": 1.0, "truncate": true
        },
        "cls_token_id": 1, "pad_token_id": 0
    })
}

fn laya_bundle(encoder: &Value) -> (TempDir, std::path::PathBuf) {
    let dir = tempdir().unwrap();
    fs::create_dir(dir.path().join("encoder")).unwrap();
    fs::write(
        dir.path().join("encoder/config.json"),
        encoder.to_string(),
    )
    .unwrap();
    let path = dir.path().to_path_buf();
    (dir, path)
}

fn parse_laya(encoder: &Value) -> omni_modernbert::structure::ModernBertStructure {
    let (_dir, path) = laya_bundle(encoder);
    LayaBundleAdapter::read(&path).unwrap().0
}

fn parse_hf(config: &Value) -> omni_modernbert::structure::ModernBertStructure {
    ModernBertConfigAdapter::from_value(config, "config.json#encoder_config")
        .unwrap()
        .0
}

#[test]
fn laya_bundle_adapter_parses_the_profile_with_provenance() {
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

#[test]
fn laya_bundle_adapter_rejects_marker_and_family_violations() {
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
fn hf_adapter_parses_the_vela_profile_with_derived_schedule() {
    let (structure, meta) =
        ModernBertConfigAdapter::from_value(&vela_encoder_config(), "config.json#encoder_config")
            .unwrap();
    assert_eq!(structure.hidden_size, 768);
    assert_eq!(structure.intermediate_size, 1152);
    assert_eq!(structure.num_hidden_layers, 22);
    assert_eq!(structure.num_attention_heads, 12);
    assert_eq!(structure.head_dim, 64);
    assert_eq!(structure.vocab_size, 256008);
    assert_eq!(structure.type_embedding_rows, None);
    assert_eq!(structure.max_positions, 32768);
    let full_layers: Vec<_> = structure
        .schedule
        .iter()
        .enumerate()
        .filter(|(_, k)| **k == AttentionKind::Full)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(full_layers, [0, 3, 6, 9, 12, 15, 18, 21]);
    assert_eq!(structure.rope_full.theta, 160000.0);
    assert_eq!(structure.rope_sliding.theta, 160000.0);
    let yarn = structure.rope_full.scaling.unwrap();
    assert_eq!(yarn.factor, 4.0);
    assert_eq!(yarn.original_max_positions, 8192);
    assert_eq!(structure.rope_full.scaling, structure.rope_sliding.scaling);
    assert_eq!(meta.unrecognized_keys, ["cls_token_id", "pad_token_id"]);
    match structure.provenance.get("schedule") {
        Some(FieldOrigin::Derived(how)) => {
            assert!(how.contains("global_attn_every_n_layers=3"), "{how}")
        }
        other => panic!("schedule provenance {other:?}"),
    }
}

#[test]
fn hf_adapter_accepts_explicit_layer_types_over_the_interval() {
    let mut config = vela_encoder_config();
    config["global_attn_every_n_layers"] = json!(2);
    config["layer_types"] = json!((vec!["sliding_attention"; 22]));
    let structure = parse_hf(&config);
    assert!(structure.schedule.iter().all(|k| *k == AttentionKind::Sliding));
    assert!(matches!(
        structure.provenance.get("schedule"),
        Some(FieldOrigin::Source(_))
    ));
}

#[test]
fn hf_adapter_rejects_unknown_variants() {
    for (name, config) in [
        ("model_type", {
            let mut c = vela_encoder_config();
            c["model_type"] = json!("vela2-unified");
            c
        }),
        ("no schedule source", {
            let mut c = vela_encoder_config();
            c.as_object_mut().unwrap().remove("global_attn_every_n_layers");
            c
        }),
        ("scaling type", {
            let mut c = vela_encoder_config();
            c["rope_scaling"]["rope_type"] = json!("linear");
            c
        }),
        ("scaling truncate", {
            let mut c = vela_encoder_config();
            c["rope_scaling"]["truncate"] = json!(false);
            c
        }),
        ("position embedding", {
            let mut c = vela_encoder_config();
            c["position_embedding_type"] = json!("absolute");
            c
        }),
        ("norm eps", {
            let mut c = vela_encoder_config();
            c.as_object_mut().unwrap().remove("norm_eps");
            c.as_object_mut().unwrap().remove("layer_norm_eps");
            c
        }),
        ("yarn factor", {
            let mut c = vela_encoder_config();
            c["rope_scaling"]["factor"] = json!(1.0);
            c
        }),
    ] {
        assert!(
            ModernBertConfigAdapter::from_value(&config, "config.json#encoder_config").is_err(),
            "accepted {name}"
        );
    }
}

#[test]
fn structure_invariants_name_the_violated_field() {
    let mut structure = parse_laya(&laya_encoder_config());
    structure.num_attention_heads = 15;
    assert!(structure.validate().unwrap_err().to_string().contains("15"));

    let mut structure = parse_laya(&laya_encoder_config());
    structure.schedule.pop();
    let message = structure.validate().unwrap_err().to_string();
    assert!(message.contains("schedule"), "{message}");

    for mutate in [
        ("local_window", 0),
        ("intermediate_size", 0),
        ("vocab_size", 0),
        ("num_hidden_layers", 0),
    ] {
        let mut structure = parse_laya(&laya_encoder_config());
        match mutate.0 {
            "local_window" => structure.local_window = mutate.1,
            "intermediate_size" => structure.intermediate_size = mutate.1,
            "vocab_size" => structure.vocab_size = mutate.1,
            _ => structure.num_hidden_layers = mutate.1,
        }
        assert!(
            structure.validate().is_err(),
            "accepted {} = 0",
            mutate.0
        );
    }
    let mut structure = parse_laya(&laya_encoder_config());
    structure.rope_full.theta = 0.0;
    assert!(structure.validate().is_err());
}

#[test]
fn laya_profile_admits_its_structure_and_rejects_field_drift() {
    let profile = gate::laya_0_3_20();
    let structure = parse_laya(&laya_encoder_config());
    let identity = profile.admit(&structure).unwrap();
    assert_eq!(identity.schema_version, 1);
    assert_eq!(identity.family, "modernbert");
    assert_eq!(identity.backend, "laya_cuda");
    assert_eq!(identity.abi_version, 1);

    let vela = parse_hf(&vela_encoder_config());
    let message = profile.admit(&vela).unwrap_err().to_string();
    assert!(message.contains("hidden_size 768"), "{message}");
    assert!(message.contains("1024"), "{message}");

    let mut drifted = parse_laya(&laya_encoder_config());
    drifted.rope_sliding.theta = 160000.0;
    assert!(profile.admit(&drifted).is_err());

    let mut drifted = parse_laya(&laya_encoder_config());
    drifted.schedule[1] = AttentionKind::Full;
    assert!(profile.admit(&drifted).is_err());

    let mut drifted = parse_laya(&laya_encoder_config());
    drifted.type_embedding_rows = None;
    assert!(profile.admit(&drifted).is_err());
}

#[test]
fn backend_limits_check_the_request_envelope() {
    let limits = &gate::laya_0_3_20().limits;
    limits.check_request(16, 512).unwrap();
    limits.check_request(1, 16).unwrap();
    for (batch, len) in [(32, 512), (3, 512), (1, 8), (1, 513), (1, 520)] {
        assert!(
            limits.check_request(batch, len).is_err(),
            "accepted batch {batch} len {len}"
        );
    }
}

#[test]
fn structure_identity_is_stable_and_sensitive() {
    let laya = parse_laya(&laya_encoder_config());
    let again = parse_laya(&laya_encoder_config());
    assert_eq!(
        structure_identity(&laya, "laya_cuda", 1),
        structure_identity(&again, "laya_cuda", 1)
    );
    let mut drifted = laya.clone();
    drifted.rope_sliding.theta = 160000.0;
    assert_ne!(
        structure_identity(&laya, "laya_cuda", 1),
        structure_identity(&drifted, "laya_cuda", 1)
    );
    let vela = parse_hf(&vela_encoder_config());
    assert_ne!(
        structure_identity(&laya, "laya_cuda", 1),
        structure_identity(&vela, "laya_cuda", 1)
    );
    assert_ne!(
        structure_identity(&laya, "laya_cuda", 1),
        structure_identity(&laya, "laya_cuda", 2)
    );
    // Provenance is not part of identity: same semantics, same digest.
    let mut reworded = laya.clone();
    reworded.provenance.clear();
    assert_eq!(
        structure_identity(&laya, "laya_cuda", 1),
        structure_identity(&reworded, "laya_cuda", 1)
    );
}

#[test]
fn derived_geometry_matches_the_laya_checkpoint_inventory() {
    let laya = parse_laya(&laya_encoder_config());
    let tensors = geometry::encoder_tensors(&laya);
    assert_eq!(tensors.len(), 171);
    assert_eq!(tensors.len(), 3 + 28 * 5 + 27 + 1);
    let find = |name: &str| tensors.iter().find(|(n, _)| n == name).unwrap();
    assert_eq!(
        find("encoder.embeddings.tok_embeddings.weight").1,
        vec![50368, 1024]
    );
    assert_eq!(find("encoder.layers.0.mlp_norm.weight").1, vec![1024]);
    assert_eq!(
        find("encoder.layers.0.mlp.Wi.weight").1,
        vec![5248, 1024]
    );
    let attn_norm_count = tensors
        .iter()
        .filter(|(name, _)| name.ends_with("attn_norm.weight"))
        .count();
    assert_eq!(attn_norm_count, 27);
    assert_eq!(tensors.last().unwrap().0, "type_emb.weight");
    assert_eq!(tensors.last().unwrap().1, vec![3, 1024]);

    let vela = parse_hf(&vela_encoder_config());
    let vela_tensors = geometry::encoder_tensors(&vela);
    assert_eq!(vela_tensors.len(), 3 + 22 * 5 + 21);
    let find_vela = |name: &str| vela_tensors.iter().find(|(n, _)| n == name).unwrap();
    assert_eq!(
        find_vela("encoder.layers.0.mlp.Wi.weight").1,
        vec![2304, 768]
    );
    assert!(vela_tensors.iter().all(|(name, _)| name != "type_emb.weight"));

    assert_eq!(geometry::rotary_half(&laya), 32);
    assert_eq!(geometry::rope_table_bytes(&laya), 512 * 32 * 4);
}

#[test]
fn encoder_scratch_sizes_follow_the_structure() {
    let laya = parse_laya(&laya_encoder_config());
    let scratch = EncoderScratch::new(1, 512, &laya);
    assert_eq!(
        scratch.sizes(),
        vec![512 * 8, 4, 512 * 1024 * 4, 512 * 1024 * 2, 512 * 3072 * 2, 512 * 1024 * 2, 512 * 2624 * 2]
    );
    let total: usize = scratch.sizes().iter().sum();
    assert_eq!(scratch.total(), total);
    let vela = parse_hf(&vela_encoder_config());
    assert_eq!(
        EncoderScratch::new(1, 512, &vela).qkv,
        512 * 3 * 768 * 2
    );
}
