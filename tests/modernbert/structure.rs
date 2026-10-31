use omni_modernbert::adapter::ModernBertConfigAdapter;
use omni_modernbert::gate::{self, structure_identity};
use omni_modernbert::geometry::{self, EncoderScratch};
use omni_modernbert::structure::{
    Activation, AttentionKind, FieldOrigin, ModernBertStructure, RopeSpec,
};
use serde_json::{Value, json};

/// A 1024-hidden family member with an explicit schedule, a type
/// embedding and plain rotary tables — the same geometry the Laya
/// bundle dialect produces, expressed as a stock HF config so the
/// family tests stay model-agnostic.
fn family_config() -> Value {
    json!({
        "model_type": "modernbert", "hidden_activation": "gelu",
        "attention_bias": false, "mlp_bias": false, "norm_bias": false,
        "hidden_size": 1024, "intermediate_size": 2624,
        "num_attention_heads": 16, "num_hidden_layers": 28,
        "vocab_size": 50368, "norm_eps": 0.00001, "local_attention": 128,
        "type_vocab_size": 3, "max_position_embeddings": 512,
        "layer_types": (["full_attention", "sliding_attention", "sliding_attention"]
            .repeat(10))[..28],
        "global_rope_theta": 160000.0, "local_rope_theta": 10000.0
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

fn parse(config: &Value) -> ModernBertStructure {
    ModernBertConfigAdapter::from_value(config, "config.json")
        .unwrap()
        .0
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
    let structure = parse(&config);
    assert!(
        structure
            .schedule
            .iter()
            .all(|k| *k == AttentionKind::Sliding)
    );
    assert!(matches!(
        structure.provenance.get("schedule"),
        Some(FieldOrigin::Source(_))
    ));
}

#[test]
fn hf_adapter_falls_back_to_the_generic_rope_theta() {
    let mut config = family_config();
    let object = config.as_object_mut().unwrap();
    object.remove("global_rope_theta");
    object.remove("local_rope_theta");
    object.insert("rope_theta".into(), json!(500000.0));
    let structure = parse(&config);
    assert_eq!(structure.rope_full.theta, 500000.0);
    assert_eq!(structure.rope_sliding.theta, 500000.0);
    for field in ["rope_full", "rope_sliding"] {
        match structure.provenance.get(field) {
            Some(FieldOrigin::Source(key)) => assert_eq!(key, "config.json#rope_theta"),
            other => panic!("{field} provenance {other:?}"),
        }
    }
}

#[test]
fn hf_adapter_prefers_the_per_kind_theta_over_the_generic_key() {
    let mut config = family_config();
    let object = config.as_object_mut().unwrap();
    object.remove("global_rope_theta");
    object.insert("rope_theta".into(), json!(500000.0));
    let structure = parse(&config);
    assert_eq!(structure.rope_full.theta, 500000.0);
    assert_eq!(structure.rope_sliding.theta, 10000.0);
    assert!(matches!(
        structure.provenance.get("rope_sliding"),
        Some(FieldOrigin::Source(_))
    ));
}

#[test]
fn hf_adapter_reads_type_vocab_and_position_members() {
    let (structure, meta) =
        ModernBertConfigAdapter::from_value(&family_config(), "encoder/config.json").unwrap();
    assert_eq!(structure.hidden_size, 1024);
    assert_eq!(structure.head_dim, 64);
    assert_eq!(structure.type_embedding_rows, Some(3));
    assert_eq!(structure.max_positions, 512);
    assert_eq!(structure.rope_full.theta, 160000.0);
    assert_eq!(structure.rope_sliding.theta, 10000.0);
    assert_eq!(meta.unrecognized_keys, Vec::<String>::new());
    assert_eq!(
        structure.provenance.get("type_embedding_rows"),
        Some(&FieldOrigin::Source(
            "encoder/config.json#type_vocab_size".into()
        ))
    );
    assert_eq!(
        structure.provenance.get("head_dim"),
        Some(&FieldOrigin::Derived(
            "hidden_size / num_attention_heads".into()
        ))
    );
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
            c.as_object_mut()
                .unwrap()
                .remove("global_attn_every_n_layers");
            c
        }),
        ("zero interval", {
            // The derived schedule tests i % every; zero must be an
            // error, not a division-by-zero panic.
            let mut c = vela_encoder_config();
            c["global_attn_every_n_layers"] = json!(0);
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
    let mut structure = parse(&family_config());
    structure.num_attention_heads = 15;
    assert!(structure.validate().unwrap_err().to_string().contains("15"));

    let mut structure = parse(&family_config());
    structure.schedule.pop();
    let message = structure.validate().unwrap_err().to_string();
    assert!(message.contains("schedule"), "{message}");

    for mutate in [
        ("local_window", 0),
        ("intermediate_size", 0),
        ("vocab_size", 0),
        ("num_hidden_layers", 0),
    ] {
        let mut structure = parse(&family_config());
        match mutate.0 {
            "local_window" => structure.local_window = mutate.1,
            "intermediate_size" => structure.intermediate_size = mutate.1,
            "vocab_size" => structure.vocab_size = mutate.1,
            _ => structure.num_hidden_layers = mutate.1,
        }
        assert!(structure.validate().is_err(), "accepted {} = 0", mutate.0);
    }
    let mut structure = parse(&family_config());
    structure.rope_full.theta = 0.0;
    assert!(structure.validate().is_err());
}

/// A test-local registered profile: the gate is family machinery, so
/// its accept/reject set is pinned here against a synthetic profile,
/// independent of any model crate's registered constants.
fn test_profile() -> gate::RegisteredProfile {
    let structure = parse(&family_config());
    gate::RegisteredProfile {
        name: "test-profile",
        structure,
        limits: gate::BackendLimits {
            hidden_sizes: vec![1024],
            batch_max: 16,
            len_min: 16,
            len_max: 512,
            len_step: 16,
            yarn_scaling: false,
        },
        backend: "test_backend",
        abi_version: 1,
    }
}

#[test]
fn admission_accepts_the_registered_structure_and_rejects_drift() {
    let profile = test_profile();
    let structure = parse(&family_config());
    let identity = profile.admit(&structure).unwrap();
    assert_eq!(identity.schema_version, 1);
    assert_eq!(identity.family, "modernbert");
    assert_eq!(identity.backend, "test_backend");
    assert_eq!(identity.abi_version, 1);

    let vela = parse(&vela_encoder_config());
    let message = profile.admit(&vela).unwrap_err().to_string();
    assert!(message.contains("hidden_size 768"), "{message}");
    assert!(message.contains("1024"), "{message}");

    for mutate in [
        ("rope", {
            let mut s = parse(&family_config());
            s.rope_sliding.theta = 160000.0;
            s
        }),
        ("schedule", {
            let mut s = parse(&family_config());
            s.schedule[1] = AttentionKind::Full;
            s
        }),
        ("type_embedding_rows", {
            let mut s = parse(&family_config());
            s.type_embedding_rows = None;
            s
        }),
        ("norm_eps", {
            let mut s = parse(&family_config());
            s.norm_eps = 1e-6;
            s
        }),
        ("max_positions", {
            let mut s = parse(&family_config());
            s.max_positions = 8192;
            s
        }),
    ] {
        assert!(profile.admit(&mutate.1).is_err(), "accepted {}", mutate.0);
    }
}

#[test]
fn backend_limits_check_the_request_envelope() {
    let limits = &test_profile().limits;
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
    let member = parse(&family_config());
    let again = parse(&family_config());
    assert_eq!(
        structure_identity(&member, "laya_cuda", 1),
        structure_identity(&again, "laya_cuda", 1)
    );
    let mut drifted = member.clone();
    drifted.rope_sliding.theta = 160000.0;
    assert_ne!(
        structure_identity(&member, "laya_cuda", 1),
        structure_identity(&drifted, "laya_cuda", 1)
    );
    let vela = parse(&vela_encoder_config());
    assert_ne!(
        structure_identity(&member, "laya_cuda", 1),
        structure_identity(&vela, "laya_cuda", 1)
    );
    assert_ne!(
        structure_identity(&member, "laya_cuda", 1),
        structure_identity(&member, "laya_cuda", 2)
    );
    // Provenance is not part of identity: same semantics, same digest.
    let mut reworded = member.clone();
    reworded.provenance.clear();
    assert_eq!(
        structure_identity(&member, "laya_cuda", 1),
        structure_identity(&reworded, "laya_cuda", 1)
    );
}

#[test]
fn derived_geometry_matches_the_1024_member_inventory() {
    let member = parse(&family_config());
    let tensors = geometry::encoder_tensors(&member);
    assert_eq!(tensors.len(), 171);
    assert_eq!(tensors.len(), 3 + 28 * 5 + 27 + 1);
    let find = |name: &str| tensors.iter().find(|(n, _)| n == name).unwrap();
    assert_eq!(
        find("encoder.embeddings.tok_embeddings.weight").1,
        vec![50368, 1024]
    );
    assert_eq!(find("encoder.layers.0.mlp_norm.weight").1, vec![1024]);
    assert_eq!(find("encoder.layers.0.mlp.Wi.weight").1, vec![5248, 1024]);
    let attn_norm_count = tensors
        .iter()
        .filter(|(name, _)| name.ends_with("attn_norm.weight"))
        .count();
    assert_eq!(attn_norm_count, 27);
    assert_eq!(tensors.last().unwrap().0, "type_emb.weight");
    assert_eq!(tensors.last().unwrap().1, vec![3, 1024]);

    let vela = parse(&vela_encoder_config());
    let vela_tensors = geometry::encoder_tensors(&vela);
    assert_eq!(vela_tensors.len(), 3 + 22 * 5 + 21);
    let find_vela = |name: &str| vela_tensors.iter().find(|(n, _)| n == name).unwrap();
    assert_eq!(
        find_vela("encoder.layers.0.mlp.Wi.weight").1,
        vec![2304, 768]
    );
    assert!(
        vela_tensors
            .iter()
            .all(|(name, _)| name != "type_emb.weight")
    );

    assert_eq!(geometry::rotary_half(&member), 32);
    assert_eq!(geometry::rope_table_bytes(&member), 512 * 32 * 4);
}

#[test]
fn encoder_scratch_sizes_follow_the_structure() {
    let member = parse(&family_config());
    let scratch = EncoderScratch::new(1, 512, &member);
    assert_eq!(
        scratch.sizes(),
        vec![
            512 * 8,
            4,
            512 * 1024 * 4,
            512 * 1024 * 2,
            512 * 3072 * 2,
            512 * 1024 * 2,
            512 * 2624 * 2
        ]
    );
    let total: usize = scratch.sizes().iter().sum();
    assert_eq!(scratch.total(), total);
    let vela = parse(&vela_encoder_config());
    assert_eq!(EncoderScratch::new(1, 512, &vela).qkv, 512 * 3 * 768 * 2);
}

#[test]
fn admitted_structures_carry_the_family_activation() {
    // The frozen-equality check includes the activation; today the
    // family admits gelu only, and both adapters reject anything else
    // at the source, so this pins the field's presence in the IR.
    let member = parse(&family_config());
    assert_eq!(member.activation, Activation::Gelu);
    assert_eq!(
        member.rope_full,
        RopeSpec {
            theta: 160000.0,
            scaling: None
        }
    );
}
