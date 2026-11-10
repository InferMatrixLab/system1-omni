mod common;

use omni_modernbert::plan::{self, EncoderPlanOptions, PlanStep};
use omni_vela_native::{profile, read_structure};
use serde_json::json;
use std::fs;
use tempfile::tempdir;

use tempfile::TempDir;

fn bundle_dir() -> (TempDir, std::path::PathBuf) {
    common::bundle_dir()
}

#[test]
fn the_pinned_bundle_parses_into_the_registered_profile() {
    let (_keep, path) = bundle_dir();
    let (structure, meta) = read_structure(&path).unwrap();
    let identity = profile().admit(&structure).unwrap();
    assert_eq!(identity.backend, "laya_cuda");
    assert_eq!(identity.abi_version, 2);
    assert_eq!(meta.model_type, "modernbert");
    let registered = profile().structure;
    assert_eq!(structure.hidden_size, registered.hidden_size);
    assert_eq!(structure.schedule, registered.schedule);
    assert_eq!(structure.rope_full, registered.rope_full);
    assert_eq!(structure.type_embedding_rows, None);
}

#[test]
fn the_bundle_rejects_foreign_roots_and_encoders() {
    for (name, mutate) in [("root class", "vela2-decoder"), ("encoder class", "bert")] {
        let dir = tempdir().unwrap();
        let mut config = json!({
            "model_type": "vela2-unified",
            "encoder_config": {
                "model_type": "modernbert", "hidden_activation": "gelu",
                "attention_bias": false, "mlp_bias": false, "norm_bias": false,
                "hidden_size": 768, "intermediate_size": 1152,
                "num_attention_heads": 12, "num_hidden_layers": 22,
                "vocab_size": 256008, "norm_eps": 1e-5, "local_attention": 128,
                "global_attn_every_n_layers": 3,
                "global_rope_theta": 160000, "local_rope_theta": 160000,
                "max_position_embeddings": 32768
            }
        });
        match name {
            "root class" => config["model_type"] = json!(mutate),
            _ => config["encoder_config"]["model_type"] = json!(mutate),
        }
        fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
        assert!(read_structure(dir.path()).is_err(), "accepted {name}");
    }
    let dir = tempdir().unwrap();
    fs::write(
        dir.path().join("config.json"),
        json!({"model_type": "vela2-unified"}).to_string(),
    )
    .unwrap();
    assert!(read_structure(dir.path()).is_err());
}

#[test]
fn profile_drift_is_rejected_with_field_diffs() {
    let (_keep, path) = bundle_dir();
    let (mut structure, _) = read_structure(&path).unwrap();
    structure.vocab_size = 50368;
    let message = profile().admit(&structure).unwrap_err().to_string();
    assert!(message.contains("vocab_size 50368"), "{message}");

    let (_keep, path) = bundle_dir();
    let (mut structure, _) = read_structure(&path).unwrap();
    structure.schedule[3] = omni_modernbert::AttentionKind::Sliding;
    assert!(profile().admit(&structure).is_err());

    let (_keep, path) = bundle_dir();
    let (mut structure, _) = read_structure(&path).unwrap();
    structure.rope_full.scaling = None;
    assert!(profile().admit(&structure).is_err());
}

#[test]
fn the_encoder_plan_runs_the_profile_schedule_without_a_type_add() {
    let (_keep, path) = bundle_dir();
    let (structure, _) = read_structure(&path).unwrap();
    let steps = plan::encoder_plan(&structure, &EncoderPlanOptions::default());
    let launches = steps
        .iter()
        .filter(|s| matches!(s, PlanStep::Launch { .. }))
        .count();
    assert_eq!(launches, 1 + 22 * 8);
    assert!(!steps.iter().any(|s| s.render().contains("type_emb")));
    let limits = &profile().limits;
    limits.check_request(1, 8192).unwrap();
    limits.check_request(4, 512).unwrap();
    for (batch, len) in [(8, 512), (1, 8208), (1, 8190)] {
        assert!(
            limits.check_request(batch, len).is_err(),
            "accepted batch {batch} len {len}"
        );
    }
}
