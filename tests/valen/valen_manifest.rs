//! CPU tests for the Valen export-manifest vision checks (spec 03, N1;
//! ported to the shared Qwen vision executor).
//!
//! The pure manifest validation runs without weights: it pins the export
//! format, the checkpoint/base provenance and the trained vision-top +
//! merger replacement records against the configuration-derived inventory.
//! The layout cases replace the old cua_s1-side unpinning tests: the shared
//! `VisionConfig` whitelist accepts the 4B (2560) and Valen 2B (2048) merger
//! widths on the width-1024 kernel path and rejects every other dimension
//! drift.
use super::*;
use serde_json::{Value, json};
use std::collections::BTreeMap;

fn two_b_config() -> VisionConfig {
    VisionConfig {
        depth: 24,
        hidden_size: 1024,
        intermediate_size: 4096,
        num_heads: 16,
        num_position_embeddings: 2304,
        out_hidden_size: 2048,
        in_channels: 3,
        patch_size: 16,
        temporal_patch_size: 2,
        spatial_merge_size: 2,
        hidden_act: "gelu_pytorch_tanh".into(),
        deepstack_visual_indexes: Vec::new(),
        model_type: "qwen3_5".into(),
    }
}

/// Build a manifest whose visual tensor records mirror the real export's:
/// every inventory tensor present, blocks 20-23 and the merger marked as
/// checkpoint-sourced replacements, the rest base-sourced, plus non-visual
/// records that must be ignored.
fn export_manifest() -> Value {
    let config = two_b_config();
    let mut tensors = Vec::new();
    for (name, shape) in config.inventory() {
        let trained = name.contains(".blocks.20.")
            || name.contains(".blocks.21.")
            || name.contains(".blocks.22.")
            || name.contains(".blocks.23.")
            || name.contains(".merger.");
        tensors.push(json!({
            "component": if name.contains(".merger.") {
                "merger"
            } else if trained {
                "vision-top"
            } else {
                "backbone"
            },
            "dtype": "bfloat16",
            "file": "model.safetensors",
            "name": name,
            "shape": shape,
            "source": if trained { "checkpoint" } else { "base" },
        }));
    }
    tensors.push(json!({
        "component": "lora",
        "dtype": "float32",
        "file": "lora.safetensors",
        "name": "backbone.language_model.base_model.model.layers.0.mlp.gate_proj.lora_A.default.weight",
        "shape": [32, 2048],
        "source": "checkpoint",
    }));
    json!({
        "format": "valen-checkpoint-export/2",
        "files": {
            "config.json": {"sha256": "0".repeat(64), "size": 1},
            "model.safetensors": {"sha256": "0".repeat(64), "size": 1},
        },
        "sources": {
            "base": {"repo": "Qwen/Qwen3.5-2B", "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc"},
            "checkpoint": {
                "revision": "81b9c6396ded3a45a92ae8079af85f988aa40501",
                "sha256": "836622efe78fe757e2627aa6050c223d461c424ea430a1c110c15e6f42f5a012",
                "stage": "vision_top",
            },
        },
        "tensors": tensors,
    })
}

#[test]
fn valid_manifest_passes() {
    validate_manifest(&export_manifest(), &two_b_config()).unwrap();
}

#[test]
fn format_and_provenance_pins_are_enforced() {
    let cases = [
        |m: &mut Value| m["format"] = json!("valen-checkpoint-export/1"),
        |m: &mut Value| m["sources"]["checkpoint"]["revision"] = json!("deadbeef"),
        |m: &mut Value| m["sources"]["checkpoint"]["sha256"] = json!("0".repeat(64)),
        |m: &mut Value| m["sources"]["checkpoint"]["stage"] = json!("joint"),
        |m: &mut Value| m["sources"]["base"]["repo"] = json!("Qwen/Qwen3.5-4B"),
        |m: &mut Value| m["sources"]["base"]["revision"] = json!("0".repeat(40)),
    ];
    for corrupt in cases {
        let mut value = export_manifest();
        corrupt(&mut value);
        assert!(validate_manifest(&value, &two_b_config()).is_err());
    }
}

#[test]
fn trained_replacements_must_be_recorded() {
    // A trained tensor relabeled as a base passthrough must fail.
    let mut value = export_manifest();
    let records = value["tensors"].as_array_mut().unwrap();
    let record = records
        .iter_mut()
        .find(|r| r["name"] == "model.visual.blocks.20.attn.qkv.weight")
        .unwrap();
    record["source"] = json!("base");
    record["component"] = json!("backbone");
    assert!(validate_manifest(&value, &two_b_config()).is_err());

    // A base tensor relabeled as trained must fail too.
    let mut value = export_manifest();
    let records = value["tensors"].as_array_mut().unwrap();
    let record = records
        .iter_mut()
        .find(|r| r["name"] == "model.visual.blocks.19.attn.qkv.weight")
        .unwrap();
    record["source"] = json!("checkpoint");
    record["component"] = json!("vision-top");
    assert!(validate_manifest(&value, &two_b_config()).is_err());
}

#[test]
fn visual_records_must_match_the_inventory() {
    let cases = [
        // A visual tensor with the wrong shape.
        |records: &mut Vec<Value>| {
            records
                .iter_mut()
                .find(|r| r["name"] == "model.visual.merger.linear_fc2.weight")
                .unwrap()["shape"] = json!([2560, 4096]);
        },
        // The wrong dtype.
        |records: &mut Vec<Value>| {
            records
                .iter_mut()
                .find(|r| r["name"] == "model.visual.blocks.0.norm1.weight")
                .unwrap()["dtype"] = json!("float32");
        },
        // Weights claimed from a different file.
        |records: &mut Vec<Value>| {
            records
                .iter_mut()
                .find(|r| r["name"] == "model.visual.blocks.0.attn.qkv.weight")
                .unwrap()["file"] = json!("lora.safetensors");
        },
        // A missing visual record.
        |records: &mut Vec<Value>| {
            records.retain(|r| r["name"] != "model.visual.pos_embed.weight");
        },
        // An unexpected extra visual record.
        |records: &mut Vec<Value>| {
            records.push(json!({
                "component": "backbone",
                "dtype": "bfloat16",
                "file": "model.safetensors",
                "name": "model.visual.blocks.24.norm1.weight",
                "shape": [1024],
                "source": "base",
            }));
        },
    ];
    for corrupt in cases {
        let mut value = export_manifest();
        let mut owned = value["tensors"].as_array().unwrap().clone();
        corrupt(&mut owned);
        value["tensors"] = Value::Array(owned);
        assert!(validate_manifest(&value, &two_b_config()).is_err());
    }
}

#[test]
fn trained_inventory_covers_last_four_blocks_and_merger() {
    let trained = trained_tensor_names(&two_b_config());
    assert_eq!(trained.len(), 48 + 6);
    assert!(trained.contains("model.visual.blocks.20.mlp.linear_fc2.bias"));
    assert!(trained.contains("model.visual.blocks.23.norm1.weight"));
    assert!(trained.contains("model.visual.merger.linear_fc2.weight"));
    assert!(!trained.contains("model.visual.blocks.19.attn.qkv.weight"));
    assert!(trained.contains("model.visual.merger.norm.weight"));
    // Every trained name is part of the config inventory.
    let inventory: BTreeMap<String, Vec<usize>> = two_b_config().inventory();
    assert!(trained.iter().all(|name| inventory.contains_key(name)));
}

fn model_config(out_hidden_size: u64, text_hidden_size: u64) -> Value {
    json!({
        "model_type": "qwen3_5",
        "text_config": {"hidden_size": text_hidden_size},
        "vision_config": {
            "deepstack_visual_indexes": [],
            "depth": 24,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 1024,
            "in_channels": 3,
            "intermediate_size": 4096,
            "model_type": "qwen3_5",
            "num_heads": 16,
            "num_position_embeddings": 2304,
            "out_hidden_size": out_hidden_size,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2
        }
    })
}

/// The shared layout accepts the Valen 2B merger width and keeps the 4B path.
#[test]
fn valen_and_cua_merger_widths_load() {
    let valen = export_vision_config(&model_config(2048, 2048)).unwrap();
    assert_eq!(valen.out_hidden_size, 2048);
    let inventory = valen.inventory();
    assert_eq!(
        inventory["model.visual.merger.linear_fc2.weight"],
        &[2048_usize, 4096]
    );
    assert_eq!(
        inventory["model.visual.merger.linear_fc2.bias"],
        &[2048_usize]
    );
    assert_eq!(
        inventory["model.visual.merger.linear_fc1.weight"],
        &[4096_usize, 4096]
    );
    assert_eq!(
        inventory["model.visual.patch_embed.proj.weight"],
        &[1024_usize, 3, 2, 16, 16]
    );
    let cua = export_vision_config(&model_config(2560, 2560)).unwrap();
    assert_eq!(cua.out_hidden_size, 2560);
    assert_eq!(
        cua.inventory()["model.visual.merger.linear_fc2.weight"],
        &[2560_usize, 4096]
    );
}

/// The merger output width must equal the text hidden size it feeds.
#[test]
fn text_hidden_mismatch_is_rejected() {
    assert!(export_vision_config(&model_config(2048, 2560)).is_err());
    assert!(export_vision_config(&model_config(2560, 2048)).is_err());
}

/// Every other vision dimension stays pinned to the layout the kernels
/// implement; changing any of them must stay a load error.
#[test]
fn shared_layout_dimensions_stay_pinned() {
    let cases = [
        ("depth", json!(23)),
        ("hidden_size", json!(512)),
        ("intermediate_size", json!(2048)),
        ("num_heads", json!(8)),
        ("num_position_embeddings", json!(1152)),
        ("in_channels", json!(1)),
        ("patch_size", json!(14)),
        ("temporal_patch_size", json!(1)),
        ("spatial_merge_size", json!(4)),
        ("out_hidden_size", json!(1024)),
    ];
    for (field, value) in cases {
        let mut config = model_config(2048, 2048);
        config["vision_config"][field] = value;
        assert!(
            VisionConfig::from_value(config["vision_config"].clone()).is_err()
                || export_vision_config(&config).is_err(),
            "{field} must stay pinned"
        );
    }
}
