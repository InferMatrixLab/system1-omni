//! CPU tests for the configuration-driven vision layout (spec 03, N1).
//!
//! The shared vision tower keeps every dimension the CUDA kernels implement
//! pinned to the Qwen3.5 vision layout, except the merger output width, which
//! now follows `vision_config.out_hidden_size` and must equal the text hidden
//! size: 2560 for the Cua-S1 4B default path, 2048 for the Valen 2B export.
use super::*;
use serde_json::{Value, json};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

static SEQUENCE: AtomicUsize = AtomicUsize::new(0);

fn config(out_hidden_size: u64, text_hidden_size: u64) -> Value {
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

fn dir_with(config: &Value) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "cua_s1_vision_cfg_{}_{}",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("config.json"), serde_json::to_vec(config).unwrap()).unwrap();
    dir
}

/// The Cua-S1 4B layout must keep loading unchanged after the unpinning.
#[test]
fn four_b_out_width_loads() {
    let config = VisionConfig::load(&dir_with(&config(2560, 2560))).unwrap();
    assert_eq!(config.out_hidden_size, 2560);
    let inventory = config.inventory();
    assert_eq!(
        inventory["model.visual.merger.linear_fc2.weight"],
        &[2560_usize, 4096]
    );
    assert_eq!(
        inventory["model.visual.patch_embed.proj.weight"],
        &[1024_usize, 3, 2, 16, 16]
    );
}

/// The Valen 2B layout differs only in the merger output width (2048).
#[test]
fn two_b_out_width_loads() {
    let config = VisionConfig::load(&dir_with(&config(2048, 2048))).unwrap();
    assert_eq!(config.out_hidden_size, 2048);
    let inventory = config.inventory();
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
        inventory["model.visual.blocks.23.mlp.linear_fc2.weight"],
        &[1024_usize, 4096]
    );
}

/// The merger output width must equal the text hidden size it feeds.
#[test]
fn text_hidden_mismatch_is_rejected() {
    assert!(VisionConfig::load(&dir_with(&config(2048, 2560))).is_err());
    assert!(VisionConfig::load(&dir_with(&config(2560, 2048))).is_err());
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
    ];
    for (field, value) in cases {
        let mut config = config(2048, 2048);
        config["vision_config"][field] = value;
        assert!(
            VisionConfig::load(&dir_with(&config)).is_err(),
            "{field} must stay pinned"
        );
    }
}
