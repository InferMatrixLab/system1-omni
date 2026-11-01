use omni_kev_native::checkpoint::{PINS, validate_manifest};
use serde_json::{Value, json};

fn manifest() -> Value {
    let mut value: Value = serde_json::from_str(PINS).unwrap();
    let mut files = serde_json::Map::new();
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
        files.insert(name.into(), value["inputs"]["base"][name].clone());
    }
    for name in [
        "head.safetensors",
        "model.safetensors.index.json",
        "model.safetensors-00001-of-00002.safetensors",
        "model.safetensors-00002-of-00002.safetensors",
    ] {
        files.insert(name.into(), json!("0".repeat(64)));
    }
    files.insert(
        "head.safetensors".into(),
        json!("65698f2568fc1b6ebb60a1c2a24c7bb6622dcc23e1c6d9efe8d3274ad918a9a7"),
    );
    value["files"] = Value::Object(files);
    value
}
#[test]
fn pinned_metadata_and_complete_inventory_are_required() {
    let good = manifest();
    validate_manifest(&good).unwrap();
    for key in [
        "revision",
        "base_revision",
        "reference_revision",
        "temperature",
        "special_tokens",
        "head_dtype",
        "hidden_size",
        "merged_targets",
        "inputs",
        "merge_numerics",
    ] {
        let mut bad = good.clone();
        bad[key] = Value::Null;
        assert!(validate_manifest(&bad).is_err(), "{key}");
    }
    let mut bad = good.clone();
    bad["files"]
        .as_object_mut()
        .unwrap()
        .remove("head.safetensors");
    assert!(validate_manifest(&bad).is_err());
    let mut bad = good;
    bad["files"]["../other"] = json!("0".repeat(64));
    assert!(validate_manifest(&bad).is_err());
}

#[test]
fn corrupted_export_fails_before_device_loading() {
    let root = std::env::temp_dir().join(format!("kev-bad-export-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    std::fs::write(
        root.join("kev_export.json"),
        serde_json::to_vec(&manifest()).unwrap(),
    )
    .unwrap();
    std::fs::write(root.join("config.json"), b"{}").unwrap();
    let error = omni_kev_native::checkpoint::verify(&root)
        .unwrap_err()
        .to_string();
    std::fs::remove_dir_all(&root).unwrap();
    assert!(
        error.contains("export hash mismatch: config.json"),
        "{error}"
    );
}
