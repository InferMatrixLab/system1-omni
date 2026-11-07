use omni_vela_native::{Weights, head_tensors, read_structure, storage_dtype};
use serde_json::{Value, json};
use std::fs;
use tempfile::{TempDir, tempdir};

/// The recorded census of the pinned checkpoint's safetensors header
/// (vllm-sr/Vela-2.0-0.3B@d6f03aa9, model.safetensors): every tensor
/// name with dtype and shape, fetched from the first bytes of the file.
fn census() -> Value {
    serde_json::from_str(include_str!("data/pinned-checkpoint-census.json")).unwrap()
}

/// Builds a synthetic checkpoint holding exactly the given tensors.
fn checkpoint(tensors: &Value) -> (TempDir, std::path::PathBuf) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    let mut entries: Vec<(String, Vec<usize>, &str, Vec<u8>)> = Vec::new();
    for (name, spec) in tensors.as_object().unwrap() {
        let shape: Vec<usize> = spec["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let dtype = spec["dtype"].as_str().unwrap();
        let width = if dtype == "F32" { 4 } else { 2 };
        let bytes: Vec<u8> = vec![0; shape.iter().product::<usize>() * width];
        entries.push((name.clone(), shape, dtype, bytes));
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    let mut offsets = 0usize;
    let mut header = serde_json::Map::new();
    let mut blob: Vec<u8> = Vec::new();
    for (name, shape, dtype, bytes) in &entries {
        let end = offsets + bytes.len();
        header.insert(
            name.clone(),
            json!({"dtype": dtype, "shape": shape, "data_offsets": [offsets, end]}),
        );
        offsets = end;
        blob.extend_from_slice(bytes);
    }
    let mut header_text = Value::Object(header).to_string();
    header_text.extend(std::iter::repeat_n(' ', (8 - header_text.len() % 8) % 8));
    let mut file = (header_text.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header_text.as_bytes());
    file.extend_from_slice(&blob);
    fs::write(&path, file).unwrap();
    (dir, path)
}

mod common;

fn structure() -> omni_modernbert::structure::ModernBertStructure {
    let (_keep, path) = common::bundle_dir();
    let (structure, _) = read_structure(&path).unwrap();
    structure
}

#[test]
fn the_geometry_inventory_matches_the_pinned_census_exactly() {
    let expected = omni_modernbert::geometry::encoder_tensors(&structure());
    let heads = head_tensors();
    assert_eq!(expected.len(), 134);
    assert_eq!(heads.len(), 18);
    for (name, shape) in expected.iter().chain(heads.iter()) {
        let entry = &census()[name];
        assert!(entry.is_object(), "{name} missing from the census");
        let census_shape: Vec<usize> = entry["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(&census_shape, shape, "{name}");
        assert_eq!(entry["dtype"], "F32", "{name}");
    }
    // And the census holds nothing beyond these two inventories.
    assert_eq!(census().as_object().unwrap().len(), 134 + 18);
}

#[test]
fn a_checkpoint_matching_the_profile_validates_completely() {
    let (_dir, path) = checkpoint(&census());
    let weights = Weights::open(&path).unwrap();
    assert_eq!(weights.validate(&structure()).unwrap(), 152);
    // Storage classes mirror the Laya baseline: embedding f16, norms,
    // calibration scalars and head biases f32, GEMM-class bf16.
    assert_eq!(
        storage_dtype("encoder.embeddings.tok_embeddings.weight"),
        "f16"
    );
    assert_eq!(storage_dtype("encoder.layers.0.attn_norm.weight"), "f32");
    assert_eq!(storage_dtype("encoder.layers.0.attn.Wqkv.weight"), "bf16");
    assert_eq!(storage_dtype("norm.weight"), "f32");
    assert_eq!(storage_dtype("log_tau"), "f32");
    assert_eq!(storage_dtype("log_tau_span"), "f32");
    assert_eq!(storage_dtype("cls_mlp.3.bias"), "f32");
    assert_eq!(storage_dtype("w_e.bias"), "f32");
    assert_eq!(storage_dtype("cls_mlp.0.weight"), "bf16");
    assert_eq!(storage_dtype("w_o.weight"), "bf16");
    let scalar = weights.f32("log_tau", &[]).unwrap();
    assert_eq!(scalar.len(), 1);
}

#[test]
fn inventory_drift_fails_naming_both_sides() {
    for (name, mutate) in [
        ("missing", {
            let mut c = census();
            c.as_object_mut()
                .unwrap()
                .remove("encoder.layers.5.attn.Wqkv.weight");
            c
        }),
        ("extra", {
            let mut c = census();
            c.as_object_mut().unwrap().insert(
                "encoder.layers.22.attn.Wqkv.weight".into(),
                json!({"dtype": "F32", "shape": [2304, 768]}),
            );
            c
        }),
        ("shape", {
            let mut c = census();
            c["encoder.layers.0.mlp.Wi.weight"]["shape"] = json!([2303, 768]);
            c
        }),
        ("dtype", {
            let mut c = census();
            c["encoder.layers.0.attn.Wo.weight"]["dtype"] = json!("F16");
            c
        }),
    ] {
        let (_dir, path) = checkpoint(&mutate);
        let weights = Weights::open(&path).unwrap();
        let result = weights.validate(&structure());
        assert!(result.is_err(), "accepted {name}");
        let message = result.unwrap_err().to_string();
        match name {
            "missing" => assert!(message.contains("missing"), "{message}"),
            "extra" => assert!(message.contains("unexpected"), "{message}"),
            "shape" => assert!(message.contains("expected"), "{message}"),
            "dtype" => assert!(message.contains("f32"), "{message}"),
            _ => {}
        }
    }
}
