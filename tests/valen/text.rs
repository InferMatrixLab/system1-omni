//! The unmerged Valen text path (spec 03, N2): export-policy validation and
//! the loading seam over the shared #143 machinery.
//!
//! CPU-only tests build synthetic export directories and pin the validation
//! contract (manifest format, LoRA profile, adapter bytes, the full inventory)
//! before any CUDA loading; the device-gated parity tests live beside the
//! recorded fixtures and are `#[ignore]`d on the environment.
use omni_valen_native::text;
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "valen-text-export-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

/// The miniature configuration: the Valen 2B layer mix (full attention every
/// fourth layer) over four layers, with the shared loader's pinned kernel
/// dimensions (head 256 / linear 128) and the tiny widths the CPU tests use.
fn write_config(dir: &Path) {
    let config = json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "image_token_id": 248056,
        "model_type": "qwen3_5",
        "text_config": {
            "attention_bias": false,
            "attn_output_gate": true,
            "head_dim": 256,
            "hidden_act": "silu",
            "hidden_size": 8,
            "intermediate_size": 8,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_value_head_dim": 128,
            "max_position_embeddings": 512,
            "num_attention_heads": 1,
            "num_hidden_layers": 4,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10],
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25
            }
        }
    });
    fs::write(dir.join("config.json"), config.to_string()).unwrap();
}

/// One paired FP32 tensor per inventory target, in the pinned peft key layout,
/// with the manifest's rank-32 shapes. `complete = false` keeps only a
/// foreign pair (layer 9) instead of the inventory.
fn write_adapter(dir: &Path, variant: Adapter) -> u64 {
    let prefix = "backbone.language_model.base_model.model.";
    let mut values: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    let target = |values: &mut Vec<(String, Vec<usize>, Vec<u8>)>, name: String, rows: usize, cols: usize| {
        values.push((
            format!("{prefix}{name}.lora_A.default.weight"),
            vec![32, cols],
            vec![0u8; 32 * cols * 4],
        ));
        values.push((
            format!("{prefix}{name}.lora_B.default.weight"),
            vec![rows, 32],
            vec![0u8; rows * 32 * 4],
        ));
    };
    match variant {
        Adapter::Complete => {
            for layer in 0..4 {
                if layer == 3 {
                    target(&mut values, format!("layers.{layer}.self_attn.q_proj"), 512, 8);
                    target(&mut values, format!("layers.{layer}.self_attn.k_proj"), 256, 8);
                    target(&mut values, format!("layers.{layer}.self_attn.v_proj"), 256, 8);
                    target(&mut values, format!("layers.{layer}.self_attn.o_proj"), 8, 256);
                } else {
                    target(&mut values, format!("layers.{layer}.linear_attn.in_proj_qkv"), 384, 8);
                    target(&mut values, format!("layers.{layer}.linear_attn.in_proj_z"), 128, 8);
                    target(&mut values, format!("layers.{layer}.linear_attn.in_proj_b"), 1, 8);
                    target(&mut values, format!("layers.{layer}.linear_attn.in_proj_a"), 1, 8);
                    target(&mut values, format!("layers.{layer}.linear_attn.out_proj"), 8, 128);
                }
                target(&mut values, format!("layers.{layer}.mlp.gate_proj"), 8, 8);
                target(&mut values, format!("layers.{layer}.mlp.up_proj"), 8, 8);
                target(&mut values, format!("layers.{layer}.mlp.down_proj"), 8, 8);
            }
        }
        Adapter::MissingPair => {
            for (name, rows, cols) in inventory() {
                if name != "layers.3.mlp.down_proj" {
                    target(&mut values, name.to_string(), rows, cols);
                }
            }
        }
        Adapter::ForeignTarget => {
            target(&mut values, "layers.9.mlp.down_proj".into(), 8, 8);
        }
    }
    let bytes = safetensors_bytes(&values);
    fs::write(dir.join("lora.safetensors"), &bytes).unwrap();
    bytes.len() as u64
}

/// (target name, rows, cols) for the miniature census: 31 targets.
fn inventory() -> Vec<(String, usize, usize)> {
    let mut entries = Vec::new();
    for layer in 0..4 {
        if layer == 3 {
            for (name, rows, cols) in [
                ("self_attn.q_proj", 512usize, 8usize),
                ("self_attn.k_proj", 256, 8),
                ("self_attn.v_proj", 256, 8),
                ("self_attn.o_proj", 8, 256),
            ] {
                entries.push((format!("layers.{layer}.{name}"), rows, cols));
            }
        } else {
            for (name, rows, cols) in [
                ("linear_attn.in_proj_qkv", 384usize, 8usize),
                ("linear_attn.in_proj_z", 128, 8),
                ("linear_attn.in_proj_b", 1, 8),
                ("linear_attn.in_proj_a", 1, 8),
                ("linear_attn.out_proj", 8, 128),
            ] {
                entries.push((format!("layers.{layer}.{name}"), rows, cols));
            }
        }
        for (name, rows, cols) in [
            ("mlp.gate_proj", 8usize, 8usize),
            ("mlp.up_proj", 8, 8),
            ("mlp.down_proj", 8, 8),
        ] {
            entries.push((format!("layers.{layer}.{name}"), rows, cols));
        }
    }
    entries
}

enum Adapter {
    Complete,
    MissingPair,
    ForeignTarget,
}

fn write_manifest(dir: &Path, patch: Option<Value>) {
    let adapter = fs::read(dir.join("lora.safetensors")).unwrap();
    let config = fs::read(dir.join("config.json")).unwrap();
    let mut manifest = json!({
        "format": "valen-checkpoint-export/2",
        "files": {
            "config.json": {
                "size": config.len(),
                "sha256": sha256(&config),
            },
            "lora.safetensors": {
                "size": adapter.len(),
                "sha256": sha256(&adapter),
            }
        },
        "lora": {
            "alpha": 64,
            "dtype": "float32",
            "file": "lora.safetensors",
            "keys": 62,
            "rank": 32,
            "scaling": 2.0,
            "targets": 31
        }
    });
    if let Some(patch) = patch {
        merge(&mut manifest, &patch);
    }
    fs::write(dir.join("valen_export.json"), manifest.to_string()).unwrap();
}

fn sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

/// Minimal safetensors serializer (header JSON + payload, little-endian FP32).
fn safetensors_bytes(values: &[(String, Vec<usize>, Vec<u8>)]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    for (name, shape, data) in values {
        let begin = payload.len();
        payload.extend_from_slice(data);
        header.insert(
            name.clone(),
            json!({"dtype": "F32", "shape": shape, "data_offsets": [begin, payload.len()]}),
        );
    }
    let header = serde_json::to_string(&header).unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&payload);
    bytes
}

fn merge(target: &mut Value, patch: &Value) {
    match (target, patch) {
        (Value::Object(t), Value::Object(p)) => {
            for (key, value) in p {
                merge(t.entry(key.clone()).or_insert(Value::Null), value);
            }
        }
        (target, patch) => *target = patch.clone(),
    }
}

fn export(variant: Adapter, patch: Option<Value>) -> Directory {
    let dir = Directory::new();
    write_config(&dir.0);
    write_adapter(&dir.0, variant);
    write_manifest(&dir.0, patch);
    dir
}

#[test]
fn valid_synthetic_export_validates_and_reports_the_inventory() {
    let dir = export(Adapter::Complete, None);
    let policy = text::validate_export(&dir.0).unwrap();
    assert_eq!(policy.adapter_file, Path::new("lora.safetensors"));
    assert_eq!(policy.scale, 2.0);
    assert_eq!(policy.targets, 31);
    assert_eq!(policy.keys, 62);
}

#[test]
fn unknown_manifest_format_is_rejected() {
    let dir = export(Adapter::Complete, Some(json!({"format": "valen-checkpoint-export/1"})));
    let error = text::validate_export(&dir.0).unwrap_err();
    assert!(error.to_string().contains("unsupported export"), "{error:#}");
}

#[test]
fn the_lora_profile_is_pinned_before_any_loading() {
    for patch in [
        json!({"lora": {"scaling": 1.0}}),
        json!({"lora": {"alpha": 32}}),
        json!({"lora": {"rank": 16}}),
        json!({"lora": {"dtype": "bfloat16"}}),
        json!({"lora": {"file": "other.safetensors"}}),
        json!({"lora": {"keys": 60}}),
        json!({"lora": {"targets": 30}}),
    ] {
        let dir = export(Adapter::Complete, Some(patch));
        let error = text::validate_export(&dir.0).unwrap_err();
        assert!(
            error.to_string().contains("LoRA profile"),
            "{}",
            error.to_string()
        );
    }
}

#[test]
fn tampered_adapter_bytes_are_rejected_by_the_manifest_hash() {
    let dir = export(Adapter::Complete, None);
    let adapter = fs::read(dir.0.join("lora.safetensors")).unwrap();
    let mut tampered = adapter;
    tampered[100] ^= 0xff;
    fs::write(dir.0.join("lora.safetensors"), &tampered).unwrap();
    let error = text::validate_export(&dir.0).unwrap_err();
    assert!(error.to_string().contains("checksum mismatch"), "{error:#}");
}

#[test]
fn a_partial_adapter_is_outside_the_export_contract() {
    // One pair short of the full inventory: the shared loader would accept it,
    // but the pinned export must carry every declared target.
    let dir = export(Adapter::MissingPair, None);
    let error = text::validate_export(&dir.0).unwrap_err();
    assert!(
        error.to_string().contains("inventory"),
        "{}",
        error.to_string()
    );
}

#[test]
fn foreign_adapter_targets_are_rejected_by_the_inventory_check() {
    let dir = export(Adapter::ForeignTarget, None);
    let error = text::validate_export(&dir.0).unwrap_err();
    assert!(
        error.to_string().contains("inventory"),
        "{}",
        error.to_string()
    );
}

#[test]
fn manifest_validation_precedes_cuda_loading() {
    let dir = export(Adapter::Complete, None);
    // The manifest passes; the CUDA library path does not exist. The failure
    // must be the library's, proving validation ran first and passed.
    let error = text::TextModel::load(&dir.0, Path::new("/missing/valen-cuda-library"))
        .err()
        .unwrap();
    let message = format!("{error:#}");
    assert!(
        message.contains("valen-cuda-library") || message.contains("library"),
        "{message}"
    );
    assert!(
        !message.contains("unsupported export"),
        "validation must pass before CUDA loading: {message}"
    );
}
