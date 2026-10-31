//! Bundle-gate tests: a synthetic bundle built in a unique temp
//! directory (no weights, no network) drives the manifest gate — pins,
//! inventory counts and per-file hashes.

use omni_vela_native::bundle::{self, Bundle};
use serde_json::json;

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir =
        std::env::temp_dir().join(format!("vela-native-bundle-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("tempdir");
    dir
}

fn sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::new_with_prefix(bytes).finalize();
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// The minimal manifest of a correctly exported member; `files` are the
/// bundle files the gate hashes.
fn write_bundle(dir: &std::path::Path) {
    let files: [(&str, &[u8]); 6] = [
        ("backbone.safetensors", b"backbone tensors"),
        ("heads.safetensors", b"head tensors"),
        ("config.json", b"{}"),
        ("calibration.json", b"{}"),
        ("tokenizer.json", b"{}"),
        ("tokenizer_config.json", b"{}"),
    ];
    let mut hashes = serde_json::Map::new();
    for (name, content) in files {
        std::fs::write(dir.join(name), content).expect("write");
        hashes.insert(name.to_string(), json!(sha256(content)));
    }
    let manifest = json!({
        "format": bundle::FORMAT,
        "model_id": bundle::MODEL_ID,
        "model_name": "vllm-sr/Vela-2.0-0.8B",
        "max_length": 16384,
        "source": {"repo": "vllm-sr/Vela-2.0-0.8B", "revision": bundle::SOURCE_REVISION},
        "backbone": {"tensors": 320, "bf16": 204, "fp32": 116},
        "heads": {"tensors": 37},
        "files_sha256": hashes,
    });
    std::fs::write(
        dir.join("vela_native_manifest.json"),
        serde_json::to_vec(&manifest).expect("manifest"),
    )
    .expect("write");
}

#[test]
fn a_matching_bundle_loads_and_exposes_the_member() {
    let dir = temp_dir("ok");
    write_bundle(&dir);
    let loaded = Bundle::load(&dir).expect("loads");
    assert_eq!(loaded.model_name(), "vllm-sr/Vela-2.0-0.8B");
    assert_eq!(loaded.max_length(), 16384);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_tampered_file_fails_its_hash() {
    let dir = temp_dir("tampered");
    write_bundle(&dir);
    std::fs::write(dir.join("calibration.json"), b"{\"tampered\": true}").expect("write");
    let error = Bundle::load(&dir).unwrap_err();
    assert!(error.to_string().contains("calibration.json"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_foreign_revision_is_refused() {
    let dir = temp_dir("foreign");
    write_bundle(&dir);
    let manifest_path = dir.join("vela_native_manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).expect("read")).expect("json");
    manifest["source"]["revision"] = json!("d692394127f6f87fc581d8634c9558fe4776efce");
    std::fs::write(&manifest_path, serde_json::to_vec(&manifest).expect("json")).expect("write");
    let error = Bundle::load(&dir).unwrap_err();
    assert!(error.to_string().contains("pinned"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_unexpected_inventory_is_refused() {
    let dir = temp_dir("inventory");
    write_bundle(&dir);
    let manifest_path = dir.join("vela_native_manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).expect("read")).expect("json");
    manifest["backbone"]["bf16"] = json!(320);
    manifest["backbone"]["fp32"] = json!(0);
    std::fs::write(&manifest_path, serde_json::to_vec(&manifest).expect("json")).expect("write");
    let error = Bundle::load(&dir).unwrap_err();
    assert!(error.to_string().contains("inventory"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_extra_unlisted_file_fails_the_gate() {
    let dir = temp_dir("extra");
    write_bundle(&dir);
    std::fs::write(dir.join("extra.bin"), b"not in the manifest").expect("write");
    let error = Bundle::load(&dir).unwrap_err();
    assert!(error.to_string().contains("extra.bin"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_unlisted_file_inside_an_admitted_directory_fails_the_gate() {
    let dir = temp_dir("bf16-extra");
    write_bundle(&dir);
    let manifest_path = dir.join("vela_native_manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).expect("read")).expect("json");
    std::fs::create_dir(dir.join("bf16")).expect("mkdir");
    std::fs::write(dir.join("bf16/model.safetensors"), b"bf16 tensors").expect("write");
    std::fs::write(dir.join("bf16/extra.bin"), b"unlisted").expect("write");
    manifest["files_sha256"]["bf16/model.safetensors"] = json!(sha256(b"bf16 tensors"));
    std::fs::write(&manifest_path, serde_json::to_vec(&manifest).expect("json")).expect("write");
    let error = Bundle::load(&dir).unwrap_err();
    assert!(error.to_string().contains("bf16/extra.bin"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_empty_files_map_fails_the_gate() {
    let dir = temp_dir("empty-files");
    write_bundle(&dir);
    let manifest_path = dir.join("vela_native_manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).expect("read")).expect("json");
    manifest["files_sha256"] = json!({});
    std::fs::write(&manifest_path, serde_json::to_vec(&manifest).expect("json")).expect("write");
    let error = Bundle::load(&dir).unwrap_err();
    assert!(error.to_string().contains("files_sha256"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_missing_manifest_names_the_export_step() {
    let dir = temp_dir("empty");
    std::fs::create_dir_all(&dir).expect("tempdir");
    let error = Bundle::load(&dir).unwrap_err();
    assert!(error.to_string().contains("export_native.py"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}
