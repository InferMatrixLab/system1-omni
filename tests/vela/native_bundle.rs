//! Bundle-gate tests: a synthetic bundle built in a unique temp
//! directory (no weights, no network) drives the manifest gate — pins,
//! inventory counts, per-file hashes and the manifest SHA-256 anchor.

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

/// The SHA-256 of the manifest as it currently sits on disk — the
/// anchor a loader must pin from outside the bundle directory.
fn manifest_sha(dir: &std::path::Path) -> String {
    sha256(&std::fs::read(dir.join("vela_native_manifest.json")).expect("manifest"))
}

/// Rewrite the manifest through one JSON mutation and re-serialize it
/// (the same shape an exporter or a tamperer would produce).
fn edit_manifest(dir: &std::path::Path, edit: impl FnOnce(&mut serde_json::Value)) {
    let manifest_path = dir.join("vela_native_manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).expect("read")).expect("json");
    edit(&mut manifest);
    std::fs::write(&manifest_path, serde_json::to_vec(&manifest).expect("json")).expect("write");
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
    let loaded = Bundle::load(&dir, &manifest_sha(&dir)).expect("loads");
    assert_eq!(loaded.model_name(), "vllm-sr/Vela-2.0-0.8B");
    assert_eq!(loaded.max_length(), 16384);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_tampered_file_fails_its_hash() {
    let dir = temp_dir("tampered");
    write_bundle(&dir);
    std::fs::write(dir.join("calibration.json"), b"{\"tampered\": true}").expect("write");
    let error = Bundle::load(&dir, &manifest_sha(&dir)).unwrap_err();
    assert!(error.to_string().contains("calibration.json"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_rewritten_manifest_fails_the_anchor() {
    // Re-hashing a swapped file and rewriting files_sha256 used to pass
    // the gate: the manifest now has to match an anchor fixed outside
    // the bundle directory, so the rewrite itself is refused.
    let dir = temp_dir("rewritten");
    write_bundle(&dir);
    let anchor = manifest_sha(&dir);
    edit_manifest(&dir, |manifest| {
        manifest["files_sha256"]["calibration.json"] =
            json!(sha256(b"{\"swapped\": true}").as_str());
    });
    std::fs::write(dir.join("calibration.json"), b"{\"swapped\": true}").expect("write");
    let error = Bundle::load(&dir, &anchor).unwrap_err();
    assert!(
        error.to_string().contains("pinned SHA-256 anchor"),
        "{error}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_foreign_revision_is_refused() {
    let dir = temp_dir("foreign");
    write_bundle(&dir);
    edit_manifest(&dir, |manifest| {
        manifest["source"]["revision"] = json!("d692394127f6f87fc581d8634c9558fe4776efce");
    });
    let error = Bundle::load(&dir, &manifest_sha(&dir)).unwrap_err();
    assert!(error.to_string().contains("pinned"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_unexpected_inventory_is_refused() {
    let dir = temp_dir("inventory");
    write_bundle(&dir);
    edit_manifest(&dir, |manifest| {
        manifest["backbone"]["bf16"] = json!(320);
        manifest["backbone"]["fp32"] = json!(0);
    });
    let error = Bundle::load(&dir, &manifest_sha(&dir)).unwrap_err();
    assert!(error.to_string().contains("inventory"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_extra_unlisted_file_fails_the_gate() {
    let dir = temp_dir("extra");
    write_bundle(&dir);
    std::fs::write(dir.join("extra.bin"), b"not in the manifest").expect("write");
    let error = Bundle::load(&dir, &manifest_sha(&dir)).unwrap_err();
    assert!(error.to_string().contains("extra.bin"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_absolute_files_key_is_refused() {
    let dir = temp_dir("absolute-key");
    write_bundle(&dir);
    edit_manifest(&dir, |manifest| {
        manifest["files_sha256"]["/etc/passwd"] = json!(sha256(b""));
    });
    let error = Bundle::load(&dir, &manifest_sha(&dir)).unwrap_err();
    assert!(error.to_string().contains("plain relative name"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_parent_escaping_files_key_is_refused() {
    let dir = temp_dir("parent-key");
    write_bundle(&dir);
    edit_manifest(&dir, |manifest| {
        manifest["files_sha256"]["../outside.bin"] = json!(sha256(b""));
    });
    let error = Bundle::load(&dir, &manifest_sha(&dir)).unwrap_err();
    assert!(error.to_string().contains("plain relative name"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_symlinked_files_key_is_refused() {
    let dir = temp_dir("symlink-key");
    write_bundle(&dir);
    let outside = temp_dir("symlink-outside");
    std::fs::write(outside.join("secret.bin"), b"outside the bundle").expect("write");
    std::os::unix::fs::symlink(outside.join("secret.bin"), dir.join("linked.safetensors"))
        .expect("symlink");
    edit_manifest(&dir, |manifest| {
        manifest["files_sha256"]["linked.safetensors"] = json!(sha256(b"outside the bundle"));
    });
    let error = Bundle::load(&dir, &manifest_sha(&dir)).unwrap_err();
    assert!(error.to_string().contains("symlink"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&outside);
}

#[test]
fn an_empty_files_map_fails_the_gate() {
    let dir = temp_dir("empty-files");
    write_bundle(&dir);
    edit_manifest(&dir, |manifest| {
        manifest["files_sha256"] = json!({});
    });
    let error = Bundle::load(&dir, &manifest_sha(&dir)).unwrap_err();
    assert!(error.to_string().contains("files_sha256"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_missing_manifest_names_the_export_step() {
    let dir = temp_dir("empty");
    std::fs::create_dir_all(&dir).expect("tempdir");
    let error = Bundle::load(&dir, &sha256(b"any anchor")).unwrap_err();
    assert!(error.to_string().contains("export_native.py"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}
