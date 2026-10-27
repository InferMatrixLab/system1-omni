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
/// bundle files the gate hashes. Every member of the table gets one.
fn write_bundle(dir: &std::path::Path, member: &bundle::MemberPins) {
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
        "model_id": member.model_id,
        "model_name": member.model_id,
        "max_length": member.max_length,
        "source": {"repo": member.model_id, "revision": member.revision},
        "backbone": {
            "tensors": member.backbone.0,
            "bf16": member.backbone.1,
            "fp32": member.backbone.2,
        },
        "heads": {"tensors": member.heads},
        "files_sha256": hashes,
    });
    std::fs::write(
        dir.join("vela_native_manifest.json"),
        serde_json::to_vec(&manifest).expect("manifest"),
    )
    .expect("write");
}

fn first_member() -> &'static bundle::MemberPins {
    &bundle::MEMBERS[0]
}

#[test]
fn the_table_pins_the_three_decoder_members() {
    let ids: Vec<&str> = bundle::MEMBERS.iter().map(|m| m.model_id).collect();
    assert_eq!(
        ids,
        vec![
            "vllm-sr/Vela-2.0-0.8B",
            "vllm-sr/Vela-2.0-4B",
            "vllm-sr/Vela-2.0-9B"
        ]
    );
    // 4B and 9B share one layer structure, so their inventories match;
    // the compact member differs.
    assert_eq!(bundle::MEMBERS[0].backbone, (320, 204, 116));
    assert_eq!(bundle::MEMBERS[1].backbone, (426, 272, 154));
    assert_eq!(bundle::MEMBERS[1].backbone, bundle::MEMBERS[2].backbone);
    assert_eq!(
        bundle::MEMBERS[0].revision,
        "326b01d81f61fff7fd4ae570e54b87a49119e490"
    );
    assert_eq!(
        bundle::MEMBERS[1].revision,
        "d692394127f6f87fc581d8634c9558fe4776efce"
    );
    assert_eq!(
        bundle::MEMBERS[2].revision,
        "068425bb145c05884a98b870c219047da22d9aea"
    );
}

#[test]
fn a_matching_bundle_loads_and_exposes_the_member() {
    let dir = temp_dir("ok");
    write_bundle(&dir, first_member());
    let loaded = Bundle::load(&dir).expect("loads");
    assert_eq!(loaded.model_name(), "vllm-sr/Vela-2.0-0.8B");
    assert_eq!(loaded.max_length(), 16384);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn every_member_of_the_table_loads_its_own_bundle() {
    for member in bundle::MEMBERS {
        let dir = temp_dir(&format!(
            "member-{}",
            member.model_id.rsplit('/').next().unwrap()
        ));
        write_bundle(&dir, member);
        let loaded = Bundle::load(&dir).expect("loads");
        assert_eq!(loaded.model_name(), member.model_id);
        assert_eq!(loaded.member.model_id, member.model_id);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[test]
fn a_member_with_another_members_inventory_is_refused() {
    let dir = temp_dir("mixed");
    let member = &bundle::MEMBERS[1]; // 4B identity...
    write_bundle(&dir, member);
    let manifest_path = dir.join("vela_native_manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).expect("read")).expect("json");
    *manifest.get_mut("backbone").expect("backbone") = json!({
        "tensors": bundle::MEMBERS[0].backbone.0, // ...0.8B inventory
        "bf16": bundle::MEMBERS[0].backbone.1,
        "fp32": bundle::MEMBERS[0].backbone.2,
    });
    std::fs::write(&manifest_path, serde_json::to_vec(&manifest).expect("json")).expect("write");
    let error = Bundle::load(&dir).unwrap_err();
    assert!(error.to_string().contains("inventory"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_tampered_file_fails_its_hash() {
    let dir = temp_dir("tampered");
    write_bundle(&dir, first_member());
    std::fs::write(dir.join("calibration.json"), b"{\"tampered\": true}").expect("write");
    let error = Bundle::load(&dir).unwrap_err();
    assert!(error.to_string().contains("calibration.json"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_foreign_revision_is_refused() {
    let dir = temp_dir("foreign");
    write_bundle(&dir, first_member());
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
    write_bundle(&dir, first_member());
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
    write_bundle(&dir, first_member());
    std::fs::write(dir.join("extra.bin"), b"not in the manifest").expect("write");
    let error = Bundle::load(&dir).unwrap_err();
    assert!(error.to_string().contains("extra.bin"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_empty_files_map_fails_the_gate() {
    let dir = temp_dir("empty-files");
    write_bundle(&dir, first_member());
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
