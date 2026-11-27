//! Bundle-gate tests: a synthetic bundle built in a unique temp
//! directory (no weights, no network) drives the manifest gate — pins,
//! inventory counts, per-file hashes and the manifest SHA-256 anchor,
//! across every member of the pinned table.

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

/// Load with the pin table [`write_bundle`] reports for its synthetic
/// runtime files, anchored on the manifest as it currently sits on
/// disk (production pins the selected member's own table).
fn load_with_pins(dir: &std::path::Path, pins: &[(String, String)]) -> anyhow::Result<Bundle> {
    Bundle::load_with_runtime_pins(dir, &manifest_sha(dir), Some(&pins_refs(pins)))
}

fn pins_refs(pins: &[(String, String)]) -> Vec<(&str, &str)> {
    pins.iter()
        .map(|(name, digest)| (name.as_str(), digest.as_str()))
        .collect()
}

/// The minimal manifest of a correctly exported member; `files` are the
/// bundle files the gate hashes. Returns the digests the repository
/// pin table would carry for the three verbatim runtime files of this
/// synthetic package.
fn write_bundle(dir: &std::path::Path, member: &bundle::MemberPins) -> Vec<(String, String)> {
    let files: [(&str, &[u8]); 6] = [
        ("backbone.safetensors", b"backbone tensors"),
        ("heads.safetensors", b"head tensors"),
        ("config.json", b"{}"),
        ("calibration.json", b"{\"K\": 1}"),
        ("tokenizer.json", b"{}"),
        ("tokenizer_config.json", b"{}"),
    ];
    let mut hashes = serde_json::Map::new();
    for (name, content) in files {
        std::fs::write(dir.join(name), content).expect("write");
        hashes.insert(name.to_string(), json!(sha256(content)));
    }
    let pins = [
        "calibration.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    .iter()
    .map(|name| {
        let content = files
            .iter()
            .find(|(file, _)| file == name)
            .expect("runtime file")
            .1;
        ((*name).to_string(), sha256(content))
    })
    .collect();
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
    pins
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
    // Every member carries the three verbatim runtime pins.
    for member in bundle::MEMBERS {
        assert_eq!(member.runtime_pins.len(), 3, "{}", member.model_id);
    }
}

#[test]
fn a_matching_bundle_loads_and_exposes_the_member() {
    let dir = temp_dir("ok");
    let pins = write_bundle(&dir, first_member());
    let loaded = load_with_pins(&dir, &pins).expect("loads");
    assert_eq!(loaded.model_name(), first_member().model_id);
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
        let pins = write_bundle(&dir, member);
        let loaded = load_with_pins(&dir, &pins).expect("loads");
        assert_eq!(loaded.model_name(), member.model_id);
        assert_eq!(loaded.member.model_id, member.model_id);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[test]
fn a_member_with_another_members_inventory_is_refused() {
    let dir = temp_dir("mixed");
    let member = &bundle::MEMBERS[1]; // 4B identity...
    let pins = write_bundle(&dir, member);
    edit_manifest(&dir, |manifest| {
        *manifest.get_mut("backbone").expect("backbone") = json!({
            "tensors": bundle::MEMBERS[0].backbone.0, // ...0.8B inventory
            "bf16": bundle::MEMBERS[0].backbone.1,
            "fp32": bundle::MEMBERS[0].backbone.2,
        });
    });
    let error = load_with_pins(&dir, &pins).unwrap_err();
    assert!(error.to_string().contains("inventory"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_tampered_file_fails_its_hash() {
    let dir = temp_dir("tampered");
    let pins = write_bundle(&dir, first_member());
    std::fs::write(dir.join("calibration.json"), b"{\"tampered\": true}").expect("write");
    let error = load_with_pins(&dir, &pins).unwrap_err();
    assert!(error.to_string().contains("calibration.json"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_rewritten_manifest_fails_the_anchor() {
    // Re-hashing a swapped file and rewriting files_sha256 used to pass
    // the gate: the manifest now has to match an anchor fixed outside
    // the bundle directory, so the rewrite itself is refused.
    let dir = temp_dir("rewritten");
    let pins = write_bundle(&dir, first_member());
    let anchor = manifest_sha(&dir);
    edit_manifest(&dir, |manifest| {
        manifest["files_sha256"]["calibration.json"] =
            json!(sha256(b"{\"swapped\": true}").as_str());
    });
    std::fs::write(dir.join("calibration.json"), b"{\"swapped\": true}").expect("write");
    let error = Bundle::load_with_runtime_pins(&dir, &anchor, Some(&pins_refs(&pins))).unwrap_err();
    assert!(
        error.to_string().contains("pinned SHA-256 anchor"),
        "{error}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_self_consistent_rewrite_fails_the_repository_pin() {
    let dir = temp_dir("self-consistent");
    let pins = write_bundle(&dir, first_member());
    // Rewrite a verbatim runtime file and regenerate its manifest
    // digest: the manifest's own table is satisfied, the repository
    // pin is not.
    std::fs::write(dir.join("calibration.json"), b"{\"rewritten\": true}").expect("write");
    edit_manifest(&dir, |manifest| {
        manifest["files_sha256"]["calibration.json"] =
            json!(sha256(b"{\"rewritten\": true}").as_str());
    });
    let error = load_with_pins(&dir, &pins).unwrap_err();
    assert!(error.to_string().contains("repository pin"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_pinned_bf16_directory_loads() {
    let dir = temp_dir("bf16-ok");
    let pins = write_bundle(&dir, first_member());
    std::fs::create_dir(dir.join("bf16")).expect("dir");
    std::fs::write(dir.join("bf16/model.safetensors"), b"bf16 backbone").expect("write");
    std::fs::write(dir.join("bf16/config.json"), b"{}").expect("write");
    edit_manifest(&dir, |manifest| {
        manifest["files_sha256"]["bf16/model.safetensors"] =
            json!(sha256(b"bf16 backbone").as_str());
        manifest["files_sha256"]["bf16/config.json"] = json!(sha256(b"{}").as_str());
    });
    load_with_pins(&dir, &pins).expect("loads");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_stray_file_inside_a_pinned_directory_fails_the_gate() {
    let dir = temp_dir("stray-nested");
    let pins = write_bundle(&dir, first_member());
    std::fs::create_dir(dir.join("bf16")).expect("mkdir");
    std::fs::write(dir.join("bf16/model.safetensors"), b"bf16 backbone").expect("write");
    std::fs::write(dir.join("bf16/stray.bin"), b"unlisted").expect("write");
    edit_manifest(&dir, |manifest| {
        manifest["files_sha256"]["bf16/model.safetensors"] =
            json!(sha256(b"bf16 backbone").as_str());
    });
    let error = load_with_pins(&dir, &pins).unwrap_err();
    assert!(error.to_string().contains("bf16/stray.bin"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_directory_with_no_pinned_paths_fails_the_gate() {
    let dir = temp_dir("stray-dir");
    let pins = write_bundle(&dir, first_member());
    std::fs::create_dir(dir.join("junk")).expect("mkdir");
    let error = load_with_pins(&dir, &pins).unwrap_err();
    assert!(error.to_string().contains("junk"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_foreign_revision_is_refused() {
    let dir = temp_dir("foreign");
    let pins = write_bundle(&dir, first_member());
    edit_manifest(&dir, |manifest| {
        manifest["source"]["revision"] = json!("068425bb145c05884a98b870c219047da22d9aea");
    });
    let error = load_with_pins(&dir, &pins).unwrap_err();
    assert!(error.to_string().contains("pinned"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_unexpected_inventory_is_refused() {
    let dir = temp_dir("inventory");
    let pins = write_bundle(&dir, first_member());
    edit_manifest(&dir, |manifest| {
        manifest["backbone"]["bf16"] = json!(320);
        manifest["backbone"]["fp32"] = json!(0);
    });
    let error = load_with_pins(&dir, &pins).unwrap_err();
    assert!(error.to_string().contains("inventory"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_extra_unlisted_file_fails_the_gate() {
    let dir = temp_dir("extra");
    let pins = write_bundle(&dir, first_member());
    std::fs::write(dir.join("extra.bin"), b"not in the manifest").expect("write");
    let error = load_with_pins(&dir, &pins).unwrap_err();
    assert!(error.to_string().contains("extra.bin"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_absolute_files_key_is_refused() {
    let dir = temp_dir("absolute-key");
    let pins = write_bundle(&dir, first_member());
    edit_manifest(&dir, |manifest| {
        manifest["files_sha256"]["/etc/passwd"] = json!(sha256(b""));
    });
    let error = load_with_pins(&dir, &pins).unwrap_err();
    assert!(error.to_string().contains("plain relative name"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_parent_escaping_files_key_is_refused() {
    let dir = temp_dir("parent-key");
    let pins = write_bundle(&dir, first_member());
    edit_manifest(&dir, |manifest| {
        manifest["files_sha256"]["../outside.bin"] = json!(sha256(b""));
    });
    let error = load_with_pins(&dir, &pins).unwrap_err();
    assert!(error.to_string().contains("plain relative name"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_symlinked_files_key_is_refused() {
    let dir = temp_dir("symlink-key");
    let pins = write_bundle(&dir, first_member());
    let outside = temp_dir("symlink-outside");
    std::fs::write(outside.join("secret.bin"), b"outside the bundle").expect("write");
    std::os::unix::fs::symlink(outside.join("secret.bin"), dir.join("linked.safetensors"))
        .expect("symlink");
    edit_manifest(&dir, |manifest| {
        manifest["files_sha256"]["linked.safetensors"] = json!(sha256(b"outside the bundle"));
    });
    let error = load_with_pins(&dir, &pins).unwrap_err();
    assert!(error.to_string().contains("symlink"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&outside);
}

#[test]
#[cfg(unix)]
fn a_symlinked_pinned_file_fails_the_gate() {
    let dir = temp_dir("symlink-entry");
    let pins = write_bundle(&dir, first_member());
    let target = dir.join("tokenizer.json");
    std::fs::rename(&target, dir.join("tokenizer.real")).expect("rename");
    std::os::unix::fs::symlink(dir.join("tokenizer.real"), &target).expect("symlink");
    let error = load_with_pins(&dir, &pins).unwrap_err();
    assert!(error.to_string().contains("symlink"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_empty_files_map_fails_the_gate() {
    let dir = temp_dir("empty-files");
    let pins = write_bundle(&dir, first_member());
    edit_manifest(&dir, |manifest| {
        manifest["files_sha256"] = json!({});
    });
    let error = load_with_pins(&dir, &pins).unwrap_err();
    assert!(error.to_string().contains("files_sha256"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_missing_manifest_names_the_export_step() {
    let dir = temp_dir("empty");
    std::fs::create_dir_all(&dir).expect("tempdir");
    let error = Bundle::load_with_runtime_pins(&dir, "any anchor", None).unwrap_err();
    assert!(error.to_string().contains("export_native.py"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}
