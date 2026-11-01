use omni_cua_s1_native::vision_engine::device_features_enabled;

#[test]
fn device_features_accepts_only_unset_zero_and_one() {
    assert!(!device_features_enabled(None).unwrap());
    assert!(!device_features_enabled(Some("0")).unwrap());
    assert!(device_features_enabled(Some("1")).unwrap());
    for value in ["", "true", " 1", "1 ", "2"] {
        assert!(device_features_enabled(Some(value)).is_err());
    }
}

fn rejects_before_loading(value: &std::ffi::OsStr) {
    let missing =
        std::env::temp_dir().join(format!("cua-device-control-missing-{}", std::process::id()));
    assert!(!missing.exists());
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_omni-cua-s1-vision"))
        .env("CUA_S1_DEVICE_FEATURES", value)
        .env("CUA_S1_BASE", &missing)
        .env("CUA_S1_VISION_ADAPTER", &missing)
        .env("CUA_S1_MODEL", &missing)
        .env("CUA_S1_CUDA_LIB", &missing)
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(
        error.contains("CUA_S1_DEVICE_FEATURES must be unset, 0, or 1"),
        "{error}"
    );
    assert!(!error.contains("export the multimodal language checkpoint"));
}

#[test]
fn invalid_feature_controls_fail_before_checkpoint_or_cuda_loading() {
    for value in ["", "true", "2"] {
        rejects_before_loading(value.as_ref());
    }
}

#[cfg(unix)]
#[test]
fn non_unicode_feature_control_fails_before_checkpoint_or_cuda_loading() {
    use std::os::unix::ffi::OsStrExt;
    rejects_before_loading(std::ffi::OsStr::from_bytes(&[0xff]));
}

#[test]
fn multimodal_prefix_controls_reject_invalid_and_incompatible_modes_before_loading() {
    for (prefix, graph, device, expected) in [
        (
            "",
            "0",
            "0",
            "CUA_S1_MM_PREFIX must be off, fixed, shared, or matched",
        ),
        (
            "true",
            "0",
            "0",
            "CUA_S1_MM_PREFIX must be off, fixed, shared, or matched",
        ),
        (
            "shared",
            "1",
            "0",
            "multimodal prefix modes require Graph and device features off",
        ),
        (
            "matched",
            "1",
            "0",
            "multimodal prefix modes require Graph and device features off",
        ),
        (
            "matched",
            "0",
            "1",
            "multimodal prefix modes require Graph and device features off",
        ),
        (
            "fixed",
            "0",
            "1",
            "multimodal prefix modes require Graph and device features off",
        ),
    ] {
        let missing =
            std::env::temp_dir().join(format!("cua-prefix-missing-{}", std::process::id()));
        assert!(!missing.exists());
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_omni-cua-s1-vision"))
            .env("CUA_S1_MM_PREFIX", prefix)
            .env("CUA_S1_GRAPH", graph)
            .env("CUA_S1_DEVICE_FEATURES", device)
            .env("CUA_S1_BASE", &missing)
            .env("CUA_S1_VISION_ADAPTER", &missing)
            .env("CUA_S1_MODEL", &missing)
            .env("CUA_S1_CUDA_LIB", &missing)
            .output()
            .unwrap();
        assert!(!output.status.success());
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(error.contains(expected), "{error}");
    }
}

#[test]
fn matched_prefix_is_opt_in_and_recognized_before_checkpoint_loading() {
    for value in [
        None,
        Some("off"),
        Some("fixed"),
        Some("shared"),
        Some("matched"),
    ] {
        let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_omni-cua-s1-vision"));
        cmd.env_remove("CUA_S1_MM_PREFIX")
            .env("CUA_S1_GRAPH", "0")
            .env("CUA_S1_DEVICE_FEATURES", "0")
            .env("CUA_S1_MODEL", "/nonexistent-cua-prefix-model")
            .env("CUA_S1_BASE", "/nonexistent-cua-prefix-model")
            .env("CUA_S1_VISION_ADAPTER", "/nonexistent-cua-prefix-model")
            .env("CUA_S1_CUDA_LIB", "/nonexistent-cua-prefix-model");
        if let Some(value) = value {
            cmd.env("CUA_S1_MM_PREFIX", value);
        }
        let output = cmd.output().unwrap();
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(!output.status.success());
        assert!(
            error.contains("export the multimodal language checkpoint"),
            "{value:?}: {error}"
        );
    }
}

#[cfg(unix)]
#[test]
fn non_unicode_prefix_is_rejected_before_loading() {
    use std::os::unix::ffi::OsStrExt;
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_omni-cua-s1-vision"))
        .env("CUA_S1_MM_PREFIX", std::ffi::OsStr::from_bytes(&[0xff]))
        .env("CUA_S1_DEVICE_FEATURES", "0")
        .env("CUA_S1_MODEL", "/nonexistent-cua-prefix-model")
        .env("CUA_S1_BASE", "/nonexistent-cua-prefix-model")
        .env("CUA_S1_VISION_ADAPTER", "/nonexistent-cua-prefix-model")
        .env("CUA_S1_CUDA_LIB", "/nonexistent-cua-prefix-model")
        .output()
        .unwrap();
    assert!(
        String::from_utf8(output.stderr)
            .unwrap()
            .contains("CUA_S1_MM_PREFIX must be off, fixed, shared, or matched")
    );
}
