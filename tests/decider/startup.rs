//! Combined worker switches must be validated before checkpoint or CUDA loading.
use std::process::Command;

fn rejects_before_loading(settings: &[(&str, &str)], expected: &str) {
    let missing = std::env::temp_dir().join(format!(
        "decider-startup-missing-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    assert!(!missing.exists());
    let mut command = Command::new(env!("CARGO_BIN_EXE_omni-decider"));
    for key in [
        "DECIDER_GRAPH",
        "DECIDER_PREFIX",
        "DECIDER_FIXED",
        "DECIDER_BATCH_MAX_ROWS",
        "DECIDER_BATCH_MAX_TOKENS",
        "DECIDER_MAX_PENDING_REQUESTS",
        "DECIDER_MAX_PENDING_TOKENS",
        "DECIDER_REQUEST_TIMEOUT_MS",
        "DECIDER_PORT",
    ] {
        command.env_remove(key);
    }
    let output = command
        .env("DECIDER_MODEL", missing)
        .env("DECIDER_CUDA_LIB", "unavailable-cuda-library")
        .env("CUA_S1_GRAPH", "1")
        .envs(settings.iter().copied())
        .output()
        .unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(!output.status.success());
    assert!(stderr.contains(expected), "{settings:?}: {stderr}");
    assert!(!stderr.contains("No such file"), "{stderr}");
}

#[test]
fn graph_and_prefix_validation_survives_runtime_integration() {
    for (settings, expected) in [
        (vec![("DECIDER_GRAPH", "2")], "DECIDER_GRAPH must be 0 or 1"),
        (
            vec![("DECIDER_PREFIX", "invalid")],
            "DECIDER_PREFIX must be",
        ),
        (vec![("DECIDER_FIXED", "invalid")], "DECIDER_FIXED must be"),
        (
            vec![("DECIDER_PREFIX", "1"), ("DECIDER_FIXED", "1")],
            "select only one",
        ),
        (
            vec![("DECIDER_GRAPH", "1"), ("DECIDER_PREFIX", "auto")],
            "shared/fixed execution requires DECIDER_GRAPH=0",
        ),
    ] {
        rejects_before_loading(&settings, expected);
    }
}

#[test]
fn runtime_validation_survives_every_execution_mode() {
    for mode in [
        vec![],
        vec![("DECIDER_GRAPH", "1")],
        vec![("DECIDER_PREFIX", "1")],
        vec![("DECIDER_PREFIX", "auto")],
        vec![("DECIDER_FIXED", "1")],
    ] {
        for (setting, expected) in [
            (
                "DECIDER_MAX_PENDING_REQUESTS",
                "DECIDER_MAX_PENDING_REQUESTS must be positive",
            ),
            (
                "DECIDER_MAX_PENDING_TOKENS",
                "DECIDER_MAX_PENDING_TOKENS must be positive",
            ),
            (
                "DECIDER_REQUEST_TIMEOUT_MS",
                "DECIDER_REQUEST_TIMEOUT_MS must be positive",
            ),
        ] {
            let mut settings = mode.clone();
            settings.push((setting, "0"));
            rejects_before_loading(&settings, expected);
        }
    }
}

#[tokio::test]
async fn explicit_mode_constructor_rejects_graph_with_prefix_before_loading() {
    use omni_decider_native::{
        batching::BatchLimits, engine::Engine, prefix::PrefixMode, runtime::RuntimeConfig,
    };
    for mode in [PrefixMode::Fixed, PrefixMode::Shared, PrefixMode::Auto] {
        let result = Engine::load_with_modes_and_runtime(
            std::path::Path::new("unavailable-model"),
            std::path::Path::new("unavailable-cuda-library"),
            BatchLimits::default(),
            true,
            mode,
            RuntimeConfig::default(),
        )
        .await;
        assert!(
            result
                .err()
                .unwrap()
                .to_string()
                .contains("shared/fixed execution requires Graph off")
        );
    }
}

#[tokio::test]
async fn explicit_mode_constructor_validates_runtime_before_loading() {
    use omni_decider_native::{
        batching::BatchLimits, engine::Engine, prefix::PrefixMode, runtime::RuntimeConfig,
    };
    for (graph, mode) in [
        (false, PrefixMode::Off),
        (true, PrefixMode::Off),
        (false, PrefixMode::Fixed),
        (false, PrefixMode::Shared),
        (false, PrefixMode::Auto),
    ] {
        let result = Engine::load_with_modes_and_runtime(
            std::path::Path::new("unavailable-model"),
            std::path::Path::new("unavailable-cuda-library"),
            BatchLimits::default(),
            graph,
            mode,
            RuntimeConfig {
                max_requests: Some(0),
                ..RuntimeConfig::default()
            },
        )
        .await;
        assert!(
            result
                .err()
                .unwrap()
                .to_string()
                .contains("runtime budgets must be positive")
        );
    }
}
