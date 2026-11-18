//! Frozen full-row reference replay, including two softmax layers and causal GDN.
use omni_qwen3_5_native::model::Model;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Fixture {
    schema: String,
    rows: Vec<Row>,
}

#[derive(Deserialize)]
struct Row {
    id: String,
    ids: Vec<u32>,
    state_tokens: usize,
    hidden: Vec<f32>,
}

#[test]
#[ignore = "needs a GPU, CUA_S1_CUDA_LIB and QWEN_PREFIX_BIDIR_FIXTURE"]
fn multilayer_prefix_mask_matches_frozen_reference_and_restores_causal_execution() {
    let file = PathBuf::from(std::env::var_os("QWEN_PREFIX_BIDIR_FIXTURE").expect("fixture path"));
    let fixture: Fixture = serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
    assert_eq!(fixture.schema, "qwen-prefix-bidir-reference/1");
    assert!(!fixture.rows.is_empty());
    let checkpoint = file.parent().unwrap();
    let library = PathBuf::from(std::env::var_os("CUA_S1_CUDA_LIB").expect("CUDA library"));
    let mut model = Model::load_with_graph(checkpoint, &library, false).unwrap();
    model.require_prompt_prefix_bidirectional().unwrap();
    assert!(model.cfg.full_attention.iter().filter(|&&x| x).count() >= 2);
    assert!(model.cfg.full_attention.iter().any(|&x| !x));
    let first = &fixture.rows[0];
    let causal_before = model.forward(&first.ids).unwrap();
    let mut records = Vec::new();
    let mut worst = 0.0f32;
    for row in fixture.rows.iter().chain([first]) {
        let hidden = model
            .forward_prompt_prefix_bidirectional(&row.ids, row.state_tokens)
            .unwrap();
        assert_eq!(hidden.len(), row.hidden.len());
        assert!(hidden.iter().chain(&row.hidden).all(|x| x.is_finite()));
        let scale = row.hidden.iter().map(|x| x.abs()).fold(1e-3f32, f32::max);
        let error = hidden
            .iter()
            .zip(&row.hidden)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
            / scale;
        eprintln!("{}: normalized hidden max error {error:.6e}", row.id);
        worst = worst.max(error);
        records.push(serde_json::json!({"id":row.id,"ids":row.ids,"state_tokens":row.state_tokens,"hidden":hidden,"normalized_max_error":error}));
    }
    assert_eq!(
        records[0]["hidden"],
        records.last().unwrap()["hidden"],
        "earlier row changed after scratch reuse"
    );
    assert_eq!(
        causal_before,
        model.forward(&first.ids).unwrap(),
        "prefix mask leaked into a causal call"
    );
    if let Some(output) = std::env::var_os("QWEN_PREFIX_BIDIR_OUTPUT") {
        std::fs::write(output, serde_json::to_vec(&records).unwrap()).unwrap();
    }
    // This synthetic BF16 kernel/backbone gate is deliberately separate from
    // a model adapter's decision/probability gates. It is fixed before replay.
    assert!(worst < 0.03, "full-row BF16 hidden gate failed: {worst}");
}

#[cfg(unix)]
mod cpu_support;

/// This checks host routing and ABI arguments, not attention numerics.
#[cfg(unix)]
#[test]
fn attention_modes_dispatch_independently_on_cpu() {
    const CHILD: &str = "QWEN_TEST_ATTENTION_DISPATCH";
    let Ok(mode) = std::env::var(CHILD) else {
        // The CUDA API is process-global, so each library needs its own process.
        for mode in ["complete", "without-prefix"] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "attention_modes_dispatch_independently_on_cpu",
                    "--nocapture",
                ])
                .env(CHILD, mode)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{mode}: {}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        return;
    };
    let directory =
        std::env::temp_dir().join(format!("qwen-attention-dispatch-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    cpu_support::write_tiny_checkpoint(&directory);
    let symbol = "cs1_attention_gated_prefix_bidirectional";
    let mut extra = include_str!("attention_dispatch_fake.c").to_owned();
    let missing = mode == "without-prefix";
    if missing {
        extra = extra.replace(symbol, "omitted_prefix_attention");
    }
    let omitted = [symbol];
    let library =
        cpu_support::build_fake_library(&directory, &extra, if missing { &omitted } else { &[] });
    // SAFETY: the library and model remain alive while these metric symbols are used.
    let metrics = unsafe { libloading::Library::new(&library).unwrap() };
    let metric =
        |name: &[u8]| unsafe { metrics.get::<unsafe extern "C" fn() -> i32>(name).unwrap()() };
    let counts = || {
        [
            metric(b"dispatch_causal\0"),
            metric(b"dispatch_bidir\0"),
            metric(b"dispatch_prefix\0"),
            metric(b"dispatch_gdn\0"),
        ]
    };
    let mut model = Model::load_with_graph(&directory, &library, false).unwrap();
    let ids = vec![0; 129];
    model.forward(&ids).unwrap();
    assert_eq!(counts(), [1, 0, 0, 1]);
    model.forward_prompt_bidirectional(&ids).unwrap();
    assert_eq!(counts(), [1, 1, 0, 2]);
    if missing {
        let error = model
            .forward_prompt_prefix_bidirectional(&ids, 65)
            .unwrap_err();
        let detail = format!("{error:#}");
        assert!(
            detail.contains(symbol) && detail.contains("rebuild"),
            "{detail}"
        );
        assert_eq!(counts(), [1, 1, 0, 2]);
    } else {
        for (index, state) in [1, 64, 65, 128].into_iter().enumerate() {
            model
                .forward_prompt_prefix_bidirectional(&ids, state)
                .unwrap();
            assert_eq!(metric(b"dispatch_tokens\0"), 129);
            assert_eq!(metric(b"dispatch_state\0"), state as i32);
            assert_eq!(counts(), [1, 1, index as i32 + 1, index as i32 + 3]);
        }
    }
    let before = counts();
    model.forward_prompt_bidirectional(&ids).unwrap();
    model.forward_batch(&[&ids[..3], &ids[..67]]).unwrap();
    assert_eq!(
        counts(),
        [before[0] + 2, before[1] + 1, before[2], before[3] + 3]
    );
    assert_eq!(metric(b"dispatch_tokens\0"), 67);
    drop(model);
    drop(metrics);
    std::fs::remove_dir_all(directory).unwrap();
}
