//! GPU executor check: the backbone through the shared qwen3_5 crate,
//! full-sequence per readout position, replayed into the Rust heads
//! and compared with the recorded reference logits. Opt-in:
//!
//!     VELA_NATIVE_BUNDLE=.../vela-2.0-0.8b-native \
//!     VELA_CUDA_LIB=.../libqwen3_5_cuda.so \
//!     cargo test -p omni-vela-native --test vela_native_executor -- --ignored --nocapture
//!
//! The recorded reference ran CPU FP32 (0.8B) or GPU bf16-autocast
//! (4B/9B); this path runs the uniform-BF16
//! backbone, so the gates are backbone-level numerical tolerances, not
//! the reference worker's exact-match bar: logit drift is measured and
//! asserted against the gate.

use std::path::PathBuf;

use omni_vela_native::executor::Executor;
use omni_vela_native::render::Block;
use serde_json::Value;

mod native_parity;
use native_parity::max_logit_drift;

/// The member whose fixtures this run replays: the bundle's manifest
/// member when VELA_NATIVE_BUNDLE selects one, else the 0.8B baseline
/// that ships unlabeled history-free under the labeled scheme.
fn member_label() -> String {
    if let Ok(dir) = std::env::var("VELA_NATIVE_BUNDLE") {
        let manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(std::path::Path::new(&dir).join("vela_native_manifest.json"))
                .expect("the bundle's manifest"),
        )
        .expect("vela_native_manifest.json must parse");
        return manifest["model_id"]
            .as_str()
            .expect("model_id")
            .rsplit('/')
            .next()
            .unwrap()
            .to_string();
    }
    "Vela-2.0-0.8B".to_string()
}

const CASES: [&str; 4] = [
    "example-request",
    "example-request-set-threshold",
    "example-request-cjk",
    "example-request-score",
];

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../tests/vela/fixtures")
}

fn env_dir(name: &str) -> PathBuf {
    let path = PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name}")));
    assert!(path.is_dir(), "{} is not a directory", path.display());
    path
}

#[test]
#[ignore = "requires VELA_NATIVE_BUNDLE at an exported bundle and VELA_CUDA_LIB at the built CUDA library; GPU"]
fn the_gpu_backbone_replays_the_reference_readouts() {
    let bundle = env_dir("VELA_NATIVE_BUNDLE");
    let library = PathBuf::from(
        std::env::var("VELA_CUDA_LIB").expect("set VELA_CUDA_LIB to the built CUDA library"),
    );
    assert!(library.exists(), "{} does not exist", library.display());
    let mut executor = Executor::load(&bundle, &library).expect("executor loads");
    let logit_gate = std::env::var("VELA_GPU_LOGIT_GATE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.35);

    let m = member_label();

    for case in CASES {
        let head_fixture: Value = serde_json::from_str(
            &std::fs::read_to_string(fixtures_dir().join(format!("head-input-{m}-{case}.json")))
                .expect("fixture"),
        )
        .expect("json");
        let plan_fixture: Value = serde_json::from_str(
            &std::fs::read_to_string(fixtures_dir().join(format!("native-plan-{m}-{case}.json")))
                .expect("fixture"),
        )
        .expect("json");
        let request = plan_fixture["request"]
            .as_object()
            .expect("request")
            .clone();
        let processor =
            omni_vela_native::processing::Processor::load(&bundle).expect("processor loads");
        let planned = processor.plan(&request).expect("plan");
        let raw = std::fs::read(fixtures_dir().join(format!("head-input-{m}-{case}.safetensors")))
.unwrap_or_else(|error| panic!(
                "head-input-{m}-{case}.safetensors is missing ({error}); only the 0.8B binaries are committed — regenerate 4B/9B with recipe/vela/export_head_inputs.py"
            ));
        let _file = safetensors::SafeTensors::deserialize(&raw).expect("safetensors");

        // The recorded reference logits per call, in call order.
        let calls: Vec<&Value> = head_fixture["head_calls"]
            .as_array()
            .unwrap()
            .iter()
            .collect();
        let mut q_call_index = 0usize;
        let mut span_call_index = 0usize;

        let mut worst_logit = 0.0f32;
        for row in &planned.rows {
            let readouts = executor.row_readouts(row).expect("readouts");
            // One recorded q call carries the row's question blocks
            // stacked; compare per block, matched by option count.
            if !readouts.questions.is_empty() {
                let call = next_call(&calls, &mut q_call_index, "q");
                let want: Vec<Vec<f64>> = call["logits"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|r| {
                        r.as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_f64().unwrap())
                            .collect()
                    })
                    .collect();
                assert_eq!(
                    readouts.questions.len(),
                    want.len(),
                    "{case}: question count"
                );
                let padded_width = readouts
                    .questions
                    .iter()
                    .map(|q| q.candidates.len())
                    .max()
                    .unwrap();
                // The stacked call's rows are in block order, padded to
                // the widest block; compare each block's prefix.
                for (question, ours) in readouts.questions.iter().zip(&want) {
                    assert_eq!(ours.len(), padded_width, "{case}: padded candidate count");
                    let Block::Question { ends, .. } = &row.blocks[question.block_index] else {
                        panic!("expected question block");
                    };
                    assert_eq!(
                        question.candidates.len(),
                        ends.len(),
                        "{case}: candidate readouts"
                    );
                    let query = question.query.as_ref().expect("question query");
                    let got = executor
                        .heads
                        .forward_candidate(&question.candidates, query);
                    assert_eq!(
                        got.len(),
                        question.candidates.len(),
                        "{case}: candidate count"
                    );
                    worst_logit = worst_logit.max(max_logit_drift(&got, &ours[..got.len()]) as f32);
                }
            }
            if let Some(span) = &readouts.span {
                let which = row.head.clone();
                let head = if which == "broad" {
                    &executor.heads.broad
                } else {
                    &executor.heads.router
                };
                let got = head.forward(&span.words, &span.labels);
                let call = next_call(&calls, &mut span_call_index, "span");
                let want: Vec<Vec<f64>> = call["logits"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|row| {
                        row.as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_f64().unwrap())
                            .collect()
                    })
                    .collect();
                assert_eq!(got.len(), want.len(), "{case}: span word count");
                let mut worst_row = 0.0f32;
                for (g, w) in got.iter().zip(&want) {
                    worst_row = worst_row.max(max_logit_drift(g, w) as f32);
                }
                println!("{case}: span logits drift {worst_row:.4}");
                worst_logit = worst_logit.max(worst_row);
            }
        }
        assert_eq!(
            q_call_index,
            calls.iter().filter(|c| c["kind"] == "q").count(),
            "{case}: question calls"
        );
        assert_eq!(
            span_call_index,
            calls.iter().filter(|c| c["kind"] == "span").count(),
            "{case}: span calls"
        );
        println!("{case}: worst logit drift {worst_logit:.4}");
        assert!(
            worst_logit <= logit_gate,
            "{case}: drift {worst_logit} > {logit_gate}"
        );
    }
}

fn next_call<'a>(calls: &[&'a Value], index: &mut usize, kind: &str) -> &'a Value {
    let call = calls
        .iter()
        .filter(|c| c["kind"] == kind)
        .nth(*index)
        .unwrap_or_else(|| panic!("recorded {kind} call"));
    *index += 1;
    call
}
