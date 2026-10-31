//! GPU executor check: the backbone through the shared qwen3_5 crate,
//! full-sequence per readout position, replayed into the Rust heads
//! and compared with the recorded reference logits. Opt-in:
//!
//!     VELA_NATIVE_BUNDLE=.../vela-2.0-0.8b-native \
//!     VELA_CUDA_LIB=.../libqwen3_5_cuda.so \
//!     cargo test -p omni-vela-native --test vela_native_executor -- --ignored --nocapture
//!
//! The recorded reference ran CPU FP32; this path runs the uniform-BF16
//! backbone, so the gates are backbone-level numerical tolerances, not
//! the reference worker's exact-match bar: logit drift is measured and
//! asserted against the gate.

use std::path::PathBuf;

use omni_vela_native::executor::Executor;
use serde_json::Value;

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
    // The env gate may only tighten the documented ceiling: an
    // unparseable value is an error, and a wider one clamps back down.
    let logit_gate = std::env::var("VELA_GPU_LOGIT_GATE")
        .map(|v| {
            v.parse::<f32>()
                .expect("VELA_GPU_LOGIT_GATE must be a number")
        })
        .unwrap_or(0.35)
        .min(0.35);

    for case in CASES {
        let head_fixture: Value = serde_json::from_str(
            &std::fs::read_to_string(fixtures_dir().join(format!("head-input-{case}.json")))
                .expect("fixture"),
        )
        .expect("json");
        let plan_fixture: Value = serde_json::from_str(
            &std::fs::read_to_string(fixtures_dir().join(format!("native-plan-{case}.json")))
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
        let raw = std::fs::read(fixtures_dir().join(format!("head-input-{case}.safetensors")))
            .expect("safetensors fixture");
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
                // The stacked call's rows are in block order, padded to
                // the widest block; compare each block's prefix.
                for (question, ours) in readouts.questions.iter().zip(&want) {
                    let Some(query) = &question.query else {
                        continue;
                    };
                    let got = executor
                        .heads
                        .forward_candidate(&question.candidates, query);
                    for (a, b) in got.iter().zip(ours.iter()) {
                        worst_logit = worst_logit.max((a - *b as f32).abs());
                    }
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
                let mut worst_row = 0.0f32;
                for (g, w) in got.iter().zip(&want) {
                    for (a, b) in g.iter().zip(w) {
                        worst_row = worst_row.max((a - *b as f32).abs());
                    }
                }
                println!("{case}: span logits drift {worst_row:.4}");
                worst_logit = worst_logit.max(worst_row);
            }
        }
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
