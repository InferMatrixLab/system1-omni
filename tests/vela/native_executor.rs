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

fn manifest_anchor() -> String {
    std::env::var("VELA_NATIVE_MANIFEST_SHA256")
        .expect("set VELA_NATIVE_MANIFEST_SHA256 to the manifest digest the exporter prints")
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
    let anchor = manifest_anchor();
    let mut executor = Executor::load(&bundle, &library, &anchor).expect("executor loads");
    // The env gate may only tighten the documented ceiling: an
    // unparseable value is an error, and a wider one clamps back down.
    let logit_gate = std::env::var("VELA_GPU_LOGIT_GATE")
        .map(|v| {
            v.parse::<f64>()
                .expect("VELA_GPU_LOGIT_GATE must be a number")
        })
        .unwrap_or(0.35)
        .min(0.35);

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
        let processor = omni_vela_native::processing::Processor::load(&bundle, &anchor)
            .expect("processor loads");
        let planned = processor.plan(&request).expect("plan");
        let raw = std::fs::read(fixtures_dir().join(format!("head-input-{m}-{case}.safetensors")))
.unwrap_or_else(|error| panic!(
                "head-input-{m}-{case}.safetensors is missing ({error}); only the 0.8B binaries are committed — regenerate 4B/9B with recipe/vela/export_head_inputs.py"
            ));
        let _file = safetensors::SafeTensors::deserialize(&raw).expect("safetensors");

        // The recorder's execution order can differ from planned row order.
        let calls: Vec<&Value> = head_fixture["head_calls"]
            .as_array()
            .unwrap()
            .iter()
            .collect();
        let mut q_call_index = 0usize;
        let mut span_call_index = 0usize;
        let mut consumed = vec![false; calls.len()];

        let mut worst_logit = 0.0f32;
        for (row_index, row) in planned.rows.iter().enumerate() {
            let readouts = executor.row_readouts(row).expect("readouts");
            // One recorded q call carries the row's question blocks
            // stacked; compare per block, matched by option count.
            if !readouts.questions.is_empty() {
                let call = next_call(&calls, &mut consumed, row_index, "q", None);
                q_call_index += 1;
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
                let call = next_call(&calls, &mut consumed, row_index, "span", Some(&which));
                span_call_index += 1;
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
        assert!(
            consumed.iter().all(|used| *used),
            "{case}: unused head call"
        );
        println!("{case}: worst logit drift {worst_logit:.4}");
        assert!(
            worst_logit as f64 <= logit_gate,
            "{case}: drift {worst_logit} > {logit_gate}"
        );
    }
}

fn next_call<'a>(
    calls: &[&'a Value],
    consumed: &mut [bool],
    row: usize,
    kind: &str,
    head: Option<&str>,
) -> &'a Value {
    assert_eq!(calls.len(), consumed.len());
    let attributed = calls.iter().any(|call| call.get("row").is_some());
    let index = if attributed {
        assert!(
            calls.iter().all(|call| call["row"].as_u64().is_some()),
            "all recorded calls must carry valid row attribution"
        );
        let matches: Vec<usize> = calls
            .iter()
            .enumerate()
            .filter(|(index, call)| {
                !consumed[*index]
                    && call["row"].as_u64() == Some(row as u64)
                    && call["kind"] == kind
                    && head.is_none_or(|head| call["head"] == head)
            })
            .map(|(index, _)| index)
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "expected exactly one unconsumed row={row} kind={kind} head={head:?} call"
        );
        matches[0]
    } else {
        // The original CPU-recorded 0.8B fixtures predate row metadata.
        calls
            .iter()
            .enumerate()
            .position(|(index, call)| !consumed[index] && call["kind"] == kind)
            .unwrap_or_else(|| panic!("recorded {kind} call"))
    };
    if let Some(head) = head {
        assert_eq!(calls[index]["head"], head, "recorded span head");
    }
    consumed[index] = true;
    calls[index]
}

#[test]
fn attributed_calls_match_rows_when_execution_is_reordered() {
    let first = serde_json::json!({"kind": "span", "row": 1, "head": "router", "logits": [[27.0]]});
    let second =
        serde_json::json!({"kind": "span", "row": 0, "head": "router", "logits": [[23.0]]});
    let calls = [&first, &second];
    let mut consumed = [false; 2];
    assert_eq!(
        next_call(&calls, &mut consumed, 0, "span", Some("router"))["logits"],
        second["logits"]
    );
    assert_eq!(
        next_call(&calls, &mut consumed, 1, "span", Some("router"))["logits"],
        first["logits"]
    );
    assert!(consumed.iter().all(|used| *used));
}

#[test]
fn legacy_unattributed_calls_keep_the_original_kind_order() {
    let first = serde_json::json!({"kind": "span", "head": "router", "logits": [[1.0]]});
    let second = serde_json::json!({"kind": "span", "head": "router", "logits": [[2.0]]});
    let calls = [&first, &second];
    let mut consumed = [false; 2];
    assert_eq!(
        next_call(&calls, &mut consumed, 0, "span", Some("router"))["logits"],
        first["logits"]
    );
    assert_eq!(
        next_call(&calls, &mut consumed, 1, "span", Some("router"))["logits"],
        second["logits"]
    );
    assert!(consumed.iter().all(|used| *used));
}

#[test]
fn attributed_calls_reject_missing_duplicate_and_wrong_heads() {
    let one = serde_json::json!({"kind": "span", "row": 0, "head": "router", "logits": [[1.0]]});
    let other_row =
        serde_json::json!({"kind": "span", "row": 1, "head": "router", "logits": [[1.0]]});
    let other_head =
        serde_json::json!({"kind": "span", "row": 0, "head": "broad", "logits": [[1.0]]});
    for calls in [vec![&other_row], vec![&one, &one], vec![&other_head]] {
        let mut consumed = vec![false; calls.len()];
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                next_call(&calls, &mut consumed, 0, "span", Some("router"));
            }))
            .is_err()
        );
    }
}
