//! Processing tests against the frozen token-plan fixtures.
//!
//! The mapping tests (state → parts, SystemOne → predict, plan) run
//! everywhere with no tokenizer and no weights. The full render parity
//! — rows and window rows byte-for-byte against the fixtures — needs
//! the exported bundle's tokenizer and is opt-in:
//!
//!     VELA_NATIVE_BUNDLE=weights/vela-2.0-0.8b-native \
//!       cargo test -p omni-vela-native --test vela_native_processing -- --ignored --nocapture

use std::path::PathBuf;

use omni_vela_native::questions::system_one_to_predict;
use omni_vela_native::state::system_one_state;
use serde_json::Value;

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

const CASES: [&str; 5] = [
    "example-request",
    "example-request-set-threshold",
    "example-request-cjk",
    "example-request-score",
    "long-span-windowed",
];

/// Every member's labeled plan fixtures: the plans are tokenizer-free,
/// so the always-run mapping check replays all three members, not just
/// the one a bundle selects.
const MEMBERS: [&str; 3] = ["Vela-2.0-0.8B", "Vela-2.0-4B", "Vela-2.0-9B"];

fn fixture(member: &str, case: &str) -> Value {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../tests/vela/fixtures");
    serde_json::from_str(
        &std::fs::read_to_string(root.join(format!("native-plan-{member}-{case}.json")))
            .expect("fixture present; record it with recipe/vela/export_token_plan.py"),
    )
    .expect("fixture JSON")
}

// ------------------------------------------------------- mapping (no tokens)

fn mapped(member: &str, case: &str) -> Value {
    let fix = fixture(member, case);
    let request = fix["request"].as_object().expect("request");
    let state = system_one_state(&request["state"]).expect("state maps");
    let questions = request["questions"].as_object().expect("questions");
    let mapped = system_one_to_predict(state, questions).expect("maps");
    serde_json::json!({
        "predict_request": serde_json::to_value(&mapped.request).unwrap(),
        "plan": serde_json::to_value(&mapped.plan).unwrap(),
    })
}

#[test]
fn the_predict_mapping_matches_every_fixture() {
    for member in MEMBERS {
        for case in CASES {
            let got = mapped(member, case);
            let fix = fixture(member, case);
            assert_eq!(
                got["predict_request"], fix["predict_request"],
                "predict_request differs for {member} {case}"
            );
            assert_eq!(got["plan"], fix["plan"], "plan differs for {member} {case}");
        }
    }
}

#[test]
fn an_invalid_question_reports_the_pinned_triples() {
    let state = system_one_state(&serde_json::json!("s")).expect("state");
    let questions = serde_json::json!({
        "q": {"type": "choice", "instructions": "ok?", "criteria": {"a": "x"}}
    });
    let error = system_one_to_predict(state, questions.as_object().unwrap()).unwrap_err();
    assert_eq!(
        error.body,
        serde_json::json!({"detail": [{"loc": ["body", "questions", "q", "criteria"],
            "msg": "choice criteria must be an object with 2 to 255 entries", "type": "value_error"}]})
    );

    let state = system_one_state(&serde_json::json!("s")).expect("state");
    let questions = serde_json::json!({
        "q": {"type": "span", "instructions": "which?", "criteria": {"a": "x"},
              "head": "other", "over": "request"}
    });
    let error = system_one_to_predict(state, questions.as_object().unwrap()).unwrap_err();
    assert_eq!(
        error.body["detail"][0]["loc"],
        serde_json::json!(["body", "questions", "q", "head"])
    );
    assert_eq!(error.body["detail"][0]["type"], "literal_error");
}

#[test]
fn a_set_label_collision_is_rejected_after_the_walk() {
    let state = system_one_state(&serde_json::json!("s")).expect("state");
    let questions = serde_json::json!({
        "issues": {"type": "set", "instructions": "which?", "criteria": {"billing": "b"}},
        "issues.billing": {"type": "noul", "instructions": "which?"}
    });
    let error = system_one_to_predict(state, questions.as_object().unwrap()).unwrap_err();
    assert_eq!(
        error.body["detail"][0]["msg"],
        "collides with the Noul view of a set label"
    );
}

// ------------------------------------------------- full render (opt-in)

fn bundle_dir() -> PathBuf {
    let path = PathBuf::from(
        std::env::var("VELA_NATIVE_BUNDLE").expect("set VELA_NATIVE_BUNDLE to an exported bundle"),
    );
    assert!(path.is_dir(), "{} is not a directory", path.display());
    path
}

#[test]
#[ignore = "requires VELA_NATIVE_BUNDLE at an exported bundle; CPU only"]
fn the_rendered_rows_match_every_fixture() {
    let bundle = bundle_dir();
    let processor = omni_vela_native::processing::Processor::load(&bundle)
        .expect("processor loads the exported bundle");
    for case in CASES {
        let fix = fixture(&member_label(), case);
        let request = fix["request"].as_object().expect("request");
        let planned = match processor.plan(request) {
            Ok(planned) => planned,
            Err(error) => panic!(
                "plan failed for {case}: {}",
                serde_json::to_string(&error.body).unwrap()
            ),
        };
        let got = serde_json::json!({
            "rows": serde_json::to_value(&planned.rows).unwrap(),
            "window_rows": serde_json::to_value(&planned.window_rows).unwrap(),
        });
        assert_eq!(got["rows"], fix["rows"], "rows differ for {case}");
        assert_eq!(
            got["window_rows"], fix["window_rows"],
            "window rows differ for {case}"
        );
    }
}
