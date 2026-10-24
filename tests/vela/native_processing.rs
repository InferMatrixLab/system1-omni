//! Processing tests against the frozen token-plan fixtures.
//!
//! The mapping tests (state → parts, SystemOne → predict, plan) run
//! everywhere with no tokenizer and no weights. The full render parity
//! — rows and window rows byte-for-byte against the fixtures — needs
//! the exported bundle's tokenizer and is opt-in:
//!
//!     VELA_NATIVE_BUNDLE=weights/vela-2.0-0.8b-native \
//!       cargo test -p omni-vela-native --test vela_native_processing -- --nocapture

use std::path::PathBuf;

use omni_vela_native::questions::system_one_to_predict;
use omni_vela_native::state::system_one_state;
use serde_json::Value;

const CASES: [&str; 5] = [
    "example-request",
    "example-request-set-threshold",
    "example-request-cjk",
    "example-request-score",
    "long-span-windowed",
];

fn fixture(case: &str) -> Value {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../tests/vela/fixtures");
    serde_json::from_str(
        &std::fs::read_to_string(root.join(format!("native-plan-{case}.json")))
            .expect("fixture present; record it with recipe/vela/export_token_plan.py"),
    )
    .expect("fixture JSON")
}

// ------------------------------------------------------- mapping (no tokens)

fn mapped(case: &str) -> Value {
    let fix = fixture(case);
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
    for case in CASES {
        let got = mapped(case);
        let fix = fixture(case);
        assert_eq!(
            got["predict_request"], fix["predict_request"],
            "predict_request differs for {case}"
        );
        assert_eq!(got["plan"], fix["plan"], "plan differs for {case}");
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

fn bundle_dir() -> Option<PathBuf> {
    let value = std::env::var("VELA_NATIVE_BUNDLE").ok()?;
    let path = PathBuf::from(value);
    path.is_dir().then_some(path)
}

#[test]
fn the_rendered_rows_match_every_fixture() {
    let Some(bundle) = bundle_dir() else {
        eprintln!("skipping: set VELA_NATIVE_BUNDLE to an exported bundle for render parity");
        return;
    };
    let processor = omni_vela_native::processing::Processor::load(&bundle)
        .expect("processor loads the exported bundle");
    for case in CASES {
        let fix = fixture(case);
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
