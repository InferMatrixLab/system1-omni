//! Processing tests against the frozen token-plan fixtures.
//!
//! The mapping tests (state → parts, SystemOne → predict, plan) run
//! everywhere with no tokenizer and no weights. The full render parity
//! — rows and window rows byte-for-byte against the fixtures — needs
//! the exported bundle's tokenizer and is opt-in:
//!
//!     VELA_NATIVE_BUNDLE=weights/vela-2.0-0.8b-native \
//!       VELA_NATIVE_MANIFEST_SHA256=<the digest the exporter prints> \
//!       cargo test -p omni-vela-native --test vela_native_processing -- --ignored --nocapture

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

fn bundle_dir() -> PathBuf {
    let path = PathBuf::from(
        std::env::var("VELA_NATIVE_BUNDLE").expect("set VELA_NATIVE_BUNDLE to an exported bundle"),
    );
    assert!(path.is_dir(), "{} is not a directory", path.display());
    path
}

fn manifest_anchor() -> String {
    std::env::var("VELA_NATIVE_MANIFEST_SHA256")
        .expect("set VELA_NATIVE_MANIFEST_SHA256 to the manifest digest the exporter prints")
}

#[test]
#[ignore = "requires VELA_NATIVE_BUNDLE at an exported bundle; CPU only"]
fn the_rendered_rows_match_every_fixture() {
    let bundle = bundle_dir();
    let processor = omni_vela_native::processing::Processor::load(&bundle, &manifest_anchor())
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

// --------------------------------------------- structured state (opt-in)

#[test]
#[ignore = "requires VELA_NATIVE_BUNDLE at an exported bundle; CPU only"]
fn the_structured_state_tokenizes_like_the_pinned_runtime() {
    let bundle = bundle_dir();
    let encoder = omni_vela_native::render::Encoder::load(&bundle.join("tokenizer.json"))
        .expect("the pinned tokenizer loads");
    // The review probe: serde_json's float forms ("1e20", "1e-7") change
    // the token stream against the pinned runtime's json.dumps forms
    // ("1e+20", "1e-07") — 15 vs 17 ids on this state.
    let state: Value = serde_json::from_str(r#"{"request":{"small":1e-7,"large":1e20}}"#).unwrap();
    let parts = omni_vela_native::state::system_one_state(&state).expect("state maps");
    assert_eq!(
        parts.parts[0].text, r#"{"large":1e+20,"small":1e-07}"#,
        "the state text must render exactly as Python's json.dumps does"
    );
    let ids = encoder
        .ids(&parts.parts[0].text)
        .expect("part text encodes");
    assert_eq!(
        ids,
        vec![
            4754u32, 16275, 763, 16, 68, 10, 17, 15, 1288, 8732, 763, 16, 68, 12, 15, 22, 92
        ],
        "the state text must tokenize exactly as the pinned runtime's tokenizer does"
    );
}
