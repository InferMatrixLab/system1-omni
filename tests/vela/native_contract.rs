//! Wire-contract tests for the native Vela 2.0 worker, mirroring the
//! reference worker's protocol tests: the envelope triples must be
//! byte-equal to the pinned runtime's rejections.

use omni_vela_native::contract::{self, RequestError};
use serde_json::json;

fn body(raw: &str) -> serde_json::Map<String, serde_json::Value> {
    contract::parse_body(raw.as_bytes()).expect("valid body")
}

fn detail(loc: &[&str], msg: &str, type_: &str) -> serde_json::Value {
    let mut path = vec![json!("body")];
    path.extend(loc.iter().map(|part| json!(part)));
    json!({"detail": [{"loc": path, "msg": msg, "type": type_}]})
}

const SERVED: &str = "vllm-sr/Vela-2.0-0.8B";

fn raw_body() -> String {
    format!(
        r#"{{"model": "{SERVED}", "state": "My card was charged twice.", "questions": {{"billing": {{"type": "noul", "instructions": "any?"}}}}}}"#
    )
}

#[test]
fn the_envelope_passes_through_unchanged() {
    let parsed = body(&raw_body());
    contract::admit_request(&parsed, SERVED).expect("admitted");
    assert_eq!(parsed.len(), 3);
}

#[test]
fn a_missing_field_reports_upstreams_missing_triple() {
    for drop in ["state", "model", "questions"] {
        let mut parsed = body(&raw_body());
        parsed.shift_remove(drop);
        let error = contract::admit_request(&parsed, SERVED).unwrap_err();
        assert_eq!(
            error,
            RequestError::envelope(&[drop], "Field required", "missing")
        );
        assert_eq!(error.body, detail(&[drop], "Field required", "missing"));
    }
}

#[test]
fn an_extra_field_reports_upstreams_extra_forbidden() {
    let mut parsed = body(&raw_body());
    parsed.insert("images".into(), json!([]));
    let error = contract::admit_request(&parsed, SERVED).unwrap_err();
    assert_eq!(error.status, 422);
    assert_eq!(
        error.body,
        detail(
            &["images"],
            "Extra inputs are not permitted",
            "extra_forbidden"
        )
    );
}

#[test]
fn a_blank_model_is_rejected_before_the_name_check() {
    let mut parsed = body(&raw_body());
    parsed.insert("model".into(), json!("  "));
    let error = contract::admit_request(&parsed, SERVED).unwrap_err();
    assert_eq!(
        error.body,
        detail(
            &["model"],
            "model must not be empty or whitespace",
            "value_error"
        )
    );
}

#[test]
fn another_members_name_is_rejected_with_the_model_triple() {
    let mut parsed = body(&raw_body());
    parsed.insert("model".into(), json!("vllm-sr/Vela-2.0-0.3B"));
    let error = contract::admit_request(&parsed, SERVED).unwrap_err();
    assert_eq!(
        error.body,
        detail(
            &["model"],
            "Model is not served by this runtime",
            "value_error.model"
        )
    );
}

#[test]
fn duplicate_keys_are_rejected_with_the_reference_message() {
    let error =
        contract::parse_body(br#"{"model": "x", "model": "y", "state": "s", "questions": {}}"#)
            .unwrap_err();
    assert_eq!(error.status, 400);
    assert_eq!(error.body, json!({"detail": "duplicate JSON key 'model'"}));
}

#[test]
fn invalid_json_is_a_generic_400() {
    let error = contract::parse_body(b"{not json").unwrap_err();
    assert_eq!(error.status, 400);
    assert_eq!(
        error.body,
        json!({"detail": "request body must contain valid JSON and UTF-8 text"})
    );
}

#[test]
fn a_non_object_body_is_rejected() {
    let error = contract::parse_body(b"[1, 2]").unwrap_err();
    assert_eq!(error.status, 400);
}

#[test]
fn compile_admits_and_returns_the_body() {
    let parsed = contract::compile(raw_body().as_bytes(), SERVED).expect("admitted");
    assert!(parsed.contains_key("questions"));
}
