//! Private-item tests for the native crate's finished-response gate,
//! loaded by the `#[path]` wiring in
//! `src/models/vela/native/src/scoring.rs`.

use super::validate_finished_response;
use serde_json::json;

#[test]
fn final_response_rejects_an_extra_answer_key() {
    let plan = vec![json!({
        "qid": "q",
        "type": "noul",
        "criteria": {},
        "range": null,
        "part": "user",
    })];
    let response = json!({
        "model": "m",
        "answers": {
            "q": {"type": "noul", "noul": 0.5},
            "extra": {"type": "noul", "noul": 0.1},
        },
        "usage": {"input_tokens": 0, "output_tokens": 0},
    });
    let error = validate_finished_response(&response, &plan)
        .expect_err("an extra answer key must not pass the envelope gate");
    assert!(error.to_string().contains("answers"), "{error}");
}
