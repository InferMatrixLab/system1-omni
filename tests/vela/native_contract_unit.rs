//! Private-item tests for the native crate's wire contract, loaded by
//! the `#[path]` wiring in `src/models/vela/native/src/contract.rs`.

use super::*;

#[test]
fn duplicate_keys_mirror_the_reference_message() {
    let error = normalize_parse_error("request body is not valid JSON: duplicate key \"state\"");
    assert_eq!(error.status, 400);
    assert_eq!(
        error.body,
        serde_json::json!({"detail": "duplicate JSON key 'state'"})
    );
}
