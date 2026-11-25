//! Private-item tests for the native crate's state canonicalization,
//! loaded by the `#[path]` wiring in
//! `src/models/vela/native/src/state.rs`.

use super::*;

/// The review probe: `serde_json`'s float forms (`1e20`, `1e-7`)
/// differ from the pinned runtime's `json.dumps` forms
/// (`1e+20`, `1e-07`) and change the token stream (15 vs 17 ids).
#[test]
fn canonical_renders_floats_like_python_json_dumps() {
    let value: Value = serde_json::from_str(r#"{"small":1e-7,"large":1e20}"#).unwrap();
    assert_eq!(canonical(&value), r#"{"large":1e+20,"small":1e-07}"#);
}

/// Scalar rendering tracks Python `repr` per json.dumps: the
/// exponent is always signed with at least two digits, decimal
/// notation runs down to 1e-4, integral floats keep their `.0`,
/// integers stay bare.
#[test]
fn canonical_scalar_cases_track_the_python_repr() {
    let cases = [
        ("1.5e-7", "1.5e-07"),
        ("1e-5", "1e-05"),
        ("0.0001", "0.0001"),
        ("42.0", "42.0"),
        ("42", "42"),
        ("-0.0", "-0.0"),
    ];
    for (input, expected) in cases {
        let value: Value = serde_json::from_str(input).unwrap();
        assert_eq!(canonical(&value), expected, "input {input}");
    }
}

#[test]
fn state_parts_carry_the_canonical_probe_text() {
    let state: Value = serde_json::from_str(r#"{"request":{"small":1e-7,"large":1e20}}"#).unwrap();
    let parts = system_one_state(&state).unwrap();
    assert_eq!(parts.parts[0].text, r#"{"large":1e+20,"small":1e-07}"#);
}
