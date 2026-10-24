//! The wire layer's inline-suite bodies, wired in from
//! `src/models/valen/native/src/wire.rs` under `#[cfg(test)]` (tests of
//! private items keep only the wiring in `src/`, per CONTRIBUTING).

use super::*;

#[test]
fn repr_quotes_like_python() {
    assert_eq!(python_repr("plain"), "'plain'");
    assert_eq!(python_repr("it's"), "\"it's\"");
    assert_eq!(python_repr("say \"hi\""), "'say \"hi\"'");
    assert_eq!(python_repr("it's \"both\""), "'it\\'s \"both\"'");
    assert_eq!(python_repr("a\nb\x01"), "'a\\nb\\x01'");
}

#[test]
fn trim_strips_python_whitespace() {
    assert_eq!(python_strip("\u{a0}x\u{a0}"), "x");
    assert_eq!(python_strip("\u{1c}x\u{1f}"), "x");
    assert_eq!(python_strip("\u{3000}x\u{2028}"), "x");
    assert_eq!(python_strip("  hi  "), "hi");
}

#[test]
fn nonfinite_scan_matches_value_positions() {
    assert_eq!(nonfinite_spans(b"{\"a\": NaN}"), vec![(6, 9)]);
    assert_eq!(nonfinite_spans(b"[Infinity]"), vec![(1, 9)]);
    assert_eq!(nonfinite_spans(b"{\"a\": -Infinity}"), vec![(6, 15)]);
    assert_eq!(
        nonfinite_spans(b"{\"NaN\": 1}"),
        Vec::<(usize, usize)>::new()
    );
    assert_eq!(nonfinite_spans(b"{,NaN}"), Vec::<(usize, usize)>::new());
    assert_eq!(
        nonfinite_spans(b"{\"a\": 1}NaN"),
        Vec::<(usize, usize)>::new()
    );
}

#[test]
fn malformed_before_a_constant_keeps_the_generic_body() {
    // Python's scanner fails at the ',' before ever reaching the constant.
    assert_eq!(
        parse_body(b"[,NaN]").unwrap_err(),
        error(400, "request body must contain valid JSON and UTF-8 text")
    );
}

/// Recorded deviation (recipe/valen/native.md): CPython's scanner errors
/// as soon as it reaches the first constant, so a body that is also
/// malformed *after* it still earns the specific non-finite detail; this
/// layer answers specifically only when the whole substituted document is
/// well-formed, and keeps the generic body here. The status agrees (400
/// both ways); only the detail text differs, on doubly-invalid bodies.
#[test]
fn malformed_after_a_constant_keeps_the_generic_body_recorded_deviation() {
    // Missing separator between the members.
    assert_eq!(
        parse_body(b"{\"a\": NaN \"b\": NaN}").unwrap_err(),
        error(400, "request body must contain valid JSON and UTF-8 text")
    );
    // Trailing data after an otherwise complete document.
    assert_eq!(
        parse_body(b"{\"a\": NaN}trailing").unwrap_err(),
        error(400, "request body must contain valid JSON and UTF-8 text")
    );
    // A container left open after the constant.
    assert_eq!(
        parse_body(b"[NaN").unwrap_err(),
        error(400, "request body must contain valid JSON and UTF-8 text")
    );
}

/// The reverse ordering of the same deviation: a duplicate-key object
/// that *closes* before a later constant makes the reference report the
/// duplicate (its hook fires at object close, before the scanner reaches
/// the constant), while the native pre-check must not fail on duplicates
/// and so reaches the constant and answers with the non-finite detail.
#[test]
fn a_closed_duplicate_object_before_a_constant_keeps_the_nonfinite_detail() {
    assert_eq!(
        parse_body(b"[{\"a\": 1, \"a\": 2}, NaN]").unwrap_err(),
        error(400, "non-finite JSON number NaN is not supported")
    );
}

#[test]
fn body_errors_match_the_reference() {
    assert_eq!(
        parse_body(b"{\"model\": \"x\", \"model\": \"y\"}").unwrap_err(),
        error(400, "duplicate JSON key 'model'")
    );
    assert_eq!(
        parse_body(b"{\"model\": NaN}").unwrap_err(),
        error(400, "non-finite JSON number NaN is not supported")
    );
    assert_eq!(
        parse_body(b"[1, 2]").unwrap_err(),
        error(400, "request must be a JSON object")
    );
    assert_eq!(
        parse_body(b"{\"a\": 1e999}").unwrap_err(),
        error(400, "request body must contain valid JSON and UTF-8 text")
    );
    assert_eq!(
        parse_body(b"{\"a\": \"\\ud800\"}").unwrap_err(),
        error(400, "request body must contain valid JSON and UTF-8 text")
    );
    assert_eq!(
        parse_body(&[]).unwrap_err(),
        error(400, "request body must contain valid JSON and UTF-8 text")
    );
    let mut oversize = vec![b' '; 8 * 1024 * 1024 + 1];
    oversize[0] = b'{';
    assert_eq!(
        parse_body(&oversize).unwrap_err(),
        error(413, "request body exceeds 8 MiB")
    );
}
