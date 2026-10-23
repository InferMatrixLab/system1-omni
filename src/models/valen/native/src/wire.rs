//! The wire layer's Python-behaviour helpers and the JSON body decode.
//!
//! [`parse_body`] reproduces `src/models/valen/protocol.py::parse_body` byte
//! for byte on the error surface: the 413 body limit, the three literal
//! non-finite JSON constants, duplicate keys (each with the reference's exact
//! `detail` text), and the generic malformed-JSON body for everything else
//! serde rejects (invalid UTF-8, lone surrogates, trailing data, numbers out
//! of range, non-object roots).

use std::cell::RefCell;

use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::value::RawValue;
use serde_json::{Map, Number, Value};

/// The worker's error: an HTTP status and the `detail` body text.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct RequestError {
    pub status: u16,
    pub message: String,
}

pub fn error(status: u16, message: impl Into<String>) -> RequestError {
    RequestError {
        status,
        message: message.into(),
    }
}

/// Python's `repr` for the strings error messages quote: single quotes unless
/// the text contains `'` and not `"`, with `\\`, `\n`, `\r`, `\t`, the active
/// quote and control characters escaped (`\xHH` below U+0100). Non-printable
/// non-control code points (e.g. most format characters) are left verbatim —
/// Python escapes those as well, but the wire corpus never reaches them.
pub fn python_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f || (0x80..=0x9f).contains(&(c as u32)) => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Python's `str.isspace`, which counts the four file/group/record/unit
/// separators as whitespace on top of Rust's `char::is_whitespace`.
pub fn python_isspace(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}')
}

/// Python's `str.strip`.
pub fn python_strip(s: &str) -> &str {
    s.trim_matches(python_isspace)
}

/// The spans of the literal non-finite JSON constants (`NaN`, `Infinity`,
/// `-Infinity`) that sit in value positions, in order.
///
/// Python's `json.loads` accepts them as value tokens and the reference
/// rejects them by name; a constant anywhere else (inside a string, or where a
/// key is expected) is ordinary invalid JSON and keeps the generic message, so
/// the scan tracks string and container state. Python's scanner errors the
/// moment it *reaches* the first constant, whatever follows; reproducing that
/// ordering exactly would mean porting the scanner (constants raise at reach,
/// the duplicate-key hook at object close, surrogates and out-of-range floats
/// only at the round-trip check). [`parse_body`] instead gives the constant
/// the specific detail only when the whole document — constants substituted
/// with `null` — is well-formed, and the generic malformed-JSON body
/// otherwise: a recorded detail-text deviation on doubly-invalid bodies (the
/// status is 400 either way), pinned below and in recipe/valen/native.md.
fn nonfinite_spans(raw: &[u8]) -> Vec<(usize, usize)> {
    const CONSTANTS: [&[u8]; 3] = [b"Infinity", b"-Infinity", b"NaN"];
    let mut spans = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    // A value may start at the input start, after `[`, after `:`, and after a
    // `,` inside an array; object positions expect a string key instead.
    let mut value_position = true;
    let mut containers: Vec<u8> = Vec::new();
    let mut i = 0;
    while i < raw.len() {
        let b = raw[i];
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        match b {
            b'"' => {
                in_string = true;
                value_position = false;
            }
            b' ' | b'\t' | b'\n' | b'\r' => {}
            b'[' => {
                containers.push(b'[');
                value_position = true;
            }
            b'{' => {
                containers.push(b'{');
                value_position = false;
            }
            b',' => {
                value_position = containers.last() == Some(&b'[');
            }
            b':' => value_position = true,
            b']' | b'}' => {
                containers.pop();
                value_position = false;
            }
            _ => {
                if value_position
                    && let Some(length) = CONSTANTS
                        .iter()
                        .find_map(|c| raw[i..].starts_with(c).then_some(c.len()))
                {
                    spans.push((i, i + length));
                }
                value_position = false;
            }
        }
        i += 1;
    }
    spans
}

/// Read containers as raw JSON so numeric values are handled exactly once and
/// feature unification cannot confuse private-marker object keys.
struct PythonValue<'dup> {
    depth: usize,
    duplicate: &'dup RefCell<Option<String>>,
}

impl<'de> Visitor<'de> for PythonValue<'_> {
    type Value = Value;

    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("a JSON value")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
        let mut items = Vec::new();
        while let Some(raw) = seq.next_element::<&'de RawValue>()? {
            items.push(parse_raw(raw, self.depth + 1, self.duplicate).map_err(de::Error::custom)?);
        }
        Ok(Value::Array(items))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        let mut obj = Map::new();
        while let Some((key, raw)) = map.next_entry::<String, &'de RawValue>()? {
            if obj.contains_key(&key) {
                if self.duplicate.borrow().is_none() {
                    *self.duplicate.borrow_mut() = Some(key);
                }
                return Err(de::Error::custom("duplicate key"));
            }
            obj.insert(
                key,
                parse_raw(raw, self.depth + 1, self.duplicate).map_err(de::Error::custom)?,
            );
        }
        Ok(Value::Object(obj))
    }
}

fn parse_raw(
    raw: &RawValue,
    depth: usize,
    duplicate: &RefCell<Option<String>>,
) -> serde_json::Result<Value> {
    let text = raw.get();
    if !matches!(text.as_bytes()[0], b'{' | b'[') {
        let value: Value = serde_json::from_str(text)?;
        if let Value::Number(n) = value {
            if n.is_i64() || n.is_u64() {
                return Ok(Value::Number(n));
            }
            // Python accepts out-of-range floats at parse and rejects them in
            // the round-trip check; both surface as the generic 400 body.
            return n
                .as_f64()
                .and_then(Number::from_f64)
                .map(Value::Number)
                .ok_or_else(|| de::Error::custom("number out of range"));
        }
        return Ok(value);
    }
    if depth >= 127 {
        // Python raises RecursionError, which parse_body reports as malformed.
        return Err(de::Error::custom("recursion limit exceeded"));
    }
    serde_json::Deserializer::from_str(text).deserialize_any(PythonValue { depth, duplicate })
}

/// Decode a bounded request body; errors carry the reference's exact wording.
pub fn parse_body(raw: &[u8]) -> Result<Map<String, Value>, RequestError> {
    const MAX_BODY: usize = 8 * 1024 * 1024;
    if raw.len() > MAX_BODY {
        return Err(error(413, "request body exceeds 8 MiB"));
    }
    let spans = nonfinite_spans(raw);
    if !spans.is_empty() {
        // Python's scanner reaches the first constant only when everything
        // before it parsed, so the substituted document must be valid.
        let mut substituted = Vec::with_capacity(raw.len());
        let mut at = 0;
        for (start, end) in &spans {
            substituted.extend_from_slice(&raw[at..*start]);
            substituted.extend_from_slice(b"null");
            at = *end;
        }
        substituted.extend_from_slice(&raw[at..]);
        if serde_json::from_slice::<serde::de::IgnoredAny>(&substituted).is_ok() {
            let (start, end) = spans[0];
            return Err(error(
                400,
                format!(
                    "non-finite JSON number {} is not supported",
                    std::str::from_utf8(&raw[start..end]).unwrap_or("NaN")
                ),
            ));
        }
        return Err(error(
            400,
            "request body must contain valid JSON and UTF-8 text",
        ));
    }
    let duplicate = RefCell::new(None);
    let value = serde_json::from_slice::<&RawValue>(raw)
        .map_err(|_| error(400, "request body must contain valid JSON and UTF-8 text"))
        .and_then(|raw| {
            parse_raw(raw, 0, &duplicate)
                .map_err(|_| error(400, "request body must contain valid JSON and UTF-8 text"))
        });
    match value {
        Ok(Value::Object(map)) => Ok(map),
        Ok(_) => Err(error(400, "request must be a JSON object")),
        Err(_) => Err(match duplicate.into_inner() {
            Some(key) => error(400, format!("duplicate JSON key {}", python_repr(&key))),
            None => error(400, "request body must contain valid JSON and UTF-8 text"),
        }),
    }
}

#[cfg(test)]
mod tests {
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
}
