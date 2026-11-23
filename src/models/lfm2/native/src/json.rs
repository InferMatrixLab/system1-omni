//! JSON behavior used by the Python LFM2 Choice worker and RLCD prompts.

use serde::de::{self, MapAccess, Visitor};
use serde_json::{Map, Number, Value, value::RawValue};
use std::fmt::{self, Write};

pub(crate) fn parse(bytes: &[u8]) -> Result<Value, String> {
    let input = decode(bytes)?;
    let raw =
        serde_json::from_str::<&RawValue>(&input).map_err(|error| malformed(&input, error))?;
    parse_value(raw, 0)
}

fn malformed(text: &str, error: serde_json::Error) -> String {
    let line = text
        .split('\n')
        .nth(error.line().saturating_sub(1))
        .unwrap_or("");
    let column = error.column().saturating_sub(1);
    let mut at_error = line.get(column..).unwrap_or("");
    if column > 0 && line.as_bytes().get(column - 1) == Some(&b'-') {
        at_error = line.get(column - 1..).unwrap_or("");
    }
    for constant in ["NaN", "Infinity", "-Infinity"] {
        if at_error.starts_with(constant) {
            return format!("invalid JSON constant: {constant}");
        }
    }
    format!("malformed JSON: {error}")
}

fn decode(bytes: &[u8]) -> Result<std::borrow::Cow<'_, str>, String> {
    use std::borrow::Cow;

    // json.loads accepts UTF-8/16/32. Detect BOMs and the same null-byte
    // patterns used by Python for BOM-less UTF-16 and UTF-32.
    let dec = |kind| format!("malformed JSON: invalid {kind} encoding");
    let (bytes, width, little) = match bytes {
        [0xff, 0xfe, 0, 0, rest @ ..] => (rest, 4, true),
        [0, 0, 0xfe, 0xff, rest @ ..] => (rest, 4, false),
        [0xff, 0xfe, rest @ ..] => (rest, 2, true),
        [0xfe, 0xff, rest @ ..] => (rest, 2, false),
        [0xef, 0xbb, 0xbf, rest @ ..] => (rest, 1, false),
        [0, 0, _, _, ..] => (bytes, 4, false),
        [_, 0, 0, 0, ..] => (bytes, 4, true),
        [0, _, ..] => (bytes, 2, false),
        [_, 0, ..] => (bytes, 2, true),
        _ => (bytes, 1, false),
    };
    match width {
        1 => std::str::from_utf8(bytes)
            .map(Cow::Borrowed)
            .map_err(|_| dec("UTF-8")),
        2 if bytes.len().is_multiple_of(2) => {
            let words: Vec<_> = bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let word = [chunk[0], chunk[1]];
                    if little {
                        u16::from_le_bytes(word)
                    } else {
                        u16::from_be_bytes(word)
                    }
                })
                .collect();
            String::from_utf16(&words)
                .map(Cow::Owned)
                .map_err(|_| dec("UTF-16"))
        }
        4 if bytes.len().is_multiple_of(4) => {
            let mut output = String::new();
            for chunk in bytes.chunks_exact(4) {
                let word = [chunk[0], chunk[1], chunk[2], chunk[3]];
                let scalar = if little {
                    u32::from_le_bytes(word)
                } else {
                    u32::from_be_bytes(word)
                };
                output.push(char::from_u32(scalar).ok_or_else(|| dec("UTF-32"))?);
            }
            Ok(Cow::Owned(output))
        }
        _ => Err(dec(if width == 2 { "UTF-16" } else { "UTF-32" })),
    }
}

fn parse_value(raw: &RawValue, depth: usize) -> Result<Value, String> {
    if depth > 512 {
        return Err("malformed JSON: nesting exceeds 512 levels".to_owned());
    }
    let text = raw.get();
    let invalid = |error| malformed(text, error);
    match text.as_bytes()[0] {
        b'{' => {
            struct Fields;
            impl<'de> Visitor<'de> for Fields {
                type Value = Vec<(String, &'de RawValue)>;

                fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.write_str("an object")
                }

                fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                    let mut entries = Vec::new();
                    while let Some(entry) = map.next_entry()? {
                        entries.push(entry);
                    }
                    Ok(entries)
                }
            }
            let mut deserializer = serde_json::Deserializer::from_str(text);
            let entries =
                de::Deserializer::deserialize_map(&mut deserializer, Fields).map_err(invalid)?;
            let mut object = Map::new();
            for (key, value) in entries {
                let value = parse_value(value, depth + 1)?;
                if object.contains_key(&key) {
                    return Err(format!("duplicate key: {key}"));
                }
                object.insert(key, value);
            }
            Ok(Value::Object(object))
        }
        b'[' => serde_json::from_str::<Vec<&RawValue>>(text)
            .map_err(invalid)?
            .into_iter()
            .map(|raw| parse_value(raw, depth + 1))
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        b'-' | b'0'..=b'9' => {
            let number: Number = serde_json::from_str(text).map_err(invalid)?;
            if text.contains(['.', 'e', 'E']) && !number.as_f64().is_some_and(f64::is_finite) {
                return Err("non-finite number is not allowed".to_owned());
            }
            if text == "-0" {
                return Ok(Value::from(0));
            }
            Ok(Value::Number(number))
        }
        _ => serde_json::from_str(text).map_err(invalid),
    }
}

// Insertion order, ensure_ascii=False, and Python's default ", " and ": "
// separators are part of the prompt. Preserve large integers as decimal text.
pub(crate) fn dumps(value: &Value) -> String {
    let mut output = String::new();
    render(value, &mut output, false);
    output
}

pub(crate) fn repr(value: &Value) -> String {
    let mut output = String::new();
    render(value, &mut output, true);
    output
}

fn render(value: &Value, output: &mut String, python_repr: bool) {
    match value {
        Value::Object(map) => {
            output.push('{');
            for (index, (key, value)) in map.iter().enumerate() {
                if index > 0 {
                    output.push_str(", ");
                }
                render_string(key, output, python_repr);
                output.push_str(": ");
                render(value, output, python_repr);
            }
            output.push('}');
        }
        Value::Array(items) => {
            output.push('[');
            for (index, value) in items.iter().enumerate() {
                if index > 0 {
                    output.push_str(", ");
                }
                render(value, output, python_repr);
            }
            output.push(']');
        }
        Value::String(text) => render_string(text, output, python_repr),
        Value::Null if python_repr => output.push_str("None"),
        Value::Bool(true) if python_repr => output.push_str("True"),
        Value::Bool(false) if python_repr => output.push_str("False"),
        Value::Number(number) => {
            let text = number.as_str();
            if text.contains(['.', 'e', 'E']) {
                render_float(number.as_f64().expect("validated number"), output);
            } else {
                output.push_str(text);
            }
        }
        _ => output.push_str(&value.to_string()),
    }
}

fn render_float(value: f64, output: &mut String) {
    // Python resolves equally short decimals by round-to-even. Ryu agrees
    // and uses the same exponent thresholds except at exponent -5.
    let mut buffer = ryu::Buffer::new();
    let shortest = buffer.format_finite(value);
    if let Some((mantissa, exponent)) = shortest.split_once('e') {
        let exponent: i32 = exponent.parse().expect("Ryu float exponent");
        write!(output, "{mantissa}e{exponent:+03}").unwrap();
        return;
    }
    let magnitude = shortest.trim_start_matches('-');
    if let Some(digits) = magnitude.strip_prefix("0.0000") {
        let (first, rest) = digits.split_at(1);
        if value.is_sign_negative() {
            output.push('-');
        }
        output.push_str(first);
        if !rest.is_empty() {
            output.push('.');
            output.push_str(rest);
        }
        output.push_str("e-05");
    } else {
        output.push_str(shortest);
    }
}

fn render_string(text: &str, output: &mut String, python_repr: bool) {
    if !python_repr {
        output.push_str(&serde_json::to_string(text).expect("string serialization"));
        return;
    }
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    output.push(quote);
    for c in text.chars() {
        match c {
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            c if c == quote => {
                output.push('\\');
                output.push(c);
            }
            c if c.is_control() || (c != ' ' && c.is_whitespace()) => {
                let n = u32::from(c);
                match n {
                    0..=255 => write!(output, "\\x{n:02x}").unwrap(),
                    256..=65535 => write!(output, "\\u{n:04x}").unwrap(),
                    _ => write!(output, "\\U{n:08x}").unwrap(),
                }
            }
            _ => output.push(c),
        }
    }
    output.push(quote);
}
