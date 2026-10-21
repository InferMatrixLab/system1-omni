//! Response finishing, equivalent to `src/models/valen/postprocess.py` at the
//! pinned reference: the FP64 softmax and Valen's confidence formulas, and the
//! response reconstruction with `json.dumps(..., ensure_ascii=False,
//! allow_nan=False)` byte semantics — Python's `repr` floats, `", "`/`": "`
//! separators and control-character escaping, so a native response body is
//! byte-for-byte what the reference worker sends for the same logits.
//!
//! N4 finishes `choice`; the `noul`/`score` variations are N5.

use crate::protocol::MODEL_NAME;

/// The worker's inference failure: the reference's generic handler turns any
/// non-`RequestError` exception into 500 `inference failed`.
#[derive(Debug)]
pub struct Invalid;

pub fn python_float(value: f64) -> String {
    // CPython's repr: shortest round-trip digits, fixed notation while
    // -4 < decimal point position <= 16, otherwise d.ddde±XX (two-digit
    // minimum exponent, no trailing ".0" — repr(1e16) is "1e+16").
    // Rust's LowerExp is also shortest round-trip.
    if !value.is_finite() {
        // allow_nan=False rejects these before serialization; the finisher
        // reports them as an inference failure instead.
        return String::new();
    }
    let (sign, magnitude) = if value < 0.0 || (value == 0.0 && value.is_sign_negative()) {
        ("-", -value)
    } else {
        ("", value)
    };
    let exp_form = format!("{magnitude:e}");
    let (mantissa, exponent) = exp_form.split_once('e').expect("LowerExp format");
    let exponent: i32 = exponent.parse().expect("LowerExp exponent");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let digits = digits.trim_end_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    let decpt = exponent + 1; // digits[0] sits at 10^(decpt-1)
    if decpt <= -4 || decpt > 16 {
        let mantissa = if digits.len() == 1 {
            digits.to_owned()
        } else {
            format!("{head}.{tail}", head = &digits[..1], tail = &digits[1..])
        };
        return format!("{sign}{mantissa}e{exponent:+03}");
    }
    if decpt <= 0 {
        format!("{sign}0.{}{digits}", "0".repeat((-decpt) as usize))
    } else if decpt as usize >= digits.len() {
        format!(
            "{sign}{digits}{}.0",
            "0".repeat(decpt as usize - digits.len())
        )
    } else {
        format!(
            "{sign}{}.{}",
            &digits[..decpt as usize],
            &digits[decpt as usize..]
        )
    }
}

/// `json.dumps(..., ensure_ascii=False)` string escaping: the two mandatory
/// escapes, the five short control forms, `\u00xx` for the rest of C0, and
/// every other code point (including non-ASCII) verbatim.
pub fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// `_probabilities(logits, temperature=1.0)`: FP64 softmax with Python's
/// sequential `sum`, as the reference computes it. The worker's wire contract
/// carries no temperature, so the reference default 1.0 is the only value.
pub fn probabilities(logits: &[f32]) -> Result<Vec<f64>, Invalid> {
    if logits.is_empty() || logits.iter().any(|v| !v.is_finite()) {
        return Err(Invalid);
    }
    let scaled: Vec<f64> = logits.iter().map(|&v| v as f64).collect();
    let maximum = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exponentials: Vec<f64> = scaled.iter().map(|&v| (v - maximum).exp()).collect();
    let total: f64 = exponentials.iter().sum();
    Ok(exponentials.iter().map(|&e| e / total).collect())
}

/// `_choice_confidence`: Valen's original formula, FP64.
fn choice_confidence(probabilities: &[f64]) -> f64 {
    let n = probabilities.len() as f64;
    if probabilities.len() == 1 {
        return 1.0;
    }
    let best = probabilities.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (best - 1.0 / n).max(0.0) / (1.0 - 1.0 / n)
}

/// The first argmax, as Python's `max(range(n), key=...)` picks it.
fn argmax(probabilities: &[f64]) -> usize {
    let mut best = 0;
    for (index, &value) in probabilities.iter().enumerate() {
        if value > probabilities[best] {
            best = index;
        }
    }
    best
}

/// One serialized `choice` answer object, the reference's field order.
pub fn choice_answer(keys: &[String], logits: &[f32]) -> Result<String, Invalid> {
    if logits.len() != keys.len() {
        return Err(Invalid);
    }
    let probabilities = probabilities(logits)?;
    let mode = argmax(&probabilities);
    let pairs: Vec<String> = keys
        .iter()
        .zip(&probabilities)
        .map(|(key, &value)| format!("{}: {}", json_string(key), python_float(value)))
        .collect();
    Ok(format!(
        "{{\"type\": \"choice\", \"choice\": {}, \"probabilities\": {{{}}}, \"confidence\": {}}}",
        json_string(&keys[mode]),
        pairs.join(", "),
        python_float(choice_confidence(&probabilities)),
    ))
}

/// The serialized full response body, `build_response`'s field order and
/// dual token accounting: logical input tokens plus per-branch compute tokens.
pub fn build_response(
    answers: &[(String, String)],
    logical_tokens: usize,
    compute_tokens: usize,
) -> Result<Vec<u8>, Invalid> {
    let answers: Vec<String> = answers
        .iter()
        .map(|(qid, answer)| format!("{}: {}", json_string(qid), answer))
        .collect();
    let body = format!(
        "{{\"model\": {}, \"answers\": {{{}}}, \"usage\": {{\"input_tokens\": {}, \"output_tokens\": 0}}, \"internal_usage\": {{\"compute_tokens\": {}}}}}",
        json_string(MODEL_NAME),
        answers.join(", "),
        logical_tokens,
        compute_tokens,
    );
    Ok(body.into_bytes())
}
