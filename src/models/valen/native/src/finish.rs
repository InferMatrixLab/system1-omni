//! Response finishing, equivalent to `src/models/valen/postprocess.py` at the
//! pinned reference: the FP64 softmax and Valen's three answer variants
//! (`choice`, `noul`, `score`), and the response reconstruction with
//! `json.dumps(..., ensure_ascii=False, allow_nan=False)` byte semantics —
//! Python's `repr` floats, `", "`/`": "` separators and control-character
//! escaping, so a native response body is byte-for-byte what the reference
//! worker sends for the same logits.

use crate::protocol::{Kind, MODEL_NAME};

/// The worker's inference failure: the reference's generic handler turns any
/// non-`RequestError` exception into 500 `inference failed`.
#[derive(Debug)]
pub struct Invalid;

pub fn python_float(value: f64) -> String {
    // CPython's repr: shortest round-trip digits, fixed notation while
    // -4 < decimal point position <= 16, otherwise d.ddde±XX (two-digit
    // minimum exponent, no trailing ".0" — repr(1e16) is "1e+16").
    if !value.is_finite() {
        // allow_nan=False rejects these before serialization; the finisher
        // reports them as an inference failure instead.
        return String::new();
    }
    // The shared qwen3_5 float_repr is the repository's CPython-parity repr.
    // Deriving from Rust's own {:e} here would diverge at exact
    // shortest-decimal ties, where {:e} picks the upper digit and CPython's
    // repr picks the even one (pinned by the tests/qwen3_5/json.rs oracle).
    omni_qwen3_5_native::json::float_repr(value)
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
    let best = probabilities
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
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

/// One serialized `noul` answer object (`_noul_answer`): P(true) — the wire
/// layer pins `keys` to `["true", "false"]` in that order — plus the full
/// pair and the choice confidence over the two candidates. The reference
/// answers the *wire* type `noul` even when custom wording compiled the
/// question as a `choice` record, so the finishing depends on nothing but
/// the keys and probabilities.
pub fn noul_answer(keys: &[String], logits: &[f32]) -> Result<String, Invalid> {
    if logits.len() != keys.len() {
        return Err(Invalid);
    }
    let probabilities = probabilities(logits)?;
    let pairs: Vec<String> = keys
        .iter()
        .zip(&probabilities)
        .map(|(key, &value)| format!("{}: {}", json_string(key), python_float(value)))
        .collect();
    Ok(format!(
        "{{\"type\": \"noul\", \"noul\": {}, \"probabilities\": {{{}}}, \"confidence\": {}}}",
        python_float(probabilities[0]),
        pairs.join(", "),
        python_float(choice_confidence(&probabilities)),
    ))
}

/// One serialized `score` answer object (`_score_answer`): the K level
/// probabilities, the expected level `sum(i * p)` (Python's sequential
/// `sum`, i.e. left-to-right from 0), the static legend keyed `"0"`…, and
/// the distribution-concentration confidence
/// `max(0, 1 - distance / uniform_mad)` where `distance` is the probability-
/// weighted absolute deviation from the first argmax level and
/// `uniform_mad` the mean absolute deviation of the uniform distribution
/// over K levels. The wire layer guarantees K >= 2, so `uniform_mad` is
/// never zero.
pub fn score_answer(
    keys: &[String],
    descriptions: &[String],
    logits: &[f32],
) -> Result<String, Invalid> {
    if logits.len() != keys.len() || descriptions.len() != keys.len() {
        return Err(Invalid);
    }
    let probabilities = probabilities(logits)?;
    let mode = argmax(&probabilities);
    let count = probabilities.len();
    let distance: f64 = probabilities
        .iter()
        .enumerate()
        .map(|(index, &value)| value * (index as f64 - mode as f64).abs())
        .sum();
    let uniform_mad: f64 = (0..count)
        .map(|index| (index as f64 - (count - 1) as f64 / 2.0).abs())
        .sum::<f64>()
        / count as f64;
    let expected_level: f64 = probabilities
        .iter()
        .enumerate()
        .map(|(index, &value)| index as f64 * value)
        .sum();
    let pairs: Vec<String> = keys
        .iter()
        .zip(&probabilities)
        .map(|(key, &value)| format!("{}: {}", json_string(key), python_float(value)))
        .collect();
    let legend: Vec<String> = keys
        .iter()
        .zip(descriptions)
        .map(|(key, description)| format!("{}: {}", json_string(key), json_string(description)))
        .collect();
    Ok(format!(
        "{{\"type\": \"score\", \"probabilities\": {{{}}}, \"score\": {}, \"legend\": {{{}}}, \"confidence\": {}}}",
        pairs.join(", "),
        python_float(expected_level),
        legend.join(", "),
        python_float((1.0 - distance / uniform_mad).max(0.0)),
    ))
}

/// The per-kind dispatch of `postprocess.answer`: the wire type (not the
/// compiled record type) selects the finishing variant, exactly as the
/// reference answers `response_context.questions` — the original wire
/// questions — so a custom-wording `noul` still answers `type: noul`.
pub fn answer(
    kind: Kind,
    keys: &[String],
    descriptions: &[String],
    logits: &[f32],
) -> Result<String, Invalid> {
    match kind {
        Kind::Choice => choice_answer(keys, logits),
        Kind::Noul => noul_answer(keys, logits),
        Kind::Score => score_answer(keys, descriptions, logits),
    }
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
