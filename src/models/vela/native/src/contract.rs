//! Vela 2.0 wire admission for the native worker, mirroring
//! `models.vela.protocol` and the pinned runtime's request preamble so
//! clients cannot tell this layer from the reference worker: the
//! three-field envelope with upstream's exact `{"detail": [{"loc",
//! "msg", "type"}]}` triples, and the whole-request error model.
//!
//! Deeper validation — state rendering, question semantics, the 413 —
//! lives in the processing modules and mirrors the pinned runtime
//! against the `native-plan-*` fixtures. One known residual
//! difference: a non-finite JSON number answers the generic 400
//! message here, where the Python worker names the value; duplicate
//! keys are mirrored verbatim.

use serde_json::{Map, Value, json};

/// Slice-scoped transport cap, as on the reference worker.
pub const MAX_BODY: usize = 8 * 1024 * 1024;
/// The envelope fields, in upstream's missing-field check order.
pub const REQUEST_FIELDS: [&str; 3] = ["state", "model", "questions"];

/// A request rejection: an HTTP status and the JSON body a client sees.
#[derive(Debug, PartialEq)]
pub struct RequestError {
    pub status: u16,
    pub body: Value,
}

impl RequestError {
    fn detail(status: u16, message: &str) -> Self {
        Self {
            status,
            body: json!({"detail": message}),
        }
    }

    /// One upstream `SystemOneError` as a whole-request 422 triple.
    pub fn envelope(loc: &[&str], msg: &str, type_: &str) -> Self {
        let mut path = vec![Value::from("body")];
        path.extend(loc.iter().map(|part| Value::from(*part)));
        Self {
            status: 422,
            body: json!({"detail": [{"loc": path, "msg": msg, "type": type_}]}),
        }
    }
}

/// Decode a bounded JSON body and reject ambiguous JSON values.
pub fn parse_body(raw: &[u8]) -> Result<Map<String, Value>, RequestError> {
    if raw.len() > MAX_BODY {
        return Err(RequestError::detail(413, "request body exceeds 8 MiB"));
    }
    let message =
        omni_qwen3_5_native::json::parse(raw).map_err(|error| normalize_parse_error(&error))?;
    Ok(message)
}

/// Mirror the reference worker's 400 messages: duplicate keys verbatim,
/// everything else the shared generic message.
fn normalize_parse_error(error: &str) -> RequestError {
    if let Some(rest) = error.split("duplicate key ").nth(1) {
        // serde appends its position suffix after the quoted key.
        if let Some(key) = rest
            .strip_prefix('"')
            .and_then(|tail| tail.split('"').next())
        {
            return RequestError::detail(400, &format!("duplicate JSON key '{key}'"));
        }
    }
    RequestError::detail(400, "request body must contain valid JSON and UTF-8 text")
}

/// Admit the envelope exactly as upstream's request preamble does, for
/// one served model name: the three required fields present, no extra
/// field, `model` non-blank and equal to the served name.
pub fn admit_request(body: &Map<String, Value>, served: &str) -> Result<(), RequestError> {
    for field in REQUEST_FIELDS {
        if !body.contains_key(field) {
            return Err(RequestError::envelope(
                &[field],
                "Field required",
                "missing",
            ));
        }
    }
    let mut extra: Vec<&String> = body
        .keys()
        .filter(|key| !REQUEST_FIELDS.contains(&key.as_str()))
        .collect();
    extra.sort();
    if let Some(first) = extra.first() {
        return Err(RequestError::envelope(
            &[first],
            "Extra inputs are not permitted",
            "extra_forbidden",
        ));
    }
    let model = body["model"].as_str().unwrap_or("");
    if model.trim().is_empty() {
        return Err(RequestError::envelope(
            &["model"],
            "model must not be empty or whitespace",
            "value_error",
        ));
    }
    if model != served {
        return Err(RequestError::envelope(
            &["model"],
            "Model is not served by this runtime",
            "value_error.model",
        ));
    }
    Ok(())
}

/// Admit one complete raw request body: parse, then envelope-check.
pub fn compile(raw: &[u8], served: &str) -> Result<Map<String, Value>, RequestError> {
    let body = parse_body(raw)?;
    admit_request(&body, served)?;
    Ok(body)
}

/// A pinned-runtime `ValueError` rejection as the reference worker maps
/// it (`map_runtime_error`): a whole-request 422 naming the runtime.
pub fn runtime_rejection(message: &str) -> RequestError {
    RequestError::detail(
        422,
        &format!("request rejected by the pinned runtime: {message}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_keys_mirror_the_reference_message() {
        let error =
            normalize_parse_error("request body is not valid JSON: duplicate key \"state\"");
        assert_eq!(error.status, 400);
        assert_eq!(error.body, json!({"detail": "duplicate JSON key 'state'"}));
    }
}
