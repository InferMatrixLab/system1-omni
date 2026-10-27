//! DEV2.0 canonical JSON and the state → typed-parts mapping, mirroring
//! the pinned runtime's `canonical_json`, `content_text` and
//! `system_one_state` exactly (the frozen `native-plan-*` fixtures are
//! the parity target).

use serde_json::Value;
use std::collections::BTreeMap;

use crate::contract::RequestError;

/// `json.dumps(ensure_ascii=False, sort_keys=True, separators=(",", ":"))`
/// — key-sorted compact JSON, non-ASCII literal.
pub fn canonical(value: &Value) -> String {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let parts: Vec<String> = keys
                .iter()
                .map(|key| {
                    format!(
                        "{}:{}",
                        serde_json::to_string(key).unwrap(),
                        canonical(&map[*key])
                    )
                })
                .collect();
            format!("{{{}}}", parts.join(","))
        }
        Value::Array(items) => {
            let parts: Vec<String> = items.iter().map(canonical).collect();
            format!("[{}]", parts.join(","))
        }
        other => serde_json::to_string(other).unwrap(),
    }
}

/// JSON content → model text: strings as they are, anything else as
/// canonical JSON.
pub fn content_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => canonical(other),
    }
}

/// The state-key → part-role table (unknown keys land in `context`).
pub fn key_role(key: &str) -> &'static str {
    match key.to_ascii_lowercase().as_str() {
        "request" | "user" | "prompt" => "user",
        "answer" | "response" => "answer",
        _ => "context",
    }
}

/// One rendered state part, in user / context / answer order.
#[derive(Debug, Clone, serde::Serialize, PartialEq)]
pub struct Part {
    pub r#type: String,
    pub text: String,
}

/// A named state field's span inside its part's text (Unicode code
/// points, as on the reference).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldRange {
    pub role: &'static str,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug)]
pub struct StateParts {
    pub parts: Vec<Part>,
    pub fields: BTreeMap<String, FieldRange>,
}

fn state_error(msg: &str) -> RequestError {
    RequestError::envelope(&["state"], msg, "value_error")
}

/// The SystemOne state → (parts, fields) mapping, mirroring
/// `system_one_state`: a plain string is one user part; a JSON array is
/// canonicalized into one; an object groups its keys by role (request /
/// user / prompt → user; source / context / document / tools /
/// previous_answer → context; answer / response → answer; unknown keys
/// → context) in user / context / answer order — one known-key entry
/// per role renders bare, several (or an unknown key) render as
/// `"key:\n<value>"` blocks joined by blank lines.
pub fn system_one_state(state: &Value) -> Result<StateParts, RequestError> {
    // A string holding a JSON object is read as that object.
    let state = match state {
        Value::String(text) => {
            let stripped = text.trim_start();
            if stripped.starts_with('{') {
                if let Value::Object(object) =
                    serde_json::from_str::<Value>(stripped).unwrap_or(Value::Null)
                {
                    Value::Object(object)
                } else {
                    state.clone()
                }
            } else {
                state.clone()
            }
        }
        other => other.clone(),
    };
    match &state {
        Value::String(text) => {
            if text.trim().is_empty() {
                return Err(state_error("state must not be empty or whitespace"));
            }
            let mut fields = BTreeMap::new();
            fields.insert(
                "request".into(),
                FieldRange {
                    role: "user",
                    start: 0,
                    end: text.chars().count(),
                },
            );
            Ok(StateParts {
                parts: vec![Part {
                    r#type: "user".into(),
                    text: text.clone(),
                }],
                fields,
            })
        }
        Value::Array(_) => {
            let text = canonical(&state);
            let mut fields = BTreeMap::new();
            fields.insert(
                "request".into(),
                FieldRange {
                    role: "user",
                    start: 0,
                    end: text.chars().count(),
                },
            );
            Ok(StateParts {
                parts: vec![Part {
                    r#type: "user".into(),
                    text,
                }],
                fields,
            })
        }
        Value::Object(object) if !object.is_empty() => {
            // Group per role, keeping the object's insertion order inside
            // each role and remembering which entries had a known key.
            let mut by_role: [Vec<(&String, &Value, bool)>; 3] =
                [Vec::new(), Vec::new(), Vec::new()];
            for (key, value) in object {
                let lower = key.to_ascii_lowercase();
                let known = matches!(
                    lower.as_str(),
                    "request"
                        | "user"
                        | "prompt"
                        | "source"
                        | "context"
                        | "document"
                        | "tools"
                        | "previous_answer"
                        | "answer"
                        | "response"
                );
                let slot = match key_role(&lower) {
                    "user" => 0,
                    "answer" => 2,
                    _ => 1,
                };
                by_role[slot].push((key, value, known));
            }
            let mut parts = Vec::new();
            let mut fields = BTreeMap::new();
            for (role, items) in [
                ("user", &by_role[0]),
                ("context", &by_role[1]),
                ("answer", &by_role[2]),
            ] {
                if items.is_empty() {
                    continue;
                }
                let mut text = String::new();
                let single = items.len() == 1 && items[0].2;
                for (index, (key, value, _)) in items.iter().enumerate() {
                    let rendered = content_text(value);
                    if single {
                        fields.insert(
                            (*key).clone(),
                            FieldRange {
                                role,
                                start: 0,
                                end: rendered.chars().count(),
                            },
                        );
                        text = rendered;
                    } else {
                        if index > 0 {
                            text.push_str("\n\n");
                        }
                        let head = format!("{key}:\n");
                        let start = text.chars().count() + head.chars().count();
                        fields.insert(
                            (*key).clone(),
                            FieldRange {
                                role,
                                start,
                                end: start + rendered.chars().count(),
                            },
                        );
                        text.push_str(&head);
                        text.push_str(&rendered);
                    }
                }
                if text.trim().is_empty() {
                    return Err(state_error(&format!(
                        "the {role} part of the state is empty"
                    )));
                }
                parts.push(Part {
                    r#type: role.into(),
                    text,
                });
            }
            Ok(StateParts { parts, fields })
        }
        _ => Err(state_error(
            "state must be a non-empty string, JSON object or JSON array",
        )),
    }
}
