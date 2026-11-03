//! The SystemOne (state, questions) → predict-request mapping, mirroring
//! the pinned runtime's `system_one_to_predict` exactly: the validation
//! walk and its whole-request error triples, the Noul folding into a
//! two-option choice, and the per-question plan used to build responses.

use serde_json::{Map, Value};

use crate::contract::RequestError;
use crate::state::{Part, StateParts, content_text};

/// The `questions` container rejection shared by the mapping and the
/// processor's body walk.
pub fn empty_questions_error() -> RequestError {
    qerror(
        &["questions"],
        "questions must be a non-empty object of named questions",
    )
}

pub const SYSTEM_ONE_TYPES: [&str; 5] = ["noul", "choice", "score", "span", "set"];
pub const NOUL_DEFAULT_NO: &str = "No. The statement or question is not satisfied.";
pub const NOUL_DEFAULT_YES: &str = "Yes. The statement or question is satisfied.";

/// One option / label of a predict question (name, description).
#[derive(Debug, Clone, serde::Serialize, PartialEq)]
pub struct Named {
    pub name: String,
    pub description: String,
}

/// The predict-level question, exactly as it serializes into the frozen
/// fixtures: Noul questions arrive here already folded into a choice.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PredictQuestion {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub text: String,
    pub target_part: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub labels: Option<Vec<Named>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PredictRequest {
    pub parts: Vec<Part>,
    pub questions: Vec<PredictQuestion>,
}

/// The response-side plan entry for one original question.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PlanEntry {
    pub qid: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub criteria: Value,
    pub range: Option<(usize, usize)>,
    pub part: Value,
}

#[derive(Debug)]
pub struct Mapped {
    pub request: PredictRequest,
    pub plan: Vec<PlanEntry>,
}

fn qerror(loc: &[&str], msg: &str) -> RequestError {
    RequestError::envelope(loc, msg, "value_error")
}

fn nonblank(value: &Value) -> bool {
    match value {
        Value::String(text) => !text.trim().is_empty(),
        _ => true,
    }
}

/// `over` (state key | part name | list) → (target part(s), field range).
fn resolve_over(
    over: &Value,
    parts: &StateParts,
    qloc: &[&str],
) -> Result<(Value, Option<(usize, usize)>), RequestError> {
    let roles: Vec<&str> = parts.parts.iter().map(|p| p.r#type.as_str()).collect();
    let items: Vec<&Value> = match over {
        Value::Array(list) => list.iter().collect(),
        single => vec![single],
    };
    let mut out: Vec<&str> = Vec::new();
    let mut range = None;
    for item in items {
        let Some(name) = item.as_str() else {
            return Err(qerror(
                &[qloc[0], qloc[1], "over"],
                "over must be a state key, a part name or a list of them",
            ));
        };
        if let Some(field) = parts.fields.get(name) {
            if !over.is_array() {
                range = Some((field.start, field.end));
            }
            if !out.contains(&field.role) {
                out.push(field.role);
            }
        } else if roles.contains(&name) {
            if !out.contains(&name) {
                out.push(name);
            }
        } else {
            let role = crate::state::key_role(name);
            if roles.contains(&role) && !out.contains(&role) {
                out.push(role);
            } else {
                return Err(qerror(
                    &[qloc[0], qloc[1], "over"],
                    &format!(
                        "'{name}' is not a field of the state (fields: [{fields}], parts: [{roles}])",
                        fields = parts
                            .fields
                            .keys()
                            .map(|k| format!("'{k}'"))
                            .collect::<Vec<_>>()
                            .join(", "),
                        roles = roles.join(", "),
                    ),
                ));
            }
        }
    }
    if out.is_empty() {
        return Err(qerror(
            &[qloc[0], qloc[1], "over"],
            "over must not be empty",
        ));
    }
    // Sequence order, so the pooled range covers exactly these parts.
    let ordered: Vec<&str> = roles
        .iter()
        .filter(|role| out.contains(role))
        .copied()
        .collect();
    let target = match ordered.len() {
        1 => Value::String(ordered[0].into()),
        _ => Value::Array(
            ordered
                .into_iter()
                .map(|r| Value::String(r.into()))
                .collect(),
        ),
    };
    Ok((target, range))
}

/// The full mapping. `state` and `questions` are the admitted request's
/// fields; validation runs in the pinned runtime's order so error
/// precedence matches.
pub fn system_one_to_predict(
    state_parts: StateParts,
    questions: &Map<String, Value>,
) -> Result<Mapped, RequestError> {
    if questions.is_empty() {
        return Err(qerror(
            &["questions"],
            "questions must be a non-empty object of named questions",
        ));
    }
    let roles: Vec<&str> = state_parts
        .parts
        .iter()
        .map(|p| p.r#type.as_str())
        .collect();
    let mut predict = Vec::new();
    let mut plan = Vec::new();
    for (qid, question) in questions {
        if qid.trim().is_empty() {
            return Err(qerror(
                &["questions"],
                "question IDs must not be empty or whitespace",
            ));
        }
        let Some(question) = question.as_object() else {
            return Err(qerror(
                &["questions", qid],
                "a question must be an object with a type",
            ));
        };
        let loc = ["questions", qid];
        let kind = question.get("type").and_then(Value::as_str).unwrap_or("");
        if !SYSTEM_ONE_TYPES.contains(&kind) {
            return Err(RequestError::envelope(
                &[loc[0], loc[1], "type"],
                "type must be one of ['noul', 'choice', 'score', 'span', 'set']",
                "literal_error",
            ));
        }
        let mut allowed: Vec<&str> = vec!["type", "instructions", "criteria", "over"];
        if kind == "span" || kind == "set" {
            allowed.push("threshold");
        }
        if kind == "span" {
            allowed.push("head");
        }
        let extra: Vec<&String> = question
            .keys()
            .filter(|key| !allowed.contains(&key.as_str()))
            .collect();
        if let Some(first) = extra.first() {
            return Err(RequestError::envelope(
                &[loc[0], loc[1], first],
                "Extra inputs are not permitted",
                "extra_forbidden",
            ));
        }
        let instructions = question.get("instructions").unwrap_or(&Value::Null);
        if !nonblank(instructions) {
            return Err(qerror(
                &[loc[0], loc[1], "instructions"],
                "instructions must not be empty or whitespace",
            ));
        }
        let text = match instructions {
            Value::Null => String::new(),
            other => content_text(other),
        };
        let criteria = question.get("criteria").unwrap_or(&Value::Null);
        // Target part: the resolved `over`, the span default, or all parts.
        let (target_part, range) = match question.get("over") {
            Some(Value::Null) | None => {
                if kind == "span" {
                    let role = if roles.contains(&"answer") {
                        "answer"
                    } else if roles.contains(&"user") {
                        "user"
                    } else {
                        roles[0]
                    };
                    (Value::String(role.into()), None)
                } else if roles.len() == 1 {
                    (Value::String(roles[0].into()), None)
                } else {
                    (
                        Value::Array(roles.iter().map(|r| Value::String((*r).into())).collect()),
                        None,
                    )
                }
            }
            Some(over) => resolve_over(over, &state_parts, &loc)?,
        };
        let mut predict_question = PredictQuestion {
            id: qid.clone(),
            kind: kind.to_string(),
            text: text.clone(),
            target_part: target_part.clone(),
            labels: None,
            options: None,
            threshold: None,
            head: None,
        };
        let mut plan_criteria = criteria.clone();
        match kind {
            "noul" => {
                if !criteria.is_null()
                    && (!criteria.is_object()
                        || criteria
                            .as_object()
                            .is_some_and(|map| map.keys().any(|k| k != "true" && k != "false")))
                {
                    return Err(qerror(
                        &[loc[0], loc[1], "criteria"],
                        "noul criteria is an object with optional true / false",
                    ));
                }
                let map = criteria.as_object().cloned().unwrap_or_default();
                for key in ["true", "false"] {
                    if let Some(value) = map.get(key)
                        && !nonblank(value)
                    {
                        return Err(qerror(
                            &[loc[0], loc[1], "criteria", key],
                            "Noul criterion must not be empty or whitespace",
                        ));
                    }
                }
                let no = match map.get("false") {
                    Some(Value::Null) | None => NOUL_DEFAULT_NO.to_string(),
                    Some(value) => content_text(value),
                };
                let yes = match map.get("true") {
                    Some(Value::Null) | None => NOUL_DEFAULT_YES.to_string(),
                    Some(value) => content_text(value),
                };
                predict_question.kind = "choice".into();
                predict_question.options = Some(serde_json::json!([
                    {"name": "no", "description": no},
                    {"name": "yes", "description": yes},
                ]));
            }
            "choice" | "set" | "span" => {
                let (lo, hi) = if kind == "choice" { (2, 255) } else { (1, 255) };
                let Some(map) = criteria.as_object() else {
                    return Err(qerror(
                        &[loc[0], loc[1], "criteria"],
                        &format!("{kind} criteria must be an object with {lo} to {hi} entries"),
                    ));
                };
                if !(lo..=hi).contains(&map.len()) {
                    return Err(qerror(
                        &[loc[0], loc[1], "criteria"],
                        &format!("{kind} criteria must be an object with {lo} to {hi} entries"),
                    ));
                }
                let mut named = Vec::new();
                for (name, value) in map {
                    if name.trim().is_empty() || !nonblank(value) {
                        return Err(qerror(
                            &[loc[0], loc[1], "criteria", name],
                            "option names and descriptions must not be empty or whitespace",
                        ));
                    }
                    let description = match value {
                        Value::Null => String::new(),
                        other => content_text(other),
                    };
                    named.push(Named {
                        name: name.clone(),
                        description,
                    });
                }
                if kind == "span" {
                    predict_question.labels = Some(named);
                } else {
                    predict_question.options = Some(serde_json::to_value(&named).unwrap());
                }
            }
            _ => {
                // score
                let Some(levels) = criteria.as_array() else {
                    return Err(qerror(
                        &[loc[0], loc[1], "criteria"],
                        "score criteria must be a list of 2 to 10 levels",
                    ));
                };
                if !(2..=10).contains(&levels.len()) {
                    return Err(qerror(
                        &[loc[0], loc[1], "criteria"],
                        "score criteria must be a list of 2 to 10 levels",
                    ));
                }
                let mut names = Vec::new();
                for (index, value) in levels.iter().enumerate() {
                    if value.is_null() || !nonblank(value) {
                        return Err(qerror(
                            &[loc[0], loc[1], "criteria", &index.to_string()],
                            "Score criterion must not be empty or whitespace",
                        ));
                    }
                    names.push(content_text(value));
                }
                if names.iter().collect::<std::collections::HashSet<_>>().len() != names.len() {
                    return Err(qerror(
                        &[loc[0], loc[1], "criteria"],
                        "score levels must be distinct",
                    ));
                }
                predict_question.options = Some(serde_json::to_value(&names).unwrap());
            }
        }
        if let Some(threshold) = question.get("threshold")
            && !threshold.is_null()
        {
            match threshold.as_f64().filter(|_| threshold.is_number()) {
                Some(value) if (0.0..=1.0).contains(&value) => {
                    predict_question.threshold = Some(value);
                }
                _ => {
                    return Err(qerror(
                        &[loc[0], loc[1], "threshold"],
                        "threshold must be a number in [0, 1]",
                    ));
                }
            }
        }
        if kind == "span"
            && let Some(head) = question.get("head")
            && !head.is_null()
        {
            match head.as_str() {
                Some(value @ ("router" | "broad")) => {
                    predict_question.head = Some(value.to_string());
                }
                _ => {
                    return Err(RequestError::envelope(
                        &[loc[0], loc[1], "head"],
                        "head must be one of ['router', 'broad']",
                        "literal_error",
                    ));
                }
            }
        }
        // Python's plan keeps `crit`; noul normalized null -> {} earlier.
        if kind == "noul" && plan_criteria.is_null() {
            plan_criteria = serde_json::json!({});
        }
        predict.push(predict_question);
        plan.push(PlanEntry {
            qid: qid.clone(),
            kind: kind.to_string(),
            criteria: plan_criteria,
            range,
            part: target_part,
        });
    }
    // The Noul views of set labels must not overwrite a question.
    for entry in &plan {
        if entry.kind == "set"
            && let Value::Object(map) = &entry.criteria
        {
            for label in map.keys() {
                if questions.contains_key(&format!("{}.{}", entry.qid, label)) {
                    return Err(qerror(
                        &["questions", &format!("{}.{}", entry.qid, label)],
                        "collides with the Noul view of a set label",
                    ));
                }
            }
        }
    }
    Ok(Mapped {
        request: PredictRequest {
            parts: state_parts.parts,
            questions: predict,
        },
        plan,
    })
}
