//! Kev's API mapping, adapted from kev/api.py at the pinned reference revision.
//! See ../LICENSE.kev. Preserve insertion order and Python scalar rendering.
use anyhow::{Context, Result, bail, ensure};
use omni_qwen3_5_native::json;
use serde_json::{Map, Value};

pub const MODEL: &str = "jaredpalmer/kev-4b";
pub const MAX_QUESTIONS: usize = 64;
pub const MAX_STATE: usize = 8192;
pub const MAX_ROW: usize = 16384;
pub const MAX_REQUEST_TOKENS: usize = 65536;
pub const MAX_BODY: usize = 4 << 20;
pub const TEMPERATURE: f64 = 2.406050072164233;
pub const SPECIAL: [(&str, u32); 5] = [
    ("<|fim_prefix|>", 248060),
    ("<|fim_middle|>", 248061),
    ("<|box_start|>", 248049),
    ("<|box_end|>", 248050),
    ("<|fim_suffix|>", 248062),
];

#[derive(Debug, Clone, PartialEq)]
pub enum Kind {
    Choice,
    Noul,
    Score,
}

#[derive(Debug, Clone)]
pub struct Question {
    pub id: String,
    pub kind: Kind,
    pub instructions: String,
    pub options: Vec<String>,
    pub keys: Vec<String>,
    pub legend: Option<Value>,
}

pub struct Request {
    pub model: String,
    pub state: String,
    pub questions: Vec<Question>,
}

pub fn render(value: &Value, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    match value {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::Number(_) => json::dumps(value),
        Value::Array(items) => items
            .iter()
            .map(|v| {
                format!(
                    "{pad}- {}",
                    render(v, indent + 1).trim_start_matches(|c: char| {
                        c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
                    })
                )
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Object(items) => items
            .iter()
            .map(|(k, v)| {
                if v.is_array() || v.is_object() {
                    format!("{pad}{k}:\n{}", render(v, indent + 1))
                } else {
                    format!("{pad}{k}: {}", render(v, 0))
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn option(name: &str, value: &Value) -> String {
    if value.is_null() || value.as_str() == Some("") {
        name.into()
    } else {
        format!("{name}: {}", render(value, 0))
    }
}

pub fn compile(request: &Map<String, Value>) -> Result<Request> {
    let model = match request.get("model") {
        None => "kev-latest",
        Some(value) => value.as_str().context("model must be a string")?,
    };
    ensure!(
        [MODEL, "kev-latest", "kev-4b"].contains(&model),
        "requested model is not loaded"
    );
    let state = render(request.get("state").context("request requires state")?, 0);
    let definitions = request
        .get("questions")
        .and_then(Value::as_object)
        .context("questions must be an object")?;
    ensure!(
        (1..=MAX_QUESTIONS).contains(&definitions.len()),
        "questions must contain 1..={MAX_QUESTIONS} entries"
    );
    let mut questions = Vec::with_capacity(definitions.len());
    for (id, value) in definitions {
        let d = value
            .as_object()
            .context("question definition must be an object")?;
        let criteria = d.get("criteria").unwrap_or(&Value::Null);
        let (kind, keys, options, legend) = match d.get("type").and_then(Value::as_str) {
            Some("choice") => {
                let c = criteria
                    .as_object()
                    .context("Choice requires an object criteria")?;
                (
                    Kind::Choice,
                    c.keys().cloned().collect(),
                    c.iter().map(|(k, v)| option(k, v)).collect::<Vec<_>>(),
                    None,
                )
            }
            Some("score") => {
                let c = criteria
                    .as_array()
                    .context("Score requires an array criteria")?;
                let keys: Vec<String> = (0..c.len()).map(|i| i.to_string()).collect();
                let options: Vec<String> = c.iter().map(|v| render(v, 0)).collect();
                let legend: Map<String, Value> = keys
                    .iter()
                    .cloned()
                    .zip(options.iter().cloned().map(Value::String))
                    .collect();
                (Kind::Score, keys, options, Some(Value::Object(legend)))
            }
            Some("noul") => {
                ensure!(
                    criteria.is_null() || criteria.is_object(),
                    "Noul criteria must be null or an object"
                );
                // The pinned reference ignores other keys and fills absent descriptions.
                let options = [("no", "false"), ("yes", "true")]
                    .map(|(name, key)| option(name, criteria.get(key).unwrap_or(&Value::Null)))
                    .to_vec();
                (
                    Kind::Noul,
                    vec!["false".into(), "true".into()],
                    options,
                    None,
                )
            }
            _ => bail!("question type must be choice, noul, or score"),
        };
        ensure!(
            (1..=255).contains(&options.len()),
            "question {id}: expected 1..=255 candidates"
        );
        questions.push(Question {
            id: id.clone(),
            kind,
            keys,
            options,
            legend,
            instructions: render(d.get("instructions").unwrap_or(&Value::Null), 0),
        });
    }
    Ok(Request {
        model: model.into(),
        state,
        questions,
    })
}
