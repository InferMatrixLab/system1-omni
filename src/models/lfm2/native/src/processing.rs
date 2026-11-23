//! Model-owned Choice preparation and same-score response interpretation.
//!
//! Prompt and branch construction follow notnotsamuel/LFM2.5-350M-RLCD,
//! deb589d803d141cabd158ef55f6617b128529f36, MIT (c) 2026 notnotsamuel.
//! See ../../LICENSE.reference for the license.

use crate::{MAX_CRITERIA, MODEL_ALIAS, SERVED_MODEL, VOCAB_SIZE, json};
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use serde_json::{Value, json};
use std::{collections::HashSet, fmt, ops::Range};
use tokenizers::Tokenizer;

#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct RequestError {
    pub status: u16,
    pub detail: String,
}

impl fmt::Display for RequestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.detail.fmt(f)
    }
}

impl std::error::Error for RequestError {}

#[derive(Debug, Serialize)]
pub struct RenderedRequest {
    pub questions: Vec<RenderedQuestion>,
}

#[derive(Debug, Serialize)]
pub struct RenderedQuestion {
    pub id: String,
    pub schema: RenderedSchema,
}

#[derive(Debug, Serialize)]
pub struct RenderedSchema {
    pub prefix_text: String,
    pub fields: Vec<RenderedField>,
}

#[derive(Debug, Serialize)]
pub struct RenderedField {
    pub name: String,
    pub suffix_text: String,
    pub candidates: Vec<RenderedCandidate>,
}

#[derive(Debug, Serialize)]
pub struct RenderedCandidate {
    pub value: Value,
    pub value_text: String,
}

#[derive(Debug, Serialize)]
pub struct PreparedRequest {
    pub questions: Vec<PreparedQuestion>,
}

#[derive(Debug, Serialize)]
pub struct PreparedQuestion {
    pub id: String,
    pub plan: TokenPlan,
}

#[derive(Debug, Serialize)]
pub struct TokenPlan {
    pub prefix_ids: Vec<u32>,
    pub fields: Vec<FieldPlan>,
    pub input_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct FieldPlan {
    pub name: String,
    pub suffix_ids: Vec<u32>,
    pub candidates: Vec<CandidatePlan>,
}

#[derive(Debug, Serialize)]
pub struct CandidatePlan {
    pub value: Value,
    // Includes the scored JSON newline, which can merge with the quote token.
    pub value_ids: Vec<u32>,
    // Half-open logit rows predicting value_ids, relative to suffix + value.
    pub continuation_readout: Range<usize>,
    // The same interval in prefix + suffix + value.
    pub full_readout: Range<usize>,
}

fn text_or_container(value: &Value) -> bool {
    matches!(value, Value::String(_) | Value::Object(_) | Value::Array(_))
}

fn invalid(detail: impl Into<String>) -> RequestError {
    RequestError {
        status: 422,
        detail: detail.into(),
    }
}

pub fn validate_request(raw: &[u8]) -> Result<Value, RequestError> {
    let payload = json::parse(raw).map_err(|detail| RequestError {
        status: 400,
        detail,
    })?;
    let object = payload
        .as_object()
        .ok_or_else(|| invalid("request body must be a JSON object"))?;
    if object.get("model").and_then(Value::as_str) != Some(MODEL_ALIAS) {
        return Err(invalid(format!(
            "unsupported model; expected '{MODEL_ALIAS}'"
        )));
    }
    if !object
        .get("state")
        .and_then(Value::as_str)
        .is_some_and(|state| {
            !state
                .trim_matches(|c: char| c.is_whitespace() || ('\x1c'..='\x1f').contains(&c))
                .is_empty()
        })
    {
        return Err(invalid("state must be a non-empty text string"));
    }
    let questions = object
        .get("questions")
        .and_then(Value::as_object)
        .filter(|q| !q.is_empty())
        .ok_or_else(|| invalid("questions must be a non-empty object"))?;
    for (id, question) in questions {
        let id = json::repr(&Value::String(id.clone()));
        let question = question
            .as_object()
            .ok_or_else(|| invalid(format!("question {id} must be an object")))?;
        let kind = question.get("type").unwrap_or(&Value::Null);
        if kind.as_str() != Some("choice") {
            return Err(invalid(format!(
                "question {id} has unsupported type {}; only 'choice' is supported",
                json::repr(kind)
            )));
        }
        if !question.get("instructions").is_some_and(text_or_container) {
            return Err(invalid(format!(
                "question {id} instructions must be a string, object or array"
            )));
        }
        let criteria = question
            .get("criteria")
            .and_then(Value::as_object)
            .filter(|criteria| (1..=MAX_CRITERIA).contains(&criteria.len()))
            .ok_or_else(|| {
                invalid(format!(
                    "question {id} criteria must map 1..{MAX_CRITERIA} keys"
                ))
            })?;
        if !criteria
            .values()
            .all(|value| value.is_null() || text_or_container(value))
        {
            return Err(invalid(format!(
                "question {id} criteria values must be strings, objects, arrays or null"
            )));
        }
    }
    Ok(payload)
}

pub fn render_request(raw: &[u8]) -> Result<RenderedRequest, RequestError> {
    let payload = validate_request(raw)?;
    let state = payload["state"].as_str().expect("validated state");
    let questions = payload["questions"]
        .as_object()
        .expect("validated questions");
    let questions = questions.iter().map(|(id, question)| {
        let criteria = question["criteria"].as_object().expect("validated criteria");
        let description = json::dumps(&json!({
            "instructions": question["instructions"],
            "criteria": question["criteria"],
        }));
        let schema = json!({
            "type": "object",
            "properties": {
                "answer": { "type": "string", "description": description, "enum": criteria.keys().collect::<Vec<_>>() }
            },
            "required": ["answer"],
            "additionalProperties": false,
        });
        let fields = vec![render_field("answer", criteria.keys().cloned().map(Value::String))];
        RenderedQuestion { id: id.clone(), schema: render_schema(state, &schema, fields) }
    }).collect();
    Ok(RenderedRequest { questions })
}

/// Render the pinned RLCD flat-schema adapter used by the internal diagnostics.
/// Public Choice requests use strings even when their criteria keys look like booleans.
pub fn render_flat_schema(state: &str, schema: &Value) -> Result<RenderedSchema> {
    let object = schema.as_object().context("schema must be an object")?;
    ensure!(
        schema["type"] == "object"
            && schema["additionalProperties"] == false
            && object.keys().all(|key| matches!(
                key.as_str(),
                "type" | "properties" | "required" | "additionalProperties"
            )),
        "only closed, flat object schemas are supported"
    );
    let properties = schema["properties"]
        .as_object()
        .filter(|props| !props.is_empty())
        .context("schema must declare at least one property")?;
    let required = schema["required"]
        .as_array()
        .context("all fields must be required")?;
    let required_names: HashSet<_> = required.iter().filter_map(Value::as_str).collect();
    ensure!(
        required_names.len() == required.len()
            && required.len() == properties.len()
            && properties
                .keys()
                .all(|name| required_names.contains(name.as_str())),
        "all fields must be required"
    );
    let mut fields = Vec::with_capacity(properties.len());
    for (name, spec) in properties {
        let spec = spec.as_object().context("unsupported field constraints")?;
        ensure!(
            spec.keys()
                .all(|key| matches!(key.as_str(), "type" | "description" | "enum"))
                && spec.get("description").is_none_or(Value::is_string),
            "unsupported field constraints"
        );
        match spec.get("type").and_then(Value::as_str) {
            Some("boolean") if !spec.contains_key("enum") => {
                fields.push(render_field(name, [Value::Bool(true), Value::Bool(false)]));
            }
            Some("string") => {
                let values = spec
                    .get("enum")
                    .and_then(Value::as_array)
                    .filter(|values| !values.is_empty())
                    .context("fields must be booleans or nonempty string enums")?;
                let unique: HashSet<_> = values.iter().filter_map(Value::as_str).collect();
                ensure!(
                    unique.len() == values.len(),
                    "string enums must be unique strings"
                );
                fields.push(render_field(name, values.iter().cloned()));
            }
            _ => anyhow::bail!("fields must be booleans or nonempty string enums"),
        }
    }
    Ok(render_schema(state, schema, fields))
}

fn render_schema(state: &str, schema: &Value, fields: Vec<RenderedField>) -> RenderedSchema {
    // Fixed text-only system/user/generation branch of checkpoint chat_template.jinja.
    let schema = json::dumps(schema);
    let prefix_text = format!(
        "<|startoftext|><|im_start|>system\n\
        Extract the attributes from the text. Return only a JSON object matching this schema. \
        Use the exact allowed values. No explanation or markdown.\n\
        {schema}<|im_end|>\n<|im_start|>user\n{state}<|im_end|>\n<|im_start|>assistant\n{{\n"
    );
    RenderedSchema {
        prefix_text,
        fields,
    }
}

fn render_field(name: &str, values: impl IntoIterator<Item = Value>) -> RenderedField {
    let quoted_name = json::dumps(&Value::String(name.to_owned()));
    RenderedField {
        name: name.to_owned(),
        suffix_text: format!("  {quoted_name}: "),
        candidates: values
            .into_iter()
            .map(|value| RenderedCandidate {
                value_text: format!("{}\n", json::dumps(&value)),
                value,
            })
            .collect(),
    }
}

impl RenderedRequest {
    pub fn tokenize(self, tokenizer: &Tokenizer) -> Result<PreparedRequest> {
        self.tokenize_with(|text| encode(tokenizer, text))
    }

    pub(crate) fn tokenize_with(
        self,
        mut encode: impl FnMut(&str) -> Result<Vec<u32>>,
    ) -> Result<PreparedRequest> {
        let questions = self
            .questions
            .into_iter()
            .map(|question| {
                Ok(PreparedQuestion {
                    id: question.id,
                    plan: question.schema.tokenize_with(&mut encode)?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(PreparedRequest { questions })
    }
}

impl RenderedSchema {
    pub fn tokenize(self, tokenizer: &Tokenizer) -> Result<TokenPlan> {
        self.tokenize_with(&mut |text| encode(tokenizer, text))
    }

    pub(crate) fn tokenize_with(
        self,
        encode: &mut impl FnMut(&str) -> Result<Vec<u32>>,
    ) -> Result<TokenPlan> {
        let prefix_ids = encode(&self.prefix_text)?;
        ensure!(!prefix_ids.is_empty(), "empty prefix tokenization");
        let prefix_len = prefix_ids.len();
        let mut input_tokens = prefix_len;
        let mut fields = Vec::with_capacity(self.fields.len());
        for field in self.fields {
            let suffix_ids = encode(&field.suffix_text)?;
            let suffix_len = suffix_ids.len();
            ensure!(suffix_len > 0, "empty field suffix tokenization");
            let mut candidates = Vec::with_capacity(field.candidates.len());
            for candidate in field.candidates {
                let value_ids = encode(&candidate.value_text)?;
                ensure!(!value_ids.is_empty(), "empty candidate tokenization");
                input_tokens = input_tokens
                    .checked_add(suffix_len)
                    .and_then(|n| n.checked_add(value_ids.len()))
                    .context("input token count overflow")?;
                let relative = suffix_len - 1..suffix_len - 1 + value_ids.len();
                candidates.push(CandidatePlan {
                    value: candidate.value,
                    value_ids,
                    full_readout: prefix_len + relative.start..prefix_len + relative.end,
                    continuation_readout: relative,
                });
            }
            fields.push(FieldPlan {
                name: field.name,
                suffix_ids,
                candidates,
            });
        }
        Ok(TokenPlan {
            prefix_ids,
            fields,
            input_tokens,
        })
    }
}

fn encode(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|error| anyhow::anyhow!("tokenization failed: {error}"))?;
    let ids = encoding.get_ids();
    ensure!(
        ids.iter().all(|&id| (id as usize) < VOCAB_SIZE),
        "token ID outside model vocabulary"
    );
    Ok(ids.to_vec())
}

/// Interpret one full-candidate FP32 log-likelihood per item in prepared order.
/// The executor owns all vocabulary projection, log-softmax, and token sums.
pub fn finish_request(prepared: &PreparedRequest, scores: &[Vec<f32>]) -> Result<Value> {
    ensure!(
        prepared.questions.len() == scores.len(),
        "question score count mismatch"
    );
    let mut answers = serde_json::Map::new();
    let mut input_tokens = 0usize;
    for (question, scores) in prepared.questions.iter().zip(scores) {
        let [field] = question.plan.fields.as_slice() else {
            anyhow::bail!("expected one Choice field");
        };
        ensure!(field.name == "answer", "expected the answer field");
        let (selected, probabilities, confidence) = interpret(field, scores)?;
        let probabilities = field
            .candidates
            .iter()
            .zip(probabilities)
            .map(|(candidate, probability)| {
                let value = candidate
                    .value
                    .as_str()
                    .context("Choice candidate must be a string")?;
                Ok((value.to_owned(), json!(probability)))
            })
            .collect::<Result<serde_json::Map<_, _>>>()?;
        answers.insert(
            question.id.clone(),
            json!({
                "type": "choice",
                "choice": selected,
                "probabilities": probabilities,
                "confidence": confidence,
            }),
        );
        input_tokens = input_tokens
            .checked_add(question.plan.input_tokens)
            .context("input token count overflow")?;
    }
    Ok(json!({
        "model": SERVED_MODEL,
        "answers": answers,
        "usage": { "input_tokens": input_tokens, "output_tokens": 0 },
    }))
}

pub fn interpret(field: &FieldPlan, scores: &[f32]) -> Result<(Value, Vec<f64>, f64)> {
    ensure!(
        !scores.is_empty() && scores.len() == field.candidates.len(),
        "candidate score count mismatch"
    );
    ensure!(
        scores.iter().all(|score| score.is_finite()),
        "model returned non-finite candidate scores"
    );
    let mut index = 0;
    for candidate in 1..scores.len() {
        if scores[candidate] > scores[index] {
            index = candidate;
        }
    }
    let largest = f64::from(scores[index]);
    let mut probabilities: Vec<_> = scores
        .iter()
        .map(|&score| (f64::from(score) - largest).exp())
        .collect();
    let total: f64 = probabilities.iter().sum();
    for probability in &mut probabilities {
        *probability /= total;
    }
    let confidence = if scores.len() == 1 {
        1.0
    } else {
        let entropy = -probabilities
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|p| p * p.ln())
            .sum::<f64>();
        (1.0 - entropy / (scores.len() as f64).ln()).clamp(0.0, 1.0)
    };
    Ok((
        field.candidates[index].value.clone(),
        probabilities,
        confidence,
    ))
}
