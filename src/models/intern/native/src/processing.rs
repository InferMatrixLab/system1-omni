//! Text-only specialization of the published Intern-Decision-4B prompt.

use std::{fmt, path::Path};

use anyhow::{Result, anyhow, ensure};
use omni_qwen3_5_native::json;
use serde_json::{Map, Value};
use tokenizers::Tokenizer;

pub const MAX_INPUT_TOKENS: usize = 8192;
pub const SYMBOLS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
const MARKER: &str = "<decision>";
const MARKER_ID: u32 = 248077;
const SYSTEM: &str = "You are a careful decision assistant. Use the state and decision schema in the user message to make the requested decisions. For every field, choose exactly one answer symbol (e.g. A, B, C, ...) from its listed options and return one valid JSON object mapping each field name to its chosen symbol. Use the field names and symbols exactly as given. Do not include explanations, Markdown, or extra text.";
const RESERVED: [&str; 10] = [
    MARKER,
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<|endoftext|>",
    "<think>",
    "</think>",
];

#[derive(Debug)]
pub struct RequestError {
    pub status: u16,
    message: String,
}

impl fmt::Display for RequestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RequestError {}

fn invalid(message: impl Into<String>) -> RequestError {
    RequestError {
        status: 422,
        message: message.into(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Choice,
    Noul,
    Score,
}

#[derive(Debug)]
pub struct Field {
    pub name: String,
    pub kind: Kind,
    pub options: Vec<(String, String)>,
}

pub struct Prepared {
    pub prompt: String,
    pub ids: Vec<u32>,
    pub readout: Vec<usize>,
    pub fields: Vec<Field>,
}

pub struct Processor {
    tokenizer: Tokenizer,
    pub symbol_ids: Vec<u32>,
}

impl Processor {
    pub fn load(tokenizer: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(tokenizer).map_err(|e| anyhow!("{e}"))?;
        ensure!(
            tokenizer.token_to_id(MARKER) == Some(MARKER_ID),
            "wrong Intern marker ID"
        );
        let symbol_ids = SYMBOLS
            .chars()
            .map(|symbol| {
                let ids = tokenizer
                    .encode(symbol.to_string(), false)
                    .map_err(|e| anyhow!("{e}"))?;
                ensure!(
                    ids.len() == 1,
                    "answer symbol {symbol} must encode to one token"
                );
                Ok(ids.get_ids()[0])
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            symbol_ids
                .iter()
                .copied()
                .eq((32..58).chain(64..90).chain(15..25)),
            "wrong Intern symbol IDs"
        );
        Ok(Self {
            tokenizer,
            symbol_ids,
        })
    }

    pub fn prepare(&self, raw: &[u8]) -> Result<Prepared> {
        let request = json::parse(raw).map_err(|message| RequestError {
            status: 400,
            message,
        })?;
        let (prompt, fields) = compile(&request)?;
        let encoded = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|e| anyhow!("{e}"))?;
        let ids = encoded.get_ids();
        if ids.len() > MAX_INPUT_TOKENS {
            return Err(RequestError {
                status: 413,
                message: format!(
                    "input has {} tokens; limit is {MAX_INPUT_TOKENS}",
                    ids.len()
                ),
            }
            .into());
        }
        let readout = ids
            .iter()
            .enumerate()
            .filter(|&(_, &id)| id == MARKER_ID)
            .map(|(row, _)| row.checked_sub(1))
            .collect::<Option<Vec<_>>>()
            .filter(|rows| rows.len() == fields.len())
            .ok_or_else(|| invalid("decision marker count or position mismatch"))?;
        Ok(Prepared {
            prompt,
            ids: ids.to_vec(),
            readout,
            fields,
        })
    }
}

pub fn compile(request: &Map<String, Value>) -> Result<(String, Vec<Field>), RequestError> {
    let state = request
        .get("state")
        .ok_or_else(|| invalid("state is required"))?;
    for key in ["images", "image", "videos", "video", "media"] {
        if request.get(key).is_some_and(nonempty) {
            return Err(invalid("the Intern text worker does not accept media"));
        }
    }
    let questions = request
        .get("questions")
        .and_then(Value::as_object)
        .filter(|questions| (1..=16).contains(&questions.len()))
        .ok_or_else(|| invalid("supply 1–16 questions"))?;
    let mut fields = Vec::with_capacity(questions.len());
    let mut schema = Vec::new();
    let mut skeleton = Map::new();
    for (name, question) in questions {
        if name.is_empty() {
            return Err(invalid("question names must be nonempty"));
        }
        let kind = match question["type"].as_str() {
            Some("choice") => Kind::Choice,
            Some("noul") => Kind::Noul,
            Some("score") => Kind::Score,
            _ => return Err(invalid("question type must be choice, noul, or score")),
        };
        let options = options(kind, &question["criteria"])?;
        let instructions = question
            .get("instructions")
            .map(py_string)
            .unwrap_or_default();
        schema.push(format!("{name}: {instructions}"));
        for (symbol, (value, description)) in SYMBOLS.chars().zip(&options) {
            schema.push(format!("    {symbol} = {value}: {description}"));
        }
        fields.push(Field {
            name: name.clone(),
            kind,
            options,
        });
        skeleton.insert(name.clone(), Value::String(MARKER.into()));
    }
    let state = pretty(state, 2, 0);
    let schema = schema.join("\n");
    if RESERVED
        .iter()
        .any(|marker| state.contains(marker) || schema.contains(marker))
    {
        return Err(invalid("reserved control marker in input"));
    }
    let skeleton = pretty(&Value::Object(skeleton), 4, 0);
    let prompt = format!(
        "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\nReturn one answer for every field using the supplied answer symbols.\n\n## State\n{state}\n## Decision schema\n{schema}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{skeleton}<|im_end|>\n"
    );
    Ok((prompt, fields))
}

fn nonempty(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
        Value::Number(value) => value.as_f64() != Some(0.0),
    }
}

fn options(kind: Kind, criteria: &Value) -> Result<Vec<(String, String)>, RequestError> {
    if kind == Kind::Noul {
        let description = |aliases: &[&str], default: &str| {
            criteria
                .as_object()
                .and_then(|map| {
                    map.iter()
                        .find(|(key, _)| aliases.contains(&key.to_lowercase().as_str()))
                })
                .map(|(_, value)| py_string(value))
                .unwrap_or_else(|| default.to_owned())
        };
        return Ok(vec![
            (
                "no".into(),
                description(
                    &["no", "false", "0"],
                    "The answer is no (negative, or disagree with the claim).",
                ),
            ),
            (
                "yes".into(),
                description(
                    &["yes", "true", "1"],
                    "The answer is yes (affirmative, or align with the claim).",
                ),
            ),
        ]);
    }
    let options: Vec<_> = match criteria {
        Value::Object(map) => {
            if kind == Kind::Score && map.keys().any(|level| score_level(level).is_none()) {
                return Err(invalid("Score option keys must be finite numbers"));
            }
            map.iter()
                .map(|(name, value)| (name.clone(), py_string(value)))
                .collect()
        }
        Value::Array(array) if kind == Kind::Score => array
            .iter()
            .enumerate()
            .map(|(index, value)| (index.to_string(), py_string(value)))
            .collect(),
        _ => {
            return Err(invalid(
                "Choice criteria must be an object and Score criteria a list or object",
            ));
        }
    };
    if !(1..=SYMBOLS.len()).contains(&options.len()) {
        return Err(invalid("supply 1–62 options"));
    }
    Ok(options)
}

pub fn score_level(label: &str) -> Option<f64> {
    let label = label.trim();
    if label.as_bytes().iter().enumerate().any(|(index, &byte)| {
        byte == b'_'
            && !(index > 0
                && label.as_bytes()[index - 1].is_ascii_digit()
                && label
                    .as_bytes()
                    .get(index + 1)
                    .is_some_and(u8::is_ascii_digit))
    }) {
        return None;
    }
    label
        .replace('_', "")
        .parse::<f64>()
        .ok()
        .filter(|level| level.is_finite())
}

fn py_string(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        _ => py_repr(value),
    }
}

fn py_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(false) => "False".into(),
        Value::Bool(true) => "True".into(),
        Value::String(text) => {
            let quote = if text.contains('\'') && !text.contains('"') {
                '"'
            } else {
                '\''
            };
            let mut out = String::from(quote);
            for ch in text.chars() {
                match ch {
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    '\\' => out.push_str("\\\\"),
                    ch if ch == quote => {
                        out.push('\\');
                        out.push(ch);
                    }
                    ch if ch.is_control() || (ch.is_whitespace() && ch != ' ') => {
                        use fmt::Write;
                        let code = u32::from(ch);
                        if code <= 0xff {
                            write!(out, "\\x{code:02x}").unwrap();
                        } else if code <= 0xffff {
                            write!(out, "\\u{code:04x}").unwrap();
                        } else {
                            write!(out, "\\U{code:08x}").unwrap();
                        }
                    }
                    ch => out.push(ch),
                }
            }
            out.push(quote);
            out
        }
        Value::Array(items) => format!(
            "[{}]",
            items.iter().map(py_repr).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(items) => format!(
            "{{{}}}",
            items
                .iter()
                .map(|(key, value)| {
                    format!(
                        "{}: {}",
                        py_repr(&Value::String(key.clone())),
                        py_repr(value)
                    )
                })
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Number(_) => json::dumps(value),
    }
}

fn pretty(value: &Value, indent: usize, depth: usize) -> String {
    let (open, entries, close) = match value {
        Value::Array(items) if !items.is_empty() => (
            '[',
            items
                .iter()
                .map(|v| pretty(v, indent, depth + 1))
                .collect::<Vec<_>>(),
            ']',
        ),
        Value::Object(items) if !items.is_empty() => (
            '{',
            items
                .iter()
                .map(|(name, v)| format!("{}: {}", json::quote(name), pretty(v, indent, depth + 1)))
                .collect(),
            '}',
        ),
        _ => return json::dumps(value),
    };
    let next = " ".repeat(indent * (depth + 1));
    let current = " ".repeat(indent * depth);
    format!(
        "{open}\n{next}{}\n{current}{close}",
        entries.join(&format!(",\n{next}"))
    )
}
