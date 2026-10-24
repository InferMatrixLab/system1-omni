//! Instant prompt, letter readout and `/v1/systemone` answers.
//!
//! The strings and formulas follow StartLux-Decision
//! `a99977d9332decc9fb27a2488b4a83b533b6cd5f` (`prompt.py`, `readout.py`,
//! `pipeline.py`, `controls.py`). This module does not generate text.

use serde_json::{Map, Number, Value};

pub const MODEL_ID: &str = "StartLux-Decision-v1.1-4B-Instant";
pub const WEIGHTS_REPO: &str = "startlux-models/StartLux-Decision-v1.1-4B-Instant";
pub const WEIGHTS_REVISION: &str = "416088f77cfce09633906a3675c7cd2e5dabcf10";
pub const CODE_REVISION: &str = "a99977d9332decc9fb27a2488b4a83b533b6cd5f";
pub const LICENSE: &str = "CC-BY-NC-4.0";

pub const SYSTEM: &str = "Apply the criterion to the evidence. Choose exactly one listed option. Answer with its letter only.";
pub const THINK_OFF: &str = "<think>\n\n</think>\n\n";
pub const NONE_OPTION: &str = "None of the above";
pub const NOUL_INSTRUCTIONS: &str = "Which answer fits the evidence?";

pub const MAX_OPTIONS: usize = 26;
pub const MAX_LEVELS: usize = 10;
pub const ANNOTATE_MIN: usize = 8;
pub const WIDE_GROUP: usize = 25;
pub const WIDE_KEEP: usize = 3;
pub const WIDE_RESIDUAL: f64 = 0.001;
pub const DEFAULT_MAX_MODEL_LEN: usize = 65_536;
pub const NATIVE_CONTEXT: usize = 262_144;

pub const TEMP_CHOICE: f64 = 1.1103;
pub const TEMP_NOUL: f64 = 1.2662;
pub const TEMP_SCORE: f64 = 1.1344;

/// A–Z as pinned in `decision_config.json` for this checkpoint.
pub const LETTER_IDS: [u32; 26] = [
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57,
];

const LETTERS: &[u8; 26] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Choice,
    Noul,
    Score,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Choice => "choice",
            Self::Noul => "noul",
            Self::Score => "score",
        }
    }

    pub fn temperature(self) -> f64 {
        match self {
            Self::Choice => TEMP_CHOICE,
            Self::Noul => TEMP_NOUL,
            Self::Score => TEMP_SCORE,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Settings {
    pub max_model_len: usize,
    /// `startlux-decide serve` turns this on unless `--no-none-check` is set.
    pub none_check: bool,
    pub allow_none: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            max_model_len: DEFAULT_MAX_MODEL_LEN,
            none_check: true,
            allow_none: false,
        }
    }
}

#[derive(Debug)]
pub struct RequestError {
    pub status: u16,
    pub message: String,
    pub loc: Vec<String>,
}

impl RequestError {
    fn new(status: u16, loc: Vec<String>, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
            loc,
        }
    }

    pub fn body(&self) -> Value {
        json_map([
            ("error", Value::String(self.message.clone())),
            (
                "detail",
                Value::Array(vec![json_map([
                    (
                        "loc",
                        Value::Array(self.loc.iter().cloned().map(Value::String).collect()),
                    ),
                    ("msg", Value::String(self.message.clone())),
                    ("type", Value::String("value_error".into())),
                ])]),
            ),
        ])
    }
}

impl std::fmt::Display for RequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for RequestError {}

#[derive(Debug, Clone)]
struct OptionSpec {
    id: String,
    criterion: Option<String>,
}

#[derive(Debug, Clone)]
struct Row {
    kind: Kind,
    state: String,
    instructions: String,
    options: Vec<OptionSpec>,
    none_option: bool,
}

#[derive(Debug)]
pub struct Prompt {
    pub ids: Vec<u32>,
    pub letters: usize,
}

#[derive(Debug)]
enum Body {
    Only {
        id: String,
    },
    Direct {
        kind: Kind,
        prompt: usize,
        readout: Vec<String>,
        legend: Vec<Value>,
        none: bool,
        null_ok: bool,
        tokens: usize,
    },
    Wide {
        groups: Vec<Vec<String>>,
        group_prompts: Vec<usize>,
        criteria: Map<String, Value>,
        state: String,
        instructions: String,
        none: bool,
        null_ok: bool,
        group_tokens: usize,
    },
}

#[derive(Debug)]
struct Question {
    key: String,
    body: Body,
}

#[derive(Debug)]
pub struct Prepared {
    pub prompts: Vec<Prompt>,
    questions: Vec<Question>,
}

#[derive(Debug)]
pub struct FinalRound {
    pub prompts: Vec<Prompt>,
    slots: Vec<FinalSlot>,
}

#[derive(Debug)]
struct FinalSlot {
    question: usize,
    prompt: usize,
    finalists: Vec<String>,
    first: Vec<(String, f64)>,
    tokens: usize,
    none: bool,
    null_ok: bool,
}

pub trait PromptEncoder {
    fn encode(&mut self, text: &str) -> Result<Vec<u32>, String>;
}

impl<F> PromptEncoder for F
where
    F: FnMut(&str) -> Result<Vec<u32>, String>,
{
    fn encode(&mut self, text: &str) -> Result<Vec<u32>, String> {
        self(text)
    }
}

pub fn chat_prompt(user: &str) -> String {
    let user = user.trim();
    format!(
        "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{THINK_OFF}"
    )
}

pub fn probabilities(logits: &[f64], temperature: f64) -> Vec<f64> {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = logits
        .iter()
        .map(|logit| ((logit - max) / temperature).exp())
        .collect();
    let sum: f64 = weights.iter().sum();
    weights.into_iter().map(|weight| weight / sum).collect()
}

pub fn choice_confidence(probs: &[f64]) -> f64 {
    let n = probs.len();
    if n < 2 {
        return 1.0;
    }
    let top = probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    ((n as f64 * top - 1.0) / (n as f64 - 1.0)).clamp(0.0, 1.0)
}

pub fn score_confidence(probs: &[f64]) -> f64 {
    let n = probs.len();
    let mut mode = 0;
    for (i, prob) in probs.iter().enumerate().skip(1) {
        if *prob > probs[mode] {
            mode = i;
        }
    }
    let spread: f64 = probs
        .iter()
        .enumerate()
        .map(|(i, prob)| prob * (i as f64 - mode as f64).abs())
        .sum();
    let even = (0..n)
        .map(|i| (i as f64 - (n as f64 - 1.0) / 2.0).abs())
        .sum::<f64>()
        / n as f64;
    if even == 0.0 {
        1.0
    } else {
        (1.0 - spread / even).max(0.0)
    }
}

pub fn prepare_request(
    raw: &[u8],
    settings: &Settings,
    encode: &mut dyn PromptEncoder,
) -> Result<Prepared, RequestError> {
    let body: Value = serde_json::from_slice(raw).map_err(|_| {
        RequestError::new(
            400,
            vec!["body".into()],
            "the request body is not valid JSON",
        )
    })?;
    let root = body.as_object().ok_or_else(|| {
        RequestError::new(
            400,
            vec!["body".into()],
            "the request body must be an object",
        )
    })?;
    reject_thinking(root)?;
    reject_images(root)?;
    let (none_check, allow_none) = controls(root, settings)?;
    let state = root.get("state").ok_or_else(|| {
        RequestError::new(
            422,
            vec!["body".into(), "state".into()],
            "state is required",
        )
    })?;
    let questions = root
        .get("questions")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            RequestError::new(
                422,
                vec!["body".into(), "questions".into()],
                "questions must be a non-empty object",
            )
        })?;
    if questions.is_empty() {
        return Err(RequestError::new(
            422,
            vec!["body".into(), "questions".into()],
            "questions must be a non-empty object",
        ));
    }
    let mut prompts = Vec::new();
    let mut planned = Vec::new();
    for (key, spec) in questions {
        let loc = vec!["body".into(), "questions".into(), key.clone()];
        let spec = spec
            .as_object()
            .ok_or_else(|| RequestError::new(422, loc.clone(), "a question must be an object"))?;
        let question = plan_question(
            key.clone(),
            state,
            spec,
            none_check,
            allow_none,
            settings,
            encode,
            &mut prompts,
        )
        .map_err(|err| {
            let mut full = loc;
            full.extend(err.loc);
            RequestError::new(err.status, full, err.message)
        })?;
        planned.push(question);
    }
    Ok(Prepared {
        prompts,
        questions: planned,
    })
}

pub fn plan_finals(
    prepared: &Prepared,
    logits: &[Vec<f64>],
    settings: &Settings,
    encode: &mut dyn PromptEncoder,
) -> Result<FinalRound, RequestError> {
    ensure_logits(logits, &prepared.prompts)?;
    let mut prompts = Vec::new();
    let mut slots = Vec::new();
    for (index, question) in prepared.questions.iter().enumerate() {
        let Body::Wide {
            groups,
            group_prompts,
            criteria,
            state,
            instructions,
            none,
            null_ok,
            ..
        } = &question.body
        else {
            continue;
        };
        let loc = vec!["body".into(), "questions".into(), question.key.clone()];
        let reads: Vec<Vec<f64>> = group_prompts
            .iter()
            .map(|&prompt| probabilities(&logits[prompt], TEMP_CHOICE))
            .collect();
        if reads
            .iter()
            .any(|row| row.iter().any(|prob| !prob.is_finite()))
        {
            return Err(RequestError::new(
                422,
                loc,
                "non-finite choice probabilities",
            ));
        }
        let (first, finalists) =
            knockout_finalists(groups, &reads, WIDE_KEEP, MAX_OPTIONS - usize::from(*none));
        let row = choice_row(state, instructions, &finalists, criteria, *none)
            .map_err(|err| RequestError::new(err.status, loc.clone(), err.message))?;
        let text = chat_prompt(&user_content(&row));
        let ids = encode.encode(&text).map_err(|err| {
            RequestError::new(422, loc.clone(), format!("tokenizer failed: {err}"))
        })?;
        if ids.is_empty() || ids.len() > settings.max_model_len || ids.len() > NATIVE_CONTEXT {
            return Err(RequestError::new(
                422,
                loc,
                format!(
                    "prompt has {} tokens, above the {} token limit; this worker does not truncate",
                    ids.len(),
                    settings.max_model_len.min(NATIVE_CONTEXT)
                ),
            ));
        }
        let prompt = prompts.len();
        let tokens = ids.len();
        prompts.push(Prompt {
            letters: row.options.len(),
            ids,
        });
        slots.push(FinalSlot {
            question: index,
            prompt,
            finalists,
            first,
            tokens,
            none: *none,
            null_ok: *null_ok,
        });
    }
    Ok(FinalRound { prompts, slots })
}

pub fn finish(
    prepared: &Prepared,
    first_logits: &[Vec<f64>],
    finals: &FinalRound,
    final_logits: &[Vec<f64>],
) -> Result<Value, RequestError> {
    ensure_logits(first_logits, &prepared.prompts)?;
    ensure_logits(final_logits, &finals.prompts)?;
    let mut answers = Map::new();
    let mut input_tokens = 0usize;
    for (index, question) in prepared.questions.iter().enumerate() {
        let answer = match &question.body {
            Body::Only { id } => choice_value(id, &[(id.clone(), 1.0)], None, false),
            Body::Direct {
                kind,
                prompt,
                readout,
                legend,
                none,
                null_ok,
                tokens,
            } => {
                input_tokens += tokens;
                let probs = probabilities(&first_logits[*prompt], kind.temperature());
                typed_answer(*kind, readout, legend, &probs, *none, *null_ok, None, None)?
            }
            Body::Wide { group_tokens, .. } => {
                input_tokens += group_tokens;
                let slot = finals
                    .slots
                    .iter()
                    .find(|slot| slot.question == index)
                    .expect("wide question has a final round");
                input_tokens += slot.tokens;
                let probs = probabilities(&final_logits[slot.prompt], TEMP_CHOICE);
                let mut readout = slot.finalists.clone();
                if slot.none {
                    readout.push(NONE_OPTION.to_string());
                }
                typed_answer(
                    Kind::Choice,
                    &readout,
                    &[],
                    &probs,
                    slot.none,
                    slot.null_ok,
                    Some(&slot.first),
                    Some(WIDE_RESIDUAL),
                )?
            }
        };
        answers.insert(question.key.clone(), answer);
    }
    Ok(json_map([
        ("answers", Value::Object(answers)),
        (
            "usage",
            json_map([
                ("input_tokens", Value::from(input_tokens)),
                ("output_tokens", Value::from(0)),
            ]),
        ),
        ("model", Value::String(MODEL_ID.into())),
    ]))
}

#[allow(clippy::too_many_arguments)]
fn plan_question(
    key: String,
    state: &Value,
    spec: &Map<String, Value>,
    none_check: bool,
    allow_none: bool,
    settings: &Settings,
    encode: &mut dyn PromptEncoder,
    prompts: &mut Vec<Prompt>,
) -> Result<Question, RequestError> {
    let kind = question_kind(spec.get("type"))?;
    let mut allow = if kind == Kind::Choice {
        allow_none
    } else {
        false
    };
    if kind == Kind::Choice
        && let Some(flag) = spec.get("allow_none")
    {
        allow = flag.as_bool().ok_or_else(|| {
            RequestError::new(
                422,
                vec!["allow_none".into()],
                format!("allow_none must be true or false, not {flag}"),
            )
        })?;
    }
    let none = kind == Kind::Choice && (allow || none_check);
    let null_ok = kind == Kind::Choice && allow;
    if kind == Kind::Choice {
        let criteria = choice_criteria(spec)?;
        if criteria.len() == 1 && !none {
            let id = criteria.keys().next().expect("len checked").clone();
            validate_id(&id)?;
            return Ok(Question {
                key,
                body: Body::Only { id },
            });
        }
        if criteria.len() > MAX_OPTIONS - usize::from(none) {
            let order: Vec<String> = criteria.keys().cloned().collect();
            let groups = knockout_groups(&order, WIDE_GROUP);
            let state_text = state_text(state);
            let instructions = required_instructions(spec)?;
            let mut group_prompts = Vec::new();
            let mut group_tokens = 0;
            for group in &groups {
                let row = choice_row(&state_text, &instructions, group, &criteria, false)?;
                let (ids, tokens) = encode_row(&row, settings, encode)?;
                group_tokens += tokens;
                group_prompts.push(prompts.len());
                prompts.push(Prompt {
                    letters: row.options.len(),
                    ids,
                });
            }
            return Ok(Question {
                key,
                body: Body::Wide {
                    groups,
                    group_prompts,
                    criteria,
                    state: state_text,
                    instructions,
                    none,
                    null_ok,
                    group_tokens,
                },
            });
        }
    }
    let (row, legend) = render_row(state, spec, kind, none)?;
    let (ids, tokens) = encode_row(&row, settings, encode)?;
    let prompt = prompts.len();
    prompts.push(Prompt {
        letters: row.options.len(),
        ids,
    });
    Ok(Question {
        key,
        body: Body::Direct {
            kind,
            prompt,
            readout: row.options.into_iter().map(|option| option.id).collect(),
            legend,
            none: row.none_option,
            null_ok,
            tokens,
        },
    })
}

fn encode_row(
    row: &Row,
    settings: &Settings,
    encode: &mut dyn PromptEncoder,
) -> Result<(Vec<u32>, usize), RequestError> {
    let text = chat_prompt(&user_content(row));
    let ids = encode
        .encode(&text)
        .map_err(|err| RequestError::new(422, vec![], format!("tokenizer failed: {err}")))?;
    if ids.is_empty() || ids.len() > settings.max_model_len || ids.len() > NATIVE_CONTEXT {
        return Err(RequestError::new(
            422,
            vec![],
            format!(
                "prompt has {} tokens, above the {} token limit; this worker does not truncate",
                ids.len(),
                settings.max_model_len.min(NATIVE_CONTEXT)
            ),
        ));
    }
    let tokens = ids.len();
    Ok((ids, tokens))
}

fn render_row(
    state: &Value,
    spec: &Map<String, Value>,
    kind: Kind,
    none: bool,
) -> Result<(Row, Vec<Value>), RequestError> {
    let state = state_text(state);
    match kind {
        Kind::Choice => {
            let criteria = choice_criteria(spec)?;
            let ids: Vec<String> = criteria.keys().cloned().collect();
            let row = choice_row(&state, &required_instructions(spec)?, &ids, &criteria, none)?;
            Ok((row, Vec::new()))
        }
        Kind::Noul => {
            let criteria = spec
                .get("criteria")
                .or_else(|| spec.get("options"))
                .and_then(Value::as_object);
            let description =
                |key: &str| criteria.and_then(|map| map.get(key)).and_then(value_text);
            let mut instructions = optional_instructions(spec)?;
            if instructions.is_empty() {
                instructions = NOUL_INSTRUCTIONS.into();
            }
            let row = Row {
                kind,
                state,
                instructions,
                options: vec![
                    OptionSpec {
                        id: "true".into(),
                        criterion: description("true"),
                    },
                    OptionSpec {
                        id: "false".into(),
                        criterion: description("false"),
                    },
                ],
                none_option: false,
            };
            validate_row(&row)?;
            Ok((row, Vec::new()))
        }
        Kind::Score => {
            let (levels, legend) = score_levels(spec)?;
            if !(2..=MAX_LEVELS).contains(&levels.len()) {
                return Err(RequestError::new(
                    422,
                    vec![],
                    format!(
                        "a score question needs 2 to {MAX_LEVELS} levels, got {}",
                        levels.len()
                    ),
                ));
            }
            let options = levels
                .into_iter()
                .enumerate()
                .map(|(i, criterion)| OptionSpec {
                    id: i.to_string(),
                    criterion,
                })
                .collect();
            let row = Row {
                kind,
                state,
                instructions: required_instructions(spec)?,
                options,
                none_option: false,
            };
            validate_row(&row)?;
            Ok((row, legend))
        }
    }
}

fn choice_row(
    state: &str,
    instructions: &str,
    ids: &[String],
    criteria: &Map<String, Value>,
    none: bool,
) -> Result<Row, RequestError> {
    if none && ids.iter().any(|id| norm(id) == norm(NONE_OPTION)) {
        return Err(RequestError::new(
            422,
            vec![],
            format!(
                "allow_none adds the option \"{NONE_OPTION}\" itself: remove the option of that name or leave allow_none out"
            ),
        ));
    }
    let mut options: Vec<OptionSpec> = ids
        .iter()
        .map(|id| OptionSpec {
            id: id.clone(),
            criterion: criteria.get(id).and_then(value_text),
        })
        .collect();
    if none {
        options.push(OptionSpec {
            id: NONE_OPTION.into(),
            criterion: None,
        });
    }
    let row = Row {
        kind: Kind::Choice,
        state: state.to_string(),
        instructions: instructions.to_string(),
        options,
        none_option: none,
    };
    validate_row(&row)?;
    Ok(row)
}

fn choice_criteria(spec: &Map<String, Value>) -> Result<Map<String, Value>, RequestError> {
    let raw = spec.get("criteria").or_else(|| spec.get("options"));
    match raw {
        Some(Value::Object(map)) => Ok(map.clone()),
        Some(Value::Array(items)) => {
            let mut map = Map::new();
            for item in items {
                let id = match item {
                    Value::String(text) => text.clone(),
                    Value::Number(number) => number.to_string(),
                    _ => {
                        return Err(RequestError::new(
                            422,
                            vec![],
                            "choice list entries must be strings or numbers",
                        ));
                    }
                };
                map.insert(id, Value::Null);
            }
            Ok(map)
        }
        _ => Err(RequestError::new(
            422,
            vec![],
            "a choice question needs criteria: {option: description or null} or a list of options",
        )),
    }
}

fn score_levels(
    spec: &Map<String, Value>,
) -> Result<(Vec<Option<String>>, Vec<Value>), RequestError> {
    let raw = spec
        .get("criteria")
        .or_else(|| spec.get("options"))
        .ok_or_else(|| {
            RequestError::new(
                422,
                vec![],
                "a score question needs criteria: a list of levels, lowest first",
            )
        })?;
    let values: Vec<Value> = match raw {
        Value::Array(items) => items.clone(),
        Value::Object(map) => score_keys(map)?
            .into_iter()
            .map(|key| map.get(&key).expect("key from map").clone())
            .collect(),
        _ => {
            return Err(RequestError::new(
                422,
                vec![],
                "a score question needs criteria: a list of levels, lowest first",
            ));
        }
    };
    let text = values.iter().map(value_text).collect();
    Ok((text, values))
}

fn score_keys(map: &Map<String, Value>) -> Result<Vec<String>, RequestError> {
    let keys: Vec<&String> = map.keys().collect();
    let numbers: Option<Vec<f64>> = keys
        .iter()
        .map(|key| key.parse::<f64>().ok().filter(|value| value.is_finite()))
        .collect();
    let mut order: Vec<usize> = (0..keys.len()).collect();
    if let Some(numbers) = numbers {
        order.sort_by(|&left, &right| {
            numbers[left]
                .partial_cmp(&numbers[right])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    Ok(order.into_iter().map(|index| keys[index].clone()).collect())
}

fn question_kind(value: Option<&Value>) -> Result<Kind, RequestError> {
    let Some(value) = value else {
        return Ok(Kind::Choice);
    };
    let Some(text) = value.as_str() else {
        return Err(RequestError::new(
            422,
            vec!["type".into()],
            format!("unknown question type {value}"),
        ));
    };
    match text {
        "choice" => Ok(Kind::Choice),
        "noul" | "bool" => Ok(Kind::Noul),
        "score" => Ok(Kind::Score),
        other => Err(RequestError::new(
            422,
            vec!["type".into()],
            format!("unknown question type {other:?}: use choice, noul or score"),
        )),
    }
}

fn required_instructions(spec: &Map<String, Value>) -> Result<String, RequestError> {
    let text = optional_instructions(spec)?;
    if text.is_empty() {
        return Err(RequestError::new(
            422,
            vec![],
            "instructions are required for choice and score questions",
        ));
    }
    Ok(text)
}

fn optional_instructions(spec: &Map<String, Value>) -> Result<String, RequestError> {
    let raw = spec.get("instructions").or_else(|| spec.get("question"));
    Ok(raw.and_then(value_text).unwrap_or_default())
}

fn validate_row(row: &Row) -> Result<(), RequestError> {
    if row.instructions.trim().is_empty() {
        return Err(RequestError::new(
            422,
            vec![],
            "instructions are required for choice and score questions",
        ));
    }
    if !(2..=MAX_OPTIONS).contains(&row.options.len()) {
        return Err(RequestError::new(
            422,
            vec![],
            format!(
                "a question needs 2 to {MAX_OPTIONS} options in one pass, got {}",
                row.options.len()
            ),
        ));
    }
    let mut seen = Vec::new();
    for option in &row.options {
        validate_id(&option.id)?;
        if seen.iter().any(|id| id == &option.id) {
            return Err(RequestError::new(422, vec![], "duplicate option id"));
        }
        seen.push(option.id.clone());
    }
    if row.kind == Kind::Noul {
        let mut ids = seen.clone();
        ids.sort();
        if ids != ["false".to_string(), "true".to_string()] {
            return Err(RequestError::new(
                422,
                vec![],
                "yes/no options must be 'true' and 'false'",
            ));
        }
    }
    if row.kind == Kind::Score && !(2..=MAX_LEVELS).contains(&row.options.len()) {
        return Err(RequestError::new(
            422,
            vec![],
            format!(
                "a score question needs 2 to {MAX_LEVELS} levels, got {}",
                row.options.len()
            ),
        ));
    }
    Ok(())
}

fn validate_id(id: &str) -> Result<(), RequestError> {
    if id.trim().is_empty() || id.contains('\n') || id.contains('\r') || id.chars().count() > 200 {
        return Err(RequestError::new(
            422,
            vec![],
            format!(
                "bad option id {id:?}: ids must be non-empty single-line strings of at most 200 characters"
            ),
        ));
    }
    Ok(())
}

fn user_content(row: &Row) -> String {
    let state = if row.state.trim().is_empty() {
        "(none)"
    } else {
        row.state.as_str()
    };
    format!(
        "Evidence:\n{state}\n\nQuestion: {}\nOptions:\n{}",
        row.instructions.trim(),
        option_lines(row).join("\n")
    )
}

fn option_lines(row: &Row) -> Vec<String> {
    let extra = row.none_option.then(|| row.options.len() - 1);
    let listed = match extra {
        Some(index) => &row.options[..index],
        None => &row.options,
    };
    let hide = row.kind == Kind::Choice
        && listed.iter().all(|option| is_bare(&option.id))
        && listed.iter().all(|option| {
            option
                .criterion
                .as_ref()
                .is_some_and(|text| !text.is_empty())
        });
    row.options
        .iter()
        .enumerate()
        .map(|(index, option)| {
            let name = if row.kind == Kind::Noul {
                if option.id == "true" { "yes" } else { "no" }
            } else {
                option.id.as_str()
            };
            let body = if hide && Some(index) != extra {
                option.criterion.as_deref().unwrap_or(name)
            } else {
                match option.criterion.as_deref() {
                    Some(text) if !text.trim().is_empty() && norm(text) != norm(name) => {
                        return format!("{}) {name}: {text}", letter(index));
                    }
                    _ => name,
                }
            };
            format!("{}) {body}", letter(index))
        })
        .collect()
}

fn letter(index: usize) -> char {
    LETTERS[index] as char
}

fn is_bare(id: &str) -> bool {
    let lower = id.to_ascii_lowercase();
    if let Some(rest) = lower
        .strip_prefix("option")
        .or_else(|| lower.strip_prefix("opt"))
    {
        let rest = rest.strip_prefix(['_', ' ']).unwrap_or(rest);
        if (1..=3).contains(&rest.len()) && rest.bytes().all(|byte| byte.is_ascii_digit()) {
            return true;
        }
    }
    let bytes = lower.as_bytes();
    let bytes = bytes.strip_prefix(b"(").unwrap_or(bytes);
    let bytes = bytes
        .strip_suffix(b")")
        .or_else(|| bytes.strip_suffix(b"."))
        .unwrap_or(bytes);
    (bytes.len() == 1 && bytes[0].is_ascii_alphabetic())
        || ((1..=3).contains(&bytes.len()) && bytes.iter().all(|byte| byte.is_ascii_digit()))
}

fn norm(text: &str) -> String {
    let mut out = String::new();
    let mut separator = false;
    for ch in text.chars() {
        if ch.is_whitespace() || ch == '_' || ch == '-' {
            if !out.is_empty() {
                separator = true;
            }
        } else if separator {
            out.push(' ');
            separator = false;
            out.extend(ch.to_lowercase());
        } else {
            out.extend(ch.to_lowercase());
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn typed_answer(
    kind: Kind,
    readout: &[String],
    legend: &[Value],
    probs: &[f64],
    none: bool,
    null_ok: bool,
    first: Option<&[(String, f64)]>,
    residual: Option<f64>,
) -> Result<Value, RequestError> {
    if probs.len() != readout.len() || probs.iter().any(|prob| !prob.is_finite()) {
        return Err(RequestError::new(
            422,
            vec![],
            "non-finite or mismatched option probabilities",
        ));
    }
    match kind {
        Kind::Noul => {
            let index = readout.iter().position(|id| id == "true").ok_or_else(|| {
                RequestError::new(422, vec![], "yes/no options must be 'true' and 'false'")
            })?;
            Ok(json_map([
                ("type", Value::String("noul".into())),
                ("noul", json_f64(probs[index])),
            ]))
        }
        Kind::Score => {
            let score = probs
                .iter()
                .enumerate()
                .map(|(index, prob)| index as f64 * prob)
                .sum::<f64>();
            let mut probabilities = Map::new();
            let mut legend_map = Map::new();
            for (index, prob) in probs.iter().enumerate() {
                let key = index.to_string();
                probabilities.insert(key.clone(), json_f64(*prob));
                let label = legend
                    .get(index)
                    .cloned()
                    .unwrap_or(Value::String(key.clone()));
                legend_map.insert(key, label);
            }
            Ok(json_map([
                ("type", Value::String("score".into())),
                ("score", json_f64(score)),
                ("confidence", json_f64(score_confidence(probs))),
                ("legend", Value::Object(legend_map)),
                ("probabilities", Value::Object(probabilities)),
            ]))
        }
        Kind::Choice => {
            let (listed, none_prob, abstain) = if none {
                split_none(probs)
            } else {
                (probs.to_vec(), None, false)
            };
            let ids: Vec<String> = if none {
                readout[..readout.len() - 1].to_vec()
            } else {
                readout.to_vec()
            };
            let combined = if let (Some(first), Some(residual)) = (first, residual) {
                knockout_combine(&ids, &listed, first, residual)
            } else {
                ids.iter().cloned().zip(listed).collect()
            };
            // Response keys follow the caller's listed order when there is no
            // knock-out, and the original option order carried in `first` otherwise.
            let ordered = if let Some(first) = first {
                first
                    .iter()
                    .map(|(id, _)| {
                        let prob = combined
                            .iter()
                            .find(|(key, _)| key == id)
                            .map(|(_, prob)| *prob)
                            .unwrap_or(0.0);
                        (id.clone(), prob)
                    })
                    .collect()
            } else {
                combined
            };
            Ok(choice_value(
                &argmax(&ordered),
                &ordered,
                none_prob,
                abstain && null_ok,
            ))
        }
    }
}

fn choice_value(id: &str, probs: &[(String, f64)], none: Option<f64>, abstain: bool) -> Value {
    let mut probabilities = Map::new();
    for (key, prob) in probs {
        probabilities.insert(key.clone(), json_f64(*prob));
    }
    let mut answer = Map::new();
    answer.insert("type".into(), Value::String("choice".into()));
    answer.insert(
        "choice".into(),
        if abstain {
            Value::Null
        } else {
            Value::String(id.into())
        },
    );
    answer.insert(
        "confidence".into(),
        json_f64(choice_confidence(
            &probs.iter().map(|(_, prob)| *prob).collect::<Vec<_>>(),
        )),
    );
    answer.insert("probabilities".into(), Value::Object(probabilities));
    if let Some(none) = none {
        answer.insert("none_probability".into(), json_f64(none));
    }
    Value::Object(answer)
}

fn split_none(probs: &[f64]) -> (Vec<f64>, Option<f64>, bool) {
    let (listed, none) = probs.split_at(probs.len() - 1);
    let none = none[0];
    let total: f64 = listed.iter().sum();
    let renormalised = if total > 0.0 {
        listed.iter().map(|prob| prob / total).collect()
    } else {
        vec![1.0 / listed.len() as f64; listed.len()]
    };
    let top = listed.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (renormalised, Some(none), none > top)
}

fn knockout_groups(keys: &[String], group: usize) -> Vec<Vec<String>> {
    let groups = keys.len().div_ceil(group);
    let (size, extra) = (keys.len() / groups, keys.len() % groups);
    let mut out = Vec::new();
    let mut start = 0;
    for index in 0..groups {
        let end = start + size + usize::from(index < extra);
        out.push(keys[start..end].to_vec());
        start = end;
    }
    out
}

fn knockout_finalists(
    groups: &[Vec<String>],
    reads: &[Vec<f64>],
    keep: usize,
    max_options: usize,
) -> (Vec<(String, f64)>, Vec<String>) {
    let mut first = Vec::new();
    let mut finalists = Vec::new();
    for (group, probs) in groups.iter().zip(reads) {
        for (id, prob) in group.iter().zip(probs) {
            first.push((id.clone(), *prob));
        }
        let mut ranked: Vec<usize> = (0..group.len()).collect();
        ranked.sort_by(|&left, &right| {
            probs[right]
                .partial_cmp(&probs[left])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for index in ranked.into_iter().take(keep) {
            finalists.push(group[index].clone());
        }
    }
    if finalists.len() > max_options {
        finalists.sort_by(|left, right| {
            let lp = lookup(&first, left);
            let rp = lookup(&first, right);
            rp.partial_cmp(&lp).unwrap_or(std::cmp::Ordering::Equal)
        });
        finalists.truncate(max_options);
    }
    (first, finalists)
}

fn knockout_combine(
    finalists: &[String],
    final_probs: &[f64],
    first: &[(String, f64)],
    residual: f64,
) -> Vec<(String, f64)> {
    let rest: Vec<&str> = first
        .iter()
        .map(|(id, _)| id.as_str())
        .filter(|id| !finalists.iter().any(|finalist| finalist == id))
        .collect();
    let mass = rest.iter().map(|id| lookup(first, id)).sum::<f64>();
    let mass = if mass == 0.0 { 1.0 } else { mass };
    let mut dist: Vec<(String, f64)> = finalists
        .iter()
        .zip(final_probs)
        .map(|(id, prob)| (id.clone(), prob * (1.0 - residual)))
        .collect();
    for id in rest {
        dist.push((id.to_string(), residual * lookup(first, id) / mass));
    }
    let sum: f64 = dist.iter().map(|(_, prob)| *prob).sum();
    if sum > 0.0 {
        for (_, prob) in &mut dist {
            *prob /= sum;
        }
    }
    dist
}

fn lookup(pairs: &[(String, f64)], id: &str) -> f64 {
    pairs
        .iter()
        .find(|(key, _)| key == id)
        .map(|(_, prob)| *prob)
        .unwrap_or(0.0)
}

fn argmax(pairs: &[(String, f64)]) -> String {
    let mut best = 0;
    for (index, (_, prob)) in pairs.iter().enumerate().skip(1) {
        if *prob > pairs[best].1 {
            best = index;
        }
    }
    pairs[best].0.clone()
}

fn ensure_logits(logits: &[Vec<f64>], prompts: &[Prompt]) -> Result<(), RequestError> {
    if logits.len() != prompts.len()
        || logits
            .iter()
            .zip(prompts)
            .any(|(row, prompt)| row.len() != prompt.letters)
    {
        return Err(RequestError::new(
            422,
            vec![],
            "the readout does not match the prepared prompts",
        ));
    }
    Ok(())
}

fn reject_thinking(root: &Map<String, Value>) -> Result<(), RequestError> {
    if let Some(mode) = root.get("think_mode") {
        let canonical = match mode.as_str() {
            Some("auto" | "hybrid") => "auto",
            Some("always" | "think") => "always",
            Some("never" | "plain") => "never",
            _ => {
                return Err(RequestError::new(
                    422,
                    vec!["body".into(), "think_mode".into()],
                    "think_mode must be one of auto, never, always (or hybrid, plain, think)",
                ));
            }
        };
        if canonical != "never" {
            return Err(RequestError::new(
                422,
                vec!["body".into(), "think_mode".into()],
                "this is an Instant model: it answers in one pass and does not think; leave think_mode out or set it to \"never\" (thinking needs an AutoThink model)",
            ));
        }
    }
    if let Some(generation) = root.get("thought_generation") {
        match generation.as_str() {
            Some("auto" | "chunked") => {}
            Some("stream") => {
                return Err(RequestError::new(
                    422,
                    vec!["body".into(), "thought_generation".into()],
                    "this engine does not stream thoughts (vLLM 0.31 or later does): leave thought_generation out or set it to \"chunked\"",
                ));
            }
            _ => {
                return Err(RequestError::new(
                    422,
                    vec!["body".into(), "thought_generation".into()],
                    "thought_generation must be one of auto, stream, chunked",
                ));
            }
        }
    }
    for name in ["return_details", "return_thought"] {
        if let Some(value) = root.get(name) {
            match value.as_bool() {
                Some(false) => {}
                Some(true) => {
                    return Err(RequestError::new(
                        422,
                        vec!["body".into(), name.into()],
                        "return_details and return_thought are not served by this native worker",
                    ));
                }
                None => {
                    return Err(RequestError::new(
                        422,
                        vec!["body".into(), name.into()],
                        format!("{name} must be true or false"),
                    ));
                }
            }
        }
    }
    Ok(())
}

fn reject_images(root: &Map<String, Value>) -> Result<(), RequestError> {
    match root.get("images") {
        None | Some(Value::Null) => Ok(()),
        Some(Value::Array(items)) if items.is_empty() => Ok(()),
        Some(Value::Array(_)) => Err(RequestError::new(
            422,
            vec!["body".into(), "images".into()],
            "this native worker reads text only: send the evidence without images",
        )),
        Some(_) => Err(RequestError::new(
            422,
            vec!["body".into(), "images".into()],
            "images must be a list of base64 strings or data URIs",
        )),
    }
}

fn controls(root: &Map<String, Value>, settings: &Settings) -> Result<(bool, bool), RequestError> {
    let mut none_check = settings.none_check;
    let mut allow_none = settings.allow_none;
    if let Some(value) = root.get("none_check") {
        none_check = value.as_bool().ok_or_else(|| {
            RequestError::new(
                422,
                vec!["body".into(), "none_check".into()],
                "none_check must be true or false",
            )
        })?;
    }
    if let Some(value) = root.get("allow_none") {
        allow_none = value.as_bool().ok_or_else(|| {
            RequestError::new(
                422,
                vec!["body".into(), "allow_none".into()],
                "allow_none must be true or false",
            )
        })?;
    }
    if let Some(value) = root.get("gate") {
        fraction(value, "gate")?;
    }
    if let Some(value) = root.get("mix") {
        fraction(value, "mix")?;
    }
    if let Some(value) = root.get("think_budget") {
        let Some(budget) = value.as_u64() else {
            return Err(RequestError::new(
                422,
                vec!["body".into(), "think_budget".into()],
                "think_budget must be an integer of at least 1",
            ));
        };
        if budget < 1 {
            return Err(RequestError::new(
                422,
                vec!["body".into(), "think_budget".into()],
                "think_budget must be an integer of at least 1",
            ));
        }
    }
    Ok((none_check, allow_none))
}

fn fraction(value: &Value, name: &str) -> Result<f64, RequestError> {
    let Some(number) = value.as_f64() else {
        return Err(RequestError::new(
            422,
            vec!["body".into(), name.into()],
            format!("{name} must be a number from 0 to 1"),
        ));
    };
    if !(0.0..=1.0).contains(&number) {
        return Err(RequestError::new(
            422,
            vec!["body".into(), name.into()],
            format!("{name} must be a number from 0 to 1"),
        ));
    }
    Ok(number)
}

pub fn state_text(state: &Value) -> String {
    match state {
        Value::String(text) => text.clone(),
        Value::Null => String::new(),
        Value::Object(map) if map.is_empty() => String::new(),
        Value::Array(items) if items.is_empty() => String::new(),
        other => py_dumps(&annotate(other)),
    }
}

fn annotate(value: &Value) -> Value {
    match value {
        Value::Array(items) if items.len() >= ANNOTATE_MIN => Value::Array(
            items
                .iter()
                .enumerate()
                .map(|(index, item)| {
                    let mut object = Map::new();
                    object.insert("_index".into(), Value::from(index));
                    match annotate(item) {
                        Value::Object(inner) => {
                            for (key, value) in inner {
                                object.insert(key, value);
                            }
                        }
                        other => {
                            object.insert("value".into(), other);
                        }
                    }
                    Value::Object(object)
                })
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(annotate).collect()),
        Value::Object(map) => {
            let mut object = Map::new();
            for (key, value) in map {
                object.insert(key.clone(), annotate(value));
            }
            Value::Object(object)
        }
        other => other.clone(),
    }
}

fn py_dumps(value: &Value) -> String {
    let mut out = String::new();
    write_py(&mut out, value);
    out
}

fn write_py(out: &mut String, value: &Value) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
        Value::Number(number) => out.push_str(&py_number(number)),
        Value::String(text) => push_py_string(out, text),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_py(out, item);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (index, (key, item)) in map.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                push_py_string(out, key);
                out.push_str(": ");
                write_py(out, item);
            }
            out.push('}');
        }
    }
}

fn py_number(number: &Number) -> String {
    if let Some(value) = number.as_i64() {
        return value.to_string();
    }
    if let Some(value) = number.as_u64() {
        return value.to_string();
    }
    serde_json::to_string(number).unwrap_or_else(|_| "null".into())
}

fn push_py_string(out: &mut String, text: &str) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if (ch as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out.push('"');
}

fn value_text(value: &Value) -> Option<String> {
    if value.is_null() {
        return None;
    }
    let text = match value {
        Value::String(text) => text.clone(),
        other => py_dumps(other),
    };
    let text = text.trim();
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

fn json_f64(value: f64) -> Value {
    Number::from_f64(value)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

fn json_map<const N: usize>(pairs: [(&str, Value); N]) -> Value {
    let mut map = Map::new();
    for (key, value) in pairs {
        map.insert(key.to_string(), value);
    }
    Value::Object(map)
}

pub fn probe_prompt() -> String {
    chat_prompt("Evidence:\ns\n\nQuestion: q\nOptions:\nA) yes\nB) no")
}

/// Refuse AutoThink, 27B and any other folder. The numeric pins are the Instant 4B
/// `config.json` / `decision_config.json` at `WEIGHTS_REVISION`.
pub fn require_instant_configs(config: &Value, decision: &Value) -> Result<(), String> {
    let text = config.get("text_config").unwrap_or(config);
    let arch = config
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|items| {
            items
                .iter()
                .any(|item| item.as_str() == Some("Qwen3_5ForConditionalGeneration"))
                .then_some(())
        });
    if arch.is_none() {
        return Err(
            "config.json architectures must include Qwen3_5ForConditionalGeneration".into(),
        );
    }
    if config.get("tie_word_embeddings").and_then(Value::as_bool) != Some(true) {
        return Err(
            "Instant 4B ties the letter head to embed_tokens; tie_word_embeddings must be true"
                .into(),
        );
    }
    if config.get("jev_prompt_attention").and_then(Value::as_str) != Some("bidirectional") {
        return Err("config.json jev_prompt_attention must be bidirectional; this worker does not serve causal AutoThink".into());
    }
    expect_u64(text, "hidden_size", 2560)?;
    expect_u64(text, "intermediate_size", 9216)?;
    expect_u64(text, "num_hidden_layers", 32)?;
    expect_u64(text, "num_attention_heads", 16)?;
    expect_u64(text, "num_key_value_heads", 4)?;
    expect_u64(text, "head_dim", 256)?;
    expect_u64(text, "vocab_size", 248_320)?;
    expect_u64(text, "max_position_embeddings", 262_144)?;
    expect_u64(text, "linear_num_key_heads", 16)?;
    expect_u64(text, "linear_num_value_heads", 32)?;
    expect_u64(text, "linear_key_head_dim", 128)?;
    expect_u64(text, "linear_value_head_dim", 128)?;
    let layers = text
        .get("layer_types")
        .and_then(Value::as_array)
        .ok_or("config.json text_config.layer_types is missing")?;
    if layers.len() != 32
        || layers.iter().enumerate().any(|(index, layer)| {
            layer.as_str()
                != Some(if index % 4 == 3 {
                    "full_attention"
                } else {
                    "linear_attention"
                })
        })
    {
        return Err(
            "Instant 4B full attention is every fourth layer; this folder is not that model".into(),
        );
    }
    let rope = text
        .get("rope_parameters")
        .ok_or("config.json rope_parameters is missing")?;
    if rope.get("partial_rotary_factor").and_then(Value::as_f64) != Some(0.25) {
        return Err("partial_rotary_factor must be 0.25".into());
    }
    if rope.get("rope_theta").and_then(Value::as_f64) != Some(10_000_000.0) {
        return Err("rope_theta must be 10000000".into());
    }
    let section = rope.get("mrope_section").and_then(Value::as_array);
    let section_ok = section.is_some_and(|items| {
        items.len() == 3
            && items[0].as_u64() == Some(11)
            && items[1].as_u64() == Some(11)
            && items[2].as_u64() == Some(10)
    });
    if !section_ok {
        return Err("mrope_section must be [11, 11, 10]".into());
    }
    if decision.get("auto_think").is_some() {
        return Err("decision_config.json has auto_think; this worker serves Instant only".into());
    }
    if decision.get("format").and_then(Value::as_str) != Some("lodestar-letter-readout-v1") {
        return Err("decision_config.json format must be lodestar-letter-readout-v1".into());
    }
    if decision.get("prompt_attention").and_then(Value::as_str) != Some("bidirectional") {
        return Err("decision_config.json prompt_attention must be bidirectional".into());
    }
    if decision.get("max_options_per_pass").and_then(Value::as_u64) != Some(26) {
        return Err("max_options_per_pass must be 26".into());
    }
    let letters = decision
        .get("letter_token_ids")
        .and_then(Value::as_array)
        .ok_or("letter_token_ids is missing")?;
    if letters.len() != LETTER_IDS.len()
        || letters
            .iter()
            .enumerate()
            .any(|(index, id)| id.as_u64() != Some(u64::from(LETTER_IDS[index])))
    {
        return Err("letter_token_ids must be 32..=57".into());
    }
    let temps = decision
        .get("temperature_by_type")
        .and_then(Value::as_object)
        .ok_or("temperature_by_type is missing")?;
    expect_temp(temps, "choice", TEMP_CHOICE)?;
    expect_temp(temps, "noul", TEMP_NOUL)?;
    expect_temp(temps, "score", TEMP_SCORE)?;
    let wide = decision
        .get("wide_choice")
        .and_then(Value::as_object)
        .ok_or("wide_choice is missing")?;
    if wide.get("group").and_then(Value::as_u64) != Some(25)
        || wide.get("keep").and_then(Value::as_u64) != Some(3)
    {
        return Err("wide_choice group/keep must be 25/3".into());
    }
    let residual = wide.get("residual").and_then(Value::as_f64).unwrap_or(-1.0);
    if (residual - WIDE_RESIDUAL).abs() > 1e-12 {
        return Err("wide_choice residual must be 0.001".into());
    }
    Ok(())
}

fn expect_u64(value: &Value, key: &str, expected: u64) -> Result<(), String> {
    if value.get(key).and_then(Value::as_u64) == Some(expected) {
        Ok(())
    } else {
        Err(format!(
            "config.json {key} must be {expected}; this worker serves Instant 4B only"
        ))
    }
}

fn expect_temp(temps: &Map<String, Value>, key: &str, expected: f64) -> Result<(), String> {
    let value = temps.get(key).and_then(Value::as_f64).unwrap_or(f64::NAN);
    if (value - expected).abs() <= 1e-9 {
        Ok(())
    } else {
        Err(format!("temperature_by_type.{key} must be {expected}"))
    }
}
