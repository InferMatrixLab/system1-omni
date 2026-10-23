//! The `/v1/systemone` wire contract, equivalent to
//! `src/models/valen/protocol.py` at the pinned reference: the same limits,
//! the same acceptance rules, the same validation order and the same 4xx
//! error bodies (`{"detail": ...}` with the recorded status).

use base64::Engine;
use serde_json::{Map, Value};

use crate::wire::{RequestError, error, python_repr, python_strip};

pub const MODEL_NAME: &str = "valen-preview-0923";
pub const MAX_IMAGE_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_PIXELS: usize = 1024 * 1024;
pub const MAX_SIDE: usize = 2048;
pub const MAX_ASPECT_RATIO: usize = 200;
pub const MAX_QUESTIONS: usize = 8;
pub const MAX_OPTIONS: usize = 255;
pub const MAX_TEXT: usize = 16384;
pub const MIN_SCORE_LEVELS: usize = 2;
pub const MAX_SCORE_LEVELS: usize = 10;

/// Verbatim from the pinned `valen.data.schema.candidates` noul wording at
/// revision 750bfcf; the compiler emits these for every noul question, so the
/// wire reuses them whenever a side is left unspecified.
pub const NOUL_TRUE_DESCRIPTION: &str = "True / 是：满足问题中的条件。";
pub const NOUL_FALSE_DESCRIPTION: &str = "False / 否：不满足问题中的条件。";

/// The blocked tokenizer control tokens when no tokenizer set is injected.
pub const DEFAULT_SPECIAL_TOKENS: [&str; 4] = [
    "<|image_pad|>",
    "<|video_pad|>",
    "<|vision_start|>",
    "<|vision_end|>",
];

/// The three wire question types. `noul` and `score` finishing variations are
/// later layers; the wire shape and the compiled branch structure carry them
/// already.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Choice,
    Noul,
    Score,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Choice => "choice",
            Kind::Noul => "noul",
            Kind::Score => "score",
        }
    }
}

/// A validated inline image: decoded bytes plus the header's dimensions.
#[derive(Debug)]
pub struct ImageData {
    pub format: image::ImageFormat,
    pub data: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

/// One validated wire question. `descriptions` keeps `None` for a `noul` side
/// left to the reference's fixed wording.
#[derive(Debug)]
pub struct Question {
    pub name: String,
    pub kind: Kind,
    pub instructions: String,
    pub keys: Vec<String>,
    pub descriptions: Vec<Option<String>>,
}

/// The request's state: plain or JSON-serialized text, or one inline image.
#[derive(Debug)]
pub enum State {
    Text(String),
    Image(ImageData),
}

#[derive(Debug)]
pub struct Request {
    pub model: &'static str,
    pub state: State,
    pub questions: Vec<Question>,
}

/// `protocol._text`: a string as is, an object or array as
/// `json.dumps(value, ensure_ascii=False)`, within the character limit and
/// free of the tokenizer's control tokens.
fn text(value: &Value, field: &str, special: &[String]) -> Result<String, RequestError> {
    let result = match value {
        Value::String(s) => s.clone(),
        Value::Object(_) | Value::Array(_) => omni_qwen3_5_native::json::dumps(value),
        _ => {
            return Err(error(
                422,
                format!("{field} must be a string, object or array"),
            ));
        }
    };
    if result.chars().count() > MAX_TEXT {
        return Err(error(422, format!("{field} exceeds {MAX_TEXT} characters")));
    }
    if special.iter().any(|token| result.contains(token.as_str())) {
        return Err(error(
            422,
            format!("{field} contains an unsupported tokenizer control token"),
        ));
    }
    Ok(result)
}

/// A PNG reports more than one frame through its `acTL` animation chunk; PIL
/// surfaces the same fact as `n_frames != 1` and the wire rejects it.
fn png_is_animated(data: &[u8]) -> bool {
    if data.len() < 8 || &data[..8] != b"\x89PNG\r\n\x1a\n" {
        return false;
    }
    let mut at = 8;
    while at + 8 <= data.len() {
        let length =
            u32::from_be_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]]) as usize;
        let kind = &data[at + 4..at + 8];
        if kind == b"acTL" {
            return true;
        }
        if kind == b"IEND" {
            return false;
        }
        at += 12 + length; // length + type + data + crc
    }
    false
}

fn image_state(value: &Value) -> Result<ImageData, RequestError> {
    let invalid = || error(422, "state must contain exactly one image data URL");
    let Value::Object(state) = value else {
        return Err(invalid());
    };
    if state.len() != 1 || !state.contains_key("image") {
        return Err(invalid());
    }
    let Value::String(url) = &state["image"] else {
        return Err(error(422, "state.image must be a PNG/JPEG base64 data URL"));
    };
    let Some((prefix, encoded)) = url.split_once(',') else {
        return Err(error(422, "only inline PNG/JPEG images are supported"));
    };
    let format = match prefix {
        "data:image/png;base64" => image::ImageFormat::Png,
        "data:image/jpeg;base64" => image::ImageFormat::Jpeg,
        _ => return Err(error(422, "only inline PNG/JPEG images are supported")),
    };
    if encoded.len() > 4 * MAX_IMAGE_BYTES.div_ceil(3) {
        // 4 * ((MAX_IMAGE_BYTES + 2) // 3), as the reference computes it.
        return Err(error(422, "encoded image exceeds 4 MiB"));
    }
    let data = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|_| error(422, "invalid image data"))?;
    if data.len() > MAX_IMAGE_BYTES {
        return Err(error(422, "image exceeds 4 MiB"));
    }
    // PIL's Image.open identifies the format from the magic bytes and reads
    // the dimensions from the header; both decode failures surface as the
    // reference's generic "invalid image data".
    let identified = image::guess_format(&data).map_err(|_| error(422, "invalid image data"))?;
    if identified != format {
        return Err(error(422, "image format does not match its MIME type"));
    }
    let reader = image::ImageReader::with_format(std::io::Cursor::new(&data), format);
    let (width, height) = reader
        .into_dimensions()
        .map_err(|_| error(422, "invalid image data"))?;
    let (width, height) = (width as usize, height as usize);
    if width.max(height) > MAX_ASPECT_RATIO * width.min(height) {
        return Err(error(
            422,
            format!("image aspect ratio must not exceed {MAX_ASPECT_RATIO}:1"),
        ));
    }
    let animated = format == image::ImageFormat::Png && png_is_animated(&data);
    if width.max(height) > MAX_SIDE || width * height > MAX_PIXELS || animated {
        return Err(error(
            422,
            "image must be single-frame, at most 2048 per side and 1048576 pixels",
        ));
    }
    // PIL's source.load(): a header that passes every limit can still fail on
    // truncated or corrupt pixel data.
    image::load_from_memory(&data)
        .map_err(|_| error(422, "invalid image data"))
        .map(|_| ImageData {
            format,
            data,
            width,
            height,
        })
}

fn state(value: &Value, special: &[String]) -> Result<State, RequestError> {
    // A dict carrying an "image" key always enters image validation, so a
    // mistyped image request fails loudly instead of silently becoming text.
    if let Value::Object(map) = value
        && map.contains_key("image")
    {
        return image_state(value).map(State::Image);
    }
    let text = match value {
        Value::String(_) | Value::Object(_) | Value::Array(_) => text(value, "state", special)?,
        _ => return Err(error(422, "state must be an image data URL or text")),
    };
    if text.is_empty() {
        return Err(error(422, "state must not be empty"));
    }
    if python_strip(&text).is_empty() {
        return Err(error(422, "state must not be only whitespace"));
    }
    Ok(State::Text(text))
}

fn question(name: &str, value: &Value, special: &[String]) -> Result<Question, RequestError> {
    let name_chars = name.chars().count();
    if name_chars == 0 || name_chars > 256 {
        return Err(error(
            422,
            "question names must contain 1 to 256 characters",
        ));
    }
    let place = format!("question {}", python_repr(name));
    let Value::Object(question) = value else {
        return Err(error(422, format!("{place} contains unsupported fields")));
    };
    if !question
        .keys()
        .all(|key| matches!(key.as_str(), "type" | "instructions" | "criteria"))
    {
        return Err(error(422, format!("{place} contains unsupported fields")));
    }
    let kind = match question.get("type").and_then(Value::as_str) {
        Some("choice") => Kind::Choice,
        Some("noul") => Kind::Noul,
        Some("score") => Kind::Score,
        _ => {
            return Err(error(
                422,
                format!("{place}: type must be one of choice, noul, score"),
            ));
        }
    };
    let Some(instructions_value) = question.get("instructions") else {
        return Err(error(422, format!("{place}: instructions is required")));
    };
    let instructions = text(
        instructions_value,
        &format!("{place} instructions"),
        special,
    )?;
    if python_strip(&instructions).is_empty() {
        return Err(error(
            422,
            format!("{place}: instructions must be nonempty"),
        ));
    }
    let mut keys: Vec<String> = Vec::new();
    let mut descriptions: Vec<Option<String>> = Vec::new();
    match kind {
        Kind::Noul => {
            let criteria = question.get("criteria");
            if let Some(criteria) = criteria.filter(|value| !value.is_null()) {
                let valid = matches!(criteria, Value::Object(map)
                    if !map.is_empty()
                        && map.keys().all(|k| k == "true" || k == "false"));
                if !valid {
                    return Err(error(
                        422,
                        format!("{place}: noul criteria must describe true and/or false"),
                    ));
                }
            }
            for key in ["true", "false"] {
                let label = criteria
                    .and_then(Value::as_object)
                    .and_then(|map| map.get(key))
                    .filter(|value| !value.is_null());
                let description = match label {
                    None => None,
                    Some(label) => {
                        let field = format!("{place} criteria {}", python_repr(key));
                        let description = text(label, &field, special)?;
                        if python_strip(&description).is_empty() {
                            return Err(error(
                                422,
                                format!("{place}: criterion {} must be nonempty", python_repr(key)),
                            ));
                        }
                        Some(description)
                    }
                };
                keys.push(key.to_owned());
                descriptions.push(description);
            }
        }
        Kind::Score => {
            let valid = matches!(question.get("criteria"), Some(Value::Array(list))
                if (MIN_SCORE_LEVELS..=MAX_SCORE_LEVELS).contains(&list.len()));
            if !valid {
                return Err(error(
                    422,
                    format!(
                        "{place}: score requires an ordered list of {MIN_SCORE_LEVELS} to \
                         {MAX_SCORE_LEVELS} level descriptions"
                    ),
                ));
            }
            let criteria = question.get("criteria").and_then(Value::as_array).unwrap();
            for (index, label) in criteria.iter().enumerate() {
                let description = text(label, &format!("{place} level {index}"), special)?;
                if python_strip(&description).is_empty() {
                    return Err(error(
                        422,
                        format!("{place}: level {index} must be nonempty"),
                    ));
                }
                keys.push(index.to_string());
                descriptions.push(Some(description));
            }
        }
        Kind::Choice => {
            let valid = matches!(question.get("criteria"), Some(Value::Object(map))
                if (1..=MAX_OPTIONS).contains(&map.len()));
            if !valid {
                return Err(error(
                    422,
                    format!("{place}: choice requires 1 to {MAX_OPTIONS} options"),
                ));
            }
            let criteria = question.get("criteria").and_then(Value::as_object).unwrap();
            for (key, label) in criteria {
                let key_chars = key.chars().count();
                if key_chars == 0 || python_strip(key).is_empty() || key_chars > 256 {
                    return Err(error(
                        422,
                        "option keys must be nonempty and at most 256 characters",
                    ));
                }
                if special.iter().any(|token| key.contains(token.as_str())) {
                    return Err(error(
                        422,
                        format!(
                            "{place}: option key {} contains an unsupported tokenizer \
                             control token",
                            python_repr(key)
                        ),
                    ));
                }
                keys.push(key.clone());
                let description = match label {
                    Value::Null => key.clone(),
                    label => text(
                        label,
                        &format!("{place} criteria {}", python_repr(key)),
                        special,
                    )?,
                };
                if python_strip(&description).is_empty() {
                    return Err(error(
                        422,
                        format!("{place}: criterion {} must be nonempty", python_repr(key)),
                    ));
                }
                descriptions.push(Some(description));
            }
        }
    }
    let combined = instructions.chars().count()
        + descriptions
            .iter()
            .filter_map(|d| d.as_ref().map(|d| d.chars().count()))
            .sum::<usize>();
    if combined > MAX_TEXT {
        return Err(error(
            422,
            "combined question text exceeds 16384 characters",
        ));
    }
    Ok(Question {
        name: name.to_owned(),
        kind,
        instructions,
        keys,
        descriptions,
    })
}

/// `protocol.parse_request` for an already-decoded body. `special` extends the
/// blocked control tokens; the worker injects the loaded tokenizer's full
/// special-token set (see [`crate::Processor`]).
pub fn parse_request(
    body: &Map<String, Value>,
    special: &[String],
) -> Result<Request, RequestError> {
    if body.len() != 3
        || !["model", "state", "questions"]
            .iter()
            .all(|k| body.contains_key(*k))
    {
        return Err(error(
            422,
            "request must contain model, state and questions only",
        ));
    }
    if body["model"].as_str() != Some(MODEL_NAME) {
        return Err(error(422, format!("model must be {MODEL_NAME}")));
    }
    let questions = match &body["questions"] {
        Value::Object(questions) if (1..=MAX_QUESTIONS).contains(&questions.len()) => questions,
        _ => {
            return Err(error(
                422,
                format!("questions must contain 1 to {MAX_QUESTIONS} questions"),
            ));
        }
    };
    let mut parsed = Vec::with_capacity(questions.len());
    for (name, value) in questions {
        parsed.push(question(name, value, special)?);
    }
    // Questions validate before the state, as the reference orders it.
    let state = state(&body["state"], special)?;
    Ok(Request {
        model: MODEL_NAME,
        state,
        questions: parsed,
    })
}

/// The default special-token set as a owned list, for callers that do not
/// inject a tokenizer's set.
pub fn default_special_tokens() -> Vec<String> {
    DEFAULT_SPECIAL_TOKENS
        .iter()
        .map(|token| token.to_string())
        .collect()
}
