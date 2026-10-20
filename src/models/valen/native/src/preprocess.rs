//! The reference worker's materialization semantics, tokenizer-free: the
//! question records (`models.valen.preprocess.question_record`), the
//! chat-template-equivalent state rendering, and the image grid from the
//! pinned `Qwen2VLImageProcessor`'s smart resize.

use serde_json::{Value, json};

use crate::protocol::{
    ImageData, Kind, NOUL_FALSE_DESCRIPTION, NOUL_TRUE_DESCRIPTION, Question, State,
};

/// The pinned `Qwen2VLImageProcessor` smart resize (Apache-2.0; as the sibling
/// native crates compute it): both dimensions rounded to a multiple of the
/// factor with ties to even, then clamped into `[min_pixels, max_pixels]` by
/// scaling. The result is the grid `[1, height / patch, width / patch]`.
pub fn smart_resize(
    height: usize,
    width: usize,
    patch: usize,
    merge: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> [usize; 3] {
    let factor = patch * merge;
    // Python's round is ties-to-even.
    let mut h = (height as f64 / factor as f64).round_ties_even() as usize * factor;
    let mut w = (width as f64 / factor as f64).round_ties_even() as usize * factor;
    if h * w > max_pixels {
        let beta = ((width * height) as f64 / max_pixels as f64).sqrt();
        h = ((height as f64 / beta / factor as f64).floor() as usize * factor).max(factor);
        w = ((width as f64 / beta / factor as f64).floor() as usize * factor).max(factor);
    } else if h * w < min_pixels {
        let beta = (min_pixels as f64 / (width * height) as f64).sqrt();
        h = (height as f64 * beta / factor as f64).ceil() as usize * factor;
        w = (width as f64 * beta / factor as f64).ceil() as usize * factor;
    }
    [1, h / patch, w / patch]
}

/// Image tokens after the 2x2 spatial merge, per grid.
pub fn visual_tokens(grid: [usize; 3], merge: usize) -> usize {
    grid[1] / merge * (grid[2] / merge)
}

/// One question as the pinned compiler's record: the type that drives the
/// prompt (`Task: <type>`), the branch grouping and the candidate list.
#[derive(Debug, Clone)]
pub struct QuestionRecord {
    pub qid: String,
    /// The wire type, kept for response finishing.
    pub wire_kind: Kind,
    /// The record type: `noul` with custom wording compiles as `choice`.
    pub record_kind: Kind,
    /// The record's `instructions` field.
    pub instructions: String,
    /// `candidates()` output: candidate key and description, in order.
    pub pairs: Vec<(String, String)>,
}

impl QuestionRecord {
    /// The record as the Python layer builds it for the pinned compiler,
    /// for assertions against recorded fixtures.
    pub fn to_json(&self) -> Value {
        match self.record_kind {
            Kind::Noul => json!({"type": "noul", "instructions": self.instructions}),
            Kind::Score => json!({
                "type": "score",
                "instructions": self.instructions,
                "criteria": self.pairs.iter().map(|(_, d)| d.as_str()).collect::<Vec<_>>(),
            }),
            Kind::Choice => json!({
                "type": "choice",
                "instructions": self.instructions,
                "criteria": self.pairs.iter()
                    .map(|(k, d)| (k.clone(), Value::String(d.clone())))
                    .collect::<serde_json::Map<String, Value>>(),
            }),
        }
    }
}

/// `models.valen.preprocess.question_record`: `noul` without custom wording
/// compiles as the reference's native noul record; custom true/false wording
/// flows through a `choice` record keyed `true`/`false` with the unspecified
/// sides filled from the fixed wording; `score` passes its levels through and
/// the schema assigns the keys `"0"`….
pub fn question_record(question: &Question) -> QuestionRecord {
    match question.kind {
        Kind::Noul => {
            let filled = [
                question.descriptions[0]
                    .clone()
                    .unwrap_or_else(|| NOUL_TRUE_DESCRIPTION.to_owned()),
                question.descriptions[1]
                    .clone()
                    .unwrap_or_else(|| NOUL_FALSE_DESCRIPTION.to_owned()),
            ];
            let native = filled[0] == NOUL_TRUE_DESCRIPTION && filled[1] == NOUL_FALSE_DESCRIPTION;
            let (record_kind, pairs) = if native {
                (
                    Kind::Noul,
                    vec![
                        ("true".to_owned(), NOUL_TRUE_DESCRIPTION.to_owned()),
                        ("false".to_owned(), NOUL_FALSE_DESCRIPTION.to_owned()),
                    ],
                )
            } else {
                (
                    Kind::Choice,
                    vec![
                        ("true".to_owned(), filled[0].clone()),
                        ("false".to_owned(), filled[1].clone()),
                    ],
                )
            };
            QuestionRecord {
                qid: question.name.clone(),
                wire_kind: question.kind,
                record_kind,
                pairs,
                instructions: question.instructions.clone(),
            }
        }
        Kind::Score => QuestionRecord {
            qid: question.name.clone(),
            wire_kind: question.kind,
            record_kind: Kind::Score,
            pairs: question
                .descriptions
                .iter()
                .enumerate()
                .map(|(index, description)| {
                    (index.to_string(), description.clone().unwrap_or_default())
                })
                .collect(),
            instructions: question.instructions.clone(),
        },
        Kind::Choice => QuestionRecord {
            qid: question.name.clone(),
            wire_kind: question.kind,
            record_kind: Kind::Choice,
            pairs: question
                .keys
                .iter()
                .cloned()
                .zip(
                    question
                        .descriptions
                        .iter()
                        .map(|d| d.clone().unwrap_or_default()),
                )
                .collect(),
            instructions: question.instructions.clone(),
        },
    }
}

/// The rendered state. `prompt` is the chat template's text before the image
/// placeholder is expanded; for text states it is the final prompt.
#[derive(Debug)]
pub enum RenderedState {
    Text {
        /// The user content after the template's `trim`.
        user_content: String,
        prompt: String,
    },
    Image {
        format: &'static str,
        width: usize,
        height: usize,
        /// `[1, height patches, width patches]` after smart resize.
        grid_thw: [usize; 3],
        visual_tokens: usize,
        /// `<|vision_start|><|image_pad|><|vision_end|>`.
        user_content: String,
        /// The prompt with a single unexpanded image placeholder.
        prompt: String,
    },
}

/// The prepared request: the rendered state plus the per-question records,
/// everything the compiler needs without touching the tokenizer.
#[derive(Debug)]
pub struct Prepared {
    pub state: RenderedState,
    pub questions: Vec<QuestionRecord>,
}

/// `prepare_request` without the filesystem: the reference materializes one
/// request-scoped file because the pinned compiler wants a local media path;
/// the native path needs only the geometry that materialization produces.
pub fn prepare(request: &crate::protocol::Request) -> Prepared {
    let state = match &request.state {
        State::Text(text) => {
            // The pinned chat template wraps one user message and trims its
            // content; `add_generation_prompt` is false and no vision id is
            // added, so the whole rendering reduces to this fixed shape.
            let trimmed = crate::wire::python_strip(text).to_owned();
            RenderedState::Text {
                user_content: trimmed.clone(),
                prompt: format!("<|im_start|>user\n{trimmed}<|im_end|>\n"),
            }
        }
        State::Image(ImageData {
            format,
            width,
            height,
            ..
        }) => {
            // The checkpoint's media_kwargs: shortest_edge 65536,
            // longest_edge 1048576, patch 16, merge 2.
            let grid = smart_resize(*height, *width, 16, 2, 65_536, 1_048_576);
            let tokens = visual_tokens(grid, 2);
            RenderedState::Image {
                format: match format {
                    image::ImageFormat::Png => "PNG",
                    image::ImageFormat::Jpeg => "JPEG",
                    other => unreachable!("validated to PNG/JPEG: {other:?}"),
                },
                width: *width,
                height: *height,
                grid_thw: grid,
                visual_tokens: tokens,
                user_content: "<|vision_start|><|image_pad|><|vision_end|>".to_owned(),
                prompt: "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n"
                    .to_owned(),
            }
        }
    };
    Prepared {
        state,
        questions: request.questions.iter().map(question_record).collect(),
    }
}
