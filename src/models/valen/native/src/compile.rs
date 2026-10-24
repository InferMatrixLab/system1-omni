//! The pinned compiler, in Rust: branch suffix construction, candidate-end
//! and decision-end position capture, `max_length` enforcement and token
//! accounting, equivalent to `valen.data.compiler.Compiler.compile` at
//! revision 750bfcf behind the reference worker's error mapping.
//!
//! The token source is the [`Encoder`] trait: the real `tokenizers` backend
//! behind [`Processor`], or a deterministic stub in the CPU-only tests, so
//! the layout logic itself runs in CI without the weights.

use std::path::Path;

use tokenizers::Tokenizer as HFTokenizer;

use crate::preprocess::{Prepared, RenderedState, visual_tokens};
use crate::protocol::Kind;
use crate::wire::{RequestError, error};

/// The pinned engine constants (`models.valen.engine`): the Preview
/// checkpoint's `max_length` and the base artifact's identity.
pub const MAX_LENGTH: usize = 8192;
pub const BASE_REVISION: &str = "15852e8c16360a2fea060d615a32b45270f8a8fc";

/// The structural tokens the prompt and suffixes are built from, asserted at
/// processor load exactly as the sibling native crates pin theirs.
pub const IM_START: u32 = 248_045;
pub const IM_END: u32 = 248_046;
pub const VISION_START: u32 = 248_053;
pub const VISION_END: u32 = 248_054;
pub const IMAGE_PAD: u32 = 248_056;
const IM_START_USER: &str = "<|im_start|>user\n";

/// Segment tokenization, as the pinned compiler's `append` uses it.
pub trait Encoder {
    /// `tokenizer.encode(text, add_special_tokens=False)`.
    fn segment(&self, text: &str) -> Result<Vec<u32>, String>;
}

impl Encoder for HFTokenizer {
    fn segment(&self, text: &str) -> Result<Vec<u32>, String> {
        self.encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|e| e.to_string())
    }
}

impl Encoder for &HFTokenizer {
    fn segment(&self, text: &str) -> Result<Vec<u32>, String> {
        (**self).segment(text)
    }
}

/// The state sequence every branch is built on. `mm_token_type_ids` marks the
/// image-pad run (1) against text (0), as the processor's
/// `create_mm_token_type_ids` does; the suffix extends it with zeros.
#[derive(Debug)]
pub struct BaseSequence {
    pub input_ids: Vec<u32>,
    pub mm_token_type_ids: Vec<u8>,
    /// The image layout the vision path consumes; `None` for text.
    pub image: Option<ImageLayout>,
}

/// The image's patch grid and expanded placeholder count.
#[derive(Debug)]
pub struct ImageLayout {
    pub grid_thw: [usize; 3],
    pub visual_tokens: usize,
}

/// One branch: the full token sequence (`base ++ suffix`) with the captured
/// positions. `candidate_positions[i]` is the last token of candidate `i`'s
/// description; `decision_position` is the last token of `Decision:` (the
/// sequence's last token). The reference also builds an all-ones attention
/// mask of the same length — trivially derivable, not stored.
#[derive(Debug)]
pub struct Branch {
    pub tokens: Vec<u32>,
    pub mm_token_type_ids: Vec<u8>,
    pub candidate_positions: Vec<usize>,
    pub decision_position: usize,
    pub length: usize,
}

/// One compiled question, with the wire identity the finisher needs.
#[derive(Debug)]
pub struct CompiledQuestion {
    pub qid: String,
    pub wire_kind: Kind,
    pub record_kind: Kind,
    pub keys: Vec<String>,
    pub descriptions: Vec<String>,
    pub branches: Vec<Branch>,
}

/// The compiled request, the seam the native executor consumes.
#[derive(Debug)]
pub struct Compiled {
    pub base: BaseSequence,
    pub questions: Vec<CompiledQuestion>,
    /// The logical input tokens: the base once, then every suffix.
    pub logical_tokens: usize,
    /// The compute tokens: every branch's full length.
    pub compute_tokens: usize,
}

/// The compiler over any [`Encoder`].
pub struct Compiler<E> {
    encoder: E,
    max_length: usize,
}

impl<E: Encoder> Compiler<E> {
    pub fn new(encoder: E, max_length: usize) -> Self {
        Self {
            encoder,
            max_length,
        }
    }

    /// Compile one prepared request.
    ///
    /// The only client-visible rejection is the token budget, reported with
    /// the reference worker's exact mapping: the pinned compiler raises
    /// `ValueError` and `frontend.valen.decide` turns it into a 422 with
    /// `request rejected by the pinned compiler: ...`. The compiler's other
    /// checks (reserved control tokens in question text and state) are
    /// subsumed by the wire layer, which blocks a superset of the same tokens
    /// before compilation is reached.
    pub fn compile(&self, prepared: &Prepared) -> Result<Compiled, RequestError> {
        let (base_ids, image) = match &prepared.state {
            RenderedState::Text { prompt, .. } => {
                let ids = self.encode_segment(prompt)?;
                (ids, None)
            }
            RenderedState::Image {
                prompt, grid_thw, ..
            } => {
                // The processor expands the single placeholder to the image's
                // token count, then tokenizes the expanded prompt in one go.
                let tokens = visual_tokens(*grid_thw, 2);
                let mut pad = "<|image_pad|>".to_owned();
                for _ in 1..tokens {
                    pad.push_str("<|image_pad|>");
                }
                let expanded = prompt.replacen("<|image_pad|>", &pad, 1);
                let ids = self.encode_segment(&expanded)?;
                (
                    ids,
                    Some(ImageLayout {
                        grid_thw: *grid_thw,
                        visual_tokens: tokens,
                    }),
                )
            }
        };
        let base_mm: Vec<u8> = base_ids
            .iter()
            .map(|&id| if id == IMAGE_PAD { 1 } else { 0 })
            .collect();
        let base_len = base_ids.len();
        let mut logical = base_len;
        let mut compute = 0usize;
        let mut questions = Vec::with_capacity(prepared.questions.len());
        for record in &prepared.questions {
            let mut branches = Vec::new();
            // Score compiles one branch per level; the other types one branch
            // with every candidate. Score candidates never see their level
            // id, nor any other level's description.
            let groups: Vec<Vec<usize>> = if record.record_kind == Kind::Score {
                (0..record.pairs.len()).map(|i| vec![i]).collect()
            } else {
                vec![(0..record.pairs.len()).collect()]
            };
            for group in groups {
                let mut suffix = Vec::new();
                let push = |encoder: &E, suffix: &mut Vec<u32>, text: &str| {
                    let ids = encoder.segment(text).map_err(|_| internal())?;
                    suffix.extend(ids);
                    Ok::<(), RequestError>(())
                };
                push(&self.encoder, &mut suffix, IM_START_USER)?;
                let header = format!(
                    "Task: {}\nQuestion: {}\nCandidates:\n",
                    record.record_kind.as_str(),
                    record.instructions
                );
                push(&self.encoder, &mut suffix, &header)?;
                let mut positions = Vec::with_capacity(group.len());
                for &index in &group {
                    let (key, description) = &record.pairs[index];
                    let candidate = if record.record_kind == Kind::Score {
                        description.clone()
                    } else {
                        format!("{key}: {description}")
                    };
                    push(&self.encoder, &mut suffix, &candidate)?;
                    positions.push(base_len + suffix.len() - 1);
                    push(&self.encoder, &mut suffix, "\n")?;
                }
                push(&self.encoder, &mut suffix, "Decision:")?;
                let length = base_len + suffix.len();
                if length > self.max_length {
                    return Err(error(
                        422,
                        format!(
                            "request rejected by the pinned compiler: {}: {length} tokens \
                             exceed max_length={}; no truncation",
                            record.qid, self.max_length
                        ),
                    ));
                }
                let mut mm = base_mm.clone();
                mm.resize(length, 0);
                let mut tokens = base_ids.clone();
                tokens.extend_from_slice(&suffix);
                compute += length;
                logical += suffix.len();
                branches.push(Branch {
                    tokens,
                    mm_token_type_ids: mm,
                    candidate_positions: positions,
                    decision_position: length - 1,
                    length,
                });
            }
            questions.push(CompiledQuestion {
                qid: record.qid.clone(),
                wire_kind: record.wire_kind,
                record_kind: record.record_kind,
                keys: record.pairs.iter().map(|(k, _)| k.clone()).collect(),
                descriptions: record.pairs.iter().map(|(_, d)| d.clone()).collect(),
                branches,
            });
        }
        Ok(Compiled {
            base: BaseSequence {
                input_ids: base_ids,
                mm_token_type_ids: base_mm,
                image,
            },
            questions,
            logical_tokens: logical,
            compute_tokens: compute,
        })
    }

    fn encode_segment(&self, text: &str) -> Result<Vec<u32>, RequestError> {
        self.encoder.segment(text).map_err(|_| internal())
    }
}

/// The worker's catch-all for a failing backend: 500 with `inference failed`.
fn internal() -> RequestError {
    error(500, "inference failed")
}

/// The loaded processor: the pinned tokenizer plus the derived special-token
/// set, exactly what the reference worker injects into wire validation.
pub struct Processor {
    pub tokenizer: HFTokenizer,
    /// `DEFAULT_SPECIAL_TOKENS` plus the tokenizer's full special-token set,
    /// in the reference's first-seen order.
    pub special_tokens: Vec<String>,
}

impl Processor {
    /// Load from the pinned Qwen3.5-2B base directory (`tokenizer.json` and
    /// `tokenizer_config.json`).
    pub fn load(base: &Path) -> Result<Self, String> {
        let tokenizer = HFTokenizer::from_file(base.join("tokenizer.json"))
            .map_err(|e| format!("tokenizer load failed: {e}"))?;
        for (text, id) in [
            ("<|im_start|>", IM_START),
            ("<|im_end|>", IM_END),
            ("<|vision_start|>", VISION_START),
            ("<|vision_end|>", VISION_END),
            ("<|image_pad|>", IMAGE_PAD),
        ] {
            if tokenizer.token_to_id(text) != Some(id) {
                return Err(format!("tokenizer does not map {text} to {id}"));
            }
        }
        let config: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(base.join("tokenizer_config.json"))
                .map_err(|e| format!("tokenizer_config.json unreadable: {e}"))?,
        )
        .map_err(|e| format!("tokenizer_config.json invalid: {e}"))?;
        let mut special: Vec<String> = crate::protocol::DEFAULT_SPECIAL_TOKENS
            .iter()
            .map(|t| t.to_string())
            .collect();
        // Python's all_special_tokens collects the special_tokens_map fields
        // (eos/pad/…, additional_special_tokens, the model's own *token keys
        // and extra_special_tokens); every one of them is a top-level
        // `*_token`/`*_tokens` entry of tokenizer_config.json here.
        for (key, value) in config.as_object().into_iter().flatten() {
            if !key.ends_with("_token") && !key.ends_with("_tokens") {
                continue;
            }
            if key == "add_bos_token" || key == "added_tokens_decoder" {
                continue;
            }
            collect_tokens(value, &mut special);
        }
        Ok(Self {
            tokenizer,
            special_tokens: special,
        })
    }

    /// The worker's compile step over this processor's tokenizer.
    pub fn compiler(&self) -> Compiler<&HFTokenizer> {
        Compiler::new(&self.tokenizer, MAX_LENGTH)
    }
}

fn collect_tokens(value: &serde_json::Value, out: &mut Vec<String>) {
    let push = |text: &str, out: &mut Vec<String>| {
        if !out.iter().any(|t| t == text) {
            out.push(text.to_owned());
        }
    };
    match value {
        // A token descriptor (`{"content": ...}`), a plain name, a list of
        // either, or `extra_special_tokens`'s name-to-token map.
        serde_json::Value::String(text) => push(text, out),
        serde_json::Value::Object(map) => {
            if let Some(text) = map.get("content").and_then(|v| v.as_str()) {
                push(text, out);
            } else {
                for item in map.values() {
                    collect_tokens(item, out);
                }
            }
        }
        serde_json::Value::Array(list) => {
            for item in list {
                collect_tokens(item, out);
            }
        }
        _ => {}
    }
}
