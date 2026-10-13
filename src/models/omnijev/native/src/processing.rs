//! Prompt rendering, tokenization and the branch layout, following OmniJev @ 14dbec4
//! (`MSO1._prompt`, `MSO1._encode_many`, `MSO1.ask_branch`, `mso/branch.py`).

use std::path::Path;

use anyhow::{Context, Result, ensure};
use tokenizers::Tokenizer;

use crate::contract::{self, Image, Question, Request};

pub const IMAGE_TOKEN: u32 = 248_056;
pub const OPTION_OPEN: u32 = 248_077;
pub const OPTION_CLOSE: u32 = 248_078;
/// `MSO1`'s default image budget, which overrides the processor config's 401,408.
pub const MAX_PIXELS: usize = 768 * 28 * 28;
pub const MIN_PIXELS: usize = 65_536;
const FACTOR: usize = 32; // patch 16 × spatial merge 2
/// A single question's tokens: prompt, image and every option.
pub const MAX_SEQUENCE: usize = 8192;
/// Tokens the reference processes for a request: its prefix once, then every row.
pub const MAX_PROCESSED: usize = 65_536;

const PAD: &str = "<|image_pad|>";

/// `[1, height / 16, width / 16]` after the processor's resize: Transformers'
/// `Qwen2VLImageProcessor` smart resize (Apache-2.0; see ../../README.md), as Cua-S1's.
pub fn image_grid(width: usize, height: usize) -> [usize; 3] {
    // Python's round uses ties-to-even.
    let mut w = (width as f64 / FACTOR as f64).round_ties_even() as usize * FACTOR;
    let mut h = (height as f64 / FACTOR as f64).round_ties_even() as usize * FACTOR;
    if w * h > MAX_PIXELS {
        let beta = ((width * height) as f64 / MAX_PIXELS as f64).sqrt();
        w = ((width as f64 / beta / FACTOR as f64).floor() as usize * FACTOR).max(FACTOR);
        h = ((height as f64 / beta / FACTOR as f64).floor() as usize * FACTOR).max(FACTOR);
    } else if w * h < MIN_PIXELS {
        let beta = (MIN_PIXELS as f64 / (width * height) as f64).sqrt();
        w = (width as f64 * beta / FACTOR as f64).ceil() as usize * FACTOR;
        h = (height as f64 * beta / FACTOR as f64).ceil() as usize * FACTOR;
    }
    [1, h / 16, w / 16]
}

/// Image tokens after the 2×2 spatial merge.
pub fn image_tokens(grid: [usize; 3]) -> usize {
    grid[1] / 2 * (grid[2] / 2)
}

/// Python's `str.isspace`, which Jinja's `trim` uses.
fn python_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// The chat-template text of one question with its option blocks appended, with one
/// image placeholder that tokenization expands.
pub fn prompt(question: &Question) -> String {
    // The template trims the user content, which starts with the image, so only the
    // instructions' trailing whitespace goes.
    let content = format!(
        "<|vision_start|>{PAD}<|vision_end|>{}",
        question.instructions.trim_end_matches(python_space)
    );
    let mut text =
        format!("<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n");
    for option in &question.options {
        text += "<|opt|>";
        text += option;
        text += "<|/opt|>";
    }
    text
}

/// One (question, option) row: the rest of the question after the shared prefix,
/// then one option block.
#[derive(Debug, PartialEq)]
pub struct Row {
    pub tokens: Vec<u32>,
    /// The token before the option marker, where the question state `zq` is read.
    pub zq: usize,
    /// The closing marker, where the option state `u` is read; the row's last token.
    pub u: usize,
    /// For each option-text token: the position that predicts it and the token.
    pub targets: Vec<(usize, u32)>,
    /// The first option-text token, predicted from `zq`.
    pub first: Option<u32>,
}

pub struct PreparedQuestion {
    /// The single-question sequence: prompt, image and every option block.
    pub token_ids: Vec<u32>,
    /// `[3, sequence]` T/H/W rotary positions of `token_ids`.
    pub positions: [Vec<i64>; 3],
    pub rows: Vec<Row>,
}

/// The reference's split of a request into a shared prefix and rows.
pub struct Layout {
    /// The prefix length: the questions' common token prefix, cut at least one token
    /// before any question's first option marker.
    pub prefix: usize,
    /// The rotary position of the first token after the prefix.
    pub next_position: i64,
    /// The reference's input-token count: the prefix once, then every row.
    pub processed_tokens: usize,
}

/// Option-marker positions in a single-question sequence.
pub fn spans(ids: &[u32]) -> (Vec<usize>, Vec<usize>) {
    let at = |token| (0..ids.len()).filter(|&i| ids[i] == token).collect();
    (at(OPTION_OPEN), at(OPTION_CLOSE))
}

/// Rows and readouts of every question, as `ask_branch` and `branch_forward` build them.
pub fn layout(sequences: &[Vec<u32>], grid: [usize; 3]) -> Result<(Layout, Vec<Vec<Row>>)> {
    ensure!(!sequences.is_empty(), "no questions");
    let shortest = sequences.iter().map(Vec::len).min().unwrap_or(0);
    let mut prefix = 0;
    while prefix < shortest && sequences.iter().all(|s| s[prefix] == sequences[0][prefix]) {
        prefix += 1;
    }
    let mut spans_of = Vec::with_capacity(sequences.len());
    for ids in sequences {
        let (opens, closes) = spans(ids);
        ensure!(
            !opens.is_empty()
                && opens.len() == closes.len()
                && opens.iter().zip(&closes).all(|(o, c)| o < c)
                && closes.iter().zip(&opens[1..]).all(|(c, o)| c < o),
            "every option needs one opening and one closing marker, in order"
        );
        // Every row keeps at least the token before its option marker.
        ensure!(opens[0] >= 2, "options must follow the prompt");
        prefix = prefix.min(opens[0] - 1);
        spans_of.push((opens, closes));
    }
    let prefix = prefix.max(1);
    let mut rows = Vec::with_capacity(sequences.len());
    let mut processed = prefix;
    for (ids, (opens, closes)) in sequences.iter().zip(&spans_of) {
        let head = &ids[prefix..opens[0]];
        let question_rows: Vec<Row> = opens
            .iter()
            .zip(closes)
            .map(|(&o, &c)| {
                let tokens: Vec<u32> = head.iter().chain(&ids[o..=c]).copied().collect();
                let open = head.len();
                let last = tokens.len() - 1;
                let inner = open + 1..last;
                Row {
                    zq: open - 1,
                    u: last,
                    targets: inner.clone().map(|t| (t - 1, tokens[t])).collect(),
                    first: inner.clone().next().map(|t| tokens[t]),
                    tokens,
                }
            })
            .collect();
        processed += question_rows.iter().map(|r| r.tokens.len()).sum::<usize>();
        rows.push(question_rows);
    }
    let positions = image_positions(&sequences[0], grid)?;
    let next_position = positions
        .iter()
        .map(|axis| axis[prefix - 1])
        .max()
        .unwrap_or(0)
        + 1;
    Ok((
        Layout {
            prefix,
            next_position,
            processed_tokens: processed,
        },
        rows,
    ))
}

/// Qwen3.5's T/H/W rotary positions (Transformers' `get_rope_index`, Apache-2.0; see
/// ../../README.md) for a sequence with one contiguous image span:
/// text before it counts up, the image's patches take the position of its first token
/// plus their row (H) and column (W), and text after it continues from the image's
/// largest position plus one.
pub fn image_positions(ids: &[u32], grid: [usize; 3]) -> Result<[Vec<i64>; 3]> {
    let start = ids
        .iter()
        .position(|&id| id == IMAGE_TOKEN)
        .context("missing image placeholders")?;
    let count = ids[start..]
        .iter()
        .take_while(|&&id| id == IMAGE_TOKEN)
        .count();
    let end = start + count;
    ensure!(
        !ids[end..].contains(&IMAGE_TOKEN),
        "image placeholders must be one contiguous span"
    );
    let [t, h, w] = grid;
    ensure!(
        t == 1 && h > 0 && w > 0 && h % 2 == 0 && w % 2 == 0 && h / 2 * (w / 2) == count,
        "image span does not match its grid"
    );
    let mut positions: [Vec<i64>; 3] = std::array::from_fn(|_| (0..start as i64).collect());
    for y in 0..h / 2 {
        for x in 0..w / 2 {
            positions[0].push(start as i64);
            positions[1].push((start + y) as i64);
            positions[2].push((start + x) as i64);
        }
    }
    let next = start + h.max(w) / 2;
    for axis in &mut positions {
        axis.extend((next..next + ids.len() - end).map(|p| p as i64));
    }
    Ok(positions)
}

pub struct Processor {
    tokenizer: Tokenizer,
}

/// A request ready for execution; `questions` keeps identity and order for finishing.
pub struct PreparedRequest {
    pub image: Image,
    pub grid: [usize; 3],
    pub questions: Vec<Question>,
    pub inputs: Vec<PreparedQuestion>,
    pub layout: Layout,
}

impl Processor {
    pub fn load(dir: &Path) -> Result<Self> {
        let tokenizer =
            Tokenizer::from_file(dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
        for (text, id) in [
            ("<|image_pad|>", IMAGE_TOKEN),
            ("<|opt|>", OPTION_OPEN),
            ("<|/opt|>", OPTION_CLOSE),
        ] {
            ensure!(
                tokenizer.token_to_id(text) == Some(id),
                "tokenizer does not map {text} to {id}"
            );
        }
        Ok(Self { tokenizer })
    }

    /// The single-question sequence: the prompt's text tokens around the image's
    /// expanded placeholder.
    pub fn tokenize(&self, question: &Question, grid: [usize; 3]) -> Result<Vec<u32>> {
        let text = prompt(question);
        let (before, after) = text.split_once(PAD).context("prompt has no image")?;
        let encode = |part: &str| -> Result<Vec<u32>> {
            Ok(self
                .tokenizer
                .encode(part, false)
                .map_err(anyhow::Error::msg)?
                .get_ids()
                .to_vec())
        };
        let mut ids = encode(before)?;
        ids.extend(std::iter::repeat_n(IMAGE_TOKEN, image_tokens(grid)));
        ids.extend(encode(after)?);
        Ok(ids)
    }

    /// Validate and lay out the whole request before any device work.
    pub fn prepare(&self, raw: &[u8]) -> Result<PreparedRequest> {
        let Request { image, questions } = contract::compile(raw)?;
        let grid = image_grid(image.width, image.height);
        let sequences = questions
            .iter()
            .map(|q| {
                let ids = self.tokenize(q, grid)?;
                ensure!(
                    ids.len() <= MAX_SEQUENCE,
                    "question {}: {} tokens exceeds {MAX_SEQUENCE}",
                    q.id,
                    ids.len()
                );
                Ok(ids)
            })
            .collect::<Result<Vec<_>>>()?;
        let (layout, rows) = layout(&sequences, grid)?;
        ensure!(
            layout.processed_tokens <= MAX_PROCESSED,
            "request needs {} tokens, above {MAX_PROCESSED}",
            layout.processed_tokens
        );
        let inputs = sequences
            .into_iter()
            .zip(rows)
            .map(|(token_ids, rows)| {
                Ok(PreparedQuestion {
                    positions: image_positions(&token_ids, grid)?,
                    token_ids,
                    rows,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(PreparedRequest {
            image,
            grid,
            questions,
            inputs,
            layout,
        })
    }
}
