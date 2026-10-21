//! Execution orchestration (spec 03, N4): compiled branch → hidden states →
//! CPU FP32 head → logits, one question at a time over the shared backends.
//!
//! The two modality paths converge at the language boundary exactly as the
//! reference's `Qwen3_5Model` forward does: a text state feeds the branch
//! tokens straight to the unmerged 2B language model; an image state decodes
//! and preprocesses the wire image once, runs the manifest-verified vision
//! tower once per request, and splices its adapted output rows into the
//! language forward at the placeholder positions with the meshgrid mrope
//! positions (the `jev_vl` port of HF's `get_rope_index`). Both paths read
//! the final-norm hidden state at the compiler's captured candidate-end and
//! decision-end positions and hand those rows to the FP32 decision head —
//! the reference `Valen.forward`'s structure, eagerly, one request at a time.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use omni_qwen3_5_native::inputs::MultimodalInput;
use omni_qwen3_5_native::vision::VisionModel;
use omni_qwen3_5_native::image_preprocess::{ImageLimits, preprocess_rgb8_with_limits};

use crate::compile::{Compiled, Processor, IMAGE_PAD};
use crate::head::Head;
use crate::preprocess::{self, Prepared, RenderedState};
use crate::protocol::{Kind, Request, State};
use crate::text::TextModel;
use crate::wire::{RequestError, error};
use crate::{finish, protocol};

/// The checkpoint's media_kwargs as preprocessing limits: shortest edge
/// 65 536 pixels, longest edge 1 048 576, patch 16, merge 2.
pub const VALEN_IMAGE_LIMITS: ImageLimits = ImageLimits {
    max_source_side: protocol::MAX_SIDE,
    max_source_pixels: protocol::MAX_PIXELS,
    min_pixels: 65_536,
    max_pixels: protocol::MAX_PIXELS,
    max_patches: 4096,
};

/// The request's outcome: serialized answers in request order plus the dual
/// token accounting (`logical_tokens` once over the base and every suffix,
/// `compute_tokens` over every full branch).
pub struct Outcome {
    pub answers: Vec<(String, String)>,
    pub logical_tokens: usize,
    pub compute_tokens: usize,
}

/// The loaded model-owned engine: processor, backbones and head.
pub struct Engine {
    pub processor: Processor,
    text: TextModel,
    vision: VisionModel,
    head: Head,
}

impl Engine {
    /// Validate the export policy (text, head and vision provenance) and load
    /// every backend before the first request.
    pub fn load(export: &Path, base: &Path, library: &Path) -> Result<Self> {
        let processor =
            Processor::load(base).map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?;
        let head = Head::load(export)?;
        let text = TextModel::load(export, library)?;
        let language_cfg = text.config().clone();
        ensure!(
            language_cfg.hidden == head.hidden_size(),
            "the head's hidden size does not match the backbone"
        );
        ensure!(
            language_cfg.image_token_id == Some(IMAGE_PAD),
            "the checkpoint's image_token_id must be {IMAGE_PAD}"
        );
        let vision = VisionModel::load_valen(export, library)?;
        Ok(Self {
            processor,
            text,
            vision,
            head,
        })
    }

    /// The complete request path of the reference worker's `decide`:
    /// wire validation, materialization, compilation, execution, finishing.
    pub fn decide(&mut self, raw: &[u8]) -> Result<Outcome, RequestError> {
        let request = protocol::parse_request(
            &crate::wire::parse_body(raw)?,
            &self.processor.special_tokens,
        )?;
        self.decide_request(&request)
    }

    /// `decide` over an already-parsed request (the warmup path reuses it).
    pub fn decide_request(&mut self, request: &Request) -> Result<Outcome, RequestError> {
        choice_only_guard(request)?;
        let prepared = preprocess::prepare(request);
        let compiled = self.processor.compiler().compile(&prepared)?;
        let logits = self
            .execute(request, &prepared, &compiled)
            .map_err(|_| error(500, "inference failed"))?;
        let mut answers = Vec::with_capacity(compiled.questions.len());
        for (question, logits) in compiled.questions.iter().zip(&logits) {
            let answer = finish::choice_answer(&question.keys, logits)
                .map_err(|_| error(500, "inference failed"))?;
            answers.push((question.qid.clone(), answer));
        }
        finish::build_response(&answers, compiled.logical_tokens, compiled.compute_tokens)
            .map_err(|_| error(500, "inference failed"))?;
        Ok(Outcome {
            answers,
            logical_tokens: compiled.logical_tokens,
            compute_tokens: compiled.compute_tokens,
        })
    }

    /// One forward per branch; the head readout happens on the captured
    /// positions' hidden rows. Returns one logit row per question, in order.
    fn execute(
        &mut self,
        request: &Request,
        prepared: &Prepared,
        compiled: &Compiled,
    ) -> Result<Vec<Vec<f32>>> {
        let image_rows = match (&prepared.state, &compiled.base.image) {
            (RenderedState::Image { grid_thw, .. }, Some(layout)) => Some((
                self.image_rows(request, *grid_thw, layout.visual_tokens)?,
                layout.grid_thw,
            )),
            (RenderedState::Text { .. }, None) => None,
            _ => anyhow::bail!("prepared state and compiled image layout disagree"),
        };
        let mut rows = Vec::with_capacity(compiled.questions.len());
        for question in &compiled.questions {
            let mut logits = Vec::new();
            for branch in &question.branches {
                let mut positions = branch.candidate_positions.clone();
                positions.push(branch.decision_position);
                let mut hidden = match &image_rows {
                    None => self.text.forward_positions(&branch.tokens, &positions)?,
                    Some((embeddings, grid_thw)) => {
                        let mrope = mrope_positions(&branch.tokens, IMAGE_PAD, grid_thw)?;
                        self.text
                            .language()
                            .forward_multimodal_positions(
                            &MultimodalInput {
                                token_ids: &branch.tokens,
                                image_token_indices: &mrope.1,
                                image_embeddings: embeddings,
                                position_ids: [
                                    mrope.0[0].as_slice(),
                                    mrope.0[1].as_slice(),
                                    mrope.0[2].as_slice(),
                                ],
                            },
                            &positions,
                        )?
                    }
                };
                let decision = hidden
                    .pop()
                    .context("the decision position is missing from the readout")?;
                logits.extend(self.head.logits(&hidden, &decision)?);
            }
            rows.push(logits);
        }
        self.text.synchronize()?;
        Ok(rows)
    }

    /// Decode, resize and run the vision tower once for the whole request;
    /// `branch()` reuses the rows for every branch of every question.
    fn image_rows(
        &mut self,
        request: &Request,
        grid_thw: [usize; 3],
        visual_tokens: usize,
    ) -> Result<Vec<half::bf16>> {
        let State::Image(image) = &request.state else {
            anyhow::bail!("image rows requested for a text state");
        };
        let decoded = image::load_from_memory(&image.data)
            .context("the wire-validated image could not be decoded")?
            .into_rgb8();
        let processed = preprocess_rgb8_with_limits(
            decoded.width() as usize,
            decoded.height() as usize,
            decoded.as_raw(),
            &VALEN_IMAGE_LIMITS,
        )?;
        ensure!(
            processed.image_grid_thw == grid_thw && processed.image_tokens() == visual_tokens,
            "the preprocessing grid does not match the compiled image layout"
        );
        let rows = self.vision.forward(&processed)?;
        ensure!(
            rows.len() == visual_tokens * self.head.hidden_size(),
            "the vision tower output does not match the image layout"
        );
        Ok(rows)
    }
}

/// The single image's placeholder indices and the three mrope axes over the
/// full branch sequence, following the HF `get_rope_index` port the `jev_vl`
/// worker gates: text tokens advance all three axes by one, image rows carry
/// the meshgrid (t outer, h mid, w fastest) offset by the run's start, and
/// the text after the image resumes at start + max(merged h, merged w).
///
/// The grid is the compiled image layout's `[1, patches_h, patches_w]`; the
/// merged meshgrid axes are the patch grid halved by the 2×2 spatial merge.
fn mrope_positions(
    tokens: &[u32],
    image_pad: u32,
    grid_thw: &[usize; 3],
) -> Result<([Vec<i64>; 3], Vec<usize>)> {
    let start = tokens
        .iter()
        .position(|&id| id == image_pad)
        .context("the compiled branch has no image placeholder")?;
    let end = start
        + tokens[start..]
            .iter()
            .position(|&id| id != image_pad)
            .unwrap_or(tokens.len() - start);
    ensure!(
        tokens[end..].iter().all(|&id| id != image_pad),
        "the compiled branch carries more than one image run"
    );
    let [gt, gh, gw] = *grid_thw;
    let (lh, lw) = (gh / 2, gw / 2);
    let count = gt * lh * lw;
    ensure!(
        end - start == count,
        "the placeholder run does not match the image grid"
    );
    let indices: Vec<usize> = (start..end).collect();
    let mut axes: [Vec<i64>; 3] = std::array::from_fn(|_| Vec::with_capacity(tokens.len()));
    let mut current = 0i64;
    for token in &tokens[..start] {
        let _ = token;
        for axis in &mut axes {
            axis.push(current);
        }
        current += 1;
    }
    let base = current;
    for ti in 0..gt as i64 {
        for hi in 0..lh as i64 {
            for wi in 0..lw as i64 {
                axes[0].push(base + ti);
                axes[1].push(base + hi);
                axes[2].push(base + wi);
            }
        }
    }
    current = base + lh.max(lw) as i64;
    for _ in &tokens[end..] {
        for axis in &mut axes {
            axis.push(current);
        }
        current += 1;
    }
    Ok((axes, indices))
}

/// The N4 milestone guard: `choice` is served end-to-end; `noul` and `score`
/// are compiled by the wire and compiler layers but their finishing
/// variations are N5, so the worker fails closed with a deterministic 422
/// instead of serving an ungated type. Recorded deviation from the reference
/// contract (the reference serves all three); see recipe/valen/native.md.
pub fn choice_only_guard(request: &Request) -> Result<(), RequestError> {
    for question in &request.questions {
        if question.kind != Kind::Choice {
            return Err(error(
                422,
                format!(
                    "question {}: the {} question type is not served by this worker \
                     milestone; see recipe/valen/native.md",
                    crate::wire::python_repr(&question.name),
                    question.kind.as_str(),
                ),
            ));
        }
    }
    Ok(())
}
