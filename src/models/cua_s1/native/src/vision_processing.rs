//! Screenshot preparation and response interpretation, independent of execution.
use crate::{
    contract::{self, Question},
    image_preprocess::{ProcessedImage, preprocess_rgb8},
    image_request::MODEL_ID,
    multimodal::{ImagePrompt, prepare_prompt},
};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use tokenizers::Tokenizer;

pub struct VisionProcessor {
    tokenizer: Tokenizer,
    image_token: u32,
    resize_workers: usize,
}

pub struct PreparedRequest {
    pub image: ProcessedImage,
    pub prompts: Vec<ImagePrompt>,
    pub context: ResponseContext,
}

pub struct ResponseContext {
    questions: Vec<Question>,
    input_tokens: usize,
}

impl VisionProcessor {
    pub fn new(tokenizer: Tokenizer, image_token: u32) -> Self {
        Self {
            tokenizer,
            image_token,
            resize_workers: 0,
        }
    }

    /// Explicit CPU resize scheduling for this processor; zero retains the
    /// existing shared serial entry point. All child work joins during prepare.
    pub fn with_resize_workers(mut self, workers: usize) -> Result<Self> {
        ensure!(workers <= 4, "resize workers must be 0..=4");
        self.resize_workers = workers;
        Ok(self)
    }

    /// Validate every question and prompt before submitting model work.
    pub fn prepare(
        &self,
        width: usize,
        height: usize,
        rgb: &[u8],
        questions: &[Question],
    ) -> Result<PreparedRequest> {
        crate::multimodal::validate_questions(questions)?;
        let image = if self.resize_workers == 0 {
            preprocess_rgb8(width, height, rgb)?
        } else {
            crate::image_preprocess::preprocess_rgb8_workers(
                width,
                height,
                rgb,
                self.resize_workers,
            )?
        };
        let prompts = questions
            .iter()
            .map(|q| prepare_prompt(&self.tokenizer, q, image.image_grid_thw, self.image_token))
            .collect::<Result<Vec<_>>>()?;
        let input_tokens = prompts.iter().map(|p| p.token_ids.len()).sum();
        Ok(PreparedRequest {
            image,
            prompts,
            context: ResponseContext {
                questions: questions.to_vec(),
                input_tokens,
            },
        })
    }
}

impl ResponseContext {
    pub fn option_counts(&self) -> impl Iterator<Item = usize> + '_ {
        self.questions.iter().map(|q| q.keys.len())
    }

    pub fn finish(&self, logits: Vec<Vec<f32>>) -> Result<Value> {
        ensure!(
            logits.len() == self.questions.len(),
            "question/output count mismatch"
        );
        let mut answers = serde_json::Map::new();
        for (q, row) in self.questions.iter().zip(logits) {
            ensure!(row.len() == q.keys.len(), "candidate/output count mismatch");
            answers.insert(q.name.clone(), contract::answer(q, &probabilities(&row)?));
        }
        Ok(
            json!({"model": MODEL_ID, "answers": answers, "usage": {"input_tokens": self.input_tokens, "output_tokens": 0}}),
        )
    }
}

pub(crate) fn probabilities(logits: &[f32]) -> Result<Vec<f32>> {
    ensure!(
        !logits.is_empty() && logits.iter().all(|x| x.is_finite()),
        "non-finite or empty candidate logits"
    );
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let exps: Vec<f64> = logits.iter().map(|&x| (x as f64 - max).exp()).collect();
    let total: f64 = exps.iter().sum();
    Ok(exps.iter().map(|x| (x / total) as f32).collect())
}

/// Serving switch is local to the Cua screenshot adapter; no process-wide pool
/// or behavior change in other shared-preprocessor consumers.
pub(crate) fn resize_workers(value: Option<&str>) -> Result<usize> {
    match value {
        None => Ok(0),
        Some(v @ ("0" | "1" | "2" | "3" | "4")) => Ok(usize::from(v.as_bytes()[0] - b'0')),
        _ => anyhow::bail!("CUA_S1_RESIZE_WORKERS must be unset or 0..=4"),
    }
}

#[cfg(test)]
#[path = "../../../../../tests/cua_s1/vision_processing.rs"]
mod tests;
