//! Request/token preparation and calibrated response interpretation.

use std::path::Path;
use std::time::Instant;

use anyhow::{Result, ensure};
use serde_json::{Map, Value, json};
use tokenizers::Tokenizer;

use crate::contract::{self, Checkpoint, Kind, Question};
use crate::executor::PrefixUse;
use omni_qwen3_5_native::prefix::PrefixMode;

pub struct Processor {
    checkpoint: &'static Checkpoint,
    parallel_tokenization: bool,
    tokenizer: Tokenizer,
    prefix: String,
    suffix: String,
    temperature: f64,
    max_length: usize,
}

pub struct PreparedRequest {
    pub inputs: Vec<Vec<Vec<u32>>>,
    pub context: ResponseContext,
}

/// The original question/candidate mapping, calibration, usage, and timing.
pub struct ResponseContext {
    checkpoint: &'static Checkpoint,
    questions: Vec<Question>,
    input_tokens: usize,
    candidates: usize,
    temperature: f64,
    max_length: usize,
    start: Instant,
}

impl Processor {
    pub(crate) fn load(
        dir: &Path,
        checkpoint: &'static Checkpoint,
        prefix: String,
        suffix: String,
        temperature: f64,
        max_length: usize,
    ) -> Result<Self> {
        let tokenizer =
            Tokenizer::from_file(dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
        Ok(Self {
            checkpoint,
            parallel_tokenization: checkpoint.parallel_tokenization,
            tokenizer,
            prefix,
            suffix,
            temperature,
            max_length,
        })
    }

    pub(crate) fn disable_parallel_tokenization(&mut self) {
        self.parallel_tokenization = false;
    }

    /// Validate every candidate length before inference; never truncate.
    pub fn prepare(&self, raw: &[u8]) -> Result<PreparedRequest> {
        let questions = contract::compile(raw, self.checkpoint)?;
        let start = Instant::now();
        let inputs = encode_questions(
            &self.tokenizer,
            &self.prefix,
            &self.suffix,
            self.max_length,
            &questions,
            self.parallel_tokenization,
        )?;
        let input_tokens = inputs.iter().flatten().map(Vec::len).sum();
        let candidates = inputs.iter().map(Vec::len).sum();
        Ok(PreparedRequest {
            inputs,
            context: ResponseContext {
                checkpoint: self.checkpoint,
                questions,
                input_tokens,
                candidates,
                temperature: self.temperature,
                max_length: self.max_length,
                start,
            },
        })
    }
}

impl ResponseContext {
    /// Restore the Noul baseline and normalize across each complete question.
    pub fn finish(self, rows: Vec<Vec<f32>>) -> Result<Value> {
        self.finish_with_prefix(rows, None)
    }

    /// `finish`, reporting in `metadata.prefix_cache` the configured policy, whether the
    /// candidates ran on shared prefixes, within one question or across questions, and
    /// the prefix tokens that were not recomputed.
    pub fn finish_with_prefix(
        self,
        rows: Vec<Vec<f32>>,
        prefix: Option<&PrefixUse>,
    ) -> Result<Value> {
        let used = prefix.copied().unwrap_or_default();
        let policy = match used.mode {
            PrefixMode::Off => "default",
            mode => mode.name(),
        };
        let scope = match (used.shared, used.across_questions) {
            (false, _) => "none",
            (true, false) => "within_question",
            (true, true) => "across_questions",
        };
        let prefix_cache = json!({"policy": policy, "enabled": used.shared, "scope": scope,
            "saved_tokens": used.saved_tokens});
        ensure!(rows.len() == self.questions.len(), "invalid logit rows");
        let mut answers = Map::new();
        for (q, mut logits) in self.questions.iter().zip(rows) {
            ensure!(logits.len() == q.prompts.len(), "invalid logits");
            if q.kind == Kind::Noul {
                logits.insert(0, 0.0); // logits [false=0, true=trained score]
            }
            answers.insert(
                q.id.clone(),
                contract::answer(q, &logits, self.temperature)?,
            );
        }
        Ok(
            json!({"model": self.checkpoint.model_id, "answers": answers,
            "usage": {"input_tokens": self.input_tokens, "output_tokens": 0},
            "metadata": {"method": "native_merged_lora_decision_head", "temperature": self.temperature,
                "candidate_sequences": self.candidates, "inference_seconds": self.start.elapsed().as_secs_f64(),
                "base_revision": self.checkpoint.base_revision, "max_length": self.max_length,
                "prefix_cache": prefix_cache}}),
        )
    }
}

fn encode_questions(
    tokenizer: &Tokenizer,
    prefix: &str,
    suffix: &str,
    max_length: usize,
    questions: &[Question],
    parallel: bool,
) -> Result<Vec<Vec<Vec<u32>>>> {
    questions
        .iter()
        .map(|q| {
            // BatchLongest padding would change each candidate's IDs and length.
            // Keep the existing per-candidate semantics for all padded tokenizers.
            let chunk_size = if parallel && tokenizer.get_padding().is_none() {
                16
            } else {
                1
            };
            let mut inputs = Vec::with_capacity(q.prompts.len());
            for chunk in q.prompts.chunks(chunk_size) {
                let chats: Vec<_> = chunk
                    .iter()
                    .map(|prompt| format!("{prefix}{}{suffix}", prompt.trim()))
                    .collect();
                let validate = |enc: tokenizers::Encoding| -> Result<Vec<u32>> {
                    ensure!(
                        enc.len() <= max_length,
                        "question {}: {} tokens exceeds max_length={max_length}; no silent truncation",
                        q.id,
                        enc.len()
                    );
                    Ok(enc.get_ids().to_vec())
                };
                if chats.len() > 1
                    && let Ok(encodings) = tokenizer.encode_batch_fast(chats.iter().map(String::as_str).collect(), true)
                {
                    // Only token IDs are consumed; skip unused character offsets.
                    // Batch encoding preserves input order; validate in that order too.
                    for enc in encodings {
                        inputs.push(validate(enc)?);
                    }
                } else {
                    // A later tokenizer failure must not hide an earlier length error.
                    // Retrying serially also preserves the original first-error message.
                    for chat in chats {
                        inputs.push(validate(
                            tokenizer.encode(chat, true).map_err(anyhow::Error::msg)?,
                        )?);
                    }
                }
            }
            Ok(inputs)
        })
        .collect()
}

#[cfg(test)]
#[path = "../../../../../tests/open_jev/tokenization.rs"]
mod tokenization_tests;

#[cfg(test)]
#[path = "../../../../../tests/open_jev/processing.rs"]
mod tests;
