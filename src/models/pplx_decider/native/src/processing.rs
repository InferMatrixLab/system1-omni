//! Token preparation: each question's system and user messages in the checkpoint's chat template with thinking
//! off, as `DecisionModel.prepare` renders them, plus the readout rows and answer-code token ids in option order.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::contract::{self, CODES, MAX_OPTIONS, MAX_PROMPT_TOKENS, Question, SYSTEM, TEMPERATURE};

/// `chat_template.jinja` for a system and a user message, `add_generation_prompt=True`, `enable_thinking=False`;
/// the template trims each message's content.
pub fn chat_text(user_text: &str) -> String {
    format!(
        "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        user_text.trim()
    )
}

pub struct Processor {
    tokenizer: Tokenizer,
    /// `decision_config.json["token_ids"]`: the vocabulary row of each answer code, for an `lm_head` export.
    token_ids: Vec<u32>,
    max_tokens: usize,
}

/// One question ready for a single noncausal prefill: its prompt ids, and the readout rows read at the last
/// position (`0..rows`) with the matching vocabulary ids.
#[derive(Debug)]
pub struct Prepared {
    pub question: Question,
    pub input_ids: Vec<u32>,
    pub rows: usize,
    pub candidate_token_ids: Vec<u32>,
}

impl Processor {
    /// Loads `tokenizer.json` and checks `decision_config.json` against this contract, as `DecisionModel.__init__`
    /// does: format 1, the same codes, one distinct single token per code, the temperature, noncausal full
    /// attention and last-token pooling.
    pub fn load(dir: &Path) -> Result<Self> {
        let tokenizer =
            Tokenizer::from_file(dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
        let raw =
            std::fs::read(dir.join("decision_config.json")).context("decision_config.json")?;
        let config: Value = serde_json::from_slice(&raw)?;
        ensure!(
            config["format_version"] == 1,
            "unsupported decision checkpoint format"
        );
        ensure!(
            config["attention_mode"] == "noncausal_full_attention" && config["pooling"] == "last",
            "this contract expects noncausal full attention and last-token pooling"
        );
        ensure!(
            config["temperature"].as_f64() == Some(TEMPERATURE),
            "decision_config.json temperature differs from the contract's"
        );
        let codes: Vec<&str> = config["codes"]
            .as_array()
            .context("codes")?
            .iter()
            .filter_map(Value::as_str)
            .collect();
        ensure!(
            codes == CODES,
            "decision_config.json codes differ from the contract's"
        );
        let token_ids: Vec<u32> = config["token_ids"]
            .as_array()
            .context("token_ids")?
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .context("token id")
            })
            .collect::<Result<_>>()?;
        ensure!(
            token_ids.len() == MAX_OPTIONS,
            "expected {MAX_OPTIONS} token ids"
        );
        for (code, id) in CODES.iter().zip(&token_ids) {
            let ids = tokenizer
                .encode(*code, false)
                .map_err(anyhow::Error::msg)?
                .get_ids()
                .to_vec();
            ensure!(
                ids == [*id],
                "answer code {code} is not the single token {id}"
            );
        }
        let mut unique = token_ids.clone();
        unique.sort_unstable();
        unique.dedup();
        ensure!(
            unique.len() == MAX_OPTIONS,
            "answer-code token ids are not distinct"
        );
        Ok(Self {
            tokenizer,
            token_ids,
            max_tokens: MAX_PROMPT_TOKENS,
        })
    }

    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Every question's ids, after validating every prompt length; never truncates.
    pub fn prepare(&self, raw: &[u8]) -> Result<Vec<Prepared>> {
        contract::compile(raw)?
            .into_iter()
            .map(|question| {
                let input_ids = self
                    .tokenizer
                    .encode(chat_text(&question.user_text), false)
                    .map_err(anyhow::Error::msg)?
                    .get_ids()
                    .to_vec();
                ensure!(
                    input_ids.len() <= self.max_tokens,
                    "questions.{} prompt is {} tokens; the limit is {}, and no input is truncated",
                    question.id,
                    input_ids.len(),
                    self.max_tokens
                );
                let rows = question.keys.len();
                Ok(Prepared {
                    candidate_token_ids: self.token_ids[..rows].to_vec(),
                    question,
                    input_ids,
                    rows,
                })
            })
            .collect()
    }
}
