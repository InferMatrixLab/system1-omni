//! Independent preparation: Kev fragments, complete causal question rows and readouts.
use crate::contract::{self, Question, SPECIAL};
use crate::postprocess;
use anyhow::{Result, ensure};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::path::Path;
use std::sync::{Arc, LazyLock};
use tokenizers::Tokenizer;

static CONTROL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<\|([A-Za-z0-9_]+)\|>").unwrap());

pub fn sanitize(text: &str) -> String {
    CONTROL.replace_all(text, "<¦${1}¦>").into_owned()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Row {
    pub ids: Vec<u32>,
    pub positions: Vec<i64>,
    /// Candidate closing markers in order, followed by the decision marker.
    pub readouts: Vec<usize>,
}

pub struct Prepared {
    pub rows: Vec<Row>,
    pub context: ResponseContext,
}

pub struct ResponseContext {
    pub model: String,
    pub questions: Vec<Question>,
    pub input_tokens: usize,
    tokenizer: Arc<Tokenizer>,
}

#[derive(Clone)]
pub struct Processor {
    tokenizer: Arc<Tokenizer>,
}

impl Processor {
    pub fn load(dir: &Path) -> Result<Self> {
        Self::new(Tokenizer::from_file(dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?)
    }

    pub fn new(mut tokenizer: Tokenizer) -> Result<Self> {
        for (text, id) in SPECIAL {
            ensure!(
                tokenizer.token_to_id(text) == Some(id),
                "base tokenizer delimiter mismatch: {text}"
            );
        }
        // Fragment encoding must never pad or truncate silently.
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(anyhow::Error::msg)?;
        Ok(Self {
            tokenizer: Arc::new(tokenizer),
        })
    }

    fn user_tokens(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(sanitize(text), false)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec())
    }

    pub fn prepare(&self, request: &Map<String, Value>) -> Result<Prepared> {
        let request = contract::compile(request)?;
        let mut state = vec![SPECIAL[0].1];
        state.extend(self.user_tokens(&request.state)?);
        ensure!(
            state.len() <= contract::MAX_STATE,
            "state exceeds {} tokens; no truncation",
            contract::MAX_STATE
        );
        let mut rows = Vec::with_capacity(request.questions.len());
        let mut input_tokens = state.len();
        let mut compute_tokens = 0;
        for q in &request.questions {
            let mut ids = state.clone();
            ids.push(SPECIAL[1].1);
            ids.extend(self.user_tokens(&q.instructions)?);
            let mut readouts = Vec::with_capacity(q.options.len() + 1);
            for option in &q.options {
                ids.push(SPECIAL[2].1);
                ids.extend(self.user_tokens(option)?);
                ids.push(SPECIAL[3].1);
                readouts.push(ids.len() - 1);
                ensure!(
                    ids.len() < contract::MAX_ROW,
                    "question {} exceeds {} row tokens",
                    q.id,
                    contract::MAX_ROW
                );
            }
            ids.push(SPECIAL[4].1);
            readouts.push(ids.len() - 1);
            ensure!(
                ids.len() <= contract::MAX_ROW,
                "question {} exceeds {} row tokens",
                q.id,
                contract::MAX_ROW
            );
            compute_tokens += ids.len();
            ensure!(
                compute_tokens <= contract::MAX_REQUEST_TOKENS,
                "request exceeds {} computed row tokens",
                contract::MAX_REQUEST_TOKENS
            );
            input_tokens += ids.len() - state.len();
            let positions = (0..ids.len() as i64).collect();
            rows.push(Row {
                ids,
                positions,
                readouts,
            });
        }
        Ok(Prepared {
            rows,
            context: ResponseContext {
                model: request.model,
                questions: request.questions,
                input_tokens,
                tokenizer: self.tokenizer.clone(),
            },
        })
    }
}

impl ResponseContext {
    pub fn finish(self, logits: Vec<Vec<f32>>) -> Result<Value> {
        ensure!(
            logits.len() == self.questions.len(),
            "question output count mismatch"
        );
        let probs = logits
            .iter()
            .map(|v| {
                postprocess::probabilities(v, contract::TEMPERATURE)
                    .map(|p| p.into_iter().map(f64::from).collect::<Vec<_>>())
            })
            .collect::<Result<Vec<_>>>()?;
        self.finish_probabilities(&probs)
    }

    pub fn finish_probabilities(&self, probabilities: &[Vec<f64>]) -> Result<Value> {
        let answers = postprocess::answers(&self.questions, probabilities)?;
        let serialized = postprocess::billing_json(&answers);
        let output_tokens = self
            .tokenizer
            .encode(serialized, false)
            .map_err(anyhow::Error::msg)?
            .len();
        Ok(serde_json::json!({"model":self.model,"answers":answers,
            "usage":{"input_tokens":self.input_tokens,"output_tokens":output_tokens}}))
    }
}
