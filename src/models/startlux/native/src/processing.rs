//! Tokenizer probe and request preparation. No device work.

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail, ensure};
use tokenizers::Tokenizer;

use crate::contract::{self, LETTER_IDS, Prepared, RequestError, Settings, THINK_OFF};

pub struct Processor {
    tokenizer: Tokenizer,
    settings: Settings,
}

pub struct Encoder {
    tokenizer: Tokenizer,
}

impl contract::PromptEncoder for Encoder {
    fn encode(&mut self, text: &str) -> Result<Vec<u32>, String> {
        encode_text(&self.tokenizer, text)
    }
}

impl Processor {
    pub fn load(dir: &Path, settings: Settings) -> Result<Self> {
        ensure!(
            (1..=contract::NATIVE_CONTEXT).contains(&settings.max_model_len),
            "STARTLUX_MAX_MODEL_LEN must be 1..=262144"
        );
        let path = dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&path)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("reading {}", path.display()))?;
        probe(&tokenizer)?;
        Ok(Self {
            tokenizer,
            settings,
        })
    }

    pub fn prepare(&self, raw: &[u8]) -> Result<Prepared, RequestError> {
        let mut encoder = self.encoder();
        contract::prepare_request(raw, &self.settings, &mut encoder)
    }

    pub fn encoder(&self) -> Encoder {
        Encoder {
            tokenizer: self.tokenizer.clone(),
        }
    }

    pub fn settings(&self) -> &Settings {
        &self.settings
    }
}

fn encode_text(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>, String> {
    tokenizer
        .encode(text, false)
        .map(|encoding| encoding.get_ids().to_vec())
        .map_err(|err| err.to_string())
}

/// Letters A–Z are one token each and stay one token when appended to the probe.
fn probe(tokenizer: &Tokenizer) -> Result<()> {
    let text = contract::probe_prompt();
    ensure!(
        text.ends_with(THINK_OFF),
        "the Instant prompt must end with the empty think block"
    );
    let base = encode_text(tokenizer, &text).map_err(|err| anyhow!(err))?;
    for (index, letter) in b"ABCDEFGHIJKLMNOPQRSTUVWXYZ".iter().enumerate() {
        let letter = char::from(*letter).to_string();
        let alone = encode_text(tokenizer, &letter).map_err(|err| anyhow!(err))?;
        let extended =
            encode_text(tokenizer, &format!("{text}{letter}")).map_err(|err| anyhow!(err))?;
        if alone.len() != 1
            || alone[0] != LETTER_IDS[index]
            || extended != [&base[..], &alone[..]].concat()
        {
            bail!(
                "the letter {letter} is not token {} immediately after the Instant prompt",
                LETTER_IDS[index]
            );
        }
    }
    Ok(())
}
