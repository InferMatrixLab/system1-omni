//! One Instant folder: processor, serial admission, bidirectional executor.

use std::path::Path;

use anyhow::{Context, Result, bail};
use omni_runtime::SerialScheduler;
use serde_json::{Value, json};

use crate::contract::{self, Settings};
use crate::executor::Executor;
use crate::processing::Processor;

pub struct Engine {
    processor: Processor,
    scheduler: SerialScheduler,
    executor: Executor,
}

impl Engine {
    pub async fn load(dir: &Path, library: &Path, settings: Settings) -> Result<Self> {
        let config = read_json(&dir.join("config.json"))?;
        let decision = read_json(&dir.join("decision_config.json"))?;
        contract::require_instant_configs(&config, &decision)
            .map_err(|err| anyhow::anyhow!("{err}"))?;
        let processor = Processor::load(dir, settings)?;
        let executor = Executor::load(dir, library).await?;
        Ok(Self {
            processor,
            scheduler: SerialScheduler::default(),
            executor,
        })
    }

    /// Prepare on the caller, then run every prompt of the request under one admission.
    pub async fn decide(&self, raw: &[u8]) -> Result<Value> {
        let prepared = self.processor.prepare(raw)?;
        let settings = self.processor.settings().clone();
        let mut encoder = self.processor.encoder();
        let executor = self.executor.clone();
        self.scheduler
            .run(move || {
                let first = executor.logits(&prepared.prompts)?;
                let finals = contract::plan_finals(&prepared, &first, &settings, &mut encoder)
                    .map_err(anyhow::Error::from)?;
                let second = executor.logits(&finals.prompts)?;
                contract::finish(&prepared, &first, &finals, &second).map_err(anyhow::Error::from)
            })
            .await
    }
}

pub fn settings_from_env() -> Result<Settings> {
    let mut settings = Settings::default();
    if let Ok(value) = std::env::var("STARTLUX_MAX_MODEL_LEN") {
        let parsed: usize = value.parse().context("STARTLUX_MAX_MODEL_LEN")?;
        if !(1..=contract::NATIVE_CONTEXT).contains(&parsed) {
            bail!("STARTLUX_MAX_MODEL_LEN must be 1..=262144");
        }
        settings.max_model_len = parsed;
    }
    if let Ok(value) = std::env::var("STARTLUX_NONE_CHECK") {
        settings.none_check = flag(&value, "STARTLUX_NONE_CHECK")?;
    }
    if let Ok(value) = std::env::var("STARTLUX_ALLOW_NONE") {
        settings.allow_none = flag(&value, "STARTLUX_ALLOW_NONE")?;
    }
    Ok(settings)
}

fn flag(value: &str, name: &str) -> Result<bool> {
    match value {
        "1" | "true" | "TRUE" => Ok(true),
        "0" | "false" | "FALSE" => Ok(false),
        _ => bail!("{name} must be 0 or 1"),
    }
}

fn read_json(path: &Path) -> Result<Value> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))
}

pub fn health() -> Value {
    json!({
        "status": "ready",
        "model": contract::MODEL_ID,
        "prompt_attention": "bidirectional",
        "license": contract::LICENSE,
        "weights": contract::WEIGHTS_REPO,
        "revision": contract::WEIGHTS_REVISION,
    })
}
