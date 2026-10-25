//! Bidirectional Qwen prefill and the tied letter rows. No request semantics.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, ensure};
use omni_qwen3_5_native::model::Model;
use serde_json::Value as Json;

use crate::contract::{self, Prompt};

const EMBEDDING: &str = "model.language_model.embed_tokens.weight";
const HIDDEN: usize = 2560;
const VOCAB: usize = 248_320;

#[derive(Clone)]
pub struct Executor {
    model: Arc<Mutex<Model>>,
    letters: Arc<Vec<f32>>,
}

impl Executor {
    pub async fn load(dir: &Path, library: &Path) -> Result<Self> {
        let (folder, lib) = (dir.to_path_buf(), library.to_path_buf());
        let model = tokio::task::spawn_blocking(move || {
            let model = Model::load_with_graph(&folder, &lib, false)?;
            model.require_prompt_bidirectional()?;
            pin_model(&model)?;
            Ok::<_, anyhow::Error>(model)
        })
        .await??;
        let letters = Arc::new(letter_rows(dir)?);
        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            letters,
        })
    }

    /// Last-position letter logits, one row per prompt, in prompt order.
    /// The caller admits the whole request once; this does not admit again.
    pub fn logits(&self, prompts: &[Prompt]) -> Result<Vec<Vec<f64>>> {
        if prompts.is_empty() {
            return Ok(Vec::new());
        }
        let mut model = self.model.lock().map_err(|_| anyhow::anyhow!("poisoned"))?;
        let mut rows = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            let last = model.forward_prompt_bidirectional(&prompt.ids)?;
            rows.push(letter_logits(&self.letters, &last, prompt.letters)?);
        }
        Ok(rows)
    }
}

fn pin_model(model: &Model) -> Result<()> {
    let cfg = &model.cfg;
    ensure!(cfg.hidden == HIDDEN, "hidden size is not Instant 4B");
    ensure!(
        cfg.intermediate == 9216,
        "intermediate size is not Instant 4B"
    );
    ensure!(
        cfg.full_attention.len() == 32
            && cfg
                .full_attention
                .iter()
                .enumerate()
                .all(|(index, full)| *full == (index % 4 == 3)),
        "layer pattern is not Instant 4B"
    );
    ensure!(
        cfg.heads == 16 && cfg.kv_heads == 4 && cfg.head_dim == 256,
        "attention shape is not Instant 4B"
    );
    ensure!(
        cfg.lin_k_heads == 16
            && cfg.lin_v_heads == 32
            && cfg.lin_k_dim == 128
            && cfg.lin_v_dim == 128,
        "linear-attention shape is not Instant 4B"
    );
    ensure!(
        cfg.max_positions == contract::NATIVE_CONTEXT,
        "native context is not 262144"
    );
    ensure!(cfg.rotary_half == 32, "rotary half is not 32");
    ensure!(
        cfg.mrope_section == [11, 11, 10],
        "mrope section is not Instant 4B"
    );
    ensure!(
        cfg.rope_theta == 10_000_000.0,
        "rope theta is not Instant 4B"
    );
    Ok(())
}

fn letter_logits(letters: &[f32], last: &[f32], n_options: usize) -> Result<Vec<f64>> {
    ensure!(last.len() == HIDDEN, "last hidden state is not {HIDDEN}");
    ensure!(n_options <= contract::MAX_OPTIONS, "more than 26 letters");
    let (rows, _) = letters.as_chunks::<HIDDEN>();
    let logits: Vec<f64> = rows
        .iter()
        .take(n_options)
        .map(|row| {
            let dot = row
                .iter()
                .zip(last)
                .map(|(&weight, &hidden)| f64::from(weight) * f64::from(hidden))
                .sum::<f64>() as f32;
            f64::from(dot)
        })
        .collect();
    ensure!(logits.len() == n_options, "letter head is short");
    ensure!(
        logits.iter().all(|logit| logit.is_finite()),
        "non-finite letter logit"
    );
    Ok(logits)
}

fn letter_rows(dir: &Path) -> Result<Vec<f32>> {
    let index: Json = serde_json::from_str(&std::fs::read_to_string(
        dir.join("model.safetensors.index.json"),
    )?)?;
    let file = index
        .get("weight_map")
        .and_then(|map| map.get(EMBEDDING))
        .and_then(Json::as_str)
        .context(EMBEDDING)?;
    let file = std::fs::File::open(dir.join(file))?;
    // SAFETY: the checkpoint is not modified while the worker runs.
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let tensors = safetensors::SafeTensors::deserialize(&mmap)?;
    let view = tensors.tensor(EMBEDDING)?;
    ensure!(
        view.dtype() == safetensors::Dtype::BF16 && view.shape() == [VOCAB, HIDDEN],
        "{EMBEDDING}: {:?} {:?}",
        view.dtype(),
        view.shape()
    );
    let mut rows = Vec::with_capacity(contract::LETTER_IDS.len() * HIDDEN);
    for &id in &contract::LETTER_IDS {
        let start = id as usize * HIDDEN * 2;
        let row = &view.data()[start..start + HIDDEN * 2];
        let (pairs, rest) = row.as_chunks::<2>();
        ensure!(rest.is_empty(), "short embedding row");
        rows.extend(
            pairs
                .iter()
                .map(|&bytes| half::bf16::from_le_bytes(bytes).to_f32()),
        );
    }
    Ok(rows)
}
