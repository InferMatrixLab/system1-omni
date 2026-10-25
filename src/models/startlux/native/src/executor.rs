//! Bidirectional Qwen prefill and the tied letter rows. No request semantics.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, ensure};
use omni_qwen3_5_native::model::{
    self, Config, Model, PROMPT_SCRATCH_TOKEN_QUANTUM, PromptScratchSize,
};
use serde_json::Value as Json;

use crate::contract::{self, Prompt};

const EMBEDDING: &str = "model.language_model.embed_tokens.weight";
const HIDDEN: usize = 2560;
const INTERMEDIATE: usize = 9216;
const LAYERS: usize = 32;
const HEADS: usize = 16;
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 256;
const LIN_K_HEADS: usize = 16;
const LIN_V_HEADS: usize = 32;
const LIN_K_DIM: usize = 128;
const LIN_V_DIM: usize = 128;
const ROTARY_HALF: usize = 32;
const VOCAB: usize = 248_320;

/// Startup scratch reservation for one serve limit.
///
/// `tokens` is the serve limit after the same 1024-token quantum on-demand
/// growth uses. It is not a benchmark's longest sample. `bytes` is the
/// default-kernel scratch buffer at that cap. `warmup_quantum_bytes` is the
/// same buffer at one quantum, which is what a short warmup allocates when
/// nothing was reserved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScratchPresize {
    pub tokens: usize,
    pub bytes: usize,
    pub warmup_quantum_bytes: usize,
}

impl ScratchPresize {
    /// Bytes this reservation holds beyond a one-quantum scratch.
    pub fn extra_bytes(self) -> usize {
        self.bytes.saturating_sub(self.warmup_quantum_bytes)
    }
}

/// Scratch ceiling for `serve_limit` (`Settings::max_model_len`).
///
/// The cap is that limit rounded up with [`PROMPT_SCRATCH_TOKEN_QUANTUM`].
/// Valid limits are `1..=262144`. The byte count uses the pinned Instant 4B
/// shape and the default GDN workspace, not a Public231 sample length.
pub fn scratch_presize(serve_limit: usize) -> Result<ScratchPresize> {
    ensure!(
        (1..=contract::NATIVE_CONTEXT).contains(&serve_limit),
        "scratch pre-size serve limit must be 1..=262144"
    );
    let tokens = model::prompt_scratch_token_cap(serve_limit);
    ensure!(
        tokens <= contract::NATIVE_CONTEXT,
        "aligned scratch cap {tokens} exceeds the native context"
    );
    let cfg = instant_scratch_config();
    let bytes = model::prompt_scratch_device_bytes(&cfg, tokens);
    let warmup_quantum_bytes =
        model::prompt_scratch_device_bytes(&cfg, PROMPT_SCRATCH_TOKEN_QUANTUM);
    Ok(ScratchPresize {
        tokens,
        bytes,
        warmup_quantum_bytes,
    })
}

fn instant_scratch_config() -> Config {
    Config {
        hidden: HIDDEN,
        intermediate: INTERMEDIATE,
        eps: 1e-6,
        full_attention: (0..LAYERS).map(|index| index % 4 == 3).collect(),
        heads: HEADS,
        kv_heads: KV_HEADS,
        head_dim: HEAD_DIM,
        rotary_half: ROTARY_HALF,
        rope_theta: 10_000_000.0,
        mrope_section: [11, 11, 10],
        max_positions: contract::NATIVE_CONTEXT,
        image_token_id: None,
        lin_k_heads: LIN_K_HEADS,
        lin_v_heads: LIN_V_HEADS,
        lin_k_dim: LIN_K_DIM,
        lin_v_dim: LIN_V_DIM,
    }
}

#[derive(Clone)]
pub struct Executor {
    model: Arc<Mutex<Model>>,
    letters: Arc<Vec<f32>>,
}

impl Executor {
    pub async fn load(dir: &Path, library: &Path, serve_limit: usize) -> Result<Self> {
        let expected = scratch_presize(serve_limit)?;
        let (folder, lib) = (dir.to_path_buf(), library.to_path_buf());
        let model = tokio::task::spawn_blocking(move || {
            let mut model = Model::load_with_graph(&folder, &lib, false)?;
            model.require_prompt_bidirectional()?;
            pin_model(&model)?;
            let reserved = model.reserve_prompt_scratch(serve_limit)?;
            ensure!(
                reserved.tokens == expected.tokens,
                "reserved scratch cap {} != serve-limit cap {}",
                reserved.tokens,
                expected.tokens
            );
            log_reserved_scratch(serve_limit, &model.cfg, reserved, expected);
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

fn log_reserved_scratch(
    serve_limit: usize,
    cfg: &Config,
    reserved: PromptScratchSize,
    expected: ScratchPresize,
) {
    let formula = model::prompt_scratch_device_bytes(cfg, reserved.tokens);
    let extra = reserved.bytes.saturating_sub(expected.warmup_quantum_bytes);
    eprintln!(
        "StartLux Instant prompt scratch pre-size from serve limit {serve_limit}: \
         cap {} tokens, {} bytes. The {}-token quantum is {} bytes; this pre-size \
         is {extra} bytes more. On-demand grow remains for a longer prompt.",
        reserved.tokens,
        reserved.bytes,
        PROMPT_SCRATCH_TOKEN_QUANTUM,
        expected.warmup_quantum_bytes,
    );
    if formula != reserved.bytes {
        eprintln!(
            "StartLux Instant scratch formula is {formula} bytes; the device buffer is {} bytes",
            reserved.bytes
        );
    }
}

fn pin_model(model: &Model) -> Result<()> {
    let cfg = &model.cfg;
    ensure!(cfg.hidden == HIDDEN, "hidden size is not Instant 4B");
    ensure!(
        cfg.intermediate == INTERMEDIATE,
        "intermediate size is not Instant 4B"
    );
    ensure!(
        cfg.full_attention.len() == LAYERS
            && cfg
                .full_attention
                .iter()
                .enumerate()
                .all(|(index, full)| *full == (index % 4 == 3)),
        "layer pattern is not Instant 4B"
    );
    ensure!(
        cfg.heads == HEADS && cfg.kv_heads == KV_HEADS && cfg.head_dim == HEAD_DIM,
        "attention shape is not Instant 4B"
    );
    ensure!(
        cfg.lin_k_heads == LIN_K_HEADS
            && cfg.lin_v_heads == LIN_V_HEADS
            && cfg.lin_k_dim == LIN_K_DIM
            && cfg.lin_v_dim == LIN_V_DIM,
        "linear-attention shape is not Instant 4B"
    );
    ensure!(
        cfg.max_positions == contract::NATIVE_CONTEXT,
        "native context is not 262144"
    );
    ensure!(cfg.rotary_half == ROTARY_HALF, "rotary half is not 32");
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
