//! CUDA prefill and the learned answer-letter projection. No request semantics.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, ensure};
use omni_runtime::SerialScheduler;
use serde_json::Value as Json;

use crate::model::Model;
use crate::prefix_cache::{PrefixCache, PrefixCacheStats};
use crate::provenance::TextExport;

/// Tied with the output projection in Qwen3.5-4B.
const EMBEDDING: &str = "model.language_model.embed_tokens.weight";

/// One unpadded prompt and the number of answer-letter rows to project.
pub struct Input {
    pub ids: Vec<u32>,
    pub n_options: usize,
    /// Where the prompt's leading block ends, for the prefix cache; 0 for none.
    pub checkpoint: usize,
}

pub struct Executor {
    // Declared first so that cached states are freed before the model's stream.
    cache: Option<Arc<Mutex<PrefixCache>>>,
    model: Arc<Mutex<Model>>,
    letters: Vec<f32>,
    export: TextExport,
    /// The cache's counters after the last question, readable while one runs.
    stats: Arc<Mutex<Option<PrefixCacheStats>>>,
}

/// `CUA_S1_PREFIX_CACHE_BYTES`: unset, empty or 0 leaves the cache off.
pub fn prefix_cache_bytes(value: Option<&str>) -> Result<Option<usize>> {
    match value.map(str::trim) {
        None | Some("") => Ok(None),
        Some(v) => v
            .parse::<usize>()
            .map(|n| (n > 0).then_some(n))
            .context("CUA_S1_PREFIX_CACHE_BYTES must be a byte count"),
    }
}

/// The cache continues prompts from saved states with the fixed GEMM algorithms, which
/// need merged-adapter weights and run eagerly.
pub(crate) fn check_prefix_cache(export: TextExport, graph: bool) -> Result<()> {
    ensure!(
        export == TextExport::Merged,
        "CUA_S1_PREFIX_CACHE_BYTES needs a merged-adapter export (cua-s1-text-merged/1); \
         the unmerged profile rejects prefix continuation"
    );
    ensure!(
        !graph,
        "CUA_S1_PREFIX_CACHE_BYTES requires CUA_S1_GRAPH off"
    );
    Ok(())
}

impl Executor {
    pub(crate) async fn load(
        dir: &Path,
        library: &Path,
        letter_ids: &[u32],
        export: TextExport,
        cache_bytes: Option<usize>,
    ) -> Result<Self> {
        let (d, lib) = (dir.to_path_buf(), library.to_path_buf());
        let model = tokio::task::spawn_blocking(move || match export {
            TextExport::Merged => Model::load(&d, &lib),
            TextExport::Unmerged => {
                let mut model =
                    Model::load_with_lora(&d, &lib, &d.join("adapter.safetensors"), 2.0)?;
                model.enable_reference_text_numerics()?;
                Ok(model)
            }
        })
        .await??;
        let letters = letter_rows(dir, letter_ids, model.cfg.hidden)?;
        let cache = cache_bytes.map(PrefixCache::new);
        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            letters,
            export,
            stats: Arc::new(Mutex::new(cache.as_ref().map(PrefixCache::stats))),
            cache: cache.map(|c| Arc::new(Mutex::new(c))),
        })
    }

    /// The prefix cache's counters, or None when it is off.
    pub fn prefix_cache_stats(&self) -> Option<PrefixCacheStats> {
        *self.stats.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Admit each question through the worker's shared runtime scheduler.
    /// Returns FP32 letter logits in input order; softmax belongs to processing.
    pub async fn execute(
        &self,
        scheduler: &SerialScheduler,
        inputs: Vec<Input>,
    ) -> Result<Vec<Vec<f32>>> {
        let mut rows = Vec::with_capacity(inputs.len());
        for input in inputs {
            let model = self.model.clone();
            let cache = self.cache.clone();
            let stats = self.stats.clone();
            let last = scheduler
                .run(move || {
                    let mut model = model.lock().map_err(|_| anyhow::anyhow!("poisoned"))?;
                    let Some(cache) = cache else {
                        return model.forward(&input.ids);
                    };
                    let mut cache = cache.lock().map_err(|_| anyhow::anyhow!("poisoned"))?;
                    let last = cache.forward(&mut model, &input.ids, input.checkpoint);
                    *stats.lock().unwrap_or_else(|e| e.into_inner()) = Some(cache.stats());
                    last
                })
                .await?;
            rows.push(letter_logits(
                &self.letters,
                &last,
                input.n_options,
                self.export,
            )?);
        }
        Ok(rows)
    }
}

fn letter_logits(
    letters: &[f32],
    last: &[f32],
    n_options: usize,
    export: TextExport,
) -> Result<Vec<f32>> {
    let logits: Vec<f32> = letters
        .chunks_exact(last.len())
        .take(n_options)
        .map(|w| {
            let dot = w
                .iter()
                .zip(last)
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum::<f64>() as f32;
            match export {
                // Preserve the original merged worker's FP32 readout on upgrade.
                TextExport::Merged => dot,
                // The unmerged reference profile's tied lm_head emits BF16.
                TextExport::Unmerged => half::bf16::from_f32(dot).to_f32(),
            }
        })
        .collect();
    // Abort before submitting a later prompt, as the former per-question softmax did.
    ensure!(
        logits.iter().all(|l| l.is_finite()),
        "non-finite probabilities"
    );
    Ok(logits)
}

/// The letter rows of the bfloat16 embedding, read from the safetensors files.
fn letter_rows(dir: &Path, ids: &[u32], hidden: usize) -> Result<Vec<f32>> {
    let index: Json = serde_json::from_str(&std::fs::read_to_string(
        dir.join("model.safetensors.index.json"),
    )?)?;
    let file = index["weight_map"][EMBEDDING].as_str().context(EMBEDDING)?;
    let file = std::fs::File::open(dir.join(file))?;
    // SAFETY: the checkpoint is not modified while the worker runs.
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let tensors = safetensors::SafeTensors::deserialize(&mmap)?;
    let view = tensors.tensor(EMBEDDING)?;
    ensure!(
        view.dtype() == safetensors::Dtype::BF16 && view.shape()[1] == hidden,
        "{EMBEDDING}: {:?} {:?}",
        view.dtype(),
        view.shape()
    );
    let mut rows = Vec::with_capacity(ids.len() * hidden);
    for &id in ids {
        let row = &view.data()[id as usize * hidden * 2..(id as usize + 1) * hidden * 2];
        let (pairs, _) = row.as_chunks::<2>();
        rows.extend(pairs.iter().map(|&b| half::bf16::from_le_bytes(b).to_f32()));
    }
    Ok(rows)
}

#[cfg(test)]
#[path = "../../../../../tests/cua_s1/head.rs"]
mod tests;
