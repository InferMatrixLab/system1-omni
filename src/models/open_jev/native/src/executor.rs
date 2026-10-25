//! Independent CUDA prefill and the learned scalar head, without API semantics.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, ensure};
use omni_qwen3_5_native::model::{Config, Model};
use omni_qwen3_5_native::prefix::{PrefixMode, PrefixPlan, PrefixRow};
use omni_runtime::SerialScheduler;
use serde_json::Value;

use crate::batching;
use crate::contract::Checkpoint;

#[derive(Clone)]
pub(crate) struct DecisionHead {
    weights: Vec<f32>,
    bias: f32,
}

impl DecisionHead {
    pub(crate) fn load(dir: &Path, manifest: &Value, checkpoint: &Checkpoint) -> Result<Self> {
        let cfg = Config::load(dir)?;
        ensure!(
            (
                cfg.hidden,
                cfg.intermediate,
                cfg.full_attention.len(),
                cfg.heads,
                cfg.kv_heads,
                cfg.lin_k_heads,
                cfg.lin_v_heads
            ) == checkpoint.backbone,
            "expected the {} backbone dimensions",
            checkpoint.model_id
        );
        let weights: Vec<f32> = manifest["head_weight"]
            .as_array()
            .context("head_weight")?
            .iter()
            .map(|v| {
                v.as_f64()
                    .map(|v| v as f32)
                    .context("head weight must be numeric")
            })
            .collect::<Result<_>>()?;
        let bias = manifest["head_bias"].as_f64().context("head_bias")? as f32;
        ensure!(
            weights.len() == cfg.hidden
                && weights.iter().all(|v| v.is_finite())
                && bias.is_finite(),
            "invalid scalar decision head"
        );
        Ok(Self { weights, bias })
    }

    fn score(&self, last: Vec<f32>) -> Result<f32> {
        // BF16 hidden -> trained FP32 scalar head. FP64 accumulation on the
        // CPU, rounded to FP32, can differ from PyTorch's FP32 GEMM order.
        let score = (self
            .weights
            .iter()
            .zip(last)
            .map(|(&w, h)| w as f64 * h as f64)
            .sum::<f64>() as f32)
            + self.bias;
        ensure!(score.is_finite(), "non-finite decision score");
        Ok(score)
    }
}

/// One candidate prompt and its question's position in the request.
struct Row<'a> {
    ids: &'a [u32],
    question: usize,
}

impl PrefixRow for Row<'_> {
    fn ids(&self) -> &[u32] {
        self.ids
    }

    fn same_group(&self, other: &Self) -> bool {
        self.question == other.question
    }
}

/// Candidate prompts in prepared order; a question's candidates form one group.
fn rows(ids: &[Vec<Vec<u32>>]) -> Vec<Row<'_>> {
    ids.iter()
        .enumerate()
        .flat_map(|(question, candidates)| candidates.iter().map(move |ids| Row { ids, question }))
        .collect()
}

/// How a request's candidates ran: the configured mode, whether they ran on shared
/// prefixes, and the prefix tokens that were not recomputed.
#[derive(Clone, Copy, Debug)]
pub struct PrefixUse {
    pub mode: PrefixMode,
    pub shared: bool,
    pub saved_tokens: usize,
}

pub struct Executor {
    model: Arc<Mutex<Model>>,
    head: DecisionHead,
    pack: bool,
    prefix: PrefixMode,
}

impl Executor {
    pub(crate) async fn load(
        dir: &Path,
        library: &Path,
        head: DecisionHead,
        checkpoint: &Checkpoint,
        prefix: PrefixMode,
        graph: bool,
    ) -> Result<Self> {
        let (d, lib) = (dir.to_path_buf(), library.to_path_buf());
        let model =
            tokio::task::spawn_blocking(move || Model::load_with_graph(&d, &lib, graph)).await??;
        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            head,
            pack: checkpoint.pack_candidates,
            prefix,
        })
    }

    /// Inputs and outputs are grouped by question, then candidate, in prepared order.
    pub async fn execute(
        &self,
        scheduler: &SerialScheduler,
        ids: Vec<Vec<Vec<u32>>>,
    ) -> Result<Vec<Vec<f32>>> {
        Ok(self.execute_with_prefix(scheduler, ids).await?.0)
    }

    /// Admit one whole request. With prefix reuse selected (and, for `auto`, worth it),
    /// shared prefixes run once through `forward_shared`; with fixed or forced shared
    /// execution the candidates otherwise run one at a time with `forward_fixed`.
    /// Otherwise, where the checkpoint allows it, independent candidates are packed for
    /// shared GEMMs, or run one at a time.
    pub async fn execute_with_prefix(
        &self,
        scheduler: &SerialScheduler,
        ids: Vec<Vec<Vec<u32>>>,
    ) -> Result<(Vec<Vec<f32>>, PrefixUse)> {
        let model = self.model.clone();
        let head = self.head.clone();
        let pack = self.pack;
        let mode = self.prefix;
        scheduler
            .run(move || {
                let mut model = model
                    .lock()
                    .map_err(|_| anyhow::anyhow!("poisoned model"))?;
                let inputs: Vec<&[u32]> = ids.iter().flatten().map(Vec::as_slice).collect();
                let rows = rows(&ids);
                let plan = match mode {
                    PrefixMode::Shared => PrefixPlan::new(&rows),
                    PrefixMode::Auto => {
                        PrefixPlan::new(&rows).filter(|plan| plan.worth_auto(&rows))
                    }
                    _ => None,
                };
                let mut used = PrefixUse {
                    mode,
                    shared: false,
                    saved_tokens: 0,
                };
                let mut scores = Vec::with_capacity(inputs.len());
                if let Some(plan) = plan {
                    for last in model.forward_shared(&plan.prompts)?.into_iter().flatten() {
                        scores.push(head.score(last)?);
                    }
                    used.shared = true;
                    used.saved_tokens = plan.saved_tokens;
                } else if mode == PrefixMode::Fixed || mode == PrefixMode::Shared {
                    for ids in &inputs {
                        scores.push(head.score(model.forward_fixed(ids)?)?);
                    }
                } else if pack {
                    for range in batching::ranges(&inputs) {
                        for last in model.forward_batch(&inputs[range])? {
                            scores.push(head.score(last)?);
                        }
                    }
                } else {
                    for ids in &inputs {
                        scores.push(head.score(model.forward(ids)?)?);
                    }
                }
                ensure!(
                    scores.len() == inputs.len(),
                    "candidate score count mismatch"
                );
                let mut scores = scores.into_iter();
                let rows = ids
                    .iter()
                    .map(|candidates| scores.by_ref().take(candidates.len()).collect())
                    .collect();
                Ok((rows, used))
            })
            .await
    }
}

#[cfg(test)]
#[path = "../../../../../tests/open_jev/head.rs"]
mod tests;
