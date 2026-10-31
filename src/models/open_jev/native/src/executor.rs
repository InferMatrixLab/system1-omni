//! Independent CUDA prefill and the learned scalar head, without API semantics.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, ensure};
use omni_qwen3_5_native::model::{Config, Model};
use omni_qwen3_5_native::prefix::{PrefixMode, PrefixPlan, PrefixRow};
use omni_runtime::SerialScheduler;
use serde_json::Value;

mod batched;

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
/// prefixes and whether more than one question shared one, and the prefix tokens that
/// were not recomputed.
#[derive(Clone, Copy, Debug, Default)]
pub struct PrefixUse {
    pub mode: PrefixMode,
    pub shared: bool,
    pub across_questions: bool,
    pub saved_tokens: usize,
}

/// Each loaded model owns one execution queue; only 9B default numerics batch.
pub struct Scheduler(SchedulerKind);

enum SchedulerKind {
    Serial(SerialScheduler),
    Batched(omni_runtime::batching::BatchScheduler<batched::WorkItem, batched::Output>),
}

impl Scheduler {
    pub fn is_available(&self) -> bool {
        match &self.0 {
            SchedulerKind::Serial(_) => true,
            SchedulerKind::Batched(scheduler) => scheduler.is_available(),
        }
    }
}

pub struct Executor {
    model: Arc<Mutex<Model>>,
    head: DecisionHead,
    pack: bool,
    share_packed_prefix: bool,
    share_packed_inputs: bool,
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
        pack: bool,
    ) -> Result<Self> {
        let (d, lib) = (dir.to_path_buf(), library.to_path_buf());
        let sharing = matches!(prefix, PrefixMode::Shared | PrefixMode::Auto);
        let pack = pack && checkpoint.pack_candidates;
        let batch_gdn = pack;
        let equivalent_tiles = pack && checkpoint.alias == "open-jev-9b";
        let model = tokio::task::spawn_blocking(move || {
            let mut model = Model::load_with_graph(&d, &lib, graph)?;
            if batch_gdn {
                model.enable_batched_gdn()?;
                model.enable_packed_projections()?;
            }
            if equivalent_tiles && model.enable_equivalent_projection_tiles()? {
                eprintln!("Open-Jev-9B: validated device profile projection tiles enabled");
            }
            // Shared prefixes reproduce the fixed rows only where rows do not depend on M.
            if sharing {
                model
                    .check_fixed_rows()
                    .context("OPEN_JEV_PREFIX startup check failed")?;
            }
            Ok::<_, anyhow::Error>(model)
        })
        .await??;
        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            head,
            pack,
            share_packed_prefix: pack && checkpoint.share_packed_prefix,
            share_packed_inputs: pack && checkpoint.alias == "open-jev-9b",
            prefix,
        })
    }

    pub(crate) fn scheduler(&self, checkpoint: &Checkpoint) -> Result<Scheduler> {
        Ok(Scheduler(
            if self.pack && checkpoint.alias == "open-jev-9b" && self.prefix == PrefixMode::Off {
                SchedulerKind::Batched(batched::new(self.model.clone(), self.head.clone())?)
            } else {
                SchedulerKind::Serial(SerialScheduler::default())
            },
        ))
    }

    pub fn is_available(&self) -> bool {
        !self.model.is_poisoned()
    }

    /// Configure exact within-question reuse before the engine admits requests.
    pub(crate) fn set_exact_prefix(&mut self, enabled: bool) {
        self.share_packed_prefix = enabled;
    }

    /// Inputs and outputs are grouped by question, then candidate, in prepared order.
    pub async fn execute(
        &self,
        scheduler: &Scheduler,
        ids: Vec<Vec<Vec<u32>>>,
    ) -> Result<Vec<Vec<f32>>> {
        Ok(self.execute_with_prefix(scheduler, ids).await?.0)
    }

    /// 9B default submits original batches or one exclusive prefix item to the
    /// shared batch scheduler. Other modes admit a whole request as described below.
    /// With prefix reuse selected (and, for `auto`, worth it),
    /// shared prefixes run once through `forward_shared`; with fixed or forced shared
    /// execution the candidates otherwise run one at a time with `forward_fixed`.
    /// Otherwise the default path runs: where the checkpoint packs candidates, a single
    /// question's long candidates share their prefix through `try_forward_shared_prefix`
    /// when the device allows it, and other candidates are packed for shared GEMMs or run
    /// one at a time.
    pub async fn execute_with_prefix(
        &self,
        scheduler: &Scheduler,
        ids: Vec<Vec<Vec<u32>>>,
    ) -> Result<(Vec<Vec<f32>>, PrefixUse)> {
        let scheduler = match &scheduler.0 {
            SchedulerKind::Batched(scheduler) => {
                return batched::execute(scheduler, ids, self.share_packed_prefix).await;
            }
            SchedulerKind::Serial(scheduler) => scheduler,
        };
        let model = self.model.clone();
        let head = self.head.clone();
        let pack = self.pack;
        let share_packed_prefix = self.share_packed_prefix;
        let share_packed_inputs = self.share_packed_inputs;
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
                    ..PrefixUse::default()
                };
                let mut scores = Vec::with_capacity(inputs.len());
                if let Some(plan) = plan {
                    for last in model.forward_shared(&plan.prompts)?.into_iter().flatten() {
                        scores.push(head.score(last)?);
                    }
                    used.shared = true;
                    used.across_questions = plan.shares_across_groups();
                    used.saved_tokens = plan.saved_tokens;
                } else if mode == PrefixMode::Fixed || mode == PrefixMode::Shared {
                    for ids in &inputs {
                        scores.push(head.score(model.forward_fixed(ids)?)?);
                    }
                } else {
                    // The default path, which is also `auto`'s fallback.
                    let prefix_len = if pack && share_packed_prefix && ids.len() == 1 {
                        if share_packed_inputs {
                            batching::packed_shared_prefix(&inputs)
                        } else {
                            batching::shared_prefix(&inputs)
                        }
                    } else {
                        0
                    };
                    let shared = if prefix_len > 0 {
                        if share_packed_inputs && inputs.iter().any(|ids| ids.len() <= 2048) {
                            let mut packed_rows = vec![0; inputs.len()];
                            for range in batching::ranges(&inputs) {
                                let total = inputs[range.clone()].iter().map(|ids| ids.len()).sum();
                                packed_rows[range].fill(total);
                            }
                            model.try_forward_shared_packed_prefix(
                                &inputs,
                                prefix_len,
                                &packed_rows,
                            )?
                        } else {
                            model.try_forward_shared_prefix(&inputs, prefix_len)?
                        }
                    } else {
                        None
                    };
                    if let Some(output) = shared {
                        used.shared = true;
                        used.saved_tokens = output.saved_tokens;
                        for last in output.hidden {
                            scores.push(head.score(last)?);
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

#[cfg(test)]
#[path = "../../../../../tests/open_jev/rollback.rs"]
mod rollback_tests;
