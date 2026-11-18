//! Model execution units retain original numerical layouts across FIFO batching.
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::{Arc, Mutex};

use anyhow::{Result, ensure};
use omni_qwen3_5_native::model::Model;
use omni_runtime::batching::{BatchCost, BatchLimits, BatchScheduler};

use super::{DecisionHead, PrefixUse};
use crate::batching;

pub(super) struct WorkItem {
    inputs: Vec<Vec<u32>>,
    prefix: usize,
    compaction: usize,
}

pub(super) struct Output {
    scores: Vec<f32>,
    used: PrefixUse,
}

pub(super) fn new(
    model: Arc<Mutex<Model>>,
    head: DecisionHead,
) -> Result<BatchScheduler<WorkItem, Output>> {
    BatchScheduler::new(
        BatchLimits {
            items: 16,
            cost: 8192,
        },
        |item: &WorkItem| BatchCost {
            items: item.inputs.len(),
            cost: item.inputs.iter().map(Vec::len).sum(),
            exclusive: item.prefix > 0 || item.compaction > 0,
        },
        move |items: &[WorkItem]| {
            let mut model = model
                .lock()
                .map_err(|_| anyhow::anyhow!("poisoned model"))?;
            // Keep the model guard alive while retiring any partially submitted
            // work, even on a panic. The runtime closes the worker after a panic.
            let result = catch_unwind(AssertUnwindSafe(|| run(&mut model, &head, items)));
            match result {
                Ok(Ok(outputs)) => Ok(outputs),
                Ok(Err(error)) => {
                    model.synchronize()?;
                    Err(error)
                }
                Err(payload) => {
                    let _ = model.synchronize();
                    resume_unwind(payload)
                }
            }
        },
    )
}

fn run(model: &mut Model, head: &DecisionHead, items: &[WorkItem]) -> Result<Vec<Output>> {
    if items.len() == 1 {
        let item = &items[0];
        let prefix = item.prefix.max(item.compaction / 64 * 64);
        if prefix >= 64 {
            let inputs: Vec<&[u32]> = item.inputs.iter().map(Vec::as_slice).collect();
            let mut original_rows = vec![0; inputs.len()];
            for range in batching::ranges(&inputs) {
                let rows = inputs[range.clone()].iter().map(|ids| ids.len()).sum();
                original_rows[range].fill(rows);
            }
            if let Some(output) = model.try_forward_prefix_tree(&inputs, prefix, &original_rows)? {
                return Ok(vec![Output {
                    scores: output
                        .hidden
                        .into_iter()
                        .map(|last| head.score(last))
                        .collect::<Result<_>>()?,
                    used: PrefixUse {
                        shared: true,
                        saved_tokens: output.saved_tokens,
                        ..PrefixUse::default()
                    },
                }]);
            }
        }
    }
    if items.len() == 1 && items[0].compaction > 0 {
        let item = &items[0];
        let inputs: Vec<&[u32]> = item.inputs.iter().map(Vec::as_slice).collect();
        let hidden = match model.try_forward_prefix_compaction(&inputs, item.compaction)? {
            Some(hidden) => hidden,
            None => model.forward_batch(&inputs)?,
        };
        // Mixers still process every token. Do not report full-prefix token savings.
        return Ok(vec![Output {
            scores: hidden
                .into_iter()
                .map(|last| head.score(last))
                .collect::<Result<_>>()?,
            used: PrefixUse::default(),
        }]);
    }
    if items.len() == 1 && items[0].prefix > 0 {
        let item = &items[0];
        let inputs: Vec<&[u32]> = item.inputs.iter().map(Vec::as_slice).collect();
        let shared = if inputs.iter().any(|ids| ids.len() <= 2048) {
            let mut rows = vec![0; inputs.len()];
            for range in batching::ranges(&inputs) {
                let total = inputs[range.clone()].iter().map(|ids| ids.len()).sum();
                rows[range].fill(total);
            }
            model.try_forward_shared_packed_prefix(&inputs, item.prefix, &rows)?
        } else {
            model.try_forward_shared_prefix(&inputs, item.prefix)?
        };
        let mut scores = Vec::with_capacity(inputs.len());
        let mut used = PrefixUse::default();
        if let Some(output) = shared {
            used.shared = true;
            used.saved_tokens = output.saved_tokens;
            for last in output.hidden {
                scores.push(head.score(last)?);
            }
        } else {
            for range in batching::ranges(&inputs) {
                for last in model.forward_batch(&inputs[range])? {
                    scores.push(head.score(last)?);
                }
            }
        }
        return Ok(vec![Output { scores, used }]);
    }
    ensure!(
        items
            .iter()
            .all(|item| item.prefix == 0 && item.compaction == 0),
        "exclusive prefix item was batched"
    );
    let inputs: Vec<&[u32]> = items
        .iter()
        .flat_map(|item| item.inputs.iter().map(Vec::as_slice))
        .collect();
    let groups: Vec<usize> = items.iter().map(|item| item.inputs.len()).collect();
    let hidden = model.forward_batch_preserving_groups(&inputs, &groups)?;
    let mut scores = hidden.into_iter().map(|last| head.score(last));
    groups
        .into_iter()
        .map(|count| {
            Ok(Output {
                scores: scores.by_ref().take(count).collect::<Result<Vec<_>>>()?,
                used: PrefixUse::default(),
            })
        })
        .collect()
}

pub(super) async fn execute(
    scheduler: &BatchScheduler<WorkItem, Output>,
    ids: Vec<Vec<Vec<u32>>>,
    share_prefix: bool,
) -> Result<(Vec<Vec<f32>>, PrefixUse)> {
    let counts: Vec<usize> = ids.iter().map(Vec::len).collect();
    let candidates: Vec<Vec<u32>> = ids.into_iter().flatten().collect();
    let inputs: Vec<&[u32]> = candidates.iter().map(Vec::as_slice).collect();
    let prefix = if share_prefix && counts.len() == 1 {
        batching::packed_shared_prefix(&inputs)
    } else {
        0
    };
    let compaction = if share_prefix && counts.len() == 1 && prefix == 0 {
        batching::projection_prefix(&inputs)
    } else {
        0
    };
    let items = if prefix > 0 || compaction > 0 {
        vec![WorkItem {
            inputs: candidates,
            prefix,
            compaction,
        }]
    } else {
        let ranges = batching::ranges(&inputs);
        let mut candidates = candidates.into_iter();
        ranges
            .into_iter()
            .map(|range| WorkItem {
                inputs: candidates.by_ref().take(range.len()).collect(),
                prefix: 0,
                compaction: 0,
            })
            .collect()
    };
    let outputs = scheduler.run(items).await?;
    let mut used = PrefixUse::default();
    let mut scores = Vec::new();
    for output in outputs {
        scores.extend(output.scores);
        used.shared |= output.used.shared;
        used.saved_tokens += output.used.saved_tokens;
    }
    ensure!(
        scores.len() == counts.iter().sum::<usize>(),
        "candidate score count mismatch"
    );
    let mut scores = scores.into_iter();
    Ok((
        counts
            .into_iter()
            .map(|count| scores.by_ref().take(count).collect())
            .collect(),
        used,
    ))
}
