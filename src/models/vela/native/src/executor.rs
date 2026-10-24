//! The native executor: the backbone through the shared qwen3_5 crate
//! on CUDA. The family `Model` exposes only last-position hidden
//! states, so every readout position runs as its own full sequence —
//! `[parts .. position]` — and causality makes that hidden identical
//! to the in-sequence one the reference reads inside its tree forward.
//! Sequences pack into `forward_batch`.
//!
//! The backbone runs from the bundle's uniform-BF16 export — the
//! shared loader's format — while the Vela heads stay F32 on the
//! bundle's faithful tensors.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use omni_qwen3_5_native::model::Model;

use crate::heads::Heads;
use crate::render::{Block, RenderedRow};

/// Span-block bookkeeping while readouts accumulate.
struct SpanReadouts {
    block_index: usize,
    label_ranges: Vec<(usize, usize)>,
}

/// One readout sequence: full token prefix, hidden at its last row.
struct Readout {
    ids: Vec<u32>,
    /// (block index, Option<end index>) attribution; None = the query.
    tag: (usize, Option<usize>),
}

pub struct Executor {
    pub model: Model,
    pub heads: Heads,
    bundle: PathBuf,
}

impl Executor {
    pub fn load(bundle: &Path, library: &Path) -> Result<Self> {
        let model = Model::load(&bundle.join("bf16"), library)
            .context("the shared qwen3_5 model did not load the uniform-BF16 backbone")?;
        let heads = Heads::load(bundle)?;
        Ok(Self {
            model,
            heads,
            bundle: bundle.to_owned(),
        })
    }

    pub fn bundle(&self) -> &Path {
        &self.bundle
    }

    /// The candidate hiddens (per option end, `[ABS]` last when shown)
    /// and the query hidden of every question block, plus the word
    /// hiddens, label means and span-query hidden of the span block.
    pub fn row_readouts(&mut self, row: &RenderedRow) -> Result<RowReadouts> {
        let mut readouts: Vec<Readout> = Vec::new();
        for (block_index, block) in row.blocks.iter().enumerate() {
            let Block::Question { ids, ends, .. } = block else {
                continue;
            };
            for (end_index, end) in ends.iter().enumerate() {
                readouts.push(Readout {
                    ids: [row.p_ids.as_slice(), &ids[..=*end]].concat(),
                    tag: (block_index, Some(end_index)),
                });
            }
        }
        let mut span: Option<SpanReadouts> = None;
        if let Some(block) = row.blocks.iter().find(|b| matches!(b, Block::Span { .. })) {
            let Block::Span {
                ids,
                label_ranges,
                w_pos,
                words_in_block,
                ..
            } = block
            else {
                unreachable!()
            };
            let block_index = row
                .blocks
                .iter()
                .position(|b| matches!(b, Block::Span { .. }))
                .unwrap();
            for (word_index, position) in w_pos.iter().enumerate() {
                // Layout c: w_pos is span-block-relative, the hidden
                // rides [parts, block..=position]. Layout a: w_pos is
                // parts-relative, the hidden is the parts pass itself.
                let word_ids = if *words_in_block {
                    [row.p_ids.as_slice(), &ids[..=*position]].concat()
                } else {
                    row.p_ids[..=*position].to_vec()
                };
                readouts.push(Readout {
                    ids: word_ids,
                    tag: (block_index, Some(word_index)),
                });
            }
            span = Some(SpanReadouts {
                block_index,
                label_ranges: label_ranges.clone(),
            });
        }
        // Each block's query position is its last token: the full
        // sequence's final hidden.
        for (block_index, block) in row.blocks.iter().enumerate() {
            let ids = match block {
                Block::Question { ids, .. } | Block::Span { ids, .. } => ids,
            };
            // The query is each block's last token by construction.
            debug_assert!(ids.last().is_some());
            readouts.push(Readout {
                ids: [row.p_ids.as_slice(), ids].concat(),
                tag: (block_index, None),
            });
        }
        let hiddens = self.run_all(&readouts)?;
        // Split per block.
        /// Per block: (candidate/word hiddens, query hiddens).
        type PerBlock = std::collections::HashMap<usize, (Vec<Vec<f32>>, Vec<Vec<f32>>)>;
        let mut per_block: PerBlock = std::collections::HashMap::new();
        for (readout, hidden) in readouts.iter().zip(hiddens) {
            let entry = per_block
                .entry(readout.tag.0)
                .or_insert_with(|| (Vec::new(), Vec::new()));
            match readout.tag.1 {
                Some(_) => entry.0.push(hidden),
                None => entry.1.push(hidden),
            }
        }
        let mut questions = Vec::new();
        for (block_index, block) in row.blocks.iter().enumerate() {
            if let Block::Question { qid, .. } = block {
                let (candidates, queries) = per_block
                    .remove(&block_index)
                    .unwrap_or_else(|| (Vec::new(), Vec::new()));
                questions.push(QReadout {
                    block_index,
                    qid: qid.clone(),
                    candidates,
                    query: queries.into_iter().next(),
                });
            }
        }
        let mut span_readouts = None;
        if let Some(info) = span {
            let (words, queries) = per_block
                .remove(&info.block_index)
                .unwrap_or_else(|| (Vec::new(), Vec::new()));
            // The label mean hiddens: mean over each label range's
            // option-block hiddens... the reference means the raw hidden
            // states of the label tokens, so run those positions too.
            let label_runs: Vec<Readout> = info
                .label_ranges
                .iter()
                .flat_map(|(start, end)| {
                    (*start..=*end).map(|position| Readout {
                        ids: [
                            row.p_ids.as_slice(),
                            &row.span_block_ids(info.block_index)[..=position],
                        ]
                        .concat(),
                        tag: (info.block_index, None),
                    })
                })
                .collect();
            let label_hiddens = self.run_all(&label_runs)?;
            let mut means = Vec::new();
            let mut cursor = 0usize;
            for (start, end) in &info.label_ranges {
                let count = end - start + 1;
                let mut mean = vec![0.0f32; label_hiddens.first().map_or(0, |h| h.len())];
                for hidden in &label_hiddens[cursor..cursor + count] {
                    for (m, v) in mean.iter_mut().zip(hidden) {
                        *m += v;
                    }
                }
                for m in mean.iter_mut() {
                    *m /= count as f32;
                }
                means.push(mean);
                cursor += count;
            }
            span_readouts = Some(SpanBlockReadouts {
                words,
                labels: means,
                query: queries.into_iter().next(),
            });
        }
        Ok(RowReadouts {
            questions,
            span: span_readouts,
        })
    }

    fn run_all(&mut self, readouts: &[Readout]) -> Result<Vec<Vec<f32>>> {
        let mut hiddens = Vec::with_capacity(readouts.len());
        const CHUNK: usize = 16;
        for chunk in readouts.chunks(CHUNK) {
            let inputs: Vec<&[u32]> = chunk.iter().map(|r| r.ids.as_slice()).collect();
            hiddens.extend(self.model.forward_batch(&inputs)?);
        }
        Ok(hiddens)
    }
}

/// Question-block readouts: candidate hiddens in option order
/// (`[ABS]` last) and the block's query hidden.
pub struct QReadout {
    pub block_index: usize,
    pub qid: String,
    pub candidates: Vec<Vec<f32>>,
    pub query: Option<Vec<f32>>,
}

/// Span-block readouts: word hiddens, label-mean hiddens, query hidden.
pub struct SpanBlockReadouts {
    pub words: Vec<Vec<f32>>,
    pub labels: Vec<Vec<f32>>,
    pub query: Option<Vec<f32>>,
}

pub struct RowReadouts {
    pub questions: Vec<QReadout>,
    pub span: Option<SpanBlockReadouts>,
}

impl RenderedRow {
    fn span_block_ids(&self, block_index: usize) -> &[u32] {
        match &self.blocks[block_index] {
            Block::Span { ids, .. } => ids,
            _ => &[],
        }
    }
}
