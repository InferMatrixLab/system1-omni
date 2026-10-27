//! One complete causal row per question, shared Qwen backbone, Kev FP32 head.
use crate::{contract, head::PointerHead, processing::Row};
use anyhow::{Context, Result, ensure};
use omni_qwen3_5_native::{inputs::MultimodalInput, model::Model};
use std::path::Path;

pub struct Executor {
    model: Model,
    head: PointerHead,
}

pub fn validate_row(row: &Row) -> Result<()> {
    ensure!(
        !row.ids.is_empty() && row.ids.len() <= contract::MAX_ROW,
        "invalid row length"
    );
    ensure!(
        row.ids.iter().all(|&id| id < 248320),
        "token outside Kev vocabulary"
    );
    ensure!(
        row.positions.iter().copied().eq(0..row.ids.len() as i64),
        "expected consecutive text positions"
    );
    ensure!(
        (2..=256).contains(&row.readouts.len()),
        "expected candidate and decision readouts"
    );
    ensure!(
        row.readouts.windows(2).all(|p| p[0] < p[1])
            && row.readouts.last() == Some(&(row.ids.len() - 1)),
        "invalid readout order"
    );
    for &i in &row.readouts[..row.readouts.len() - 1] {
        ensure!(
            row.ids.get(i) == Some(&248050),
            "option readout must be at box_end"
        );
    }
    ensure!(
        row.ids.last() == Some(&248062),
        "decision readout must be at fim_suffix"
    );
    Ok(())
}

impl Executor {
    /// Caller has already verified the export; eager selection is explicit.
    pub(crate) fn load(dir: &Path, library: &Path) -> Result<Self> {
        let head = PointerHead::load(&dir.join("head.safetensors"))?;
        let model = Model::load_with_graph(dir, library, false)?;
        Ok(Self { model, head })
    }

    pub fn execute(&mut self, rows: &[Row]) -> Result<Vec<Vec<f32>>> {
        ensure!(
            (1..=contract::MAX_QUESTIONS).contains(&rows.len()),
            "invalid question count"
        );
        for row in rows {
            validate_row(row)?;
        }
        ensure!(
            rows.iter().map(|r| r.ids.len()).sum::<usize>() <= contract::MAX_REQUEST_TOKENS,
            "too many computed tokens"
        );
        let result = rows
            .iter()
            .map(|row| {
                let input = MultimodalInput {
                    token_ids: &row.ids,
                    image_token_indices: &[],
                    image_embeddings: &[],
                    position_ids: [&row.positions, &row.positions, &row.positions],
                };
                let read = self
                    .model
                    .forward_multimodal_readout(&input, &row.readouts, &[])?;
                let (decide, options) = read.hidden.split_last().context("empty readout")?;
                self.head.logits(
                    decide,
                    &options.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                )
            })
            .collect();
        // The permit and buffers remain owned until all submitted work has drained,
        // including when a later row fails or its HTTP caller has disconnected.
        let sync = self.model.synchronize();
        match result {
            Ok(output) => {
                sync?;
                Ok(output)
            }
            Err(error) => {
                let _ = sync;
                Err(error)
            }
        }
    }
}
