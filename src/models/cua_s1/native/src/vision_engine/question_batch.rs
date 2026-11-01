//! Bounded question groups within one admitted screenshot request.
use super::*;
use crate::model::MAX_MULTIMODAL_BATCH_TOKENS;
use std::ops::Range;

pub(super) fn enabled(value: Option<&str>, mode: prefix::Mode) -> Result<bool> {
    let enabled = match value {
        None | Some("0") => false,
        Some("1") => true,
        _ => anyhow::bail!("CUA_S1_MM_BATCH must be unset, 0, or 1"),
    };
    ensure!(
        !enabled || mode == prefix::Mode::Off,
        "CUA_S1_MM_BATCH requires CUA_S1_MM_PREFIX=off"
    );
    Ok(enabled)
}

fn groups(lengths: &[usize], shared_prefix: bool) -> Result<Vec<Range<usize>>> {
    ensure!(
        (1..=8).contains(&lengths.len()),
        "expected 1 to 8 questions"
    );
    let mut groups = Vec::new();
    let (mut begin, mut tokens) = (0, 0);
    for (index, &length) in lengths.iter().enumerate() {
        ensure!(
            (1..=crate::multimodal::MAX_TOKENS).contains(&length),
            "empty or oversized question prompt"
        );
        if !shared_prefix && tokens + length > MAX_MULTIMODAL_BATCH_TOKENS {
            groups.push(begin..index);
            begin = index;
            tokens = 0;
        }
        tokens += length;
    }
    groups.push(begin..lengths.len());
    Ok(groups)
}

pub(super) fn plan(prepared: &PreparedRequest) -> Result<Vec<Range<usize>>> {
    ensure!(
        prepared.prompts.iter().all(|p| p
            .position_ids
            .iter()
            .all(|axis| axis.len() == p.token_ids.len())),
        "position axis length must match prompt tokens"
    );
    ensure!(
        prepared
            .context
            .option_counts()
            .all(|n| (1..=26).contains(&n)),
        "expected 1 to 26 options"
    );
    groups(
        &prepared
            .prompts
            .iter()
            .map(|p| p.token_ids.len())
            .collect::<Vec<_>>(),
        prefix::common_prefix(&prepared.prompts) > 0,
    )
}

pub(super) enum Features<'a> {
    Host(&'a [bf16]),
    Device(DeviceBf16Matrix<'a>),
}

pub(super) fn execute(
    model: &mut Model,
    letters: &[f32],
    prepared: &PreparedRequest,
    groups: &[Range<usize>],
    features: Features<'_>,
) -> Result<Vec<Vec<f32>>> {
    let counts: Vec<_> = prepared.context.option_counts().collect();
    let mut logits = Vec::with_capacity(prepared.prompts.len());
    for group in groups {
        let prompts = &prepared.prompts[group.clone()];
        let hidden = match features {
            Features::Host(features) => {
                let inputs: Vec<_> = prompts
                    .iter()
                    .map(|p| MultimodalInput {
                        token_ids: &p.token_ids,
                        image_token_indices: &p.image_token_indices,
                        image_embeddings: features,
                        position_ids: [&p.position_ids[0], &p.position_ids[1], &p.position_ids[2]],
                    })
                    .collect();
                model
                    .forward_multimodal_batch_with_prefix(&inputs, prefix::common_prefix(prompts))?
            }
            Features::Device(features) => {
                let inputs: Vec<_> = prompts
                    .iter()
                    .map(|p| DeviceMultimodalInput {
                        token_ids: &p.token_ids,
                        image_token_indices: &p.image_token_indices,
                        image_embeddings: features,
                        position_ids: [&p.position_ids[0], &p.position_ids[1], &p.position_ids[2]],
                    })
                    .collect();
                model.forward_multimodal_device_batch_with_prefix(
                    &inputs,
                    prefix::common_prefix(prompts),
                )?
            }
        };
        for (row, &count) in hidden.iter().zip(&counts[group.clone()]) {
            logits.push(VisionEngine::letter_logits(letters, row, count)?);
        }
    }
    Ok(logits)
}

#[cfg(test)]
#[path = "../../../../../../tests/cua_s1/question_batch.rs"]
mod tests;
