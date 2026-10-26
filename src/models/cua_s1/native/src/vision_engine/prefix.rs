//! Request-local screenshot prefixes; all branches retain the Cua letter head.
use super::*;
use crate::model::PrefixState;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Mode {
    Off,
    Fixed,
    Shared,
    Matched,
}

impl Mode {
    pub(super) fn from_env(device_features: bool) -> Result<Self> {
        let mode = match std::env::var("CUA_S1_MM_PREFIX").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("off") => Self::Off,
            Ok("fixed") => Self::Fixed,
            Ok("shared") => Self::Shared,
            Ok("matched") => Self::Matched,
            _ => anyhow::bail!("CUA_S1_MM_PREFIX must be off, fixed, shared, or matched"),
        };
        ensure!(
            mode == Self::Off
                || (!device_features && std::env::var("CUA_S1_GRAPH").as_deref() != Ok("1")),
            "multimodal prefix modes require Graph and device features off"
        );
        Ok(mode)
    }
}

/// Compare actual prepared tokens and all rotary axes, leaving a nonempty tail.
pub(super) fn common_prefix(prompts: &[ImagePrompt]) -> usize {
    if prompts.len() < 2 {
        return 0;
    }
    let first = &prompts[0];
    let limit = prompts
        .iter()
        .map(|p| p.token_ids.len())
        .min()
        .unwrap()
        .saturating_sub(1);
    let count = (0..limit)
        .take_while(|&i| {
            prompts[1..].iter().all(|p| {
                p.token_ids[i] == first.token_ids[i]
                    && (0..3).all(|axis| p.position_ids[axis][i] == first.position_ids[axis][i])
            })
        })
        .count();
    count / 64 * 64
}

fn window(
    model: &mut Model,
    prompt: &ImagePrompt,
    features: &[bf16],
    range: std::ops::Range<usize>,
    prefix: Option<&PrefixState>,
    capture: Option<&mut PrefixState>,
    read_last: bool,
) -> Result<Vec<f32>> {
    let begin = prompt
        .image_token_indices
        .partition_point(|&i| i < range.start);
    let end = prompt
        .image_token_indices
        .partition_point(|&i| i < range.end);
    let indices: Vec<_> = prompt.image_token_indices[begin..end]
        .iter()
        .map(|i| i - range.start)
        .collect();
    let rows = if read_last {
        vec![range.end - 1]
    } else {
        vec![]
    };
    let input = MultimodalInput {
        token_ids: &prompt.token_ids[range.clone()],
        image_token_indices: &indices,
        image_embeddings: &features[begin * model.cfg.hidden..end * model.cfg.hidden],
        position_ids: std::array::from_fn(|axis| &prompt.position_ids[axis][range.clone()]),
    };
    let mut result = model.readout_window(&input, prefix, capture, &rows, &[])?;
    Ok(result.hidden.pop().unwrap_or_default())
}

pub(super) fn execute(
    model: &mut Model,
    letters: &[f32],
    prepared: &PreparedRequest,
    features: &[bf16],
    mode: Mode,
    storage: &mut Option<PrefixState>,
) -> Result<Vec<Vec<f32>>> {
    // PreparedRequest is public. Reject malformed layouts before planning or
    // slicing; readout_window validates token values and window positions.
    let image_token = model
        .cfg
        .image_token_id
        .context("checkpoint has no image_token_id")?;
    for prompt in &prepared.prompts {
        ensure!(
            !prompt.token_ids.is_empty()
                && prompt
                    .position_ids
                    .iter()
                    .all(|axis| axis.len() == prompt.token_ids.len()),
            "invalid prepared position shape"
        );
        ensure!(
            prompt.image_token_indices.iter().copied().eq(prompt
                .token_ids
                .iter()
                .enumerate()
                .filter_map(|(i, &id)| (id == image_token).then_some(i))),
            "image indices must exactly match the ordered placeholders"
        );
        ensure!(
            prompt
                .image_token_indices
                .len()
                .checked_mul(model.cfg.hidden)
                == Some(features.len()),
            "image embedding shape mismatch"
        );
    }
    let shared = if mode == Mode::Shared {
        common_prefix(&prepared.prompts)
    } else {
        0
    };
    // Reuse allocation only. Every request captures fresh contents, and a
    // request without a common prefix must not read the retained snapshot.
    let mut state = if shared > 0 {
        if storage.as_ref().is_none_or(|s| s.capacity() < shared) {
            *storage = Some(model.alloc_prefix(shared)?);
        }
        storage.as_mut()
    } else {
        None
    };
    if let Some(state) = &mut state {
        window(
            model,
            &prepared.prompts[0],
            features,
            0..shared,
            None,
            Some(state),
            false,
        )?;
    }
    prepared
        .prompts
        .iter()
        .zip(prepared.context.option_counts())
        .map(|(p, count)| {
            let hidden = window(
                model,
                p,
                features,
                shared..p.token_ids.len(),
                state.as_deref(),
                None,
                true,
            )?;
            VisionEngine::letter_logits(letters, &hidden, count)
        })
        .collect()
}

/// Validate before prefix discovery; PreparedRequest is a public boundary.
pub(super) fn matched_windows(prompts: &[ImagePrompt]) -> Result<(Vec<usize>, Vec<usize>)> {
    for prompt in prompts {
        ensure!(
            !prompt.token_ids.is_empty()
                && prompt
                    .position_ids
                    .iter()
                    .all(|axis| axis.len() == prompt.token_ids.len()),
            "invalid prepared position shape"
        );
    }
    let lengths: Vec<_> = prompts.iter().map(|p| p.token_ids.len()).collect();
    let shared = common_prefix(prompts);
    let mut windows = if shared > 0 { vec![shared] } else { vec![] };
    if shared > 0 {
        windows.extend(lengths.iter().map(|n| n - shared));
    }
    Ok((lengths, windows))
}
