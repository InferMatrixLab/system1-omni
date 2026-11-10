//! Bounded packing of independent candidate prompts within one admitted request.

use std::ops::Range;

const MAX_SEQUENCES: usize = 16;
const MAX_TOKENS: usize = 4096;

/// Keep candidate order, with long prompts executing alone at their original size.
pub(crate) fn ranges(inputs: &[&[u32]]) -> Vec<Range<usize>> {
    let mut batches = Vec::new();
    let mut start = 0;
    let mut tokens = 0;
    for (index, ids) in inputs.iter().enumerate() {
        if index > start && (index - start == MAX_SEQUENCES || tokens + ids.len() > MAX_TOKENS) {
            batches.push(start..index);
            start = index;
            tokens = 0;
        }
        tokens += ids.len();
    }
    if start < inputs.len() {
        batches.push(start..inputs.len());
    }
    batches
}

/// Only long candidates that the original packing policy executes separately.
/// Sharing is restricted to one question by the caller, with a nonempty suffix.
pub(crate) fn shared_prefix(inputs: &[&[u32]]) -> usize {
    shared_prefix_with_minimum(inputs, MAX_TOKENS / 2 + 1)
}

/// Packed-row compatibility is checked by the executor before any prefix runs.
pub(crate) fn packed_shared_prefix(inputs: &[&[u32]]) -> usize {
    shared_prefix_with_minimum(inputs, 512)
}

fn shared_prefix_with_minimum(inputs: &[&[u32]], minimum: usize) -> usize {
    if !(2..=MAX_SEQUENCES).contains(&inputs.len()) || inputs.iter().any(|ids| ids.len() < minimum)
    {
        return 0;
    }
    let shortest = inputs.iter().map(|ids| ids.len()).min().unwrap();
    let common = (0..shortest - 1)
        .take_while(|&i| inputs.iter().all(|ids| ids[i] == inputs[0][i]))
        .count();
    let aligned = common / 64 * 64;
    if aligned >= 512 { aligned } else { 0 }
}

/// Pointwise projection compaction keeps original mixers and one bounded batch.
pub(crate) fn projection_prefix(inputs: &[&[u32]]) -> usize {
    if !(2..=MAX_SEQUENCES).contains(&inputs.len())
        || inputs.iter().any(|ids| ids.is_empty())
        || inputs.iter().map(|ids| ids.len()).sum::<usize>() > MAX_TOKENS
    {
        return 0;
    }
    let shortest = inputs.iter().map(|ids| ids.len()).min().unwrap();
    let common = (0..shortest - 1)
        .take_while(|&i| inputs.iter().all(|ids| ids[i] == inputs[0][i]))
        .count();
    if (16..512).contains(&common) {
        common
    } else {
        0
    }
}

#[cfg(test)]
#[path = "../../../../../tests/open_jev/batching.rs"]
mod tests;
