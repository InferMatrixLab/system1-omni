//! Question geometry adapted from kev/cuda_graphs.py at the pinned revision.
//! See ../LICENSE.kev. Preserve padding and GEMM shapes without retaining prefixes.
use crate::{contract, processing::Row};
use anyhow::{Context, Result, ensure};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Plan {
    pub split: usize,
    pub state: usize,
    pub question: usize,
    pub prefix: usize,
    pub count: usize,
    pub index: usize,
}

fn bucket(n: usize, steps: usize, floor: usize) -> usize {
    let step = floor.max(n.next_power_of_two() / steps);
    n.div_ceil(step) * step
}

fn groups(lengths: &[usize]) -> Vec<Vec<usize>> {
    let mut order: Vec<_> = (0..lengths.len()).collect();
    order.sort_by_key(|&i| lengths[i]);
    let mut best = vec![usize::MAX; order.len() + 1];
    let mut cut = vec![0; order.len() + 1];
    best[0] = 0;
    for j in 1..=order.len() {
        for i in j.saturating_sub(32)..j {
            let cost =
                best[i] + 256.max(bucket(j - i, 4, 1) * bucket(lengths[order[j - 1]], 8, 16));
            if cost < best[j] {
                best[j] = cost;
                cut[j] = i;
            }
        }
    }
    let mut result = vec![];
    let mut j = order.len();
    while j > 0 {
        result.push(order[cut[j]..j].to_vec());
        j = cut[j];
    }
    result.reverse();
    result
}

pub(crate) fn geometry(split: usize, lengths: &[usize]) -> Vec<Plan> {
    let state = if bucket(split, 8, 16) <= 1024 {
        bucket(split, 8, 16)
    } else {
        split
    };
    let longest = *lengths.iter().max().unwrap();
    let graphed = bucket(longest, 8, 16) <= 1024 && split <= 4096;
    let prefix = if graphed {
        split.max(16).next_power_of_two()
    } else {
        split
    };
    let mut plans = vec![
        Plan {
            split,
            state,
            prefix,
            ..Plan::default()
        };
        lengths.len()
    ];
    let groups = if graphed {
        groups(lengths)
    } else {
        let count = (16384 / (split + longest)).max(1);
        (0..lengths.len())
            .collect::<Vec<_>>()
            .chunks(count)
            .map(<[usize]>::to_vec)
            .collect()
    };
    for group in groups {
        let longest = group.iter().map(|&i| lengths[i]).max().unwrap();
        let question = if graphed {
            bucket(longest, 8, 16)
        } else {
            longest
        };
        let limit = if graphed {
            let cap = 32.min(32768 / (prefix + question));
            (1..=cap).filter(|&n| bucket(n, 4, 1) <= cap).max().unwrap()
        } else {
            group.len()
        };
        for part in group.chunks(limit) {
            let count = if graphed {
                bucket(part.len(), 4, 1)
            } else {
                part.len()
            };
            for (index, &i) in part.iter().enumerate() {
                plans[i].question = question;
                plans[i].count = count;
                plans[i].index = index;
            }
        }
    }
    plans
}

pub(crate) fn plan(rows: &[Row]) -> Result<Vec<Plan>> {
    ensure!(
        (1..=contract::MAX_QUESTIONS).contains(&rows.len()),
        "invalid question count"
    );
    let split = rows[0]
        .ids
        .iter()
        .position(|&id| id == 248061)
        .context("missing question boundary")?;
    ensure!(
        (1..=contract::MAX_STATE).contains(&split),
        "invalid state length"
    );
    let state = &rows[0].ids[..split];
    for row in rows {
        ensure!(
            row.ids.len() > split && row.ids[..split] == *state && row.ids[split] == 248061,
            "questions must share the same state and boundary"
        );
        ensure!(
            row.ids[split + 1..].iter().all(|&id| id != 248061),
            "extra question boundary"
        );
    }
    Ok(geometry(
        split,
        &rows.iter().map(|r| r.ids.len() - split).collect::<Vec<_>>(),
    ))
}

#[cfg(test)]
#[path = "../../../../../tests/kev/planning.rs"]
mod tests;
