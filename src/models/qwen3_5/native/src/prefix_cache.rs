//! Prefix states kept across requests: one per leading token block that a worker names,
//! evicted least recently used within a byte budget. Hits and misses both run with the
//! fixed GEMM algorithms, so a prompt's result does not depend on what is cached.

use std::collections::HashMap;

use anyhow::Result;
use serde::Serialize;

use crate::model::{Model, PrefixState};

/// Counters reported by a worker's health endpoint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct PrefixCacheStats {
    pub budget_bytes: usize,
    pub bytes: usize,
    pub entries: usize,
    pub hits: u64,
    pub misses: u64,
    /// Prompts run uncached because their block alone was over the budget, or the
    /// device had no memory for it.
    pub uncached: u64,
    pub evictions: u64,
    /// Prompt tokens not recomputed on hits.
    pub saved_tokens: u64,
}

/// Which keys to keep within a byte budget, least recently used first out.
#[derive(Debug)]
pub struct Lru<K> {
    budget: usize,
    bytes: usize,
    tick: u64,
    sizes: HashMap<K, (usize, u64)>,
}

impl<K: std::hash::Hash + Eq + Clone> Lru<K> {
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            bytes: 0,
            tick: 0,
            sizes: HashMap::new(),
        }
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn len(&self) -> usize {
        self.sizes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sizes.is_empty()
    }

    /// Mark `key` used; false when it is not kept.
    pub fn touch(&mut self, key: &K) -> bool {
        self.tick += 1;
        match self.sizes.get_mut(key) {
            Some((_, used)) => {
                *used = self.tick;
                true
            }
            None => false,
        }
    }

    /// Drop `key` without counting an eviction.
    pub fn remove(&mut self, key: &K) {
        if let Some((freed, _)) = self.sizes.remove(key) {
            self.bytes -= freed;
        }
    }

    /// Make room for `size` more bytes; returns the keys to drop, oldest first, or
    /// None when `size` alone is over the budget (nothing is dropped then).
    pub fn admit(&mut self, key: K, size: usize) -> Option<Vec<K>> {
        if size > self.budget {
            return None;
        }
        let mut dropped = Vec::new();
        while self.bytes + size > self.budget {
            let oldest = self
                .sizes
                .iter()
                .min_by_key(|(_, (_, used))| *used)
                .map(|(k, _)| k.clone())
                .expect("bytes are held only by kept keys");
            let (freed, _) = self.sizes.remove(&oldest).unwrap();
            self.bytes -= freed;
            dropped.push(oldest);
        }
        self.tick += 1;
        self.bytes += size;
        self.sizes.insert(key, (size, self.tick));
        Some(dropped)
    }
}

/// Prefix states keyed by the exact tokens they cover.
pub struct PrefixCache {
    lru: Lru<Vec<u32>>,
    states: HashMap<Vec<u32>, PrefixState>,
    stats: PrefixCacheStats,
}

impl PrefixCache {
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            lru: Lru::new(budget_bytes),
            states: HashMap::new(),
            stats: PrefixCacheStats {
                budget_bytes,
                ..Default::default()
            },
        }
    }

    pub fn stats(&self) -> PrefixCacheStats {
        PrefixCacheStats {
            bytes: self.lru.bytes(),
            entries: self.lru.len(),
            ..self.stats
        }
    }

    /// The final-norm hidden state at the last token of `ids`. `checkpoint` is where
    /// the worker's leading block ends: a multiple of 64 below `ids.len()`, or 0 for
    /// none. The prompt continues from the cached state of `ids[..checkpoint]`, or
    /// captures it first; with no checkpoint, or a block over the budget, it runs as
    /// one `forward_fixed` pass.
    pub fn forward(
        &mut self,
        model: &mut Model,
        ids: &[u32],
        checkpoint: usize,
    ) -> Result<Vec<f32>> {
        if checkpoint == 0 || !checkpoint.is_multiple_of(64) || checkpoint >= ids.len() {
            return model.forward_fixed(ids);
        }
        let (block, rest) = ids.split_at(checkpoint);
        if self.lru.touch(&block.to_vec()) {
            let last = model.continue_fixed(rest, &self.states[block])?;
            self.stats.hits += 1;
            self.stats.saved_tokens += checkpoint as u64;
            return Ok(last);
        }
        let size = model.prefix_bytes(checkpoint);
        let Some(dropped) = self.lru.admit(block.to_vec(), size) else {
            self.stats.uncached += 1;
            return model.forward_fixed(ids);
        };
        self.stats.evictions += dropped.len() as u64;
        for key in &dropped {
            self.states.remove(key);
        }
        // The budget counts only cached states; the device may run out first.
        let Ok(mut state) = model.alloc_prefix(checkpoint) else {
            self.lru.remove(&block.to_vec());
            self.stats.uncached += 1;
            return model.forward_fixed(ids);
        };
        if let Err(e) = model.capture_fixed(block, &mut state) {
            self.lru.remove(&block.to_vec());
            return Err(e);
        }
        self.stats.misses += 1;
        let last = model.continue_fixed(rest, &state);
        self.states.insert(block.to_vec(), state);
        last
    }
}

#[cfg(test)]
#[path = "../../../../../tests/qwen3_5/prefix_cache.rs"]
mod tests;
