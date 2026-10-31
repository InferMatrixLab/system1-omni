//! Assemble the independent processor and model executor for the native worker.

use std::path::Path;

use anyhow::{Context, Result};
use omni_runtime::SerialScheduler;

use crate::contract::LETTERS;
use crate::executor::{Executor, check_prefix_cache, prefix_cache_bytes};
use crate::processing::Processor;

pub struct Engine {
    pub processor: Processor,
    pub scheduler: SerialScheduler,
    pub executor: Executor,
}

impl Engine {
    pub async fn load(dir: &Path, library: &Path) -> Result<Self> {
        let export = crate::provenance::text_export(dir)?;
        let cache_bytes = match std::env::var("CUA_S1_PREFIX_CACHE_BYTES") {
            Ok(value) => prefix_cache_bytes(Some(&value))?,
            Err(std::env::VarError::NotPresent) => None,
            Err(e) => return Err(e).context("CUA_S1_PREFIX_CACHE_BYTES"),
        };
        if cache_bytes.is_some() {
            // `Model::load` reads the switch the same way.
            check_prefix_cache(export, std::env::var("CUA_S1_GRAPH").as_deref() == Ok("1"))?;
        }
        let mut processor = Processor::load(dir)?;
        processor.checkpoints = cache_bytes.is_some();
        let ids = LETTERS
            .chars()
            .map(|c| {
                processor
                    .tokenizer
                    .token_to_id(&c.to_string())
                    .context("letter token")
            })
            .collect::<Result<Vec<_>>>()?;
        let executor = Executor::load(dir, library, &ids, export, cache_bytes).await?;
        Ok(Self {
            processor,
            scheduler: SerialScheduler::default(),
            executor,
        })
    }
}

#[cfg(test)]
#[path = "../../../../../tests/cua_s1/text_export.rs"]
mod tests;

#[cfg(test)]
#[path = "../../../../../tests/cua_s1/legacy_merged_gpu.rs"]
mod legacy_merged_tests;

#[cfg(test)]
#[path = "../../../../../tests/cua_s1/prefix_cache_gpu.rs"]
mod prefix_cache_tests;
