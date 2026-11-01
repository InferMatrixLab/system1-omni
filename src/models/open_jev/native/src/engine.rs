//! Assemble the independent processor and model executor from one pinned export.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use omni_qwen3_5_native::prefix::PrefixMode;
use omni_runtime::SerialScheduler;
use serde_json::Value;

use crate::contract::Checkpoint;
use crate::executor::{DecisionHead, Executor};
use crate::processing::Processor;

pub struct Engine {
    pub checkpoint: &'static Checkpoint,
    pub processor: Processor,
    pub scheduler: SerialScheduler,
    pub executor: Executor,
}

/// `OPEN_JEV_PREFIX` (0, 1 or auto) and `OPEN_JEV_FIXED` (0 or 1), with the meaning of
/// Decider's `DECIDER_PREFIX` and `DECIDER_FIXED`.
pub fn prefix_values(prefix: Option<&str>, fixed: Option<&str>) -> Result<PrefixMode> {
    let prefix = match prefix {
        None | Some("0") => PrefixMode::Off,
        Some("1") => PrefixMode::Shared,
        Some("auto") => PrefixMode::Auto,
        Some(_) => bail!("OPEN_JEV_PREFIX must be 0, 1 or auto"),
    };
    let fixed = match fixed {
        None | Some("0") => false,
        Some("1") => true,
        Some(_) => bail!("OPEN_JEV_FIXED must be 0 or 1"),
    };
    ensure!(
        !(prefix != PrefixMode::Off && fixed),
        "select only one of OPEN_JEV_PREFIX and OPEN_JEV_FIXED"
    );
    Ok(if fixed { PrefixMode::Fixed } else { prefix })
}

/// Like Decider's, the fixed and shared paths run eagerly, so they are refused with
/// CUDA Graphs on.
pub fn check_graph(prefix: PrefixMode, graph: bool) -> Result<()> {
    ensure!(
        prefix == PrefixMode::Off || !graph,
        "OPEN_JEV_PREFIX and OPEN_JEV_FIXED require CUA_S1_GRAPH off"
    );
    Ok(())
}

pub fn prefix_env() -> Result<PrefixMode> {
    let get = |name| match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error).context(name),
    };
    prefix_values(
        get("OPEN_JEV_PREFIX")?.as_deref(),
        get("OPEN_JEV_FIXED")?.as_deref(),
    )
}

impl Engine {
    pub async fn load(dir: &Path, library: &Path) -> Result<Self> {
        Self::load_with_prefix(dir, library, prefix_env()?).await
    }

    pub async fn load_with_prefix(
        dir: &Path,
        library: &Path,
        prefix_mode: PrefixMode,
    ) -> Result<Self> {
        // `Model::load` reads the switch the same way.
        let graph = std::env::var("CUA_S1_GRAPH").as_deref() == Ok("1");
        check_graph(prefix_mode, graph)?;
        let manifest: Value = serde_json::from_slice(
            &std::fs::read(dir.join("open_jev_export.json"))
                .context("export the merged checkpoint; see recipe/open_jev/native.md")?,
        )?;
        let checkpoint = Checkpoint::from_export(&manifest)?;
        let temperature = manifest["temperature"].as_f64().context("temperature")?;
        ensure!(
            temperature.is_finite() && temperature > 0.0,
            "invalid temperature"
        );
        let max_length = manifest["max_length"].as_u64().context("max_length")? as usize;
        ensure!(
            (1..=16384).contains(&max_length),
            "max_length must be within 1..=16384"
        );
        let prefix = manifest["chat_prefix"]
            .as_str()
            .context("chat_prefix")?
            .to_owned();
        let suffix = manifest["chat_suffix"]
            .as_str()
            .context("chat_suffix")?
            .to_owned();
        let head = DecisionHead::load(dir, &manifest, checkpoint)?;
        let processor = Processor::load(dir, checkpoint, prefix, suffix, temperature, max_length)?;
        let executor = Executor::load(dir, library, head, checkpoint, prefix_mode, graph).await?;
        Ok(Self {
            checkpoint,
            processor,
            scheduler: SerialScheduler::default(),
            executor,
        })
    }
}
