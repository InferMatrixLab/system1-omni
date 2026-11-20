use anyhow::{Context, Result, ensure};
use omni_modernbert::gate::BackendLimits;
use omni_modernbert::structure::ModernBertStructure;
use serde::Deserialize;
use std::{collections::HashMap, fs, path::Path};

use crate::profile::{LayaBundleAdapter, laya_0_3_20};

#[derive(Debug, Deserialize)]
pub struct AgentConfig {
    pub max_len: usize,
    pub head_max_len: usize,
    pub head_layers: usize,
    pub temperature: Vec<f32>,
    pub temperature_by_options: HashMap<String, f32>,
}

#[derive(Debug, Deserialize)]
pub struct EncoderConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub norm_eps: f32,
    pub local_attention: usize,
    pub layer_types: Vec<String>,
    pub rope_parameters: serde_json::Value,
}

impl From<ModernBertStructure> for EncoderConfig {
    fn from(s: ModernBertStructure) -> Self {
        Self {
            hidden_size: s.hidden_size,
            intermediate_size: s.intermediate_size,
            num_attention_heads: s.num_attention_heads,
            num_hidden_layers: s.num_hidden_layers,
            vocab_size: s.vocab_size,
            norm_eps: s.norm_eps,
            local_attention: s.local_window,
            layer_types: s.schedule.iter().map(|k| k.as_str().to_string()).collect(),
            rope_parameters: serde_json::json!({
                "full_attention": {"rope_type": "default", "rope_theta": s.rope_full.theta},
                "sliding_attention": {"rope_type": "default", "rope_theta": s.rope_sliding.theta},
            }),
        }
    }
}

pub struct Config {
    pub agent: AgentConfig,
    pub encoder: EncoderConfig,
    /// The admitted family structure; geometry and plans derive from it.
    pub structure: ModernBertStructure,
    /// The registered profile's request envelope, admitted with the
    /// structure; the workspace bounds come from here, not inline
    /// literals.
    pub limits: BackendLimits,
}
impl Config {
    pub fn load(dir: &Path) -> Result<Self> {
        let read = |name| fs::read(dir.join(name)).with_context(|| format!("read {name}"));
        let agent: AgentConfig = serde_json::from_slice(&read("rl_agent_config.json")?)?;
        let (structure, _meta) = LayaBundleAdapter::read(dir)?;
        let profile = laya_0_3_20();
        profile.admit(&structure)?;
        ensure!(
            agent.max_len == 512 && agent.head_max_len == 192 && agent.head_layers == 2,
            "native Laya supports max_len=512, head_max_len=192, head_layers=2"
        );
        ensure!(
            agent.temperature.len() == 3
                && agent
                    .temperature
                    .iter()
                    .chain(agent.temperature_by_options.values())
                    .all(|t| t.is_finite() && *t > 0.0),
            "invalid temperatures"
        );
        Ok(Self {
            encoder: EncoderConfig::from(structure.clone()),
            agent,
            structure,
            limits: profile.limits,
        })
    }
}
