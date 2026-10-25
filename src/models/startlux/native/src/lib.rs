//! StartLux-Decision v1.1 4B Instant: text decisions on the shared Qwen3.5 prefill.
//! Full-attention layers read each prompt bidirectionally. Linear attention stays causal.
//! Weights are CC BY-NC 4.0 and are not distributed here.

pub mod contract;
pub mod engine;
pub mod executor;
pub mod processing;
