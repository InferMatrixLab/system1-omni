//! Host-side preparation and artifact validation for the pinned LFM2.5-350M Choice contract.

pub mod artifact;
mod json;
pub mod processing;

pub const MODEL_ALIAS: &str = "lfm2.5-350m";
pub const SERVED_MODEL: &str = "LiquidAI/LFM2.5-350M@9e6c6ccf47cd318696e137d381a7ded8fe4df09f";
pub const VOCAB_SIZE: usize = 65_536;
pub const PAD_TOKEN_ID: u32 = 0;
pub const MAX_CRITERIA: usize = 255;

#[cfg(test)]
#[path = "../../../../../tests/lfm2/native_contract.rs"]
mod tests;
