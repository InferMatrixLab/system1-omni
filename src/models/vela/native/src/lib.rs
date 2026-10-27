//! Native library for vllm-sr/Vela-2.0-0.8B: the exported-bundle gate,
//! the wire contract, and the token-level processing (state → parts,
//! the SystemOne → predict mapping with the pinned runtime's validation
//! triples, word units, the DEV2.0 prompt rendering — mirrored against
//! the frozen `native-plan-*` fixtures).

pub mod bundle;
pub mod contract;
pub mod heads;
pub mod processing;
pub mod questions;
pub mod render;
pub mod scoring;
pub mod state;
pub mod words;
