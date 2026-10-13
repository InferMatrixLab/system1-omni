//! pplx-decider-v1.1-27b's decision contract: request compilation into one prompt per question, the answer codes
//! and their readout rows, and the typed answers computed from the readout's calibrated probabilities. CPU only;
//! the executor (the Qwen3.8 backbone with noncausal full attention and the 255-row readout) is a separate step.
pub mod contract;
pub mod processing;
