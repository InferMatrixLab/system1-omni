//! Kev-4B: independent processing and calibrated answers around shared Qwen CUDA execution.
pub mod checkpoint;
pub mod contract;
pub mod engine;
pub mod executor;
pub mod head;
pub mod http;
mod planning;
pub mod postprocess;
pub mod processing;

#[cfg(test)]
#[path = "../../../../../tests/kev/native.rs"]
mod native_tests;
