//! The Vela 2.0 0.3B model as a ModernBERT profile on the family core:
//! bundle config parsing, profile admission, the model-local checkpoint
//! inventory, and the YaRN-scaled rotary tables. Weight upload and
//! encoder execution land with the regenerated CUDA bundle; everything
//! here is CPU-verifiable against the pinned checkpoint's own files.

pub mod config;
pub mod profile;
pub mod rope;
pub mod weights;

#[cfg(feature = "cuda")]
pub mod executor;

pub use config::read_structure;
pub use profile::profile;
pub use rope::{YarnTables, cos_sin_tables, yarn};
pub use weights::{Weights, head_tensors, storage_dtype};
