//! The ModernBERT encoder family layer: structure parameters parsed from
//! checkpoint configs, the geometry derived from them, the capability
//! profiles deciding whether a CUDA bundle can execute a structure, and
//! the named-kernel execution plan for the encoder stack.
//!
//! This crate owns family data only. It never touches CUDA, device
//! memory, or weight files; loading weights, resolving kernel symbols,
//! and every task head stay with the model crates that consume a
//! profile.

pub mod adapter;
pub mod gate;
pub mod geometry;
pub mod plan;
pub mod structure;

pub use adapter::{LayaBundleAdapter, ModernBertConfigAdapter, SourceMeta};
pub use gate::{BackendLimits, RegisteredProfile, StructureIdentity};
pub use plan::{EncoderPlanOptions, PlanAnchor, PlanArg, PlanStep, ScratchSlot};
pub use structure::{
    Activation, AttentionKind, FieldOrigin, ModernBertStructure, RopeSpec, YarnScaling,
    SCHEMA_VERSION,
};
