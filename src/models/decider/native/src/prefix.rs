//! Request-local exact token prefix planning; no state is cached across requests.
//! The planner is shared with other Qwen3.5 workers.
pub use omni_qwen3_5_native::prefix::{PrefixMode, PrefixPlan};

impl omni_qwen3_5_native::prefix::PrefixRow for crate::RowInput {
    fn ids(&self) -> &[u32] {
        &self.ids
    }

    fn same_group(&self, other: &Self) -> bool {
        self.question_id == other.question_id
    }
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct PrefixStats {
    pub shared_requests: u64,
    pub auto_independent_requests: u64,
    pub saved_tokens: u64,
}
