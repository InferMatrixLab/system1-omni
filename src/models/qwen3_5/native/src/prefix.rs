//! Request-local exact token prefix planning for `Model::forward_shared`; no state is
//! cached across requests.
use crate::model::{PromptGroup, SharedPrompts};

/// One prompt of a request, in request order. Consecutive rows of the same group (a
/// question's candidates, for example) may share a prefix beyond the request's.
pub trait PrefixRow {
    fn ids(&self) -> &[u32];
    fn same_group(&self, other: &Self) -> bool;
}

pub struct PrefixPlan<'a> {
    pub prompts: SharedPrompts<'a>,
    pub saved_tokens: usize,
}

fn common<R: PrefixRow>(rows: &[R], skip: usize) -> usize {
    let first = rows[0].ids();
    let limit = rows
        .iter()
        .map(|row| row.ids().len().saturating_sub(1))
        .min()
        .unwrap();
    (skip..limit)
        .take_while(|&i| rows.iter().all(|row| row.ids()[i] == first[i]))
        .count()
}

impl<'a> PrefixPlan<'a> {
    /// Conservative opt-in policy; savings are aligned token work, not a speed guarantee.
    pub fn worth_auto<R: PrefixRow>(&self, rows: &[R]) -> bool {
        let total = rows.iter().map(|row| row.ids().len()).sum::<usize>();
        self.saved_tokens >= 4096 && self.saved_tokens >= total.div_ceil(3)
    }

    pub fn new<R: PrefixRow>(rows: &'a [R]) -> Option<Self> {
        if rows.len() < 2 || rows.iter().any(|row| row.ids().is_empty()) {
            return None;
        }
        let request = common(rows, 0);
        let p = request / 64 * 64;
        let mut saved_tokens = (rows.len() - 1) * p;
        let mut groups = Vec::new();
        let mut start = 0;
        while start < rows.len() {
            let mut end = start + 1;
            while end < rows.len() && rows[end].same_group(&rows[start]) {
                end += 1;
            }
            let group = &rows[start..end];
            let prefix = if group.len() > 1 {
                common(group, request)
            } else {
                0
            };
            let q = (request + prefix) / 64 * 64;
            saved_tokens += (group.len() - 1) * (q - p);
            groups.push(PromptGroup {
                prefix: &group[0].ids()[request..request + prefix],
                branches: group
                    .iter()
                    .map(|row| &row.ids()[request + prefix..])
                    .collect(),
            });
            start = end;
        }
        (saved_tokens > 0).then(|| Self {
            prompts: SharedPrompts {
                prefix: &rows[0].ids()[..request],
                groups,
            },
            saved_tokens,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PrefixMode {
    #[default]
    Off,
    Fixed,
    Shared,
    Auto,
}
impl PrefixMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Fixed => "fixed",
            Self::Shared => "shared",
            Self::Auto => "auto",
        }
    }
}
