use crate::structure::ModernBertStructure;

/// Scratch slots an encoder plan addresses. Allocating them is the
/// consumer's job; `ids`, `lens`, `x`, `y`, `qkv`, `o` and `gate` map to
/// the encoder scratch of [`crate::geometry::EncoderScratch`], and
/// `types` holds the per-row type ids consumed by the type-embedding
/// kernel when the structure carries one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScratchSlot {
    Ids,
    Lens,
    Types,
    X,
    Y,
    Qkv,
    O,
    Gate,
}

impl ScratchSlot {
    pub fn name(self) -> &'static str {
        match self {
            Self::Ids => "ids",
            Self::Lens => "lens",
            Self::Types => "types",
            Self::X => "x",
            Self::Y => "y",
            Self::Qkv => "qkv",
            Self::O => "o",
            Self::Gate => "g",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanArg {
    /// A weight-buffer name: checkpoint tensors under canonical family
    /// naming, or `zeros.<n>` synthetic zero-bias buffers the consumer
    /// uploads alongside them.
    Weight(String),
    Scratch(ScratchSlot),
}

/// Sequence anchors a consumer can hook instrumentation to; they carry
/// no execution of their own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanAnchor {
    AfterEmbed,
    LayerEnd(usize),
    AfterEncoder,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanStep {
    Launch { kernel: String, args: Vec<PlanArg> },
    Anchor(PlanAnchor),
}

impl PlanStep {
    /// Canonical one-line rendering used by plan fixtures.
    pub fn render(&self) -> String {
        match self {
            Self::Launch { kernel, args } => {
                let mut line = kernel.clone();
                for arg in args {
                    line.push(' ');
                    line.push_str(&match arg {
                        PlanArg::Weight(w) => w.clone(),
                        PlanArg::Scratch(s) => s.name().to_string(),
                    });
                }
                line
            }
            Self::Anchor(a) => format!("anchor {a:?}"),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EncoderPlanOptions {
    /// Select the reference-rotation kernel variant a bundle may carry
    /// beside the shipped tables.
    pub original_rope: bool,
}

/// The named-kernel encoder plan: embedding, per-layer attention and
/// gated MLP, the final norm fused into the last layer's post-MLP norm,
/// and the optional type-embedding add. Kernel names and argument roles
/// only — resolving symbols to loaded kernels and names to device
/// pointers stays with the consuming model crate.
pub fn encoder_plan(s: &ModernBertStructure, opts: &EncoderPlanOptions) -> Vec<PlanStep> {
    let h = s.hidden_size;
    let zero_bias = format!("zeros.{h}");
    let weight = |name: String| PlanArg::Weight(name);
    let scratch = PlanArg::Scratch;
    let mut steps = Vec::new();
    steps.push(PlanStep::Launch {
        kernel: "embed".into(),
        args: vec![
            scratch(ScratchSlot::Ids),
            weight("encoder.embeddings.tok_embeddings.weight".into()),
            weight("encoder.embeddings.norm.weight".into()),
            scratch(ScratchSlot::X),
            scratch(ScratchSlot::Y),
        ],
    });
    steps.push(PlanStep::Anchor(PlanAnchor::AfterEmbed));
    for (i, kind) in s.schedule.iter().enumerate() {
        let p = format!("encoder.layers.{i}");
        let suffix = kind.kernel_suffix();
        steps.push(PlanStep::Launch {
            kernel: "qkv".into(),
            args: vec![
                scratch(ScratchSlot::Y),
                weight(format!("{p}.attn.Wqkv.weight")),
                weight(format!("zeros.{}", 3 * h)),
                scratch(ScratchSlot::Qkv),
            ],
        });
        steps.push(PlanStep::Launch {
            kernel: if opts.original_rope {
                "rope_original".into()
            } else {
                "rope".into()
            },
            args: vec![
                scratch(ScratchSlot::Qkv),
                weight(format!("rope_{suffix}_cos")),
                weight(format!("rope_{suffix}_sin")),
            ],
        });
        steps.push(PlanStep::Launch {
            kernel: format!("attn_{suffix}"),
            args: vec![
                scratch(ScratchSlot::Qkv),
                scratch(ScratchSlot::Lens),
                scratch(ScratchSlot::O),
            ],
        });
        steps.push(PlanStep::Launch {
            kernel: "out".into(),
            args: vec![
                scratch(ScratchSlot::O),
                weight(format!("{p}.attn.Wo.weight")),
                weight(zero_bias.clone()),
                scratch(ScratchSlot::Y),
            ],
        });
        steps.push(PlanStep::Launch {
            kernel: "addln".into(),
            args: vec![
                scratch(ScratchSlot::X),
                scratch(ScratchSlot::Y),
                weight(format!("{p}.mlp_norm.weight")),
                weight(zero_bias.clone()),
                scratch(ScratchSlot::Y),
            ],
        });
        steps.push(PlanStep::Launch {
            kernel: "geglu".into(),
            args: vec![
                scratch(ScratchSlot::Y),
                weight(format!("{p}.mlp.Wi.weight")),
                scratch(ScratchSlot::Gate),
            ],
        });
        steps.push(PlanStep::Launch {
            kernel: "down".into(),
            args: vec![
                scratch(ScratchSlot::Gate),
                weight(format!("{p}.mlp.Wo.weight")),
                weight(zero_bias.clone()),
                scratch(ScratchSlot::Y),
            ],
        });
        let next_norm = if i + 1 < s.schedule.len() {
            format!("encoder.layers.{}.attn_norm.weight", i + 1)
        } else {
            "encoder.final_norm.weight".into()
        };
        steps.push(PlanStep::Launch {
            kernel: "addln".into(),
            args: vec![
                scratch(ScratchSlot::X),
                scratch(ScratchSlot::Y),
                weight(next_norm),
                weight(zero_bias.clone()),
                scratch(ScratchSlot::Y),
            ],
        });
        steps.push(PlanStep::Anchor(PlanAnchor::LayerEnd(i)));
    }
    steps.push(PlanStep::Anchor(PlanAnchor::AfterEncoder));
    if s.type_embedding_rows.is_some() {
        steps.push(PlanStep::Launch {
            kernel: "type".into(),
            args: vec![
                scratch(ScratchSlot::Y),
                weight("type_emb.weight".into()),
                scratch(ScratchSlot::Types),
                scratch(ScratchSlot::X),
            ],
        });
    }
    steps
}
