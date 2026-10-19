//! The encoder-plus-head execution plan as pure data: kernel names and
//! buffer roles, before any CUDA symbol or weight pointer is resolved.
//! Keeping the plan pure lets CPU fixtures pin the exact step sequence.

/// Scratch buffers the plan addresses; allocation lives in the executor.
#[derive(Clone, Copy)]
pub enum Scratch {
    Ids,
    Lens,
    Types,
    X,
    Y,
    Qkv,
    O,
    G,
    Ff,
}

impl Scratch {
    pub fn name(self) -> &'static str {
        match self {
            Self::Ids => "ids",
            Self::Lens => "lens",
            Self::Types => "types",
            Self::X => "x",
            Self::Y => "y",
            Self::Qkv => "qkv",
            Self::O => "o",
            Self::G => "g",
            Self::Ff => "ff",
        }
    }
}

#[derive(Clone)]
pub enum PlannedArg {
    Weight(String),
    Scratch(Scratch),
}

#[derive(Clone)]
pub enum PlannedStep {
    Launch {
        kernel: String,
        args: Vec<PlannedArg>,
    },
    Dump {
        name: String,
        slot: Scratch,
        bf: bool,
    },
}

impl PlannedStep {
    /// One-line canonical rendering used by the plan fixtures.
    pub fn render(&self) -> String {
        match self {
            Self::Launch { kernel, args } => {
                let mut line = format!("launch {kernel}");
                for arg in args {
                    line.push(' ');
                    line.push_str(&match arg {
                        PlannedArg::Weight(name) => name.clone(),
                        PlannedArg::Scratch(slot) => slot.name().to_string(),
                    });
                }
                line
            }
            Self::Dump { name, slot, bf } => {
                format!("dump {name} {} {}", slot.name(), u8::from(*bf))
            }
        }
    }
}

/// The single supported Laya encoder: 28 layers with full attention
/// every third layer, the task head's two transformer layers, and the
/// readout tail left to eager launches at inference time.
pub fn plan_encoder(original_rope: bool) -> Vec<PlannedStep> {
    let steps = std::cell::RefCell::new(Vec::new());
    let weight = |name: &str| PlannedArg::Weight(name.to_string());
    let scratch = |slot: Scratch| PlannedArg::Scratch(slot);
    let call = |kernel: &str, args: &[PlannedArg]| {
        steps.borrow_mut().push(PlannedStep::Launch {
            kernel: kernel.to_string(),
            args: args.to_vec(),
        });
    };
    let dump = |name: &str, slot: Scratch, bf: bool| {
        steps.borrow_mut().push(PlannedStep::Dump {
            name: name.to_string(),
            slot,
            bf,
        });
    };
    let attention = |label: &str| format!("attn_{label}");
    call(
        "embed",
        &[
            scratch(Scratch::Ids),
            weight("encoder.embeddings.tok_embeddings.weight"),
            weight("encoder.embeddings.norm.weight"),
            scratch(Scratch::X),
            scratch(Scratch::Y),
        ],
    );
    dump("embedding", Scratch::X, false);
    for i in 0..28 {
        let p = format!("encoder.layers.{i}");
        let w = |n: &str| weight(&format!("{p}.{n}"));
        call(
            "qkv",
            &[
                scratch(Scratch::Y),
                w("attn.Wqkv.weight"),
                weight("zeros.3072"),
                scratch(Scratch::Qkv),
            ],
        );
        let kind = if i % 3 == 0 { "full" } else { "local" };
        call(
            if original_rope {
                "rope_original"
            } else {
                "rope"
            },
            &[
                scratch(Scratch::Qkv),
                weight(&format!("rope_{kind}_cos")),
                weight(&format!("rope_{kind}_sin")),
            ],
        );
        call(
            &attention(if i % 3 == 0 { "full" } else { "local" }),
            &[
                scratch(Scratch::Qkv),
                scratch(Scratch::Lens),
                scratch(Scratch::O),
            ],
        );
        call(
            "out",
            &[
                scratch(Scratch::O),
                w("attn.Wo.weight"),
                weight("zeros.1024"),
                scratch(Scratch::Y),
            ],
        );
        call(
            "addln",
            &[
                scratch(Scratch::X),
                scratch(Scratch::Y),
                w("mlp_norm.weight"),
                weight("zeros.1024"),
                scratch(Scratch::Y),
            ],
        );
        call(
            "geglu",
            &[
                scratch(Scratch::Y),
                w("mlp.Wi.weight"),
                scratch(Scratch::G),
            ],
        );
        call(
            "down",
            &[
                scratch(Scratch::G),
                w("mlp.Wo.weight"),
                weight("zeros.1024"),
                scratch(Scratch::Y),
            ],
        );
        let next = if i < 27 {
            weight(&format!("encoder.layers.{}.attn_norm.weight", i + 1))
        } else {
            weight("encoder.final_norm.weight")
        };
        call(
            "addln",
            &[
                scratch(Scratch::X),
                scratch(Scratch::Y),
                next,
                weight("zeros.1024"),
                scratch(Scratch::Y),
            ],
        );
        if [0, 1, 2, 27].contains(&i) {
            dump(&format!("encoder{i}_residual"), Scratch::X, false);
            dump(&format!("encoder{i}_normalized"), Scratch::Y, true);
        }
    }
    call(
        "type",
        &[
            scratch(Scratch::Y),
            weight("type_emb.weight"),
            scratch(Scratch::Types),
            scratch(Scratch::X),
        ],
    );
    for i in 0..2 {
        let p = format!("head.layers.{i}");
        let w = |n: &str| weight(&format!("{p}.{n}"));
        call(
            "ln_bias",
            &[
                scratch(Scratch::X),
                scratch(Scratch::Y),
                w("norm1.weight"),
                w("norm1.bias"),
                scratch(Scratch::Y),
            ],
        );
        call(
            "head_in",
            &[
                scratch(Scratch::Y),
                w("self_attn.in_proj_weight"),
                w("self_attn.in_proj_bias"),
                scratch(Scratch::Qkv),
            ],
        );
        call(
            &attention("full"),
            &[
                scratch(Scratch::Qkv),
                scratch(Scratch::Lens),
                scratch(Scratch::O),
            ],
        );
        call(
            "head_out",
            &[
                scratch(Scratch::O),
                w("self_attn.out_proj.weight"),
                w("self_attn.out_proj.bias"),
                scratch(Scratch::Y),
            ],
        );
        call(
            "addln_bias",
            &[
                scratch(Scratch::X),
                scratch(Scratch::Y),
                w("norm2.weight"),
                w("norm2.bias"),
                scratch(Scratch::Y),
            ],
        );
        call(
            "ffn1",
            &[
                scratch(Scratch::Y),
                w("linear1.weight"),
                w("linear1.bias"),
                scratch(Scratch::Ff),
            ],
        );
        call(
            "ffn2",
            &[
                scratch(Scratch::Ff),
                w("linear2.weight"),
                w("linear2.bias"),
                scratch(Scratch::Y),
            ],
        );
        call(
            "residual",
            &[scratch(Scratch::X), scratch(Scratch::Y)],
        );
    }
    steps.into_inner()
}

/// Renders a full plan, one step per line, for fixture comparison.
pub fn render_plan(plan: &[PlannedStep]) -> String {
    plan.iter()
        .map(PlannedStep::render)
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}
