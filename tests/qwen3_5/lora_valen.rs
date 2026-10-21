//! The Valen-Preview unmerged adapter layout (spec 03, N2): the peft 0.18.1
//! checkpoint stores language adapters as
//! `backbone.language_model.base_model.model.<target>.lora_{A,B}.default.weight`
//! — a different prefix and a named-adapter suffix than the Cua-S1 export.
//! These tests pin that the shared loader accepts exactly that layout, maps it
//! onto the same config-driven inventory, and applies it with the manifest's
//! declared formula (base GEMM in BF16, delta in FP32, one final rounding).
use super::*;
use safetensors::{
    Dtype,
    tensor::{TensorView, serialize},
};

/// `PEFT_LANGUAGE_PREFIX` in recipe/valen/export_checkpoint.py.
const VALEN_PREFIX: &str = "backbone.language_model.base_model.model.";

fn config() -> Config {
    Config::load(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../../tests/qwen3_5/data"
    )))
    .unwrap()
}

/// The Valen 2B layer mix in miniature: 18 GDN + 6 full-attn over 24 layers
/// becomes 3 + 1 over 4 layers; every target family is present.
fn valen_config() -> Config {
    let mut cfg = config();
    cfg.hidden = 8;
    cfg.intermediate = 8;
    cfg.heads = 1;
    cfg.kv_heads = 1;
    cfg.lin_k_heads = 1;
    cfg.lin_v_heads = 1;
    cfg.full_attention = vec![false, false, false, true];
    cfg
}

fn tensors(values: &[(&str, Vec<usize>, Dtype, f32)]) -> Vec<u8> {
    let data: Vec<Vec<u8>> = values
        .iter()
        .map(|(_, shape, dtype, value)| {
            let count: usize = shape.iter().product();
            match dtype {
                Dtype::F32 => value.to_le_bytes().repeat(count),
                Dtype::BF16 => half::bf16::from_f32(*value).to_le_bytes().repeat(count),
                _ => unreachable!(),
            }
        })
        .collect();
    let views: Vec<_> = values
        .iter()
        .zip(&data)
        .map(|((name, shape, dtype, _), bytes)| {
            (
                *name,
                TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        })
        .collect();
    serialize(views, None).unwrap()
}

/// A complete paired adapter in the Valen key layout for every inventory target.
fn valen_adapter(cfg: &Config, rank: usize, a: f32, b: f32) -> Vec<u8> {
    let values = targets(cfg)
        .unwrap()
        .iter()
        .flat_map(|target| {
            [
                (
                    format!("{VALEN_PREFIX}{}.lora_A.default.weight", target.name),
                    vec![rank, target.cols],
                    Dtype::F32,
                    a,
                ),
                (
                    format!("{VALEN_PREFIX}{}.lora_B.default.weight", target.name),
                    vec![target.rows, rank],
                    Dtype::F32,
                    b,
                ),
            ]
        })
        .collect::<Vec<_>>();
    let borrowed: Vec<_> = values
        .iter()
        .map(|(name, shape, dtype, value)| (name.as_str(), shape.clone(), *dtype, *value))
        .collect();
    tensors(&borrowed)
}

/// The Valen export's target census: 18 GDN layers x 5 linear_attn targets,
/// 6 full-attn layers x 4 q/k/v/o targets, 24 layers x 3 MLP targets.
#[test]
fn valen_inventory_matches_the_manifest_census() {
    let mut cfg = config();
    // The exported layer mix: full attention at 3, 7, 11, 15, 19, 23.
    cfg.full_attention = vec![false; 24];
    for layer in (3..24).step_by(4) {
        cfg.full_attention[layer] = true;
    }
    let layers = cfg.full_attention.clone();
    let inventory = targets(&cfg).unwrap();
    assert_eq!(inventory.len(), 186);
    assert_eq!(layers.iter().filter(|&&f| f).count(), 6);
    for group_counts in [
        (Group::LinearInput, 18 * 4),
        (Group::LinearOutput, 18),
        (Group::AttentionInput, 6 * 3),
        (Group::AttentionOutput, 6),
        (Group::GateUp, 24 * 2),
        (Group::Down, 24),
    ] {
        assert_eq!(
            inventory
                .iter()
                .filter(|t| t.group == group_counts.0)
                .count(),
            group_counts.1,
            "{:?}",
            group_counts.0
        );
    }
}

#[test]
fn valen_layout_parses_the_complete_adapter_with_the_same_inventory() {
    let cfg = valen_config();
    let bytes = valen_adapter(&cfg, 1, 0.125, -0.25);
    let parsed = parse(&cfg, &bytes, 2.0).unwrap();
    assert_eq!(parsed.pairs.len(), 31);
    assert_eq!(parsed.base_width, 512);
    let find = |name: &str| {
        parsed
            .pairs
            .iter()
            .map(|p| &p.target)
            .find(|t| t.name == name)
            .unwrap()
    };
    // The named-adapter suffix must not leak into the target name, and the
    // fused-slice coordinates stay exactly the Cua-S1 ones.
    let a = find("layers.2.linear_attn.in_proj_a");
    assert_eq!(
        (a.rows, a.cols, a.offset, a.stride, a.group),
        (1, 8, 513, 514, Group::LinearInput)
    );
    let k = find("layers.3.self_attn.k_proj");
    assert_eq!((k.rows, k.cols, k.offset, k.stride), (256, 8, 512, 1024));
    let up = find("layers.1.mlp.up_proj");
    assert_eq!(
        (up.rows, up.cols, up.offset, up.stride, up.group),
        (8, 8, 8, 16, Group::GateUp)
    );
    let out = find("layers.0.linear_attn.out_proj");
    assert_eq!(out.group, Group::LinearOutput);
    assert_eq!(
        find("layers.3.self_attn.o_proj").group,
        Group::AttentionOutput
    );
    assert_eq!(find("layers.0.mlp.down_proj").group, Group::Down);
    // Rank and FP32 bytes survive the named layout unchanged.
    let pair = parsed
        .pairs
        .iter()
        .find(|p| p.target.name == "layers.0.mlp.down_proj")
        .unwrap();
    assert_eq!(pair.rank, 1);
    assert_eq!(&bytes[pair.a.clone()][..4], &0.125f32.to_le_bytes());
    assert_eq!(&bytes[pair.b.clone()][..4], &(-0.25f32).to_le_bytes());
}

#[test]
fn valen_layout_rejects_foreign_prefix_suffix_combinations() {
    let cfg = valen_config();
    let layout = targets(&cfg).unwrap();
    let target = layout
        .iter()
        .find(|t| t.name == "layers.0.mlp.down_proj")
        .unwrap();
    let shape = |letter: usize| {
        if letter == 0 {
            vec![1, target.cols]
        } else {
            vec![target.rows, 1]
        }
    };
    let names = [
        // Valen prefix without the named-adapter component.
        format!("{VALEN_PREFIX}layers.0.mlp.down_proj.lora_A.weight"),
        // Valen prefix with a non-default adapter name.
        format!("{VALEN_PREFIX}layers.0.mlp.down_proj.lora_A.other.weight"),
        // Cua-S1 prefix carrying the named suffix.
        format!("{PREFIX}layers.0.mlp.down_proj.lora_A.default.weight"),
        // A visual-tower prefix must never enter the language inventory.
        "backbone.visual.blocks.20.mlp.linear_fc1.lora_A.default.weight".to_string(),
    ];
    for name in names {
        let bytes = tensors(&[
            (&name, shape(0), Dtype::F32, 0.),
            (&name.replace("lora_A", "lora_B"), shape(1), Dtype::F32, 0.),
        ]);
        let error = parse(&cfg, &bytes, 2.0)
            .err()
            .unwrap_or_else(|| panic!("{name} must be rejected"));
        assert!(
            error.to_string().contains("unsupported LoRA"),
            "{name}: {error}"
        );
    }
}

/// The manifest application formula, as pinned by the Python suite
/// (tests/valen/test_export.py): the base GEMM runs in BF16, the delta
/// `scaling * ((x.float() @ A.t()) @ B.t())` is FP32 on the activations, the
/// sum upcasts the base output and rounds to BF16 exactly once. The fake CUDA
/// boundary performs that arithmetic on the CPU; this test is the Rust port of
/// the peft-oracle checks over synthetic tensors in the Valen key layout.
#[test]
fn valen_hand_computed_rounding_matches_the_pinned_python_cases() {
    // tests/valen/test_export.py::test_unmerged_output_hand_computed_rounding:
    // base 3.0 + delta 0.0114 crosses the 3.0078125 midpoint -> 3.015625.
    assert_eq!(
        half::bf16::from_f32(3.0f32 + 2.0 * (3.0f32 * 0.0019)),
        half::bf16::from_f32(3.015625)
    );
    // A tie at the 3.0078125 midpoint rounds to even (3.0), per the reference's
    // own downcast; an untrained (zero-B) delta leaves the base bits alone.
    assert_eq!(
        half::bf16::from_f32(3.0f32 + 0.0078125),
        half::bf16::from_f32(3.0)
    );
    assert_eq!(
        half::bf16::from_f32(3.0f32 + 2.0 * 0.0),
        half::bf16::from_f32(3.0)
    );
}

/// The manifest formula through the real dispatch, in the Valen key layout:
/// `y = bf16(fp32(bf16(x @ W.t())) + scaling * ((x.float() @ A.t()) @ B.t()))`.
///
/// The fake CUDA boundary (`lora_fake.c`) performs the CPU arithmetic of the
/// three kernels involved — BF16 base GEMM with FP32 accumulation and one
/// rounding, FP32 adapter GEMMs, and the single final rounding add — so this
/// is the Rust port of `tests/valen/test_export.py`'s peft-oracle checks,
/// executed through the production `project_components` + `Lora::apply` path
/// for one target in every group, over both mixer types.
#[cfg(unix)]
#[test]
fn valen_layout_applies_the_manifest_formula_in_every_group() {
    struct Temporary(std::path::PathBuf);
    impl Temporary {
        fn new() -> Self {
            let unique = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path =
                std::env::temp_dir().join(format!("valen-lora-{}-{unique}", std::process::id()));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Temporary {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let temporary = Temporary::new();
    let library = crate::lora::fake_cuda_library();
    let _guard = crate::lora::FAKE_CUDA_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    cuda::load(&library).unwrap();
    let cfg = valen_config();
    let scale = 2.0f32;
    let (a_value, b_value) = (0.125f32, -0.25f32);
    let bytes = valen_adapter(&cfg, 1, a_value, b_value);
    let path = temporary.0.join("adapter.safetensors");
    std::fs::write(&path, bytes).unwrap();
    let stream = cuda::new_stream().unwrap();
    let checkpoint = Checkpoint::load(&cfg, &path, scale).unwrap();
    let adapter = checkpoint.upload(cfg.full_attention.len(), stream).unwrap();
    let lengths = [1usize, 2usize];
    let rows_total: usize = lengths.iter().sum();
    for (name, group, fused) in [
        ("layers.0.linear_attn.in_proj_a", Group::LinearInput, true),
        ("layers.0.linear_attn.out_proj", Group::LinearOutput, false),
        ("layers.3.self_attn.k_proj", Group::AttentionInput, true),
        ("layers.3.self_attn.o_proj", Group::AttentionOutput, false),
        ("layers.1.mlp.up_proj", Group::GateUp, true),
        ("layers.3.mlp.down_proj", Group::Down, false),
    ] {
        let target = targets(&cfg)
            .unwrap()
            .into_iter()
            .find(|t| t.name == name)
            .unwrap();
        let (cols, stride) = (target.cols, target.stride);
        // Inputs: row r carries the BF16 value (r + 1) across the width.
        let input: Vec<_> = (0..rows_total)
            .flat_map(|r| {
                half::bf16::from_f32((r + 1) as f32)
                    .to_le_bytes()
                    .repeat(cols)
            })
            .collect();
        // Base weights: stacked row j carries the BF16 value (j + 1).
        let weight_rows = if fused {
            component_rows(&cfg, group).unwrap().iter().sum::<usize>()
        } else {
            target.rows
        };
        let weights: Vec<_> = (0..weight_rows)
            .flat_map(|j| {
                half::bf16::from_f32((j + 1) as f32)
                    .to_le_bytes()
                    .repeat(cols)
            })
            .collect();
        let output_bytes = half::bf16::ONE.to_le_bytes().repeat(rows_total * stride);
        let work = adapter.work(rows_total).unwrap();
        let x = DeviceBuffer::new(input.len()).unwrap();
        let w = DeviceBuffer::new(weights.len()).unwrap();
        let y = DeviceBuffer::new(output_bytes.len()).unwrap();
        unsafe {
            cuda::upload(x.at(0), &input, stream).unwrap();
            cuda::upload(w.at(0), &weights, stream).unwrap();
            cuda::upload(y.at(0), &output_bytes, stream).unwrap();
        }
        unsafe {
            if fused {
                let components = component_rows(&cfg, group).unwrap();
                project_components(
                    x.at(0),
                    w.at(0),
                    y.at(0),
                    &lengths,
                    cols,
                    &components,
                    &work,
                    std::ptr::null_mut(),
                    stream,
                )
                .unwrap();
            } else {
                // The sequence groups run one plain BF16 GEMM per prompt, as
                // Model::gemm_sequences does for out_proj/o_proj/down.
                let mut row = 0;
                for &length in &lengths {
                    check(
                        (cuda::api().cs1_gemm)(
                            std::ptr::null_mut(),
                            x.at(row * cols * 2),
                            w.at(0),
                            y.at(row * stride * 2),
                            length as i32,
                            target.rows as i32,
                            cols as i32,
                            stride as i32,
                            stream,
                        ),
                        "base sequence GEMM",
                    )
                    .unwrap();
                    row += length;
                }
            }
            adapter
                .apply(
                    target.layer,
                    group,
                    x.at(0),
                    y.at(0),
                    &lengths,
                    &work,
                    std::ptr::null_mut(),
                    stream,
                )
                .unwrap();
        }
        let mut actual = vec![0; output_bytes.len()];
        unsafe {
            cuda::download(&mut actual, y.at(0), stream).unwrap();
        }
        // The manifest formula, in FP32 with one final rounding. With the
        // synthetic constants every intermediate is exact, so the oracle and
        // the fake kernels must agree bit for bit.
        let mut row = 0;
        for &length in &lengths {
            for r in row..row + length {
                for j in 0..target.rows {
                    let x_value = (r + 1) as f32;
                    // The stacked base rows carry the BF16-rounded weight
                    // value: odd integers past 256 are not representable.
                    let weight = if fused {
                        half::bf16::from_f32((target.offset + j + 1) as f32).to_f32()
                    } else {
                        half::bf16::from_f32((j + 1) as f32).to_f32()
                    };
                    let base = half::bf16::from_f32(x_value * weight * cols as f32).to_f32();
                    let hidden = x_value * cols as f32 * a_value;
                    let delta = hidden * b_value;
                    let expected = half::bf16::from_f32(base + scale * delta);
                    let col = target.offset + j;
                    let offset = (r * stride + col) * 2;
                    let value =
                        half::bf16::from_le_bytes(actual[offset..offset + 2].try_into().unwrap());
                    assert_eq!(value, expected, "{name}, row {r}, col {col}");
                }
            }
            row += length;
        }
        // The complete adapter adapts every component of the fused group: the
        // neighbouring columns carry their own delta under the same formula.
        // (The sequence groups own their whole output row, so nothing remains.)
        for r in 0..rows_total {
            for col in 0..stride {
                if !fused && (target.offset..target.offset + target.rows).contains(&col) {
                    continue;
                }
                let offset = (r * stride + col) * 2;
                let value =
                    half::bf16::from_le_bytes(actual[offset..offset + 2].try_into().unwrap());
                let x_value = (r + 1) as f32;
                let weight = half::bf16::from_f32((col + 1) as f32).to_f32();
                let base = half::bf16::from_f32(x_value * weight * cols as f32).to_f32();
                let delta = x_value * cols as f32 * a_value * b_value;
                let expected = half::bf16::from_f32(base + scale * delta);
                assert_eq!(value, expected, "{name} neighbour, row {r}, col {col}");
            }
        }
        drop((work, x, w, y));
    }
}
