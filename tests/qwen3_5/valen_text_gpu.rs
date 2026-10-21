//! Device-gated layer-level Valen text parity (spec 03, N2): the base GEMM +
//! unmerged FP32 LoRA application of the shared machinery against the pinned
//! reference's own projections, on the reference's recorded activations.
//!
//! The fixture (`recipe/valen/fixtures/text-choice-reference.json`, recorded
//! by `recipe/valen/dump_text_reference.py`) carries, for the first GDN layer
//! (0) and the first full-attention layer (3) of a compiled text-choice
//! branch, the BF16 input each projection family consumes and the tail of the
//! reference's peft-adapted output. This test uploads the recorded input into
//! the model's scratch, runs the production `gemm_group`/`gemm_sequences` +
//! `apply_lora` path with the real export weights, and compares the output
//! tail elementwise under the fixture's declared projection tolerance
//! (measured first; `null` refuses to pass).
//!
//! Recorded measurement (RTX 5060 Laptop GPU, sm_120, BF16, torch
//! 2.11.0+cu128, peft 0.18.1, TF32 off, 2026-10-21): every projection family
//! agrees with the reference's peft-adapted output within one BF16 ulp per
//! element — in_proj_qkv max abs 0.03125 at output scale 30.4, q_proj
//! 0.015625 at 12.4, out_proj 0.0078 at 1.8, o_proj/down 0.00049, and the
//! fused gate/up projection bit-exact over all 98304 tail elements. The
//! declared tolerance 0.0625 doubles the measured worst case.
use super::*;
use crate::lora::Group;
use serde_json::Value;
use std::path::PathBuf;

fn fixture() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../recipe/valen/fixtures/text-choice-reference.json");
    let fixture: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(fixture["format"], "valen-text-reference/1");
    fixture
}

fn hex_bytes(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex digit"))
        .collect()
}

fn env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var_os(name).expect(name))
}

/// One projection-family check: recorded input in, native application, the
/// reference tail out.
struct Case {
    /// The fixture keys whose recorded outputs concatenate, in order, into the
    /// native fused output.
    outputs: Vec<String>,
    group: Group,
    layer: usize,
    /// (input scratch offset name, output scratch offset name).
    buffers: (&'static str, &'static str),
    /// Input width in elements.
    input_width: usize,
    /// Fused-group base projection (gemm_group) or sequence group
    /// (gemm_sequences).
    fused: bool,
}

#[test]
#[ignore = "requires CUDA, VALEN_EXPORT and VALEN_CUDA_LIB"]
fn valen_projections_match_the_recorded_reference() {
    let fixture = fixture();
    let mut measured: Vec<(String, f32)> = Vec::new();
    let tokens = fixture["tokens"].as_u64().unwrap() as usize;
    let tail: usize = fixture["projections"]["layers.0.mlp.down_proj"]["output_tail_rows"]
        .as_u64()
        .unwrap() as usize;

    let export = env_path("VALEN_EXPORT");
    let mut model = Model::load_with_lora(
        &export,
        &env_path("VALEN_CUDA_LIB"),
        &export.join("lora.safetensors"),
        2.0,
    )
    .unwrap();
    model.enable_valen_text_numerics().unwrap();
    model.prepare_scratch(tokens).unwrap();

    let fused_weight = |model: &Model, layer: usize, group: Group| -> Tensor {
        let layer = &model.layers[layer];
        match (&layer.mixer, group) {
            (Mixer::Linear(la), Group::LinearInput) => la.in_proj.clone(),
            (Mixer::Full(fa), Group::AttentionInput) => fa.qkv.clone(),
            (_, Group::GateUp) => layer.gate_up.clone(),
            (Mixer::Linear(la), Group::LinearOutput) => la.out.clone(),
            (Mixer::Full(fa), Group::AttentionOutput) => fa.o.clone(),
            (_, Group::Down) => layer.down.clone(),
            _ => unreachable!("mixer/group mismatch"),
        }
    };

    let cases = [
        Case {
            outputs: (0..4)
                .map(|i| {
                    [
                        "layers.0.linear_attn.in_proj_qkv",
                        "layers.0.linear_attn.in_proj_z",
                        "layers.0.linear_attn.in_proj_b",
                        "layers.0.linear_attn.in_proj_a",
                    ][i]
                        .to_string()
                })
                .collect(),
            group: Group::LinearInput,
            layer: 0,
            buffers: ("x", "gdn_in"),
            input_width: model.cfg.hidden,
            fused: true,
        },
        Case {
            outputs: vec!["layers.0.linear_attn.out_proj".to_string()],
            group: Group::LinearOutput,
            layer: 0,
            buffers: ("ln", "delta"),
            input_width: model.cfg.value_dim(),
            fused: false,
        },
        Case {
            outputs: vec![
                "layers.3.self_attn.q_proj".to_string(),
                "layers.3.self_attn.k_proj".to_string(),
                "layers.3.self_attn.v_proj".to_string(),
            ],
            group: Group::AttentionInput,
            layer: 3,
            buffers: ("x", "attn_in"),
            input_width: model.cfg.hidden,
            fused: true,
        },
        Case {
            outputs: vec!["layers.3.self_attn.o_proj".to_string()],
            group: Group::AttentionOutput,
            layer: 3,
            buffers: ("ao", "delta"),
            input_width: model.cfg.heads * model.cfg.head_dim,
            fused: false,
        },
        Case {
            outputs: vec![
                "layers.0.mlp.gate_proj".to_string(),
                "layers.0.mlp.up_proj".to_string(),
            ],
            group: Group::GateUp,
            layer: 0,
            buffers: ("x", "gate_up"),
            input_width: model.cfg.hidden,
            fused: true,
        },
        Case {
            outputs: vec!["layers.0.mlp.down_proj".to_string()],
            group: Group::Down,
            layer: 0,
            buffers: ("act", "delta"),
            input_width: model.cfg.intermediate,
            fused: false,
        },
    ];

    for case in &cases {
        let projections = &fixture["projections"];
        let input_key = &case.outputs[0];
        let input_bytes = hex_bytes(projections[input_key]["input_bf16_hex"].as_str().unwrap());
        assert_eq!(
            input_bytes.len(),
            tokens * case.input_width * 2,
            "{input_key}: recorded input width"
        );
        // The recorded per-component tails assemble row-major into the fused
        // layout: for each tail row, each component's row in concat order.
        let component_tails: Vec<Vec<u8>> = case
            .outputs
            .iter()
            .map(|key| {
                let entry = &projections[key];
                assert_eq!(
                    entry["output_tail_rows"].as_u64().unwrap() as usize,
                    tail,
                    "{key}: tail length"
                );
                hex_bytes(entry["output_tail_bf16_hex"].as_str().unwrap())
            })
            .collect();
        let component_widths: Vec<usize> = case
            .outputs
            .iter()
            .map(|key| projections[key]["output_shape"][1].as_u64().unwrap() as usize)
            .collect();
        let output_width: usize = component_widths.iter().sum();
        let mut expected = Vec::with_capacity(tail * output_width * 2);
        for row in 0..tail {
            for (bytes, width) in component_tails.iter().zip(&component_widths) {
                expected.extend_from_slice(&bytes[row * width * 2..(row + 1) * width * 2]);
            }
        }
        assert_eq!(expected.len(), tail * output_width * 2);

        let s = model.scratch.as_ref().unwrap();
        let (input, output) = match case.buffers {
            ("x", "gdn_in") => (s.x, s.gdn_in),
            ("ln", "delta") => (s.ln, s.delta),
            ("x", "attn_in") => (s.x, s.attn_in),
            ("ao", "delta") => (s.ao, s.delta),
            ("x", "gate_up") => (s.x, s.gate_up),
            ("act", "delta") => (s.act, s.delta),
            other => unreachable!("{other:?}"),
        };
        // SAFETY: the scratch buffer at `input` holds tokens rows of the
        // recorded width; the test serializes the model on its stream.
        unsafe {
            cuda::upload(s.at(input), &input_bytes, model.stream).unwrap();
        }
        let weight = fused_weight(&model, case.layer, case.group);
        if case.fused {
            model
                .gemm_group(s, input, &weight, output, tokens, &[tokens], case.group)
                .unwrap();
        } else {
            model
                .gemm_sequences(s, input, &weight, output, &[tokens])
                .unwrap();
        }
        model
            .apply_lora(case.layer, case.group, s, input, output, &[tokens])
            .unwrap();
        let mut actual = vec![0u8; expected.len()];
        // SAFETY: the output buffer holds tokens rows of the fused width.
        unsafe {
            cuda::download(
                &mut actual,
                s.at(output + (tokens - tail) * output_width * 2),
                model.stream,
            )
            .unwrap();
        }
        let tolerance = fixture["projection_tolerance"].as_f64().map(|t| t as f32);
        let mut max = 0.0f32;
        let mut scale = 0.0f32;
        let mut differing = 0usize;
        for index in 0..tail * output_width {
            let a = half::bf16::from_le_bytes(actual[index * 2..index * 2 + 2].try_into().unwrap())
                .to_f32();
            let e =
                half::bf16::from_le_bytes(expected[index * 2..index * 2 + 2].try_into().unwrap())
                    .to_f32();
            let delta = (a - e).abs();
            if delta > 0.0 {
                differing += 1;
            }
            if delta > max {
                max = delta;
            }
            if e.abs() > scale {
                scale = e.abs();
            }
            assert!(
                tolerance.is_none_or(|t| delta <= t),
                "{input_key} tail element {index}: native {a} vs reference {e} (delta {delta})"
            );
        }
        eprintln!(
            "valen projection {input_key} ({:?}): max abs {max:.6} at output scale {scale:.3}, {differing}/{} differ, tolerance {tolerance:?}",
            case.group,
            tail * output_width
        );
        measured.push((input_key.clone(), max));
    }
    model.synchronize().unwrap();
    if fixture["projection_tolerance"].is_null() {
        let worst = measured.iter().fold(0.0f32, |acc, (_, m)| acc.max(*m));
        panic!(
            "the fixture declares no projection tolerance; measured native max abs \
             per projection: {measured:?} (worst {worst:.6}). Declare with \
             dump_text_reference.py --projection-tolerance"
        );
    }
}
