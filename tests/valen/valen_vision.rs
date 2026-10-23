//! Device-gated Valen vision checks on the shared Qwen vision executor
//! that PR #117 introduced.
//!
//! The reference fixture is recorded by
//! `recipe/valen/dump_vision_reference.py` from the pinned reference tower
//! (Valen @ 750bfcf, Preview checkpoint 81b9c63, Qwen3.5-2B base 15852e8c)
//! on the recipe's `example-request.json` image, and committed as
//! `recipe/valen/fixtures/vision-tower-example-image.json`.
//!
//! Tolerance semantics (declared in the fixture, measured first): the maximum
//! absolute elementwise difference, compared in FP32, between the reference
//! tower output and the native tower output for the same BF16 input bits on
//! the same GPU and the same dtype. The two stacks run different kernel
//! implementations (torch eager cuBLAS vs `cs1_vision_*` cuBLASLt GEMMs and
//! custom attention/rope/norm kernels), so per-element differences at the
//! BF16 rounding scale accumulate over the 24 blocks and the merger; the
//! gate quantifies that drift instead of demanding bit equality.
//!
//! Recorded measurement on the pre-#117 executor (RTX 5060 Laptop GPU,
//! sm_120, BF16, torch 2.11.0+cu128, 2026-10-21): native vs reference max
//! abs 1.03125, mean abs 1.00e-2, 3361/131072 elements bit-exact, reference
//! output scale 2.5. The drift is cross-kernel BF16 noise, not an
//! implementation error: the reference's own attention-kernel choice (eager
//! vs sdpa, everything else identical) drifts by max abs 0.734, mean abs
//! 9.0e-3, 4401/131072 bit-exact — the same class. Per-stage tracing shows
//! the patch embedding bit-exact, the first blocks within 1-2 BF16 ulp, and
//! amplification starting where the reference's own activations jump an
//! order of magnitude (block 10, scale 7 -> 120). The declared tolerance
//! 1.25 sits above the measured native maximum with ~20% headroom and above
//! the reference's own kernel-choice spread. It gates the tower output only;
//! the end-to-end native probability gate is separate measured
//! declaration.
use super::*;
use crate::vision::ValenVisionCheckpoint;
use serde_json::Value;
use std::path::PathBuf;

fn fixture() -> (Value, PathBuf) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../recipe/valen/fixtures/vision-tower-example-image.json");
    let fixture: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert!(
        fixture["format"] == "valen-vision-reference/1",
        "unexpected fixture format"
    );
    (fixture, path)
}

fn decode_bf16(hex: &str) -> Vec<bf16> {
    let bytes = (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex digit"))
        .collect::<Vec<_>>();
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])))
        .collect()
}

fn env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var_os(name).expect(name))
}

/// CPU-only: the real export passes the manifest, hash and provenance gates
/// and yields the 2B inventory (297 visual tensors, merger width 2048).
#[test]
#[ignore = "requires VALEN_EXPORT pointing at the five-file export directory"]
fn valen_vision_checkpoint_loads_from_the_verified_export() {
    let checkpoint = ValenVisionCheckpoint::load(env_path("VALEN_EXPORT")).unwrap();
    let config = checkpoint.vision_config();
    assert_eq!(config.out_hidden_size, 2048);
    assert_eq!(config.depth, 24);
    assert_eq!(checkpoint.base_names().count(), 297);
    // The trained replacements load from the same mmap as the base tensors.
    for (name, shape) in [
        ("model.visual.blocks.20.attn.qkv.weight", vec![3072, 1024]),
        ("model.visual.blocks.23.mlp.linear_fc2.bias", vec![1024]),
        ("model.visual.merger.linear_fc2.weight", vec![2048, 4096]),
    ] {
        let tensor = checkpoint.base_tensor(name).unwrap();
        assert_eq!(tensor.shape(), &shape[..], "{name}");
        assert_eq!(tensor.dtype(), safetensors::Dtype::BF16);
    }
}

/// GPU parity: the native tower's output against the recorded reference on
/// the fixture image, inside the fixture's declared tolerance.
#[test]
#[ignore = "requires CUDA, VALEN_EXPORT and VALEN_CUDA_LIB"]
fn valen_vision_tower_matches_the_recorded_reference() {
    let (fixture, fixture_path) = fixture();
    let grid: [usize; 3] = fixture["image_grid_thw"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let [tokens, width]: [usize; 2] = fixture["tower_output_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let pixels = decode_bf16(fixture["pixel_values_bf16_hex"].as_str().unwrap());
    let expected = decode_bf16(fixture["tower_output_bf16_hex"].as_str().unwrap());
    let image = ProcessedImage {
        image_grid_thw: grid,
        resized_height: grid[1] * 16,
        resized_width: grid[2] * 16,
        pixel_values: pixels.iter().map(|v| v.to_f32()).collect(),
    };
    let mut model =
        VisionModel::load_valen(env_path("VALEN_EXPORT"), &env_path("VALEN_CUDA_LIB")).unwrap();
    model.graph_enabled = false; // parity runs eager
    let output = model.forward(&image).unwrap();
    assert_eq!(output.len(), tokens * width, "native output token count");
    let mut max_abs = 0_f32;
    let mut mean_abs = 0_f64;
    let mut scale = 0_f32;
    let mut exact = 0_usize;
    for (native, reference) in output.iter().zip(&expected) {
        let (native, reference) = (native.to_f32(), reference.to_f32());
        let delta = (native - reference).abs();
        max_abs = max_abs.max(delta);
        mean_abs += delta as f64;
        scale = scale.max(reference.abs());
        exact += usize::from(native.to_bits() == reference.to_bits());
    }
    mean_abs /= output.len() as f64;
    let tolerance = fixture["tolerance"].as_f64();
    println!(
        "valen vision parity: {exact}/{} elements bit-exact, max abs delta {max_abs:.6e}, \
         mean abs delta {mean_abs:.6e}, reference scale {scale:.3}, \
         fixture tolerance {tolerance:?}, fixture {}",
        output.len(),
        fixture_path.display(),
    );
    let tolerance = tolerance.unwrap_or_else(|| {
        panic!(
            "measured native delta above; the fixture {} declares no tolerance \
             ({}), re-record with dump_vision_reference.py --tolerance",
            fixture_path.display(),
            fixture["tolerance_semantics"].as_str().unwrap()
        )
    });
    assert!(
        max_abs <= tolerance as f32,
        "native tower drift {max_abs:.6e} exceeds the declared tolerance {tolerance:.6e}"
    );
}

#[test]
#[ignore = "diagnostic: requires VALEN_EXPORT, VALEN_CUDA_LIB, VALEN_VISION_STAGES"]
fn valen_vision_stage_drift_diagnostic() {
    let (fixture, _) = fixture();
    let grid: [usize; 3] = fixture["image_grid_thw"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let pixels = decode_bf16(fixture["pixel_values_bf16_hex"].as_str().unwrap());
    let image = ProcessedImage {
        image_grid_thw: grid,
        resized_height: grid[1] * 16,
        resized_width: grid[2] * 16,
        pixel_values: pixels.iter().map(|v| v.to_f32()).collect(),
    };
    let stages_bytes: Vec<u8> =
        std::fs::read(std::env::var("VALEN_VISION_STAGES").expect("VALEN_VISION_STAGES")).unwrap();
    let stages: Value = serde_json::from_slice(&stages_bytes).unwrap();
    let mut model =
        VisionModel::load_valen(env_path("VALEN_EXPORT"), &env_path("VALEN_CUDA_LIB")).unwrap();
    model.graph_enabled = false;
    let drift = std::cell::RefCell::new(Vec::new());
    {
        let mut drain = drift.borrow_mut();
        model
            .forward_with_trace(&image, |name, values| {
                if let Some(hex) = stages["stages"][name].as_str() {
                    let expected: Vec<bf16> = decode_bf16(hex);
                    let mut max_abs = 0_f32;
                    let mut exact = 0_usize;
                    let mut scale = 0_f32;
                    for (native, reference) in values.iter().zip(&expected) {
                        let (native, reference) = (native.to_f32(), reference.to_f32());
                        max_abs = max_abs.max((native - reference).abs());
                        scale = scale.max(reference.abs());
                        exact += usize::from(native.to_bits() == reference.to_bits());
                    }
                    drain.push(format!(
                        "{name}: max_abs {max_abs:.6e} scale {scale:.3e} exact {}/{}",
                        exact,
                        values.len()
                    ));
                }
                Ok(())
            })
            .unwrap();
    }
    for line in drift.borrow().iter() {
        println!("{line}");
    }
}
