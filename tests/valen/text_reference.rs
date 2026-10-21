//! Device-gated Valen text parity (spec 03, N2).
//!
//! The reference fixture is recorded by `recipe/valen/dump_text_reference.py`
//! from the pinned reference stack (Valen @ 750bfcf, Preview checkpoint
//! 81b9c63, Qwen3.5-2B base 15852e8c, peft 0.18.1 unmerged FP32 adapters,
//! eager attention, TF32 off) on a compiled text-choice branch, and committed
//! as `recipe/valen/fixtures/text-choice-reference.json`.
//!
//! Tolerance semantics (declared in the fixture, measured first): the maximum
//! absolute elementwise difference, compared in FP32, between the reference
//! final-norm last-position hidden state and the native `TextModel::forward`
//! output for the same token ids on the same GPU and dtype. The two stacks
//! run different kernel implementations of the same numeric profile
//! (framework-compatible cuBLASLt GEMMs/norms, FP32 LoRA branches, FP32
//! chunked GDN), so per-element differences at the BF16 rounding scale can
//! accumulate over 24 layers; the gate quantifies that drift instead of
//! demanding bit equality. A `null` tolerance refuses to pass: measure the
//! native delta first, then declare it in the fixture.
//!
//! Recorded measurement (RTX 5060 Laptop GPU, sm_120, BF16, torch
//! 2.11.0+cu128, peft 0.18.1, TF32 off, 2026-10-21): native vs reference max
//! abs 0.25 over the 2048-element final hidden state (reference output scale
//! 50.75, mean abs 2.31) — exactly one BF16 ulp at the largest element —
//! with 1765/2048 elements differing at their own 1-ulp scale. The declared
//! tolerance 0.5 doubles the measured maximum. Per-projection parity
//! (tests/qwen3_5/valen_text_gpu.rs) sits at 0..1 BF16 ulp per element, with
//! the fused gate/up projection bit-exact over all 98304 compared elements.
use omni_valen_native::text::TextModel;
use serde_json::Value;
use std::path::PathBuf;

fn fixture() -> (Value, PathBuf) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../recipe/valen/fixtures/text-choice-reference.json");
    let fixture: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert!(
        fixture["format"] == "valen-text-reference/1",
        "unexpected fixture format"
    );
    (fixture, path)
}

fn decode_bf16(hex: &str) -> Vec<f32> {
    let bytes = (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex digit"))
        .collect::<Vec<_>>();
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
        .collect()
}

fn env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var_os(name).expect(name))
}

/// CPU-only: the real five-file export passes the manifest, profile, hash and
/// inventory gates and reports the pinned census (372 FP32 tensors over 186
/// targets, scale 2).
#[test]
#[ignore = "requires VALEN_EXPORT pointing at the spec-02 export directory"]
fn valen_text_export_validates_the_pinned_inventory() {
    let policy = omni_valen_native::text::validate_export(&env_path("VALEN_EXPORT")).unwrap();
    assert_eq!(
        policy.adapter_file,
        std::path::Path::new("lora.safetensors")
    );
    assert_eq!(policy.scale, 2.0);
    assert_eq!(policy.targets, 186);
    assert_eq!(policy.keys, 372);
}

/// GPU parity: the native unmerged text forward (all 186 targets applied under
/// the reference numeric profile) against the recorded reference final output,
/// inside the fixture's declared tolerance.
#[test]
#[ignore = "requires CUDA, VALEN_EXPORT and VALEN_CUDA_LIB"]
fn valen_text_forward_matches_the_recorded_reference() {
    let (fixture, fixture_path) = fixture();
    let ids: Vec<u32> = fixture["token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let width = fixture["hidden_width"].as_u64().unwrap() as usize;
    let expected = decode_bf16(fixture["final_output_bf16_hex"].as_str().unwrap());
    assert_eq!(expected.len(), width);

    let mut model =
        TextModel::load(&env_path("VALEN_EXPORT"), &env_path("VALEN_CUDA_LIB")).unwrap();
    assert_eq!(model.config().hidden, width);
    assert_eq!(model.config().full_attention.len(), 24);
    let actual = model.forward(&ids).unwrap();
    model.synchronize().unwrap();
    assert_eq!(actual.len(), width);

    let mut max = 0.0f32;
    let mut differing = 0usize;
    for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
        let delta = (a - e).abs();
        if delta > 0.0 {
            differing += 1;
        }
        if delta > max {
            max = delta;
        }
        let tolerance = fixture["tolerance"].as_f64().map(|t| t as f32);
        assert!(
            tolerance.is_none_or(|t| delta <= t),
            "element {index}: native {a} vs reference {e} (delta {delta})"
        );
    }
    eprintln!(
        "valen text end-to-end: tokens {}, max abs {max:.6}, {differing}/{} elements differ, tolerance {:?}",
        ids.len(),
        width,
        fixture["tolerance"].as_f64()
    );
    if fixture["tolerance"].is_null() {
        panic!(
            "the fixture declares no end-to-end tolerance; measured native max abs \
             {max:.6} over {width} elements ({differing} differ). Declare with \
             dump_text_reference.py --tolerance on {}",
            fixture_path.display()
        );
    }
}
