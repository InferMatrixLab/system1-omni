use omni_vela_native::{cos_sin_tables, profile, yarn};
use serde_json::Value;

/// Reference tables dumped from the pinned runtime itself:
/// transformers 5.19.0's ModernBertRotaryEmbedding over a
/// ModernBertConfig built from the pinned encoder_config, CPU float32.
fn reference_value() -> Value {
    serde_json::from_str(include_str!("data/yarn-reference.json")).unwrap()
}

fn rope() -> omni_modernbert::structure::RopeSpec {
    profile().structure.rope_full
}

#[test]
fn inverse_frequencies_match_the_reference_runtime() {
    let tables = yarn(&rope(), 32).unwrap();
    let expected: Vec<f64> = reference_value()["inv_freq"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert_eq!(tables.inv_freq.len(), expected.len());
    let mut worst = 0.0f64;
    for (actual, wanted) in tables.inv_freq.iter().zip(&expected) {
        let diff = (*actual as f64 - wanted).abs();
        worst = worst.max(diff);
    }
    assert!(worst < 1e-7, "max inverse-frequency delta {worst}");
}

#[test]
fn the_attention_scaling_is_the_reference_mscale() {
    let tables = yarn(&rope(), 32).unwrap();
    let expected = reference_value()["attention_scaling"].as_f64().unwrap();
    assert!((tables.attention_scaling - expected).abs() < 1e-12);
    assert_eq!(tables.attention_scaling, 1.0 + 0.1 * 4.0f64.ln());
}

#[test]
fn cos_sin_tables_reproduce_the_reference_rows() {
    let structure = &profile().structure;
    let (cos, sin, tables) = cos_sin_tables(structure, 32768).unwrap();
    let half = tables.inv_freq.len();
    assert_eq!(cos.len(), 32768 * half);
    let reference = reference_value();
    let positions: Vec<usize> = reference_value()["positions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let mut worst_served = 0.0f64;
    let mut worst_full = 0.0f64;
    for (row, pos) in positions.iter().enumerate() {
        for j in 0..half {
            let cos_ref = reference["cos_half"][row][j].as_f64().unwrap();
            let sin_ref = reference["sin_half"][row][j].as_f64().unwrap();
            let cos_actual = cos[pos * half + j] as f64;
            let sin_actual = sin[pos * half + j] as f64;
            let delta = (cos_actual - cos_ref)
                .abs()
                .max((sin_actual - sin_ref).abs());
            if *pos <= 8192 {
                worst_served = worst_served.max(delta);
            }
            worst_full = worst_full.max(delta);
        }
    }
    // The reference computes f32 powers and f32 angle products; this
    // generator runs f64 and narrows once. The residual is f32-ulp
    // frequency noise amplified by the position, so the served window
    // (at most 8 192 tokens) holds a tight gate and the extension
    // range a documented looser one.
    assert!(worst_served < 1e-4, "served-window delta {worst_served}");
    assert!(worst_full < 5e-4, "full-range delta {worst_full}");
}

#[test]
fn the_ramp_upper_bound_clamps_to_the_head_dimension_like_the_reference() {
    // theta=1000, original_max=8192, beta_slow=1 puts the unclamped
    // correction range above the ramp array (high = 34 > 31): the
    // reference clamps to head_dim - 1 = 63 and lets the mask saturate
    // beyond the array, so the in-band blend denominator stays
    // 34 - low. Clamping to rotary_half - 1 would rescale every
    // in-band blend.
    let spec = omni_modernbert::structure::RopeSpec {
        theta: 1000.0,
        scaling: Some(omni_modernbert::structure::YarnScaling {
            factor: 4.0,
            original_max_positions: 8192,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }),
    };
    let tables = yarn(&spec, 32).unwrap();
    let correction = |rotations: f64| {
        (2.0 * 32.0 * (8192.0f64 / (rotations * std::f64::consts::TAU)).ln()) / (2.0 * 1000f64.ln())
    };
    let low = correction(32.0).floor().max(0.0);
    let high = correction(1.0).ceil().min(2.0 * 32.0 - 1.0);
    assert!(high > 31.0, "the case must exceed the ramp array ({high})");
    for (j, actual) in tables.inv_freq.iter().enumerate() {
        let j = j as f64;
        let pos_freq = 1000.0f64.powf(j / 32.0);
        let ramp = ((j - low) / (high - low)).clamp(0.0, 1.0);
        let expected = 1.0 / (4.0 * pos_freq) * ramp + 1.0 / pos_freq * (1.0 - ramp);
        assert!(
            (*actual as f64 - expected).abs() < 1e-6,
            "inv_freq[{j}] = {actual}, expected {expected}"
        );
    }
}

#[test]
fn table_lengths_follow_the_structure_bounds() {
    let structure = &profile().structure;
    assert!(cos_sin_tables(structure, 0).is_err());
    assert!(cos_sin_tables(structure, 32769).is_err());
    let (cos, sin, _) = cos_sin_tables(structure, 8192).unwrap();
    assert_eq!(cos.len(), 8192 * 32);
    assert_eq!(sin.len(), cos.len());
    // Inside the ramp band the frequency is a blend of extrapolation
    // and interpolation, so it sits strictly between plain theta
    // rope and the fully interpolated (factor-divided) variant.
    let tables = yarn(&rope(), 32).unwrap();
    let plain = 160000.0f64.powf(-(15.0 / 32.0));
    let blended = tables.inv_freq[15] as f64;
    assert!(blended < plain, "{blended} vs {plain}");
    assert!(blended > plain / 4.0, "{blended} vs {}", plain / 4.0);
}
