//! Logit comparisons shared by the CPU-head and GPU-executor replays.

pub fn max_logit_drift(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len(), "logit count");
    assert!(
        got.iter().all(|v| v.is_finite()),
        "non-finite replay logits"
    );
    assert!(
        want.iter().all(|v| v.is_finite()),
        "non-finite reference logits"
    );
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - *b as f32).abs() as f64)
        .fold(0.0, f64::max)
}

#[test]
fn finite_logits_keep_their_measured_drift() {
    assert_eq!(max_logit_drift(&[1.0, 2.5], &[1.0, 2.0]), 0.5);
}

#[test]
fn non_finite_logits_are_rejected() {
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(std::panic::catch_unwind(|| max_logit_drift(&[invalid], &[0.0])).is_err());
        assert!(std::panic::catch_unwind(|| max_logit_drift(&[0.0], &[invalid as f64])).is_err());
    }
}

#[test]
fn missing_or_extra_logits_are_rejected() {
    assert!(std::panic::catch_unwind(|| max_logit_drift(&[], &[0.0])).is_err());
    assert!(std::panic::catch_unwind(|| max_logit_drift(&[0.0], &[])).is_err());
}
