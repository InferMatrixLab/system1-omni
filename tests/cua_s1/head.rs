use super::*;

#[test]
fn letter_projection_matches_the_reference_bf16_output_boundary() {
    assert_eq!(
        letter_logits(&[1.0078125], &[1.0078125], 1).unwrap(),
        [1.015625]
    );
    // Both exact dot products belong to the same BF16 bin. Retaining FP32
    // invents a winning second option where the reference has an ordered tie.
    assert_eq!(
        letter_logits(&[20.0, 0.03125, 20.0, 0.046875], &[1.0, 1.0], 2).unwrap(),
        [20.0, 20.0]
    );
}

#[test]
fn letter_projection_preserves_fp64_accumulation_and_bf16_readout() {
    let letters = [
        16_777_216.0,
        1.0,
        -16_777_216.0,
        16_777_216.0,
        1.0,
        0.0,
        f32::NAN,
        f32::NAN,
        f32::NAN,
    ];
    // The first dot product would be zero with FP32 accumulation. The second
    // must round before postprocessing; unused letter rows must not be read.
    assert_eq!(
        letter_logits(&letters, &[1.0, 1.0, 1.0], 2).unwrap(),
        [1.0, 16_777_216.0]
    );
    assert_eq!(letter_logits(&letters, &[1.0, 1.0, 1.0], 1).unwrap(), [1.0]);
}

#[test]
fn letter_projection_rejects_nonfinite_scores_before_later_prompts() {
    assert!(letter_logits(&[f32::MAX], &[2.0], 1).is_err());
    assert!(letter_logits(&[f32::NAN], &[1.0], 1).is_err());
}
