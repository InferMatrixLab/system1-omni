use omni_laya::plan::{plan_encoder, render_plan};

// The golden fixtures are the rendered encoder plans of the registered
// Laya 0.3.20 profile: `encoder-plan.txt` drives the rope-table kernel,
// `encoder-plan-original-rope.txt` its `rope_original` reference
// variant. The dispatch trace test in model_dispatch.rs cross-checks
// the same plans against the frozen CUDA-ABI launch sequence.

fn laya_structure() -> omni_modernbert::structure::ModernBertStructure {
    omni_laya::profile::laya_0_3_20().structure
}

#[test]
fn encoder_plan_matches_the_recorded_sequence() {
    assert_eq!(
        render_plan(&plan_encoder(&laya_structure(), false)),
        include_str!("data/encoder-plan.txt")
    );
    assert_eq!(
        render_plan(&plan_encoder(&laya_structure(), true)),
        include_str!("data/encoder-plan-original-rope.txt")
    );
}

#[test]
fn encoder_plan_shape_is_the_supported_profile() {
    let plan = plan_encoder(&laya_structure(), false);
    let launches = plan
        .iter()
        .filter(|s| matches!(s, omni_laya::plan::PlannedStep::Launch { .. }))
        .count();
    // embed + 28 layers x 8 kernels + type add + 2 head layers x 8 kernels.
    assert_eq!(launches, 1 + 28 * 8 + 1 + 2 * 8);
    let rendered = render_plan(&plan);
    assert_eq!(rendered.matches("attn_full").count(), 12);
    assert_eq!(rendered.matches("attn_local").count(), 18);
    assert!(rendered.contains("addln x y encoder.final_norm.weight zeros.1024 y"));
    assert!(rendered.ends_with("residual x y\n"));
    // The two rope variants differ in exactly the 28 rope kernel names.
    let original = render_plan(&plan_encoder(&laya_structure(), true));
    let differences = rendered
        .lines()
        .zip(original.lines())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(differences, 28);
}
