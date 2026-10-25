use super::*;
use crate::contract::CHECKPOINTS;

#[test]
fn scalar_projection_preserves_accumulation_rounding_and_bias_order() {
    let head = DecisionHead {
        weights: vec![16_777_216.0, 1.0, -16_777_216.0],
        bias: 0.25,
    };
    assert_eq!(head.score(vec![1.0, 1.0, 1.0]).unwrap(), 1.25);
    // Bias is added after rounding the dot product to FP32, not in FP64.
    let head = DecisionHead {
        weights: vec![16_777_216.0, 1.0],
        bias: -16_777_216.0,
    };
    assert_eq!(head.score(vec![1.0, 1.0]).unwrap(), 0.0);
}

#[test]
fn scalar_projection_rejects_nonfinite_scores() {
    let head = DecisionHead {
        weights: vec![f32::MAX],
        bias: 0.0,
    };
    assert!(head.score(vec![2.0]).is_err());
    assert!(head.score(vec![f32::NAN]).is_err());
}

#[test]
fn head_loader_checks_the_selected_checkpoints_backbone() {
    // Qwen/Qwen3.8-27B @ 1d4bf0f2 and Qwen/Qwen3.5-9B @ c2022362 configurations.
    let tests = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../../tests"));
    let (qwen_27b, qwen_9b) = (
        tests.join("qwen3_5/data"),
        tests.join("open_jev/data/qwen3_5_9b"),
    );
    let (open_jev_27b, open_jev_9b) = (&CHECKPOINTS[0], &CHECKPOINTS[1]);
    let manifest =
        |width: usize| serde_json::json!({"head_weight": vec![0.0; width], "head_bias": 0.0});
    assert!(DecisionHead::load(&qwen_27b, &manifest(5120), open_jev_27b).is_ok());
    assert!(DecisionHead::load(&qwen_9b, &manifest(4096), open_jev_9b).is_ok());
    assert_eq!(
        DecisionHead::load(&qwen_9b, &manifest(4096), open_jev_27b)
            .err()
            .unwrap()
            .to_string(),
        "expected the Qwen/Qwen3.8-27B backbone dimensions"
    );
    assert!(DecisionHead::load(&qwen_27b, &manifest(5120), open_jev_9b).is_err());
    // The trained head must match the selected backbone's width.
    assert!(DecisionHead::load(&qwen_9b, &manifest(5120), open_jev_9b).is_err());
}

#[test]
fn candidates_plan_as_question_groups_and_rebuild_in_order() {
    let state: Vec<u32> = (0..200).collect();
    let prompt = |question: &[u32], candidate: &[u32]| [&state[..], question, candidate].concat();
    let ids = vec![
        vec![
            prompt(&[900; 100], &[1, 2]),
            prompt(&[900; 100], &[1, 3]),
            prompt(&[900; 100], &[4]),
        ],
        vec![prompt(&[901; 10], &[5])],
    ];
    let rows = rows(&ids);
    assert_eq!(
        rows.iter().map(|row| row.question).collect::<Vec<_>>(),
        [0, 0, 0, 1]
    );
    let plan = PrefixPlan::new(&rows).unwrap();
    assert_eq!(plan.prompts.groups.len(), 2);
    let rebuilt: Vec<Vec<u32>> = plan
        .prompts
        .groups
        .iter()
        .flat_map(|group| {
            group
                .branches
                .iter()
                .map(|branch| [plan.prompts.prefix, group.prefix, branch].concat())
        })
        .collect();
    assert_eq!(rebuilt, ids.concat());
    // The 200-token state is floored to 192; the first question's 100 extra tokens
    // reach 256 for its other two candidates.
    assert_eq!(plan.saved_tokens, 3 * 192 + 2 * (256 - 192));
}
