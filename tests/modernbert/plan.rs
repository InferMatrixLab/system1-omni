use omni_modernbert::adapter::ModernBertConfigAdapter;
use omni_modernbert::plan::{self, EncoderPlanOptions, PlanAnchor, PlanArg, PlanStep, ScratchSlot};
use omni_modernbert::structure::ModernBertStructure;
use serde_json::json;

/// A 1024-hidden member with a type embedding and plain rotary tables,
/// parsed as a stock HF config (the same geometry the Laya bundle
/// dialect produces), so the family plan tests stay model-agnostic.
fn laya_structure() -> ModernBertStructure {
    let config = json!({
        "model_type": "modernbert", "hidden_activation": "gelu",
        "attention_bias": false, "mlp_bias": false, "norm_bias": false,
        "hidden_size": 1024, "intermediate_size": 2624,
        "num_attention_heads": 16, "num_hidden_layers": 28,
        "vocab_size": 50368, "norm_eps": 0.00001, "local_attention": 128,
        "type_vocab_size": 3, "max_position_embeddings": 512,
        "layer_types": (["full_attention", "sliding_attention", "sliding_attention"]
            .repeat(10))[..28],
        "global_rope_theta": 160000.0, "local_rope_theta": 10000.0
    });
    ModernBertConfigAdapter::from_value(&config, "config.json")
        .unwrap()
        .0
}

fn vela_structure() -> ModernBertStructure {
    let config = json!({
        "model_type": "modernbert", "hidden_activation": "gelu",
        "attention_bias": false, "mlp_bias": false, "norm_bias": false,
        "hidden_size": 768, "intermediate_size": 1152,
        "num_attention_heads": 12, "num_hidden_layers": 22,
        "vocab_size": 256008, "norm_eps": 1e-5,
        "local_attention": 128, "global_attn_every_n_layers": 3,
        "global_rope_theta": 160000, "local_rope_theta": 160000,
        "max_position_embeddings": 32768,
        "rope_scaling": {
            "rope_type": "yarn", "factor": 4.0,
            "original_max_position_embeddings": 8192, "truncate": true
        }
    });
    ModernBertConfigAdapter::from_value(&config, "config.json#encoder_config")
        .unwrap()
        .0
}

fn launches(steps: &[PlanStep]) -> Vec<&PlanStep> {
    steps
        .iter()
        .filter(|s| matches!(s, PlanStep::Launch { .. }))
        .collect()
}

#[test]
fn laya_plan_reproduces_the_encoder_step_sequence() {
    let steps = plan::encoder_plan(&laya_structure(), &EncoderPlanOptions::default());
    // embed + 28 layers x 8 kernels + the type add.
    assert_eq!(launches(&steps).len(), 1 + 28 * 8 + 1);
    assert_eq!(
        steps[0].render(),
        "embed ids encoder.embeddings.tok_embeddings.weight encoder.embeddings.norm.weight x y"
    );
    assert_eq!(steps[1], PlanStep::Anchor(PlanAnchor::AfterEmbed));

    // Layer 0 runs full attention on the full-kind tables.
    assert_eq!(
        steps[2].render(),
        "qkv y encoder.layers.0.attn.Wqkv.weight zeros.3072 qkv"
    );
    assert_eq!(steps[3].render(), "rope qkv rope_full_cos rope_full_sin");
    assert_eq!(steps[4].render(), "attn_full qkv lens o");
    assert_eq!(
        steps[5].render(),
        "out o encoder.layers.0.attn.Wo.weight zeros.1024 y"
    );
    assert_eq!(
        steps[6].render(),
        "addln x y encoder.layers.0.mlp_norm.weight zeros.1024 y"
    );
    assert_eq!(
        steps[7].render(),
        "geglu y encoder.layers.0.mlp.Wi.weight g"
    );
    assert_eq!(
        steps[8].render(),
        "down g encoder.layers.0.mlp.Wo.weight zeros.1024 y"
    );
    // Layer 0's post-MLP norm is layer 1's attention norm.
    assert_eq!(
        steps[9].render(),
        "addln x y encoder.layers.1.attn_norm.weight zeros.1024 y"
    );

    // Layer 1 is sliding (layer 0 ends with its anchor step at index 10).
    assert!(steps[12].render().starts_with("rope qkv rope_local"));
    assert_eq!(steps[13].render(), "attn_local qkv lens o");

    // The last layer's post-MLP norm is the final norm.
    let last_layer_addln = steps
        .iter()
        .position(|s| s.render().contains("encoder.final_norm.weight"))
        .unwrap();
    assert_eq!(
        steps[last_layer_addln].render(),
        "addln x y encoder.final_norm.weight zeros.1024 y"
    );

    // The type add closes the encoder plan.
    let last = launches(&steps).last().unwrap().render();
    assert_eq!(last, "type y type_emb.weight types x");
    assert_eq!(
        steps.last(),
        Some(&PlanStep::Launch {
            kernel: "type".into(),
            args: vec![
                PlanArg::Scratch(ScratchSlot::Y),
                PlanArg::Weight("type_emb.weight".into()),
                PlanArg::Scratch(ScratchSlot::Types),
                PlanArg::Scratch(ScratchSlot::X),
            ]
        })
    );
}

#[test]
fn vela_plan_runs_its_own_schedule_without_a_type_add() {
    let steps = plan::encoder_plan(&vela_structure(), &EncoderPlanOptions::default());
    assert_eq!(launches(&steps).len(), 1 + 22 * 8);
    assert!(!steps.iter().any(|s| s.render().contains("type_emb")));
    // Layer 21 is the last full-attention layer; zeros follow its width.
    assert!(
        steps
            .iter()
            .any(|s| s.render() == "qkv y encoder.layers.21.attn.Wqkv.weight zeros.2304 qkv")
    );
    assert!(steps.iter().any(|s| s.render() == "attn_full qkv lens o"));
    assert_eq!(
        steps
            .iter()
            .filter(|s| s.render() == "attn_local qkv lens o")
            .count(),
        14
    );
    assert!(steps.contains(&PlanStep::Anchor(PlanAnchor::LayerEnd(21))));
    assert_eq!(
        steps
            .iter()
            .filter(|s| matches!(s, PlanStep::Anchor(PlanAnchor::LayerEnd(_))))
            .count(),
        22
    );
}

#[test]
fn original_rope_option_selects_the_reference_variant() {
    let steps = plan::encoder_plan(
        &laya_structure(),
        &EncoderPlanOptions {
            original_rope: true,
        },
    );
    assert!(
        steps
            .iter()
            .any(|s| s.render() == "rope_original qkv rope_full_cos rope_full_sin")
    );
}

#[test]
fn plan_rendering_is_deterministic() {
    let once = plan::encoder_plan(&laya_structure(), &EncoderPlanOptions::default());
    let twice = plan::encoder_plan(&laya_structure(), &EncoderPlanOptions::default());
    let render = |steps: &[PlanStep]| steps.iter().map(PlanStep::render).collect::<Vec<_>>();
    assert_eq!(render(&once), render(&twice));
    assert_eq!(render(&once).len(), once.len());
}
