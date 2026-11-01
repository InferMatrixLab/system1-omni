//! Frozen full-row reference replay, including two softmax layers and causal GDN.
use omni_qwen3_5_native::model::Model;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Fixture {
    schema: String,
    rows: Vec<Row>,
}

#[derive(Deserialize)]
struct Row {
    id: String,
    ids: Vec<u32>,
    state_tokens: usize,
    hidden: Vec<f32>,
}

#[test]
#[ignore = "needs a GPU, CUA_S1_CUDA_LIB and QWEN_PREFIX_BIDIR_FIXTURE"]
fn multilayer_prefix_mask_matches_frozen_reference_and_restores_causal_execution() {
    let file = PathBuf::from(std::env::var_os("QWEN_PREFIX_BIDIR_FIXTURE").expect("fixture path"));
    let fixture: Fixture = serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
    assert_eq!(fixture.schema, "qwen-prefix-bidir-reference/1");
    assert!(!fixture.rows.is_empty());
    let checkpoint = file.parent().unwrap();
    let library = PathBuf::from(std::env::var_os("CUA_S1_CUDA_LIB").expect("CUDA library"));
    let mut model = Model::load_with_graph(checkpoint, &library, false).unwrap();
    model.require_prompt_prefix_bidirectional().unwrap();
    assert!(model.cfg.full_attention.iter().filter(|&&x| x).count() >= 2);
    assert!(model.cfg.full_attention.iter().any(|&x| !x));
    let first = &fixture.rows[0];
    let causal_before = model.forward(&first.ids).unwrap();
    let mut records = Vec::new();
    let mut worst = 0.0f32;
    for row in fixture.rows.iter().chain([first]) {
        let hidden = model
            .forward_prompt_prefix_bidirectional(&row.ids, row.state_tokens)
            .unwrap();
        assert_eq!(hidden.len(), row.hidden.len());
        assert!(hidden.iter().chain(&row.hidden).all(|x| x.is_finite()));
        let scale = row.hidden.iter().map(|x| x.abs()).fold(1e-3f32, f32::max);
        let error = hidden
            .iter()
            .zip(&row.hidden)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
            / scale;
        eprintln!("{}: normalized hidden max error {error:.6e}", row.id);
        worst = worst.max(error);
        records.push(serde_json::json!({"id":row.id,"ids":row.ids,"state_tokens":row.state_tokens,"hidden":hidden,"normalized_max_error":error}));
    }
    assert_eq!(
        records[0]["hidden"],
        records.last().unwrap()["hidden"],
        "earlier row changed after scratch reuse"
    );
    assert_eq!(
        causal_before,
        model.forward(&first.ids).unwrap(),
        "prefix mask leaked into a causal call"
    );
    if let Some(output) = std::env::var_os("QWEN_PREFIX_BIDIR_OUTPUT") {
        std::fs::write(output, serde_json::to_vec(&records).unwrap()).unwrap();
    }
    // This synthetic BF16 kernel/backbone gate is deliberately separate from
    // a model adapter's decision/probability gates. It is fixed before replay.
    assert!(worst < 0.03, "full-row BF16 hidden gate failed: {worst}");
}
