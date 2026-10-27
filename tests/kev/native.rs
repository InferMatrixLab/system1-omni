//! Explicit GPU capture for three-way parity. Executed only under gpu run.
use crate::{checkpoint, executor::Executor, processing::Processor};
use serde_json::{Value, json};

#[test]
#[ignore = "requires KEV_EXPORT, KEV_CUDA_LIB, KEV_NATIVE_OUT and a reserved CUDA GPU"]
fn native_corpus_capture() {
    let dir = std::path::PathBuf::from(std::env::var("KEV_EXPORT").expect("KEV_EXPORT"));
    let lib = std::path::PathBuf::from(std::env::var("KEV_CUDA_LIB").expect("KEV_CUDA_LIB"));
    let out = std::path::PathBuf::from(std::env::var("KEV_NATIVE_OUT").expect("KEV_NATIVE_OUT"));
    assert!(
        !out.exists(),
        "preserve previous evidence; choose a fresh output"
    );
    checkpoint::verify(&dir).unwrap();
    let processor = Processor::load(&dir).unwrap();
    let mut executor = Executor::load(&dir, &lib).unwrap();
    let fixture: Value = serde_json::from_str(include_str!("data/processing.json")).unwrap();
    let mut captured = vec![];
    for case in fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .chain(std::iter::repeat_n(&fixture["cases"][8], 3))
    {
        let prepared = processor
            .prepare(case["request"].as_object().unwrap())
            .unwrap();
        assert_eq!(json!(prepared.rows), case["rows"]);
        let logits = executor.execute(&prepared.rows).unwrap();
        let probabilities: Vec<_> = logits
            .iter()
            .map(|row| {
                crate::postprocess::probabilities(row, crate::contract::TEMPERATURE).unwrap()
            })
            .collect();
        let body = prepared.context.finish(logits.clone()).unwrap();
        captured.push(
            json!({"name":case["name"],"logits":logits,"probabilities":probabilities,"body":body}),
        );
        std::fs::write(
            &out,
            serde_json::to_vec_pretty(&json!({"kind":"native","cases":captured})).unwrap(),
        )
        .unwrap();
    }
}
