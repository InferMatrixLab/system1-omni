//! Opt-in capture of actual CUDA readouts through the production typed pipeline.

use std::{fs, path::PathBuf, time::Instant};

use omni_vela_native::{
    bundle::Bundle, contract, executor::Executor, processing::Processor, scoring::Calibration,
};
use serde_json::{Value, json};

#[test]
#[ignore = "requires pinned Vela 0.8B bundle, CUDA and a manifest of request files"]
fn complete_cuda_pipeline_captures_typed_responses_and_raw_logits() {
    let bundle = PathBuf::from(std::env::var_os("VELA_NATIVE_BUNDLE").unwrap());
    let anchor = std::env::var("VELA_NATIVE_MANIFEST_SHA256").unwrap();
    let lib = PathBuf::from(std::env::var_os("VELA_CUDA_LIB").unwrap());
    let cases: Vec<(String, PathBuf)> = serde_json::from_slice(
        &fs::read(std::env::var_os("VELA_PIPELINE_CASES").unwrap()).unwrap(),
    )
    .unwrap();
    let out = PathBuf::from(std::env::var_os("VELA_PIPELINE_OUTPUT").unwrap());
    fs::create_dir(&out).unwrap();
    let name = Bundle::load(&bundle, &anchor)
        .unwrap()
        .model_name()
        .to_owned();
    let processor = Processor::load(&bundle, &anchor).unwrap();
    let calibration = Calibration::load(&bundle, &anchor).unwrap();
    let mut executor = Executor::load(&bundle, &lib, &anchor).unwrap();

    let mut failed = Vec::new();
    for (case, path) in cases {
        let started = Instant::now();
        println!("{case}: start");
        let request = fs::read(path).unwrap();
        let body = contract::compile(&request, &name).expect("capture case admits");
        let planned = processor.plan(&body).expect("capture case plans");
        let mut rows: Vec<Option<Value>> =
            vec![None; planned.rows.len() + planned.window_rows.len()];
        let result = executor.execute(&name, &planned, &calibration, |row, logits| {
            assert!(
                rows[row]
                    .replace(serde_json::to_value(logits).unwrap())
                    .is_none(),
                "duplicate row"
            );
        });
        let response = match result {
            Ok(response) => {
                assert!(rows.iter().all(Option::is_some), "missing row");
                json!({"status": 200, "response": response})
            }
            Err(error) => {
                failed.push(case.clone());
                json!({"status": 500, "error": format!("{error:#}")})
            }
        };
        let output = json!({
            "request": body, "response": response,
            "plan": planned.plan, "predict_request": planned.predict_request,
            "rows": planned.rows, "window_rows": planned.window_rows,
            "logits": rows, "wall_seconds": started.elapsed().as_secs_f64()
        });
        fs::write(
            out.join(format!("{case}.json")),
            serde_json::to_vec(&output).unwrap(),
        )
        .unwrap();
        println!(
            "{case}: status {} in {:?}",
            response["status"],
            started.elapsed()
        );
    }
    assert!(failed.is_empty(), "failed cases: {failed:?}");
}
