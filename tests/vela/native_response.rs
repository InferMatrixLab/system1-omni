//! The SystemOne response finishing, replayed from the recorded
//! reference logits against the frozen oracle responses (the same
//! engine produced both on CPU FP32, so this checks the finishing
//! mirror exactly). Opt-in via `VELA_NATIVE_BUNDLE` (calibration).

use std::collections::HashMap;
use std::path::PathBuf;

use omni_vela_native::scoring::{Calibration, SpanOut, row_answers, system_one_response};
use serde_json::Value;

const CASES: [&str; 4] = [
    "example-request",
    "example-request-set-threshold",
    "example-request-cjk",
    "example-request-score",
];

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../tests/vela/fixtures")
}

fn load(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(fixtures_dir().join(name)).expect("fixture"))
        .expect("json")
}

fn f32_matrix(raw: &[u8], name: &str) -> Vec<Vec<f32>> {
    let file = safetensors::SafeTensors::deserialize(raw).expect("safetensors");
    let view = file.tensor(name).expect("tensor");
    let shape = view.shape();
    let data: Vec<f32> = view
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    match shape.len() {
        1 => vec![data],
        2 => data.chunks(shape[1]).map(<[f32]>::to_vec).collect(),
        _ => panic!("unexpected shape"),
    }
}

fn f32_stack(raw: &[u8], name: &str) -> Vec<Vec<Vec<f32>>> {
    let file = safetensors::SafeTensors::deserialize(raw).expect("safetensors");
    let view = file.tensor(name).expect("tensor");
    let shape = view.shape();
    assert_eq!(shape.len(), 3);
    let data: Vec<f32> = view
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    data.chunks(shape[1] * shape[2])
        .map(|b| b.chunks(shape[2]).map(<[f32]>::to_vec).collect())
        .collect()
}

#[test]
fn the_systemone_finishing_matches_the_frozen_oracles() {
    let Ok(bundle) = std::env::var("VELA_NATIVE_BUNDLE") else {
        eprintln!("skipping: set VELA_NATIVE_BUNDLE");
        return;
    };
    let bundle = PathBuf::from(bundle);
    let cal = Calibration::load(&bundle).expect("calibration");
    let heads = omni_vela_native::heads::Heads::load(&bundle).expect("heads");
    let gate = 2e-6;

    for case in CASES {
        let head = load(&format!("head-input-{case}.json"));
        let plan = load(&format!("native-plan-{case}.json"));
        let oracle_name = if case == "example-request" {
            "oracle-hybrid-example-request.json".to_string()
        } else {
            format!("oracle-hybrid-{case}.json")
        };
        let oracle = load(&oracle_name);
        let raw = std::fs::read(fixtures_dir().join(format!("head-input-{case}.safetensors")))
            .expect("safetensors");
        let request = plan["request"].as_object().expect("request").clone();

        let mut thresholds: HashMap<String, f64> = HashMap::new();
        for (qid, question) in request["questions"].as_object().expect("questions") {
            if let Some(t) = question.get("threshold").and_then(Value::as_f64) {
                thresholds.insert(qid.clone(), t);
            }
        }

        // Drive the recorded calls through the Rust heads (as in the
        // heads test) and assemble answers per row, in plan order.
        let calls: Vec<&Value> = head["head_calls"].as_array().unwrap().iter().collect();
        let mut q_index = 0usize;
        let mut span_index = 0usize;
        let mut answers: Vec<Value> = Vec::new();
        let mut span_probs: HashMap<String, Vec<f32>> = HashMap::new();
        for (row_index, row) in plan["rows"].as_array().unwrap().iter().enumerate() {
            let q_blocks: Vec<&Value> = row["blocks"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|b| b["kind"] == "q")
                .collect();
            let mut qblocks = Vec::new();
            if !q_blocks.is_empty() {
                let call = calls
                    .iter()
                    .filter(|c| c["kind"] == "q")
                    .nth(q_index)
                    .expect("q call");
                q_index += 1;
                let cands = f32_stack(&raw, call["cand"].as_str().unwrap());
                let queries = f32_matrix(&raw, call["query"].as_str().unwrap());
                for (block, (cand, query)) in q_blocks.iter().zip(cands.iter().zip(&queries)) {
                    let names: Vec<String> = block["names"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|n| n.as_str().unwrap().to_string())
                        .collect();
                    let has_abs = block["has_abs"].as_bool().unwrap_or(false);
                    let mut logits = heads.forward_candidate(cand, query);
                    if block["qtype"] == "set" {
                        for value in logits.iter_mut() {
                            *value += heads.set_bias;
                        }
                    }
                    logits.truncate(names.len() + usize::from(has_abs));
                    qblocks.push(omni_vela_native::scoring::QBlock {
                        qid: block["qid"].as_str().unwrap().to_string(),
                        qtype: block["qtype"].as_str().unwrap().to_string(),
                        names,
                        has_abs,
                        logits,
                    });
                }
            }
            // Span: use the recorded logits directly (the heads test
            // covers head parity; here we exercise the finishing).
            let span_blocks: Vec<&Value> = row["blocks"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|b| b["kind"] == "span")
                .collect();
            let mut span_out: Option<SpanOut> = None;
            if let Some(_block) = span_blocks.first() {
                let call = calls
                    .iter()
                    .filter(|c| c["kind"] == "span")
                    .nth(span_index)
                    .expect("span call");
                span_index += 1;
                let which = call["head"].as_str().unwrap_or("router").to_string();
                let logits: Vec<Vec<f32>> = call["logits"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|r| {
                        r.as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_f64().unwrap() as f32)
                            .collect()
                    })
                    .collect();
                let names: Vec<String> = row["names"]
                    .as_array()
                    .map(|a| a.iter().map(|n| n.as_str().unwrap().to_string()).collect())
                    .unwrap_or_default();
                let w_offs: Vec<(usize, usize)> = row["w_offs"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .map(|o| {
                                (
                                    o[0].as_u64().unwrap_or(0) as usize,
                                    o[1].as_u64().unwrap_or(0) as usize,
                                )
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let over = row["span_over"].as_str().unwrap_or("user").to_string();
                let n_tokens = row["seg_lens"][&over].as_u64().unwrap_or(0) as usize;
                let qid = plan["plan"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|p| p["type"] == "span")
                    .nth(row_index)
                    .or_else(|| {
                        plan["plan"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .find(|p| p["type"] == "span")
                    })
                    .map(|p| p["qid"].as_str().unwrap().to_string())
                    .unwrap_or_default();
                // The Noul view needs the max word probability
                // (pre-decode), range-filtered by the plan's range.
                let temperature = cal.span_temperature(&which);
                let probs: Vec<f32> = logits
                    .iter()
                    .map(|row| sigmoid_f32(row, temperature))
                    .collect();
                let range = plan["plan"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|p| p["qid"] == qid.as_str())
                    .and_then(|p| p["range"].as_array().cloned());
                let filtered: Vec<f32> = match &range {
                    Some(bounds) if bounds.len() == 2 => probs
                        .iter()
                        .zip(&w_offs)
                        .filter(|(_, (a, b))| {
                            *a >= bounds[0].as_u64().unwrap_or(0) as usize
                                && *b <= bounds[1].as_u64().unwrap_or(0) as usize
                        })
                        .map(|(p, _)| *p)
                        .collect(),
                    _ => probs,
                };
                span_probs.insert(qid.clone(), filtered);
                span_out = Some(SpanOut {
                    qid: qid.clone(),
                    over,
                    names,
                    offs: w_offs,
                    logits,
                    head: which,
                    head_reason: row["head_reason"].as_str().map(str::to_string),
                    label_alias: row.get("label_alias").filter(|v| !v.is_null()).cloned(),
                    n_tokens,
                });
            }
            let span_text = row["span_over"].as_str().and_then(|role| {
                plan["predict_request"]["parts"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|p| p["type"] == *role)
                    .map(|p| p["text"].as_str().unwrap().to_string())
            });
            let mut got = row_answers(
                &cal,
                &qblocks,
                span_out.as_ref(),
                &thresholds,
                span_text.as_deref(),
            );
            answers.append(&mut got);
        }
        // Plan-order the answers, then finish the envelope.
        let order: Vec<String> = plan["plan"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p["qid"].as_str().unwrap().to_string())
            .collect();
        answers.sort_by_key(|a| {
            order
                .iter()
                .position(|q| q == a["id"].as_str().unwrap_or(""))
                .unwrap_or(usize::MAX)
        });
        let part_texts: HashMap<String, String> = plan["predict_request"]["parts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                (
                    p["type"].as_str().unwrap().to_string(),
                    p["text"].as_str().unwrap().to_string(),
                )
            })
            .collect();
        let input_tokens = oracle["system_one"]["usage"]["input_tokens"]
            .as_u64()
            .unwrap_or(0) as usize;
        let response = system_one_response(
            "vllm-sr/Vela-2.0-0.8B",
            plan["plan"].as_array().unwrap(),
            &answers,
            &span_probs,
            &part_texts,
            input_tokens,
        );
        let want = &oracle["system_one"];
        let drift = max_drift(&response, want);
        assert!(
            drift <= gate,
            "{case}: response drift {drift} > {gate}\ngot:  {}\nwant: {}",
            serde_json::to_string_pretty(&response).unwrap(),
            serde_json::to_string_pretty(want).unwrap(),
        );
        // Structure: same key sets and discrete fields.
        for key in ["answers", "spans", "sets", "thresholds", "span_heads"] {
            assert_eq!(
                response
                    .get(key)
                    .and_then(Value::as_object)
                    .map(serde_json::Map::len),
                want.get(key)
                    .and_then(Value::as_object)
                    .map(serde_json::Map::len),
                "{case}: {key} key set"
            );
        }
        println!("{case}: response drift {drift:.3e}");
    }
}

fn sigmoid_f32(row: &[f32], temperature: f64) -> f32 {
    // The reference's per-word span view: sigmoid of every label logit
    // (over the span temperature), max over the labels.
    row.iter()
        .map(|v| (1.0 / (1.0 + (-(*v as f64) / temperature).exp())) as f32)
        .fold(0.0f32, |a, b| a.max(b))
}

fn max_drift(a: &Value, b: &Value) -> f64 {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            (x.as_f64().unwrap_or(0.0) - y.as_f64().unwrap_or(0.0)).abs()
        }
        (Value::Array(x), Value::Array(y)) => x
            .iter()
            .zip(y)
            .map(|(x, y)| max_drift(x, y))
            .fold(0.0, f64::max),
        (Value::Object(x), Value::Object(y)) => x
            .iter()
            .map(|(k, v)| y.get(k).map(|w| max_drift(v, w)).unwrap_or(1.0))
            .chain(y.keys().filter(|k| !x.contains_key(*k)).map(|_| 1.0))
            .fold(0.0, f64::max),
        (Value::String(x), Value::String(y)) if x != y => 1.0,
        _ => 0.0,
    }
}
