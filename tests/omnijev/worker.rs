//! GPU checks of the native execution on an export. They need a GPU, OMNIJEV_EXPORT
//! pointing at an export from recipe/omnijev/export.py and OMNIJEV_CUDA_LIB at
//! libqwen3_5_cuda.so, both absolute:
//!
//!     OMNIJEV_EXPORT=<export> OMNIJEV_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
//!       cargo test --release -p omni-omnijev-native --test worker -- --ignored --test-threads=1 --skip reference
//! The separate reference tests below name their additional prerequisites.
mod common;

use std::path::PathBuf;

use omni_omnijev_native::executor::{self, Executor};
use serde_json::{Value, json};

/// Actual hash-pinned complete-request tensors qualify the Rust projection,
/// reduction, six features and heads without backbone/topology differences.
#[test]
#[ignore = "requires original OMNIJEV_LM_CAPTURE/OMNIJEV_LM_LAYOUT and unmerged reference model/CUDA; CUA_S1_GRAPH=0"]
fn reference_actual_lm_groups_and_features() {
    use half::bf16;
    use omni_omnijev_native::{
        contract::Kind,
        heads::{self, Heads},
    };
    use omni_qwen3_5_native::model::{Bf16HiddenRows, Model, ReferenceLmTarget};
    use safetensors::{Dtype, SafeTensors};
    use sha2::{Digest, Sha256};
    let path = |key| PathBuf::from(std::env::var_os(key).expect(key));
    let bytes = std::fs::read(path("OMNIJEV_LM_CAPTURE")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        "fb7032237649393af91922c29ff6a3317c4bcd3eb64fc683210d9f9dc03aced7"
    );
    let tensors = SafeTensors::deserialize(&bytes).unwrap();
    let layout: Value =
        serde_json::from_slice(&std::fs::read(path("OMNIJEV_LM_LAYOUT")).unwrap()).unwrap();
    assert_eq!(
        layout["request_sha256"],
        "a34c2d136a27d11a4c9085547feda474cb1f6e5d44fefcf9cfcd410c7ee5fab5"
    );
    let float = |name: &str| {
        let tensor = tensors.tensor(name).unwrap();
        match tensor.dtype() {
            Dtype::F32 => tensor
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect::<Vec<_>>(),
            Dtype::BF16 => tensor
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| bf16::from_le_bytes(*b).to_f32())
                .collect(),
            other => panic!("unexpected tensor dtype {other:?}"),
        }
    };
    let dir = path("OMNIJEV_REFERENCE_EXPORT");
    let mut model = Model::load_with_lora(
        &dir,
        &path("OMNIJEV_CUDA_LIB"),
        &dir.join("adapter_model.safetensors"),
        2.,
    )
    .unwrap();
    model.enable_reference_text_numerics().unwrap();
    let mut picked = vec![Vec::new(); 35];
    let mut first = vec![None; 35];
    let mut metrics = serde_json::Map::new();
    for (name, order_name, m) in [("option", "option_order", 112), ("first", "zq_order", 31)] {
        let tensor = tensors.tensor(&format!("{name}.hidden")).unwrap();
        assert_eq!(tensor.shape(), &[m, 2560]);
        assert_eq!(tensor.dtype(), Dtype::BF16);
        let hidden = Bf16HiddenRows::new(
            2560,
            tensor
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| bf16::from_le_bytes(*b))
                .collect(),
        )
        .unwrap();
        let order = layout["chunks"][0][order_name].as_array().unwrap();
        let targets: Vec<_> = order
            .iter()
            .enumerate()
            .map(|(row, target)| ReferenceLmTarget {
                hidden: &hidden,
                row,
                token: target["target"].as_u64().unwrap() as u32,
            })
            .collect();
        let actual = model.reference_lm_logprobs(&targets).unwrap();
        let expected = float(&format!("{name}.picked"));
        assert_eq!(actual.len(), expected.len());
        let max = actual
            .iter()
            .zip(&expected)
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max);
        assert!(max <= 1e-5, "{name} picked log-prob maxabs {max}");
        let different = actual
            .iter()
            .zip(&expected)
            .filter(|(a, e)| a.to_bits() != e.to_bits())
            .count();
        let l2 = (actual
            .iter()
            .zip(&expected)
            .map(|(a, e)| (*a as f64 - *e as f64).powi(2))
            .sum::<f64>()
            / expected.iter().map(|e| (*e as f64).powi(2)).sum::<f64>())
        .sqrt();
        metrics.insert(
            name.into(),
            json!({"rows": m, "different": different, "max_abs": max, "relative_l2": l2}),
        );
        for (target, p) in order.iter().zip(actual) {
            let i = target["request_row"].as_u64().unwrap() as usize;
            if name == "option" {
                picked[i].push(p);
            } else {
                first[i] = Some(p);
            }
        }
    }
    let heads = Heads::load(&dir.join("heads.safetensors")).unwrap();
    let (zq, u, expected_feats) = (float("zq"), float("u"), float("feats"));
    let request: Value =
        serde_json::from_slice(&std::fs::read(path("OMNIJEV_ORIGINAL_REQUEST")).unwrap()).unwrap();
    let mut feature_max = 0.0f32;
    let mut head_max = 0.0f32;
    let (mut hi, mut oi) = (0, 0);
    for (group, (_, question)) in layout["groups"]
        .as_array()
        .unwrap()
        .iter()
        .zip(request["questions"].as_object().unwrap())
    {
        let indices: Vec<_> = group
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i.as_u64().unwrap() as usize)
            .collect();
        let ps: Vec<_> = indices.iter().map(|&i| picked[i].clone()).collect();
        let fs: Vec<_> = indices.iter().map(|&i| first[i]).collect();
        let feats = heads::lm_features(&ps, &fs);
        for (&i, feature) in indices.iter().zip(&feats) {
            for (a, e) in feature.iter().zip(&expected_feats[i * 6..(i + 1) * 6]) {
                feature_max = feature_max.max((a - e).abs());
            }
        }
        let q = &zq[indices[0] * 2560..(indices[0] + 1) * 2560];
        let options: Vec<_> = indices
            .iter()
            .map(|&i| u[i * 2560..(i + 1) * 2560].to_vec())
            .collect();
        let kind = match question["type"].as_str().unwrap() {
            "choice" => Kind::Choice,
            "noul" => Kind::Noul,
            "score" => Kind::Score,
            other => panic!("{other}"),
        };
        let (actual, key) = if kind == Kind::Score {
            let key = format!("ord.{oi}");
            oi += 1;
            (heads.ordinal(q, &options).unwrap(), key)
        } else {
            let key = format!("head.{hi}");
            hi += 1;
            (
                heads
                    .option_probabilities(&options, q, kind.type_id(), &feats)
                    .unwrap(),
                key,
            )
        };
        let expected = float(&key);
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            head_max = head_max.max((a - e).abs());
        }
    }
    assert!(feature_max <= 2e-5, "features maxabs {feature_max}");
    assert!(head_max <= 1e-5, "same-hidden heads maxabs {head_max}");
    metrics.insert(
        "features".into(),
        json!({"shape": [35,6], "max_abs": feature_max}),
    );
    metrics.insert(
        "heads".into(),
        json!({"questions": 12, "max_abs": head_max}),
    );
    println!("{}", Value::Object(metrics.clone()));
    if let Some(path) = std::env::var_os("OMNIJEV_LM_RESULT") {
        std::fs::write(path, serde_json::to_vec_pretty(&metrics).unwrap()).unwrap();
    }
}

#[test]
#[ignore = "requires CUDA, OMNIJEV_REFERENCE_EXPORT, OMNIJEV_CUDA_LIB and original complete OMNIJEV_ORIGINAL_REQUEST; CUA_S1_GRAPH=0"]
fn reference_whole_request_outputs_and_recovery() {
    let path = |key| PathBuf::from(std::env::var_os(key).expect(key));
    let (mut executor, processor) = Executor::load_one_shot_reference(
        &path("OMNIJEV_REFERENCE_EXPORT"),
        &path("OMNIJEV_CUDA_LIB"),
    )
    .unwrap();
    let raw = std::fs::read(path("OMNIJEV_ORIGINAL_REQUEST")).unwrap();
    let mut prepared = processor.prepare(&raw).unwrap();
    assert_eq!(prepared.questions.len(), 12);
    assert_eq!(prepared.layout.prefix, 581);
    assert_eq!(prepared.layout.processed_tokens, 1214);
    let outputs = executor
        .head_outputs(&prepared, false)
        .expect("all three types need one-shot outputs");
    assert_eq!(outputs.len(), 12);
    for (q, output) in prepared.questions.iter().zip(&outputs) {
        assert_eq!(output.len(), q.keys.len());
        assert!(output.iter().all(|p| p.is_finite()));
    }
    assert!(executor.head_outputs(&prepared, true).is_err());
    let saved = prepared.inputs[0].rows[0].targets[0];
    prepared.inputs[0].rows[0].targets[0].1 = u32::MAX;
    assert!(executor.head_outputs(&prepared, false).is_err());
    prepared.inputs[0].rows[0].targets[0] = saved;
    assert_eq!(outputs, executor.head_outputs(&prepared, false).unwrap());
    let result = executor.execute(&prepared).unwrap();
    assert_eq!(result.len(), 12);
    assert!(result.iter().all(Value::is_object));
}

/// Real original #114 regression: compilation always sees all twelve questions.
/// The externally supplied request/oracle stay private and hash-pinned by the
/// validation recipe; no source images or checkpoint tensors are checked in.
#[test]
#[ignore = "requires CUDA, OMNIJEV_REFERENCE_EXPORT, OMNIJEV_CUDA_LIB, OMNIJEV_ORIGINAL_REQUEST and OMNIJEV_REFERENCE_ORACLE; CUA_S1_GRAPH=0"]
fn unmerged_reference_original_q2_gate() {
    use sha2::{Digest, Sha256};
    let env_path = |key| std::path::PathBuf::from(std::env::var_os(key).expect(key));
    let dir = env_path("OMNIJEV_REFERENCE_EXPORT");
    let (mut executor, processor) =
        executor::Executor::load_one_shot_reference(&dir, &env_path("OMNIJEV_CUDA_LIB")).unwrap();
    let raw = std::fs::read(env_path("OMNIJEV_ORIGINAL_REQUEST")).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&raw)),
        "a34c2d136a27d11a4c9085547feda474cb1f6e5d44fefcf9cfcd410c7ee5fab5"
    );
    let prepared = processor.prepare(&raw).unwrap();
    assert_eq!(prepared.questions.len(), 12);
    assert_eq!(prepared.layout.prefix, 581);
    assert_eq!(prepared.layout.processed_tokens, 1214);
    let index = prepared
        .questions
        .iter()
        .position(|q| q.id == "q2")
        .unwrap();
    assert!(executor.head_outputs(&prepared, true).is_err());
    assert!(
        executor
            .score_one_shot(&prepared, prepared.questions.len())
            .is_err()
    );
    let nonscore = prepared
        .questions
        .iter()
        .position(|q| q.kind != omni_omnijev_native::contract::Kind::Score)
        .unwrap();
    assert!(executor.score_one_shot(&prepared, nonscore).is_err());
    let read = executor.score_one_shot(&prepared, index).unwrap();
    let repeat = executor.score_one_shot(&prepared, index).unwrap();
    assert_eq!(read.zq, repeat.zq);
    assert_eq!(read.u, repeat.u);
    assert_eq!(read.probabilities, repeat.probabilities);
    if let Some(path) = std::env::var_os("OMNIJEV_REFERENCE_RESULT") {
        std::fs::write(
            path,
            serde_json::to_vec_pretty(&json!({
                "question_count": prepared.questions.len(), "prefix": prepared.layout.prefix,
                "input_tokens": prepared.layout.processed_tokens, "qid": "q2",
                "zq": read.zq, "u": read.u, "mu": read.probabilities, "answer": read.answer,
                "repeat_exact": true, "unsupported_modes_rejected": true,
            }))
            .unwrap(),
        )
        .unwrap();
    }
    let oracle: Value =
        serde_json::from_slice(&std::fs::read(env_path("OMNIJEV_REFERENCE_ORACLE")).unwrap())
            .unwrap();
    let expected = &oracle["official_full_answer"]["q2"];
    assert_eq!(read.answer["score"], expected["score"]);
    let delta = (read.answer["confidence"].as_f64().unwrap()
        - expected["confidence"].as_f64().unwrap())
    .abs();
    assert!(
        delta <= 0.01,
        "original q2 confidence delta {delta}: native={}, reference={expected}",
        read.answer
    );
    for (key, p) in read.answer["probabilities"].as_object().unwrap() {
        let e = &expected["probabilities"][key];
        assert!(
            (p.as_f64().unwrap() - e.as_f64().unwrap()).abs() <= 0.01,
            "level {key}"
        );
    }
}

#[test]
#[ignore = "run with CUA_S1_GRAPH=1; verifies rejection before reading artifacts or loading CUDA"]
fn reference_graph_is_rejected_before_loading() {
    assert_eq!(std::env::var("CUA_S1_GRAPH").as_deref(), Ok("1"));
    let error = Executor::load_one_shot_reference(
        std::path::Path::new("/missing-export"),
        std::path::Path::new("/missing-cuda-library"),
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("CUA_S1_GRAPH=0"));
}

fn executor() -> (Executor, omni_omnijev_native::processing::Processor) {
    let var =
        |name: &str| PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("set {name}")));
    Executor::load(&var("OMNIJEV_EXPORT"), &var("OMNIJEV_CUDA_LIB")).unwrap()
}

fn request(questions: Value) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "model": "tinnel123/OmniJev",
        "state": {"images": [common::png_url(320, 200, [200, 40, 40])]},
        "questions": questions,
    }))
    .unwrap()
}

/// The answers without the timing fields.
fn decisions(response: &Value) -> Value {
    let mut answers = response["answers"].clone();
    for answer in answers.as_object_mut().unwrap().values_mut() {
        let answer = answer.as_object_mut().unwrap();
        answer.remove("latency_s");
        answer.remove("latency_total_s");
    }
    answers
}

#[test]
#[ignore = "needs a GPU, OMNIJEV_EXPORT and OMNIJEV_CUDA_LIB"]
fn answers_are_repeatable_and_independent_of_question_order() {
    let (mut executor, processor) = executor();
    let questions = json!({
        "red": {"type": "noul", "instructions": "The image is mostly red."},
        "where": {"type": "noul", "instructions": "There is something in this region.",
                  "region": {"box": [0, 0, 160, 100]}},
        "colour": {"type": "choice", "instructions": "Which colour dominates the image?",
                   "options": [{"key": "red", "text": "red"}, {"key": "green", "text": "green"},
                               {"key": "blue", "text": "blue"}, {"text": "none", "abstain": true}]},
        "warm": {"type": "score", "instructions": "How warm is the colour?",
                 "levels": ["cold", "neutral", "warm", "hot"]},
    });
    let run = |executor: &mut Executor, questions: &Value| {
        let prepared = processor.prepare(&request(questions.clone())).unwrap();
        let answers = executor.execute(&prepared).unwrap();
        executor::response(&prepared, answers)
    };
    let first = run(&mut executor, &questions);
    assert_eq!(first["model"], "tinnel123/OmniJev");
    assert!(first["usage"]["input_tokens"].as_u64().unwrap() > 0);
    let answers = first["answers"].as_object().unwrap();
    assert_eq!(
        answers.keys().collect::<Vec<_>>(),
        ["red", "where", "colour", "warm"]
    );
    for (id, field) in [
        ("red", "noul"),
        ("where", "noul"),
        ("colour", "choice"),
        ("warm", "score"),
    ] {
        assert!(answers[id].get(field).is_some(), "{id}: {}", answers[id]);
        assert!(answers[id]["latency_total_s"].is_number(), "{id}");
    }
    assert_eq!(
        decisions(&run(&mut executor, &questions)),
        decisions(&first)
    );
    // a question's rows continue only the request's prefix and its own text, so the
    // order of the questions changes nothing
    let reversed: serde_json::Map<String, Value> = questions
        .as_object()
        .unwrap()
        .iter()
        .rev()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let mut back = decisions(&run(&mut executor, &Value::Object(reversed)));
    let mut want = decisions(&first);
    back.as_object_mut().unwrap().sort_keys();
    want.as_object_mut().unwrap().sort_keys();
    assert_eq!(back, want);
}

/// Questions on `image` whose first option marker falls at different places against the
/// 64-token chunks: among them one right on a boundary (`zq` read with the question's
/// text), one whose text crosses a boundary, the speed_bench-like and gate questions.
fn reuse_request(processor: &omni_omnijev_native::processing::Processor, image: &str) -> Vec<u8> {
    let gates: Value = serde_json::from_str(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../../recipe/omnijev/gate_questions.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let build = |padding: usize| {
        let mut questions = gates.clone();
        let q = questions.as_object_mut().unwrap();
        q.insert(
            "boundary".into(),
            json!({"type": "choice", "instructions": format!("Count the windows{}.", " again".repeat(padding)),
                   "options": [{"key": "one", "text": "one"}, {"key": "many", "text": "many windows"}]}),
        );
        // text longer than a chunk past any prefix: zq is read in rows that continue it
        q.insert(
            "long".into(),
            json!({"type": "choice", "instructions": "Look at every part of the picture, from the top left corner to the bottom right corner, including any text, icons, people, vehicles, buildings, plants, animals and the sky, then look again at the middle of the picture and at its edges, compare what you see there with what you would expect in an ordinary photograph or screenshot, and then say how much is happening in the whole scene overall.",
                   "options": [{"key": "nothing", "text": "nothing at all"}, {"key": "little", "text": "a little"}, {"key": "lot", "text": "a lot"}]}),
        );
        q.insert(
            "red".into(),
            json!({"type": "noul", "instructions": "The image is mostly red."}),
        );
        serde_json::to_vec(&json!({"model": "tinnel123/OmniJev", "state": {"images": [image]}, "questions": questions})).unwrap()
    };
    // the padding that puts the boundary question's first option on a multiple of 64
    for padding in 0..80 {
        let raw = build(padding);
        let prepared = processor.prepare(&raw).unwrap();
        let i = prepared
            .questions
            .iter()
            .position(|q| q.id == "boundary")
            .unwrap();
        let open = prepared.layout.prefix + prepared.inputs[i].rows[0].zq + 1;
        if open.is_multiple_of(64) {
            return raw;
        }
    }
    panic!("no padding puts the first option on a boundary");
}

#[test]
#[ignore = "needs a GPU, OMNIJEV_EXPORT and OMNIJEV_CUDA_LIB"]
fn prefix_reuse_matches_plain_passes_bit_for_bit() {
    let (mut executor, processor) = executor();
    for (w, h) in [(64, 48), (320, 200), (1600, 900)] {
        let raw = reuse_request(&processor, &common::png_url(w, h, [30, 120, 200]));
        let prepared = processor.prepare(&raw).unwrap();
        // the three ways a question's rows split against the 64-token chunks, each with
        // LM features read
        let l = prepared.layout.prefix;
        let p = l - l % 64;
        let cases: Vec<(bool, bool)> = prepared
            .questions
            .iter()
            .zip(&prepared.inputs)
            .filter(|(q, _)| q.kind != omni_omnijev_native::contract::Kind::Score)
            .map(|(_, input)| {
                let open = l + input.rows[0].zq + 1;
                (open - open % 64 > p, open.is_multiple_of(64))
            })
            .collect();
        for case in [(false, false), (true, false), (true, true)] {
            assert!(
                cases.contains(&case),
                "{w}x{h}: no question splits as {case:?}"
            );
        }
        let shared = executor.head_outputs(&prepared, true).unwrap();
        let plain = executor.head_outputs(&prepared, false).unwrap();
        let bits = |outputs: &[Vec<f32>]| -> Vec<u32> {
            outputs.iter().flatten().map(|v| v.to_bits()).collect()
        };
        assert_eq!(bits(&shared), bits(&plain), "{w}x{h}");
        let reused = executor::response(&prepared, executor.execute(&prepared).unwrap());
        let rows = executor::response(&prepared, executor.execute_rows(&prepared).unwrap());
        assert_eq!(decisions(&reused), decisions(&rows), "{w}x{h}");
    }
}
