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
    assert!(executor.execute(&prepared).is_err());
    assert!(executor.execute_rows(&prepared).is_err());
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
