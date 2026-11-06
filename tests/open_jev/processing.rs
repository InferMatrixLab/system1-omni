use super::*;
use crate::contract::CHECKPOINTS;
use tokenizers::{
    models::wordlevel::WordLevel, pre_tokenizers::whitespace::WhitespaceSplit,
    processors::template::TemplateProcessing,
};

fn processor(max_length: usize) -> Processor {
    let tokens = [
        "[UNK]",
        "[BOS]",
        "Context:",
        "x",
        "Question:",
        "Route",
        "Proposed",
        "answer:",
        "billing",
        "security",
        "technical",
        "Is",
        "this",
        "proposed",
        "answer",
        "correct?",
        "Answer",
        "Yes",
        "or",
        "No.",
        "Verify",
        "the",
        "to",
        "question",
        "yes?",
    ];
    let vocab = tokens
        .into_iter()
        .enumerate()
        .map(|(i, s)| (s.to_owned(), i as u32))
        .collect();
    let mut tokenizer = Tokenizer::new(
        WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .unwrap(),
    );
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    tokenizer.with_post_processor(Some(
        TemplateProcessing::builder()
            .try_single("[BOS] $A")
            .unwrap()
            .special_tokens(vec![("[BOS]", 1)])
            .build()
            .unwrap(),
    ));
    Processor {
        checkpoint: &CHECKPOINTS[0],
        parallel_tokenization: true,
        tokenizer,
        prefix: "".into(),
        suffix: "".into(),
        temperature: 0.5,
        max_length,
    }
}

const REQUEST: &[u8] = br#"{"state":"x","questions":{
    "z":{"type":"choice","instructions":"Route","criteria":{"technical":null,"billing":null}},
    "a":{"type":"noul","instructions":"Verify"}}}"#;

#[test]
fn serial_preparation_preserves_tokens_and_question_mapping() {
    let mut processor = processor(4096);
    let expected = processor.prepare(REQUEST).unwrap().inputs;
    processor.disable_parallel_tokenization();
    assert!(!processor.parallel_tokenization);
    assert_eq!(processor.prepare(REQUEST).unwrap().inputs, expected);
}

#[test]
fn preparation_applies_the_saved_chat_prefix_and_suffix() {
    let mut processor = processor(4096);
    processor.prefix = "billing ".into();
    processor.suffix = " security".into();
    let prepared = processor.prepare(REQUEST).unwrap();
    // BOS precedes the saved prefix; the suffix follows the question prompt.
    assert_eq!(
        prepared.inputs[0][0],
        [
            1, 8, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 9
        ]
    );
}

#[test]
fn prepared_candidates_have_exact_ids_and_finished_responses_keep_mapping() {
    let prepared = processor(4096).prepare(REQUEST).unwrap();
    assert_eq!(
        prepared.inputs,
        vec![
            vec![
                vec![1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                vec![1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19]
            ],
            vec![vec![
                1, 2, 3, 4, 20, 11, 21, 14, 22, 12, 23, 24, 16, 17, 18, 19
            ]]
        ]
    );
    let body = prepared
        .context
        .finish(vec![vec![3.0, 3.0], vec![0.0]])
        .unwrap();
    assert_eq!(
        body["answers"],
        json!({
        "z":{"type":"choice","choice":"technical","probabilities":{"technical":0.5,"billing":0.5},"confidence":0.0},
        "a":{"type":"noul","noul":0.5}})
    );
    assert_eq!(
        body["answers"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["z", "a"]
    );
    assert_eq!(body["usage"], json!({"input_tokens":50,"output_tokens":0}));
    assert_eq!(body["metadata"]["candidate_sequences"], 3);
    assert_eq!(body["metadata"]["temperature"], 0.5);
    assert_eq!(body["metadata"]["max_length"], 4096);
    assert_eq!(
        body["metadata"]["method"],
        "native_merged_lora_decision_head"
    );
    assert_eq!(
        body["metadata"]["base_revision"],
        CHECKPOINTS[0].base_revision
    );
    assert_eq!(
        body["metadata"]["prefix_cache"],
        json!({"policy": "default", "enabled": false, "scope": "none", "saved_tokens": 0})
    );
    assert!(body["metadata"]["inference_seconds"].as_f64().unwrap() >= 0.0);
    assert_eq!(body["model"], CHECKPOINTS[0].model_id);
}

#[test]
fn prefix_metadata_uses_executed_savings_without_changing_answers_or_usage() {
    use crate::executor::PrefixUse;
    use omni_qwen3_5_native::prefix::PrefixMode;
    let request = serde_json::to_vec(&json!({
        "state": "x ".repeat(768),
        "questions": {"q": {"type": "choice", "instructions": "Route",
            "criteria": {"billing": null, "security": null, "technical": null}}}
    }))
    .unwrap();
    let rows = vec![vec![1.0, 2.0, 3.0]];
    let baseline = processor(4096)
        .prepare(&request)
        .unwrap()
        .context
        .finish(rows.clone())
        .unwrap();
    // Same candidate count and common-prefix length, but different executed groups.
    // All three share: save 1024. Two share and one falls back: save only 512.
    for saved in [1024, 512, 0] {
        let actual = processor(4096)
            .prepare(&request)
            .unwrap()
            .context
            .finish_with_prefix(
                rows.clone(),
                Some(&PrefixUse {
                    mode: PrefixMode::Off,
                    shared: saved > 0,
                    across_questions: false,
                    saved_tokens: saved,
                }),
            )
            .unwrap();
        assert_eq!(actual["answers"], baseline["answers"]);
        assert_eq!(actual["usage"], baseline["usage"]);
        assert_eq!(actual["metadata"]["candidate_sequences"], 3);
        // Exact object equality also rejects obsolete or unexpected fields.
        assert_eq!(
            actual["metadata"]["prefix_cache"],
            json!({"policy": "default", "enabled": saved > 0,
                "scope": if saved > 0 { "within_question" } else { "none" },
                "saved_tokens": saved})
        );
    }
}

#[test]
fn responses_name_the_loaded_checkpoint() {
    let mut processor = processor(4096);
    processor.checkpoint = &CHECKPOINTS[1];
    let body = processor
        .prepare(REQUEST)
        .unwrap()
        .context
        .finish(vec![vec![3.0, 3.0], vec![0.0]])
        .unwrap();
    assert_eq!(body["model"], "Qwen/Qwen3.5-9B");
    assert_eq!(
        body["metadata"]["base_revision"],
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    );
}

#[test]
fn preparation_rejects_later_oversize_candidates_before_execution() {
    let processor = processor(17);
    processor.prepare(REQUEST).unwrap();
    let mut raw: Value = serde_json::from_slice(REQUEST).unwrap();
    raw["questions"]["a"]["instructions"] = json!("x ".repeat(18));
    let error = processor
        .prepare(&serde_json::to_vec(&raw).unwrap())
        .err()
        .unwrap();
    assert!(error.to_string().starts_with("question a: "));
    assert!(
        error
            .to_string()
            .ends_with("exceeds max_length=17; no silent truncation")
    );
    assert_eq!(
        processor
            .prepare(br#"{"state":"x","questions":{}}"#)
            .err()
            .unwrap()
            .to_string(),
        "questions must be nonempty and within the server question limit"
    );
}

fn close(actual: &Value, expected: &Value) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => assert!(
            (a.as_f64().unwrap() - b.as_f64().unwrap()).abs() < 1e-12,
            "{a} != {b}"
        ),
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(a.len(), b.len());
            for (key, value) in b {
                close(&a[key], value);
            }
        }
        _ => assert_eq!(actual, expected),
    }
}

#[test]
fn complete_questions_match_existing_choice_score_and_noul_reference() {
    // Existing golden responses generated from Open-Jev @ 3308a15, not from this processor.
    let cases: Value = serde_json::from_str(include_str!("data/contract.json")).unwrap();
    for case in cases.as_array().unwrap() {
        let mut processor = processor(4096);
        processor.temperature = case["temperature"].as_f64().unwrap();
        let prepared = processor
            .prepare(&serde_json::to_vec(&case["request"]).unwrap())
            .unwrap();
        let rows = prepared
            .context
            .questions
            .iter()
            .zip(case["logits"].as_array().unwrap())
            .map(|(q, row)| {
                let values = row.as_array().unwrap();
                if q.kind == Kind::Noul {
                    // The reference fixture has arbitrary two-class logits; the
                    // native head emits their difference with false fixed at zero.
                    vec![(values[1].as_f64().unwrap() - values[0].as_f64().unwrap()) as f32]
                } else {
                    values.iter().map(|v| v.as_f64().unwrap() as f32).collect()
                }
            })
            .collect();
        let body = prepared.context.finish(rows).unwrap();
        close(&body["answers"], &case["response"]["answers"]);
    }
}

#[test]
fn finishing_rejects_missing_candidates_and_nonfinite_scores() {
    for rows in [
        vec![vec![0.0, 0.0]],
        vec![vec![0.0], vec![0.0]],
        vec![vec![f32::NAN, 0.0], vec![0.0]],
    ] {
        assert!(
            processor(4096)
                .prepare(REQUEST)
                .unwrap()
                .context
                .finish(rows)
                .is_err()
        );
    }
}

#[test]
fn prefix_metadata_reports_how_candidates_ran() {
    use crate::executor::PrefixUse;
    use omni_qwen3_5_native::prefix::PrefixMode;
    let finish = |prefix: Option<PrefixUse>| {
        processor(4096)
            .prepare(REQUEST)
            .unwrap()
            .context
            .finish_with_prefix(vec![vec![3.0, 3.0], vec![0.0]], prefix.as_ref())
            .unwrap()["metadata"]["prefix_cache"]
            .clone()
    };
    let used = |mode, shared, across_questions, saved_tokens| {
        Some(PrefixUse {
            mode,
            shared,
            across_questions,
            saved_tokens,
        })
    };
    let off = json!({"policy": "default", "enabled": false, "scope": "none", "saved_tokens": 0});
    assert_eq!(finish(None), off);
    assert_eq!(finish(used(PrefixMode::Off, false, false, 0)), off);
    assert_eq!(
        finish(used(PrefixMode::Fixed, false, false, 0)),
        json!({"policy": "fixed", "enabled": false, "scope": "none", "saved_tokens": 0})
    );
    assert_eq!(
        finish(used(PrefixMode::Shared, true, false, 128)),
        json!({"policy": "shared", "enabled": true, "scope": "within_question", "saved_tokens": 128})
    );
    assert_eq!(
        finish(used(PrefixMode::Auto, true, true, 704)),
        json!({"policy": "auto", "enabled": true, "scope": "across_questions", "saved_tokens": 704})
    );
    assert_eq!(
        finish(used(PrefixMode::Auto, false, false, 0)),
        json!({"policy": "auto", "enabled": false, "scope": "none", "saved_tokens": 0})
    );
}

fn question(id: &str, prompts: &[&str]) -> Question {
    Question {
        id: id.into(),
        kind: Kind::Choice,
        keys: (0..prompts.len()).map(|i| i.to_string()).collect(),
        prompts: prompts.iter().map(|s| s.to_string()).collect(),
        legend: None,
    }
}

#[test]
fn parallel_tokenization_keeps_candidate_order_across_chunks_and_questions() {
    let p = processor(4096);
    let prompts: Vec<_> = (1..=33).map(|i| "x ".repeat(i)).collect();
    let refs: Vec<_> = prompts.iter().map(String::as_str).collect();
    let questions = [
        question("many", &refs),
        question("last", &["billing", "security"]),
    ];
    let actual =
        encode_questions(&p.tokenizer, "technical ", " Route", 4096, &questions, true).unwrap();
    for (row, q) in actual.iter().zip(&questions) {
        for (ids, prompt) in row.iter().zip(&q.prompts) {
            let reference = p
                .tokenizer
                .encode(format!("technical {} Route", prompt.trim()), true)
                .unwrap();
            assert_eq!(ids, reference.get_ids());
        }
    }
    assert_eq!(actual[0].len(), 33);
    assert_eq!(actual[1][0], [1, 10, 8, 5]);
    assert_eq!(actual[1][1], [1, 10, 9, 5]);
}

#[test]
fn batch_longest_padding_retains_independent_candidate_lengths() {
    let mut p = processor(4096);
    p.tokenizer
        .with_padding(Some(tokenizers::PaddingParams::default()));
    let questions = [question("padding", &["x", "x x x"])];
    let actual = encode_questions(&p.tokenizer, "", "", 4096, &questions, true).unwrap();
    assert_eq!(actual, vec![vec![vec![1, 3], vec![1, 3, 3, 3]]]);
}

#[test]
fn batch_failure_retains_the_first_candidate_error() {
    let mut p = processor(4096);
    // Known words still encode; unknown words fail because the unknown token is absent.
    p.tokenizer.with_model(
        WordLevel::builder()
            .vocab([("x".to_owned(), 3)].into_iter().collect())
            .unk_token("[missing]".into())
            .build()
            .unwrap(),
    );
    for prompts in [["x x", "unknown"], ["unknown", "x x"]] {
        let questions = [question("error-order", &prompts)];
        let reference = encode_questions(&p.tokenizer, "", "", 2, &questions, false)
            .unwrap_err()
            .to_string();
        let actual = encode_questions(&p.tokenizer, "", "", 2, &questions, true)
            .unwrap_err()
            .to_string();
        assert_eq!(actual, reference);
        if prompts[0] == "x x" {
            assert_eq!(
                actual,
                "question error-order: 3 tokens exceeds max_length=2; no silent truncation"
            );
        } else {
            assert!(actual.contains("Missing [UNK] token"), "{actual}");
        }
    }
}
