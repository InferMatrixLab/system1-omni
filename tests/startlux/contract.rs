use omni_startlux_native::contract::{
    self, DEFAULT_MAX_MODEL_LEN, Kind, Prepared, RequestError, Settings, WIDE_RESIDUAL,
};
use serde_json::{Value, json};

fn chars(text: &str) -> Result<Vec<u32>, String> {
    Ok(text.chars().map(u32::from).collect())
}

fn decode(ids: &[u32]) -> String {
    ids.iter().map(|id| char::from_u32(*id).unwrap()).collect()
}

fn prepare(raw: &Value, settings: &Settings) -> Result<Prepared, RequestError> {
    let bytes = serde_json::to_vec(raw).unwrap();
    let mut encode = chars;
    contract::prepare_request(&bytes, settings, &mut encode)
}

fn listed() -> Settings {
    Settings {
        max_model_len: DEFAULT_MAX_MODEL_LEN,
        none_check: false,
        allow_none: false,
    }
}

fn prompt_text(prepared: &Prepared, index: usize) -> String {
    decode(&prepared.prompts[index].ids)
}

fn zeros(prepared: &Prepared) -> Vec<Vec<f64>> {
    prepared
        .prompts
        .iter()
        .map(|prompt| vec![0.0; prompt.letters])
        .collect()
}

#[test]
fn pinned_instant_folder_accepts_4b_and_refuses_autothink_and_27b() {
    let config: Value =
        serde_json::from_str(include_str!("fixtures/instant_4b_config.json")).unwrap();
    let decision: Value =
        serde_json::from_str(include_str!("fixtures/decision_config.json")).unwrap();
    contract::require_instant_configs(&config, &decision).unwrap();

    let mut autothink = decision.clone();
    autothink["auto_think"] = json!({"think_mode": "auto"});
    let err = contract::require_instant_configs(&config, &autothink).unwrap_err();
    assert!(err.contains("auto_think"), "{err}");

    let mut bigger = config.clone();
    bigger["text_config"]["hidden_size"] = json!(5120);
    let err = contract::require_instant_configs(&bigger, &decision).unwrap_err();
    assert!(err.contains("Instant 4B"), "{err}");

    let mut causal = config.clone();
    causal["jev_prompt_attention"] = json!("causal");
    let err = contract::require_instant_configs(&causal, &decision).unwrap_err();
    assert!(
        err.contains("AutoThink") || err.contains("bidirectional"),
        "{err}"
    );
}

#[test]
fn choice_noul_and_score_prompts_keep_order_and_full_precision() {
    let request: Value =
        serde_json::from_str(include_str!("fixtures/choice_noul_score.json")).unwrap();
    let mut request = request;
    request["none_check"] = json!(false);
    let prepared = prepare(&request, &Settings::default()).unwrap();
    assert_eq!(prepared.prompts.len(), 3);

    let choice = prompt_text(&prepared, 0);
    let expected_choice = contract::chat_prompt(
        "Evidence:\n{\"ticket\": \"I was charged twice\"}\n\nQuestion: What should happen?\nOptions:\nA) refund: Refund the charge\nB) keep: Keep the charge",
    );
    assert_eq!(choice, expected_choice);
    assert!(choice.ends_with(contract::THINK_OFF));
    assert!(!choice.contains("None of the above"));

    let noul = prompt_text(&prepared, 1);
    assert!(noul.contains("A) yes\nB) no"), "{noul}");
    assert!(!noul.contains("true"));

    let score = prompt_text(&prepared, 2);
    assert!(score.contains("A) 0: low\nB) 1: high"), "{score}");

    let answer = contract::finish(
        &prepared,
        &zeros(&prepared),
        &contract::plan_finals(
            &prepared,
            &zeros(&prepared),
            &Settings::default(),
            &mut chars,
        )
        .unwrap(),
        &[],
    )
    .unwrap();
    assert_eq!(answer["model"], contract::MODEL_ID);
    assert_eq!(answer["usage"]["output_tokens"], 0);
    let tokens: usize = prepared.prompts.iter().map(|prompt| prompt.ids.len()).sum();
    assert_eq!(answer["usage"]["input_tokens"], tokens);
    assert_eq!(answer["answers"]["route"]["choice"], "refund");
    assert_eq!(
        answer["answers"]["route"]["probabilities"]["refund"].as_f64(),
        Some(0.5)
    );
    assert_eq!(
        answer["answers"]["route"]["probabilities"]["keep"].as_f64(),
        Some(0.5)
    );
    assert!(answer["answers"]["route"].get("none_probability").is_none());
    assert_eq!(answer["answers"]["ask"]["type"], "noul");
    assert_eq!(answer["answers"]["ask"]["noul"].as_f64(), Some(0.5));
    assert_eq!(answer["answers"]["severity"]["score"].as_f64(), Some(0.5));
    let keys: Vec<_> = answer["answers"]["severity"]["probabilities"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    assert_eq!(keys, ["0".to_string(), "1".to_string()]);
    assert_eq!(answer["answers"]["severity"]["legend"]["0"], "low");
}

#[test]
fn default_none_check_adds_the_extra_option_without_abstaining() {
    let request: Value =
        serde_json::from_str(include_str!("fixtures/choice_noul_score.json")).unwrap();
    let prepared = prepare(&request, &Settings::default()).unwrap();
    assert_eq!(prepared.prompts[0].letters, 3);
    assert!(prompt_text(&prepared, 0).contains("C) None of the above"));
    let finals = contract::plan_finals(
        &prepared,
        &zeros(&prepared),
        &Settings::default(),
        &mut chars,
    )
    .unwrap();
    let answer = contract::finish(&prepared, &zeros(&prepared), &finals, &[]).unwrap();
    assert_eq!(answer["answers"]["route"]["choice"], "refund");
    assert!(
        answer["answers"]["route"]["none_probability"]
            .as_f64()
            .unwrap()
            > 0.0
    );
    let probs = answer["answers"]["route"]["probabilities"]
        .as_object()
        .unwrap();
    assert_eq!(probs.len(), 2);
    let sum: f64 = probs.values().map(|value| value.as_f64().unwrap()).sum();
    assert!((sum - 1.0).abs() < 1e-12, "{sum}");
}

#[test]
fn bare_letter_ids_are_hidden_and_json_state_matches_python_spacing() {
    assert_eq!(
        contract::state_text(&json!({"ticket": "I was charged twice"})),
        r#"{"ticket": "I was charged twice"}"#
    );
    let long = json!(["a", "b", "c", "d", "e", "f", "g", "h"]);
    let text = contract::state_text(&long);
    assert!(text.contains(r#""_index": 0"#), "{text}");
    assert!(text.contains(", "), "{text}");
    assert!(!text.contains(r"\u"));
    assert_eq!(contract::state_text(&json!("café")), "café");
    let mixed = json!({
        "ticket": "I was charged twice",
        "note": "café",
        "city": "東京",
        "tiny": 1e-5,
        "huge": 1e16,
        "edge": 1e-4,
        "plain": 1e15
    });
    assert_eq!(
        contract::state_text(&mixed),
        r#"{"ticket": "I was charged twice", "note": "café", "city": "東京", "tiny": 1e-05, "huge": 1e+16, "edge": 0.0001, "plain": 1000000000000000.0}"#
    );
    assert_eq!(
        contract::state_text(&json!({"bell": "a\u{0001}b"})),
        "{\"bell\": \"a\\u0001b\"}"
    );
    let indexed = json!(["a", "b", "c", "d", "e", "f", "g", "café"]);
    let indexed_text = contract::state_text(&indexed);
    assert!(
        indexed_text.contains(r#""value": "café""#),
        "{indexed_text}"
    );
    assert!(!indexed_text.contains('\\'), "{indexed_text}");

    let dumped = prepare(
        &json!({
            "state": mixed,
            "none_check": false,
            "questions": {"q": {"type": "choice", "instructions": "City?", "criteria": {"A": 1e-5, "B": "東京"}}}
        }),
        &listed(),
    )
    .unwrap();
    let dumped_prompt = prompt_text(&dumped, 0);
    assert!(
        dumped_prompt.contains(r#""note": "café""#),
        "{dumped_prompt}"
    );
    assert!(
        dumped_prompt.contains(r#""city": "東京""#),
        "{dumped_prompt}"
    );
    assert!(
        dumped_prompt.contains(r#""tiny": 1e-05"#),
        "{dumped_prompt}"
    );
    assert!(
        dumped_prompt.contains(r#""huge": 1e+16"#),
        "{dumped_prompt}"
    );
    assert!(
        dumped_prompt.contains("A) 1e-05\nB) 東京"),
        "{dumped_prompt}"
    );
    assert!(!dumped_prompt.contains('\\'), "{dumped_prompt}");

    let prepared = prepare(
        &json!({
            "state": "s",
            "none_check": false,
            "questions": {"q": {"type": "choice", "instructions": "City?", "criteria": {"A": "Paris", "B": "London"}}}
        }),
        &listed(),
    )
    .unwrap();
    let prompt = prompt_text(&prepared, 0);
    assert!(prompt.contains("A) Paris\nB) London"), "{prompt}");
    assert!(!prompt.contains("A) A:"));
}

#[test]
fn single_option_skips_the_forward_and_wide_lists_are_not_dropped() {
    let prepared = prepare(
        &json!({
            "state": "s",
            "questions": {"q": {"type": "choice", "instructions": "Only one.", "criteria": {"only": "Take it"}}}
        }),
        &listed(),
    )
    .unwrap();
    assert!(prepared.prompts.is_empty());
    let finals = contract::plan_finals(&prepared, &[], &listed(), &mut chars).unwrap();
    let answer = contract::finish(&prepared, &[], &finals, &[]).unwrap();
    assert_eq!(answer["answers"]["q"]["choice"], "only");
    assert_eq!(answer["answers"]["q"]["probabilities"]["only"], 1.0);
    assert_eq!(answer["usage"]["input_tokens"], 0);

    let mut criteria = serde_json::Map::new();
    for index in 0..27 {
        criteria.insert(format!("o{index:02}"), json!(format!("c{index:02}")));
    }
    let prepared = prepare(
        &json!({"state": "s", "questions": {"q": {"type": "choice", "instructions": "Pick.", "criteria": criteria}}}),
        &Settings::default(),
    )
    .unwrap();
    assert_eq!(
        prepared.prompts.len(),
        2,
        "27 options with none_check use two groups"
    );
    assert_eq!(prepared.prompts[0].letters, 14);
    assert_eq!(prepared.prompts[1].letters, 13);
    assert!(!prompt_text(&prepared, 0).contains("None of the above"));
    let finals = contract::plan_finals(
        &prepared,
        &zeros(&prepared),
        &Settings::default(),
        &mut chars,
    )
    .unwrap();
    assert_eq!(finals.prompts.len(), 1);
    assert_eq!(finals.prompts[0].letters, 7);
    assert!(decode(&finals.prompts[0].ids).contains("None of the above"));
    let answer =
        contract::finish(&prepared, &zeros(&prepared), &finals, &zeros_final(&finals)).unwrap();
    let probs = answer["answers"]["q"]["probabilities"].as_object().unwrap();
    let keys: Vec<_> = probs.keys().cloned().collect();
    let expected: Vec<_> = (0..27).map(|index| format!("o{index:02}")).collect();
    assert_eq!(keys, expected);
    assert_eq!(answer["answers"]["q"]["choice"], "o00");
    let sum: f64 = probs.values().map(|value| value.as_f64().unwrap()).sum();
    assert!((sum - 1.0).abs() < 1e-9, "{sum}");
    let group_tokens: usize = prepared.prompts.iter().map(|prompt| prompt.ids.len()).sum();
    let total = group_tokens + finals.prompts[0].ids.len();
    assert_eq!(answer["usage"]["input_tokens"], total);
    assert!((WIDE_RESIDUAL - 0.001).abs() < 1e-15);
}

fn zeros_final(finals: &contract::FinalRound) -> Vec<Vec<f64>> {
    finals
        .prompts
        .iter()
        .map(|prompt| vec![0.0; prompt.letters])
        .collect()
}

#[test]
fn allow_none_abstains_only_when_the_extra_option_wins() {
    let request = json!({
        "state": "s",
        "allow_none": true,
        "none_check": false,
        "questions": {"q": {"type": "choice", "instructions": "Pick.", "criteria": {"a": "A", "b": "B"}}}
    });
    let prepared = prepare(&request, &listed()).unwrap();
    assert_eq!(prepared.prompts[0].letters, 3);
    let logits = vec![vec![0.0, 0.0, 10.0]];
    let finals = contract::plan_finals(&prepared, &logits, &listed(), &mut chars).unwrap();
    let answer = contract::finish(&prepared, &logits, &finals, &[]).unwrap();
    assert_eq!(answer["answers"]["q"]["choice"], Value::Null);

    let staying = json!({
        "state": "s",
        "allow_none": false,
        "none_check": true,
        "questions": {"q": {"type": "choice", "instructions": "Pick.", "criteria": {"a": "A", "b": "B"}}}
    });
    let prepared = prepare(&staying, &listed()).unwrap();
    let finals = contract::plan_finals(&prepared, &logits, &listed(), &mut chars).unwrap();
    let answer = contract::finish(&prepared, &logits, &finals, &[]).unwrap();
    assert_eq!(answer["answers"]["q"]["choice"], "a");
    assert!(answer["answers"]["q"]["none_probability"].as_f64().unwrap() > 0.5);
}

#[test]
fn rejections_match_the_instant_contract() {
    let bad = contract::prepare_request(b"not-json", &Settings::default(), &mut chars).unwrap_err();
    assert_eq!(bad.status, 400);
    assert_eq!(bad.body()["detail"][0]["type"], "value_error");

    let empty = prepare(
        &json!({"state": "s", "questions": {}}),
        &Settings::default(),
    )
    .unwrap_err();
    assert_eq!(empty.status, 422);

    let missing = prepare(
        &json!({"questions": {"q": {"type": "choice"}}}),
        &Settings::default(),
    )
    .unwrap_err();
    assert_eq!(missing.status, 422);
    assert!(missing.loc.iter().any(|part| part == "state"));

    let images = prepare(
        &json!({"state": "s", "images": ["abcd"], "questions": {"q": {"type": "noul"}}}),
        &Settings::default(),
    )
    .unwrap_err();
    assert_eq!(images.status, 422);
    assert!(images.message.contains("text only"), "{}", images.message);

    let think = prepare(
        &json!({"state": "s", "think_mode": "always", "questions": {"q": {"type": "noul"}}}),
        &Settings::default(),
    )
    .unwrap_err();
    assert_eq!(think.status, 422);
    assert!(think.message.contains("AutoThink"), "{}", think.message);

    let plain = prepare(
        &json!({"state": "s", "think_mode": "plain", "questions": {"q": {"type": "bool"}}}),
        &Settings::default(),
    )
    .unwrap();
    assert_eq!(plain.prompts.len(), 1);

    let stream = prepare(
        &json!({"state": "s", "think_mode": "never", "thought_generation": "stream", "questions": {"q": {"type": "noul"}}}),
        &Settings::default(),
    )
    .unwrap_err();
    assert_eq!(stream.status, 422);
    assert!(stream.message.contains("stream"), "{}", stream.message);

    let details = prepare(
        &json!({"state": "s", "return_details": true, "questions": {"q": {"type": "noul"}}}),
        &Settings::default(),
    )
    .unwrap_err();
    assert_eq!(details.status, 422);

    let levels: Vec<_> = (0..11).map(|index| format!("level {index}")).collect();
    let score = prepare(
        &json!({"state": "s", "questions": {"q": {"type": "score", "instructions": "Rate.", "criteria": levels}}}),
        &Settings::default(),
    )
    .unwrap_err();
    assert_eq!(score.status, 422);
    assert!(score.message.contains("10"), "{}", score.message);

    let odd = prepare(
        &json!({"state": "s", "questions": {"q": {"type": "choice", "instructions": "Pick.", "criteria": [true]}}}),
        &Settings::default(),
    )
    .unwrap_err();
    assert_eq!(odd.status, 422);
    assert!(
        odd.message.contains("strings or numbers"),
        "{}",
        odd.message
    );

    let tiny = Settings {
        max_model_len: 8,
        ..Settings::default()
    };
    let limited = prepare(
        &json!({"state": "s", "questions": {"q": {"type": "noul"}}}),
        &tiny,
    )
    .unwrap_err();
    assert_eq!(limited.status, 422);
    assert!(
        limited.message.contains("does not truncate"),
        "{}",
        limited.message
    );

    let named = prepare(
        &json!({"state": "s", "questions": {"q": {"type": "choice", "instructions": "Pick.", "criteria": {"None of the above": "extra"}}}}),
        &Settings::default(),
    )
    .unwrap_err();
    assert_eq!(named.status, 422);
}

#[test]
fn instant_4b_packs_bidirectional_v_to_hk_dh() {
    let config: Value =
        serde_json::from_str(include_str!("fixtures/instant_4b_config.json")).unwrap();
    let text = &config["text_config"];
    let heads = text["num_attention_heads"].as_u64().unwrap() as usize;
    let kv = text["num_key_value_heads"].as_u64().unwrap() as usize;
    let dh = text["head_dim"].as_u64().unwrap() as usize;
    assert_eq!((heads, kv, dh), (16, 4, 256));
    let layout = omni_qwen3_5_native::model::prompt_bidirectional_v_layout(heads, kv, dh);
    assert_eq!(layout.ldv, 1024);
    assert_eq!(layout.source_stride, 10240);
    assert_eq!(layout.source_column + layout.ldv, layout.source_stride);
    assert_eq!(layout.row_bytes(), 2048);
    assert_eq!(layout.source_pitch_bytes(), 20_480);
    assert_eq!(layout.packed_offset(0), 0);
    assert_eq!(layout.packed_offset(3), 3 * 2048);
    assert_ne!(layout.ldv, layout.source_stride);
}

#[test]
fn temperatures_and_confidence_follow_the_pinned_readout() {
    let even = contract::probabilities(&[0.0, 0.0], Kind::Choice.temperature());
    assert!((even[0] - 0.5).abs() < 1e-12);
    assert!((contract::choice_confidence(&[1.0, 0.0]) - 1.0).abs() < 1e-12);
    assert!(contract::choice_confidence(&[0.5, 0.5]).abs() < 1e-12);
    assert!((contract::score_confidence(&[1.0, 0.0]) - 1.0).abs() < 1e-12);
    assert_eq!(Kind::Noul.temperature(), contract::TEMP_NOUL);
    assert_eq!(contract::WEIGHTS_REVISION.len(), 40);
    assert_eq!(contract::CODE_REVISION.len(), 40);
}
