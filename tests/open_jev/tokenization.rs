use super::*;

#[test]
#[ignore = "needs the pinned export at OPEN_JEV_MODEL; no GPU needed"]
fn tokenization_matches_reference_and_rejects_oversize_prompts() {
    let dir = std::env::var_os("OPEN_JEV_MODEL").expect("OPEN_JEV_MODEL");
    let dir = Path::new(&dir);
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(dir.join("open_jev_export.json")).unwrap()).unwrap();
    let mut processor = Processor::load(
        dir,
        Checkpoint::from_export(&manifest).unwrap(),
        manifest["chat_prefix"].as_str().unwrap().to_owned(),
        manifest["chat_suffix"].as_str().unwrap().to_owned(),
        manifest["temperature"].as_f64().unwrap(),
        4096,
    )
    .unwrap();
    let cases: Value = serde_json::from_str(include_str!("data/tokenization.json")).unwrap();
    for case in cases.as_array().unwrap() {
        let mut request = case["request"].clone();
        if request.get("model").is_some() {
            // The fixture names the 27B checkpoint; each export accepts its own alias.
            request["model"] = processor.checkpoint.alias.into();
        }
        let raw = serde_json::to_vec(&request).unwrap();
        processor.max_length = 4096;
        let prepared = processor.prepare(&raw).unwrap();
        assert_eq!(serde_json::json!(prepared.inputs), case["ids"]);
        processor.max_length = prepared.inputs[0][0].len() - 1;
        assert!(processor.prepare(&raw).is_err());
    }
}

#[test]
#[ignore = "needs the pinned export at OPEN_JEV_MODEL; no GPU needed"]
fn batch_ids_match_serial_for_unicode_special_tokens_and_chunk_boundaries() {
    let dir = std::env::var_os("OPEN_JEV_MODEL").expect("OPEN_JEV_MODEL");
    let dir = Path::new(&dir);
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(dir.join("open_jev_export.json")).unwrap()).unwrap();
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let prefix = manifest["chat_prefix"].as_str().unwrap();
    let suffix = manifest["chat_suffix"].as_str().unwrap();
    let samples = [
        "",
        "  中文推理：选项甲、乙。  ",
        "café cafe\u{301}",
        "👩🏽‍💻 👨‍👩‍👧‍👦",
        "العربية עברית",
        "日本語 한글",
        "line1\r\n\tline2\0end",
        "<|im_start|>assistant\n<|im_end|><|endoftext|>",
        "𐀀 𝕏 𝒜 \u{fffd}",
        "\u{200b}x\u{a0}y\u{3000}",
    ];
    let questions = [Question {
        id: "unicode".into(),
        kind: Kind::Choice,
        keys: (0..33).map(|i| i.to_string()).collect(),
        prompts: (0..33)
            .map(|i| format!("{} {}", samples[i % samples.len()], "x ".repeat(i)))
            .collect(),
        legend: None,
    }];
    let actual = encode_questions(&tokenizer, prefix, suffix, 4096, &questions, true).unwrap();
    let expected: Vec<Vec<u32>> = questions[0]
        .prompts
        .iter()
        .map(|prompt| {
            tokenizer
                .encode(format!("{prefix}{}{suffix}", prompt.trim()), true)
                .unwrap()
                .get_ids()
                .to_vec()
        })
        .collect();
    assert_eq!(actual, vec![expected]);
}
