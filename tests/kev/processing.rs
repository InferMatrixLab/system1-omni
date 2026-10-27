use omni_kev_native::{
    contract,
    executor::validate_row,
    postprocess,
    processing::{Processor, Row, sanitize},
};
use omni_qwen3_5_native::json;
use serde_json::{Value, json};

fn fixture() -> Value {
    serde_json::from_str(include_str!("data/processing.json")).unwrap()
}

#[test]
fn official_api_rendering_and_answers() {
    for case in fixture()["cases"].as_array().unwrap() {
        let request = json::parse(&serde_json::to_vec(&case["request"]).unwrap()).unwrap();
        let compiled = contract::compile(&request).unwrap();
        assert_eq!(compiled.state, case["record"]["state"], "{}", case["name"]);
        for (q, expected) in compiled
            .questions
            .iter()
            .zip(case["record"]["questions"].as_array().unwrap())
        {
            assert_eq!(q.instructions, expected["instr"]);
            assert_eq!(json!(q.options), expected["options"]);
        }
        let ps: Vec<Vec<f64>> = serde_json::from_value(case["probabilities"].clone()).unwrap();
        assert_eq!(
            postprocess::answers(&compiled.questions, &ps).unwrap(),
            case["answers"],
            "{}",
            case["name"]
        );
    }
}

#[test]
fn sanitized_controls_and_json_scalars() {
    assert_eq!(
        sanitize("<|a_2|><|box_end|> <|not-valid|>"),
        "<¦a_2¦><¦box_end¦> <|not-valid|>"
    );
    assert_eq!(
        contract::render(&json!([true, false, null, 1.0, 1e-7]), 0),
        "- True\n- False\n- \n- 1.0\n- 1e-07"
    );
    assert_eq!(
        contract::render(&json!(["\u{1c}x", "\u{1f}y"]), 0),
        "- x\n- y"
    );
    assert_eq!(
        postprocess::billing_json(&json!({"😀":"中文"})),
        r#"{"\ud83d\ude00": "\u4e2d\u6587"}"#
    );
}

#[test]
fn invalid_contract_and_readouts_rejected() {
    for request in [
        json!({"questions":{"q":{"type":"noul"}}}),
        json!({"model":"other","state":"","questions":{"q":{"type":"noul"}}}),
        json!({"state":"","questions":{}}),
        json!({"state":"","questions":{"q":{"type":"span"}}}),
        json!({"state":"","questions":{"q":{"type":"choice","criteria":{}}}}),
        json!({"state":"","questions":{"q":{"type":"score","criteria":[]}}}),
        json!({"state":"","questions":{"q":{"type":"score","criteria":vec!["x";256]}}}),
        json!({"state":"","questions":{"q":{"type":"noul","criteria":[]}}}),
    ] {
        assert!(
            contract::compile(request.as_object().unwrap()).is_err(),
            "{request}"
        );
    }
    assert!(json::parse(br#"{"state":1,"state":2}"#).is_err());
    let row: Row = serde_json::from_value(fixture()["cases"][0]["rows"][0].clone()).unwrap();
    validate_row(&row).unwrap();
    let mut bad = row.clone();
    bad.readouts.reverse();
    assert!(validate_row(&bad).is_err());
    let mut bad = row.clone();
    bad.positions[0] = 1;
    assert!(validate_row(&bad).is_err());
    let mut bad = row;
    bad.ids[0] = 999999;
    assert!(validate_row(&bad).is_err());
}

#[test]
fn ties_and_whole_question_temperature() {
    let request =
        json!({"state":"", "questions":{"q":{"type":"choice","criteria":{"z":null,"a":null}}}});
    let compiled = contract::compile(request.as_object().unwrap()).unwrap();
    let answers = postprocess::answers(&compiled.questions, &[vec![0.5, 0.5]]).unwrap();
    assert_eq!(answers["q"]["choice"], "z");
    assert_eq!(answers["q"]["confidence"], 0.0);
    let p = postprocess::probabilities(&[0.0, contract::TEMPERATURE as f32], contract::TEMPERATURE)
        .unwrap();
    assert!((p[1] - 0.7310586).abs() < 1e-6);
    assert!(postprocess::answers(&compiled.questions, &[vec![0.3, 0.3]]).is_err());
}

#[test]
#[ignore = "requires KEV_EXPORT with the pinned base tokenizer; CPU only"]
fn exact_tokens_readouts_and_billing_match_official() {
    let dir = std::env::var("KEV_EXPORT").expect("KEV_EXPORT");
    let processor = Processor::load(std::path::Path::new(&dir)).unwrap();
    let unicode: Value =
        serde_json::from_str(include_str!("data/unicode-processing.json")).unwrap();
    let external: Value = std::env::var("KEV_CORPUS")
        .map(|path| serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap())
        .unwrap_or_else(|_| json!({"cases":[]}));
    for case in fixture()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .chain(unicode["cases"].as_array().unwrap())
        .chain(external["cases"].as_array().unwrap())
    {
        let request = json::parse(&serde_json::to_vec(&case["request"]).unwrap()).unwrap();
        let prepared = processor.prepare(&request).unwrap();
        let compiled = contract::compile(&request).unwrap();
        assert_eq!(compiled.state, case["record"]["state"], "{}", case["name"]);
        assert_eq!(
            compiled.questions.len(),
            case["meta"].as_array().unwrap().len()
        );
        for (q, expected) in compiled
            .questions
            .iter()
            .zip(case["record"]["questions"].as_array().unwrap())
        {
            assert_eq!(q.instructions, expected["instr"], "{}", case["name"]);
            assert_eq!(json!(q.options), expected["options"], "{}", case["name"]);
        }
        assert_eq!(json!(prepared.rows), case["rows"], "{}", case["name"]);
        assert_eq!(
            prepared.context.input_tokens as u64,
            case["input_tokens"].as_u64().unwrap()
        );
        let ps: Vec<Vec<f64>> = serde_json::from_value(case["probabilities"].clone()).unwrap();
        let body = prepared.context.finish_probabilities(&ps).unwrap();
        assert_eq!(body["answers"], case["answers"]);
        assert_eq!(
            body["usage"]["output_tokens"], case["output_tokens"],
            "{}",
            case["name"]
        );
    }
    let over =
        json!({"state":"x ".repeat(contract::MAX_STATE+1),"questions":{"q":{"type":"noul"}}});
    assert!(processor.prepare(over.as_object().unwrap()).is_err());
}
