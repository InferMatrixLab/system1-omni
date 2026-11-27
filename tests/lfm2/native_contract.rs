use crate::{
    SERVED_MODEL, artifact, json,
    processing::{self, FieldPlan, RenderedSchema, TokenPlan},
};
use anyhow::Result;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path};
use tokenizers::Tokenizer;

const FIXTURE: &str = include_str!("fixtures/q12_replay.json");
const RESPONSE_TOLERANCE: f64 = 1e-6;

#[derive(Deserialize)]
struct Fixture {
    schema: String,
    public_input_sha256: String,
    public_capture_sha256: String,
    verifier_capture_sha256: String,
    fragments: BTreeMap<String, Vec<u32>>,
    public: Vec<PublicCase>,
    internal: Vec<InternalCase>,
}

#[derive(Deserialize)]
struct PublicCase {
    id: String,
    request: Value,
    questions: Vec<PublicQuestion>,
    oracle_response: Value,
    batch8_response: Value,
}

#[derive(Deserialize)]
struct PublicQuestion {
    id: String,
    plan: ExpectedPlan,
    batch8_scores: Vec<f32>,
}

#[derive(Deserialize)]
struct InternalCase {
    id: String,
    state: String,
    schema: Value,
    plan: ExpectedPlan,
    selected: Value,
}

#[derive(Deserialize)]
struct ExpectedPlan {
    prefix: String,
    fields: Vec<ExpectedField>,
    input_tokens: usize,
}

#[derive(Deserialize)]
struct ExpectedField {
    name: String,
    suffix: String,
    values: Vec<String>,
    scores: Vec<f32>,
}

fn digest(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

fn replay_plan(
    rendered: RenderedSchema,
    expected: &ExpectedPlan,
    fragments: &BTreeMap<String, Vec<u32>>,
    tokenizer: Option<&Tokenizer>,
) -> TokenPlan {
    assert_eq!(digest(&rendered.prefix_text), expected.prefix);
    assert_eq!(rendered.fields.len(), expected.fields.len());
    for (field, expected) in rendered.fields.iter().zip(&expected.fields) {
        assert_eq!(field.name, expected.name);
        assert_eq!(digest(&field.suffix_text), expected.suffix);
        assert_eq!(field.candidates.len(), expected.values.len());
        for (candidate, expected) in field.candidates.iter().zip(&expected.values) {
            assert_eq!(digest(&candidate.value_text), *expected);
            assert!(candidate.value_text.ends_with('\n'));
        }
    }
    let plan = match tokenizer {
        Some(tokenizer) => rendered.tokenize(tokenizer),
        None => rendered
            .tokenize_with(&mut |text| -> Result<_> { Ok(fragments[&digest(text)].clone()) }),
    }
    .unwrap();

    assert_eq!(plan.prefix_ids, fragments[&expected.prefix]);
    assert_eq!(plan.input_tokens, expected.input_tokens);
    let prefix = fragments[&expected.prefix].len();
    for (field, expected) in plan.fields.iter().zip(&expected.fields) {
        assert_eq!(field.suffix_ids, fragments[&expected.suffix]);
        let suffix = fragments[&expected.suffix].len();
        for (candidate, expected) in field.candidates.iter().zip(&expected.values) {
            let value = &fragments[expected];
            assert_eq!(&candidate.value_ids, value);
            assert_eq!(
                candidate.continuation_readout,
                suffix - 1..suffix + value.len() - 1
            );
            assert_eq!(
                candidate.full_readout,
                prefix + suffix - 1..prefix + suffix + value.len() - 1
            );
        }
    }
    plan
}

fn compare_response(actual: &Value, expected: &Value) {
    assert_eq!(actual["model"], SERVED_MODEL);
    assert_eq!(actual["model"], expected["model"]);
    assert_eq!(actual["usage"], expected["usage"]);
    let actual = actual["answers"].as_object().unwrap();
    let expected = expected["answers"].as_object().unwrap();
    assert!(actual.keys().eq(expected.keys()));
    for (id, expected) in expected {
        let actual = &actual[id];
        assert_eq!(actual["type"], "choice");
        assert_eq!(actual["choice"], expected["choice"]);
        let confidence = actual["confidence"].as_f64().unwrap();
        assert!((0.0..=1.0).contains(&confidence));
        assert!(
            (confidence - expected["confidence"].as_f64().unwrap()).abs() <= RESPONSE_TOLERANCE
        );
        let probabilities = actual["probabilities"].as_object().unwrap();
        let expected = expected["probabilities"].as_object().unwrap();
        assert!(probabilities.keys().eq(expected.keys()));
        let mut sum = 0.0;
        for (candidate, expected) in expected {
            let probability = probabilities[candidate].as_f64().unwrap();
            assert!((0.0..=1.0).contains(&probability));
            assert!((probability - expected.as_f64().unwrap()).abs() <= RESPONSE_TOLERANCE);
            sum += probability;
        }
        assert!((sum - 1.0).abs() <= RESPONSE_TOLERANCE);
    }
}

fn public_replay(fixture: &Fixture, tokenizer: Option<&Tokenizer>) {
    assert_eq!(fixture.public.len(), 5);
    let mut counts = (0, 0);
    for case in &fixture.public {
        let rendered =
            processing::render_request(&serde_json::to_vec(&case.request).unwrap()).unwrap();
        assert_eq!(
            rendered.questions.len(),
            case.questions.len(),
            "{}",
            case.id
        );
        let prepared = processing::PreparedRequest {
            questions: rendered
                .questions
                .into_iter()
                .zip(&case.questions)
                .map(|(question, expected)| {
                    assert_eq!(question.id, expected.id, "{}", case.id);
                    let plan = replay_plan(
                        question.schema,
                        &expected.plan,
                        &fixture.fragments,
                        tokenizer,
                    );
                    counts.0 += 1;
                    counts.1 += plan.fields[0].candidates.len();
                    processing::PreparedQuestion {
                        id: question.id,
                        plan,
                    }
                })
                .collect(),
        };
        let oracle = case
            .questions
            .iter()
            .map(|q| q.plan.fields[0].scores.clone())
            .collect::<Vec<_>>();
        let batch8 = case
            .questions
            .iter()
            .map(|q| q.batch8_scores.clone())
            .collect::<Vec<_>>();
        compare_response(
            &processing::finish_request(&prepared, &oracle).unwrap(),
            &case.oracle_response,
        );
        compare_response(
            &processing::finish_request(&prepared, &batch8).unwrap(),
            &case.batch8_response,
        );
    }
    assert_eq!(counts, (6, 266));
}

fn internal_replay(fixture: &Fixture, tokenizer: Option<&Tokenizer>) {
    assert_eq!(fixture.internal.len(), 17);
    for case in &fixture.internal {
        let rendered = processing::render_flat_schema(&case.state, &case.schema).unwrap();
        let prepared = replay_plan(rendered, &case.plan, &fixture.fragments, tokenizer);
        for (field, expected) in prepared.fields.iter().zip(&case.plan.fields) {
            let (value, _, _) = processing::interpret(field, &expected.scores).unwrap();
            assert_eq!(
                value, case.selected[&field.name],
                "{}/{}",
                case.id, field.name
            );
        }
    }
}

fn fixture() -> Fixture {
    let fixture: Fixture = serde_json::from_str(FIXTURE).unwrap();
    assert_eq!(fixture.schema, "lfm2-q12-replay-v1");
    assert_eq!(
        fixture.public_input_sha256,
        "ca4a3723ad5d3fd80ebfba3a6908d3c283a0783b83fe15328fddb622742b23bd"
    );
    assert_eq!(
        fixture.public_capture_sha256,
        "1e8786bb3ffb29f56ffb55090e0dc3206ab4dd7ec0fd2994cb597cfc185206f4"
    );
    assert_eq!(
        fixture.verifier_capture_sha256,
        "908afa213373916f8ceae149b53a3db3a548d4e0c44e09fa94407f80940fe104"
    );
    fixture
}

#[test]
fn frozen_public_render_readout_usage_order_and_same_score_responses() {
    public_replay(&fixture(), None);
}

#[test]
fn frozen_internal_render_offsets_and_typed_choices() {
    internal_replay(&fixture(), None);
}

#[test]
#[ignore = "requires the pinned tokenizer file via LFM2_TOKENIZER_JSON"]
fn pinned_tokenizer_matches_all_public_and_internal_ids() {
    let path = std::env::var_os("LFM2_TOKENIZER_JSON").expect("LFM2_TOKENIZER_JSON");
    let tokenizer = artifact::load_tokenizer(Path::new(&path)).unwrap();
    let fixture = fixture();
    public_replay(&fixture, Some(&tokenizer));
    internal_replay(&fixture, Some(&tokenizer));
}

#[test]
fn criteria_limits_and_whole_request_validation() {
    for (count, valid) in [(0, false), (1, true), (255, true), (256, false)] {
        let criteria: serde_json::Map<_, _> =
            (0..count).map(|i| (i.to_string(), Value::Null)).collect();
        let request = json!({"model":"lfm2.5-350m","state":"boundary","questions":{
            "q":{"type":"choice","instructions":"","criteria":criteria}
        }});
        let result = processing::render_request(&serde_json::to_vec(&request).unwrap());
        if valid {
            assert_eq!(
                result.unwrap().questions[0].schema.fields[0]
                    .candidates
                    .len(),
                count
            );
        } else {
            let error = result.unwrap_err();
            assert_eq!(error.status, 422);
            assert_eq!(error.detail, "question 'q' criteria must map 1..255 keys");
        }
    }
    let error = processing::render_request(
        br#"{"model":"lfm2.5-350m","state":"ok","questions":{
        "first":{"type":"choice","instructions":"","criteria":{"":null}},
        "second":{"type":"score","instructions":"","criteria":{"x":null}}
    }}"#,
    )
    .unwrap_err();
    assert_eq!(
        error.detail,
        "question 'second' has unsupported type 'score'; only 'choice' is supported"
    );
}

#[test]
fn python_json_order_numbers_and_unicode() {
    let input = r#" {"10":1e16, "2":1e-5,"z":-0.0,"n":-0, "big":18446744073709551616000,"esc":"中😀\u0001\n\""} "#;
    let expected = r#"{"10": 1e+16, "2": 1e-05, "z": -0.0, "n": 0, "big": 18446744073709551616000, "esc": "中😀\u0001\n\""}"#;
    assert_eq!(
        json::dumps(&json::parse(input.as_bytes()).unwrap()),
        expected
    );
    let ties = br#"[-1250488981608125.25,819163341246155.25,-11996314554832.3125]"#;
    assert_eq!(
        json::dumps(&json::parse(ties).unwrap()),
        "[-1250488981608125.2, 819163341246155.2, -11996314554832.312]"
    );
    let utf16: Vec<u8> = [0xff, 0xfe]
        .into_iter()
        .chain(input.encode_utf16().flat_map(u16::to_le_bytes))
        .collect();
    assert_eq!(json::dumps(&json::parse(&utf16).unwrap()), expected);
    assert_eq!(
        json::parse(br#"{"z":[{"k":1,"\u006b":2}]}"#).unwrap_err(),
        "duplicate key: k"
    );
    assert_eq!(
        json::parse(br#"{"weird at line x":0,"weird at line x":1}"#).unwrap_err(),
        "duplicate key: weird at line x"
    );
    assert_eq!(
        json::parse(br#"{"z":[1e9999]}"#).unwrap_err(),
        "non-finite number is not allowed"
    );
    for constant in ["NaN", "Infinity", "-Infinity"] {
        let input = format!(r#"{{"z":[{constant}]}}"#);
        assert_eq!(
            json::parse(input.as_bytes()).unwrap_err(),
            format!("invalid JSON constant: {constant}")
        );
    }
}

#[test]
fn finisher_rejects_bad_shape_and_nonfinite_and_preserves_first_tie() {
    let fixture = fixture();
    let cases = &fixture.public;
    let ready = processing::render_request(&serde_json::to_vec(&cases[0].request).unwrap())
        .unwrap()
        .tokenize_with(|text| Ok(fixture.fragments[&digest(text)].clone()))
        .unwrap();
    let field: &FieldPlan = &ready.questions[0].plan.fields[0];
    assert_eq!(
        processing::interpret(field, &[-0.0, 0.0]).unwrap(),
        (json!("billing"), vec![0.5, 0.5], 0.0)
    );
    assert_eq!(
        processing::interpret(field, &[-1.0, -1000.0]).unwrap(),
        (json!("billing"), vec![1.0, 0.0], 1.0)
    );
    assert!(processing::finish_request(&ready, &[]).is_err());
    for invalid in [
        vec![],
        vec![0.0],
        vec![f32::NAN, -1.0],
        vec![-1.0, f32::INFINITY],
    ] {
        assert!(processing::finish_request(&ready, &[invalid]).is_err());
    }
}

#[test]
fn rejects_invalid_manifest_profile_tie_dtype_shape_and_offsets() {
    use crate::artifact::{TensorHeader, validate_manifest, validate_tensor_layout};

    let profile: Value =
        serde_json::from_str(include_str!("../../src/models/lfm2/native/profile.json")).unwrap();
    let mut files = BTreeMap::new();
    for (name, sha) in profile["checkpoint"].as_object().unwrap() {
        let (name, sha) = if name == "model.safetensors" {
            (
                artifact::WEIGHTS,
                include_str!("../../src/models/lfm2/native/weights.sha256").trim(),
            )
        } else {
            (name.as_str(), sha.as_str().unwrap())
        };
        files.insert(
            name.to_owned(),
            artifact::FileIdentity {
                bytes: 1,
                sha256: sha.to_owned(),
            },
        );
    }
    let mut manifest = artifact::Manifest {
        format: artifact::FORMAT.to_owned(),
        profile: profile.clone(),
        files,
    };
    assert!(validate_manifest(&manifest, &profile).is_ok());
    manifest.profile["lm_head_tie"]["lm_head.weight"] = json!("different.weight");
    assert!(validate_manifest(&manifest, &profile).is_err());
    manifest.profile = profile.clone();
    manifest.files.get_mut(artifact::WEIGHTS).unwrap().sha256 = "0".repeat(64);
    assert!(validate_manifest(&manifest, &profile).is_err());

    let shapes = artifact::expected_shapes();
    assert_eq!(shapes.len(), 148);
    assert!(!shapes.contains_key("lm_head.weight"));
    assert_eq!(
        shapes
            .values()
            .map(|shape| shape.iter().product::<usize>())
            .sum::<usize>(),
        354_483_968
    );
    let expected = BTreeMap::from([("a".to_owned(), vec![2]), ("b".to_owned(), vec![1])]);
    let mut header = BTreeMap::from([
        (
            "a".to_owned(),
            TensorHeader {
                dtype: "F32".to_owned(),
                shape: vec![2],
                data_offsets: [0, 8],
            },
        ),
        (
            "b".to_owned(),
            TensorHeader {
                dtype: "F32".to_owned(),
                shape: vec![1],
                data_offsets: [8, 12],
            },
        ),
    ]);
    assert_eq!(validate_tensor_layout(&header, &expected, 12).unwrap(), 3);
    let tensor = header.get_mut("a").unwrap();
    tensor.dtype = "BF16".to_owned();
    assert!(
        validate_tensor_layout(&header, &expected, 12)
            .unwrap_err()
            .to_string()
            .contains("must be F32")
    );
    let tensor = header.get_mut("a").unwrap();
    tensor.dtype = "F32".to_owned();
    tensor.shape = vec![1, 2];
    assert!(
        validate_tensor_layout(&header, &expected, 12)
            .unwrap_err()
            .to_string()
            .contains("shape mismatch")
    );
    header.get_mut("a").unwrap().shape = vec![2];
    header.get_mut("b").unwrap().data_offsets = [4, 8];
    assert!(
        validate_tensor_layout(&header, &expected, 12)
            .unwrap_err()
            .to_string()
            .contains("overlap")
    );
    header.get_mut("b").unwrap().data_offsets = [8, 12];
    assert!(validate_tensor_layout(&header, &expected, 16).is_err());
    header.remove("b");
    assert!(validate_tensor_layout(&header, &expected, 12).is_err());
}
