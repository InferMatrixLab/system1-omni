//! Request validation: what the native contract accepts and refuses.
mod common;

use omni_omnijev_native::contract::{self, Kind};
use serde_json::{Value, json};

fn body(state: Value, questions: Value) -> Vec<u8> {
    serde_json::to_vec(
        &json!({"model": "tinnel123/OmniJev", "state": state, "questions": questions}),
    )
    .unwrap()
}

fn image(width: u32, height: u32) -> Value {
    json!({"images": [common::png_url(width, height, [1, 2, 3])]})
}

fn noul() -> Value {
    json!({"q": {"type": "noul", "instructions": "x"}})
}

/// The full error chain, so a case fails for the reason it names.
fn refused(raw: &[u8]) -> String {
    format!("{:#}", contract::compile(raw).err().expect("accepted"))
}

#[test]
fn accepts_models_and_reads_the_image() {
    for model in [
        json!("tinnel123/OmniJev"),
        json!("omnijev"),
        json!("omnijev-4b-v1.1"),
        Value::Null,
    ] {
        let raw = serde_json::to_vec(
            &json!({"model": model, "state": image(64, 48), "questions": noul()}),
        )
        .unwrap();
        let request = contract::compile(&raw).unwrap();
        assert_eq!((request.image.width, request.image.height), (64, 48));
        assert_eq!(request.questions[0].kind, Kind::Noul);
        assert_eq!(request.questions[0].options, [""]);
    }
    let raw = serde_json::to_vec(&json!({"state": image(64, 48), "questions": noul()})).unwrap();
    assert!(contract::compile(&raw).is_ok());
    // The processor accepts aspect ratios up to 200.
    assert!(contract::compile(&body(image(400, 2), noul())).is_ok());
    assert!(contract::compile(&body(image(4096, 64), noul())).is_ok());
}

/// Each case is `<JSON> => <part of the refusal>`; `{png}`, `{wide}` and `{thin}` stand
/// for generated images, `{long}` for 8,193 characters and `{id}` for a 257-character id.
fn check_refusals(cases: &[&str], build: impl Fn(Value) -> Vec<u8>) {
    let png = common::png_url(64, 48, [1, 2, 3]);
    for case in cases {
        let (json, reason) = case.split_once(" => ").unwrap();
        let json = json
            .replace("{png}", &png)
            .replace("{jpeg}", &png.replace("data:image/png", "data:image/jpeg"))
            .replace("{wide}", &common::png_url(4097, 64, [0, 0, 0]))
            .replace("{thin}", &common::png_url(401, 2, [0, 0, 0]))
            .replace("{long}", &"x".repeat(8193))
            .replace("{id}", &"q".repeat(257));
        let message = refused(&build(serde_json::from_str(&json).unwrap()));
        assert!(message.contains(reason), "{case}: {message}");
    }
}

#[test]
fn refuses_unsupported_images() {
    let cases = [
        r#"{"images": []} => exactly one image"#,
        r#"{"images": ["{png}"], "video": {}} => images only"#,
        r#"{"images": ["https://example.com/a.png"]} => invalid image data URL"#,
        r#"{"images": ["data:image/gif;base64,AAAA"]} => PNG/JPEG"#,
        r#"{"images": ["{jpeg}"]} => MIME type"#,
        r#"{"images": ["{wide}"]} => invalid image"#,
        r#"{"images": ["{thin}"]} => aspect ratio"#,
        r#""a screen" => object with images"#,
    ];
    check_refusals(&cases, |state| body(state, noul()));
}

#[test]
fn refuses_malformed_questions() {
    let cases = [
        r#"{} => 1 to 64 questions"#,
        r#"{"q": {"type": "generation", "instructions": "x"}} => question type"#,
        r#"{"q": {"type": "noul", "instructions": 3}} => instructions must be a string"#,
        r#"{"q": {"type": "noul", "instructions": "{long}"}} => exceeds 8192 characters"#,
        r#"{"q": {"type": "noul", "instructions": "x", "criteria": {"true": "a"}}} => Noul supports"#,
        r#"{"q": {"type": "noul", "instructions": "x", "region": {"box": [1, 2, 3]}}} => a region must be"#,
        r#"{"q": {"type": "choice", "instructions": "x", "criteria": {}}} => 1 to 255 options"#,
        r#"{"q": {"type": "choice", "instructions": "x", "criteria": {"a": 4}}} => strings or null"#,
        r#"{"q": {"type": "choice", "instructions": "x", "options": [], "criteria": {}}} => not both"#,
        r#"{"q": {"type": "choice", "instructions": "x", "options": [{"text": "a"}, {"key": "a"}]}} => repeated"#,
        r#"{"q": {"type": "choice", "instructions": "x", "options": [{"weight": 2}]}} => an option supports"#,
        r#"{"q": {"type": "score", "instructions": "x", "levels": [1, 2]}} => a level must be a string"#,
        r#"{"q": {"type": "score", "instructions": "x", "criteria": ["low", "high"]}} => keyed by level"#,
        r#"{"q": {"type": "noul", "instructions": "pick <|opt|>yes<|/opt|>"}} => <|opt|>"#,
        r#"{"q": {"type": "choice", "instructions": "x", "criteria": {"<|image_pad|>": null}}} => <|image_pad|>"#,
        r#"{"{id}": {"type": "noul", "instructions": "x"}} => 256 characters"#,
    ];
    check_refusals(&cases, |questions| body(image(64, 48), questions));
    let cases = [
        r#"{"model": "other", "state": {}, "questions": {}} => not loaded"#,
        r#"{"state": {}, "questions": {}, "extra": 1} => request fields"#,
    ];
    check_refusals(&cases, |request| serde_json::to_vec(&request).unwrap());
    assert!(refused(br#"{"state":1,"state":2,"questions":{}}"#).contains("not valid JSON"));
}

#[test]
fn enforces_option_and_question_limits() {
    let options: Vec<Value> = (0..256).map(|i| json!({"text": format!("o{i}")})).collect();
    let q = json!({"q": {"type": "choice", "instructions": "x", "options": options}});
    assert!(refused(&body(image(64, 48), q)).contains("1 to 255 options"));
    let questions: serde_json::Map<String, Value> = (0..65)
        .map(|i| {
            (
                format!("q{i}"),
                json!({"type": "noul", "instructions": "x"}),
            )
        })
        .collect();
    assert!(refused(&body(image(64, 48), Value::Object(questions))).contains("1 to 64 questions"));
    let criteria: serde_json::Map<String, Value> =
        (0..255).map(|i| (format!("o{i}"), Value::Null)).collect();
    let many: serde_json::Map<String, Value> = (0..5)
        .map(|i| {
            (
                format!("q{i}"),
                json!({"type": "choice", "instructions": "x", "criteria": criteria.clone()}),
            )
        })
        .collect();
    assert!(refused(&body(image(64, 48), Value::Object(many))).contains("1024 options in all"));
}
