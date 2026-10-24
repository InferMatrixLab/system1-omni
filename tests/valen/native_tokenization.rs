//! Token-id equality against the recorded reference fixtures. Needs the
//! pinned Qwen3.5-2B base for its tokenizer:
//!
//! ```text
//! VALEN_BASE=weights/Qwen3.5-2B cargo test -p omni-valen-native --test native_tokenization
//! ```
//!
//! Follows the sibling suites' convention: the env-gated test is `#[ignore]`d
//! so the CI workspace run needs no weights.

use std::io::Cursor;

use base64::Engine;
use serde_json::Value;

use omni_valen_native::compile::IMAGE_PAD;
use omni_valen_native::preprocess;
use omni_valen_native::protocol;

fn fixture() -> Value {
    let path = format!(
        "{}/../../../../tests/valen/data/native_compiler.json",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn png_url(width: u32, height: u32, colour: [u8; 3]) -> String {
    let image = image::RgbImage::from_pixel(width, height, image::Rgb(colour));
    let mut bytes = Vec::new();
    image
        .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .unwrap();
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    )
}

fn jpeg_url(width: u32, height: u32, colour: [u8; 3]) -> String {
    let image = image::RgbImage::from_pixel(width, height, image::Rgb(colour));
    let mut bytes = Vec::new();
    image
        .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Jpeg)
        .unwrap();
    format!(
        "data:image/jpeg;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    )
}

fn case_body(case: &Value) -> Vec<u8> {
    let mut body: Value = serde_json::from_str(case["body"].as_str().unwrap()).unwrap();
    if let Some(image) = case.get("image")
        && image["kind"] == "generated"
    {
        let size = image["size"].as_array().unwrap();
        let colour = image["colour"].as_array().unwrap();
        let rgb = [0, 1, 2].map(|i| colour[i].as_u64().unwrap() as u8);
        let url = if image["format"] == "png" {
            png_url(
                size[0].as_u64().unwrap() as u32,
                size[1].as_u64().unwrap() as u32,
                rgb,
            )
        } else {
            jpeg_url(
                size[0].as_u64().unwrap() as u32,
                size[1].as_u64().unwrap() as u32,
                rgb,
            )
        };
        body["state"]["image"] = Value::String(url);
        return serde_json::to_vec(&body).unwrap();
    }
    case["body"].as_str().unwrap().as_bytes().to_vec()
}

/// Fixture token ids, with the image-placeholder run stored as its negated
/// length, expanded back to the recorded token.
fn token_ids(collapsed: &Value) -> Vec<u32> {
    let mut ids = Vec::new();
    for value in collapsed.as_array().unwrap() {
        let value = value.as_i64().unwrap();
        if value < 0 {
            ids.extend(std::iter::repeat_n(IMAGE_PAD, (-value) as usize));
        } else {
            ids.push(value as u32);
        }
    }
    ids
}

#[test]
#[ignore]
fn tokenization_and_layout_match_the_reference() {
    let base = std::env::var("VALEN_BASE").expect("set VALEN_BASE to the pinned Qwen3.5-2B base");
    let processor = omni_valen_native::Processor::load(std::path::Path::new(&base)).unwrap();

    let fixture = fixture();
    let recorded: Vec<&str> = fixture["pins"]["special_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_str().unwrap())
        .collect();
    let derived: Vec<&str> = processor
        .special_tokens
        .iter()
        .map(|t| t.as_str())
        .collect();
    // Same set; the reference unions the defaults first, order differs only
    // within the tokenizer's own contributions.
    let mut want: Vec<&str> = recorded.to_vec();
    want.sort();
    let mut got = derived.clone();
    got.sort();
    assert_eq!(
        got, want,
        "the derived special-token set must match the reference's"
    );
    assert_eq!(
        fixture["pins"]["max_length"].as_u64().unwrap() as usize,
        omni_valen_native::compile::MAX_LENGTH
    );

    let compiler = processor.compiler();
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let raw = case_body(case);
        let body = omni_valen_native::wire::parse_body(&raw).unwrap();
        let request = protocol::parse_request(&body, &processor.special_tokens).unwrap();
        let prepared = preprocess::prepare(&request);
        let compiled = compiler.compile(&prepared).unwrap();
        let expected = &case["expected"];

        let base_ids = token_ids(&expected["base_input_ids"]);
        assert_eq!(compiled.base.input_ids, base_ids, "{name}: base ids");
        if let Some(span) = expected["image_pad_span"].as_array() {
            let start = span[0].as_u64().unwrap() as usize;
            let end = span[1].as_u64().unwrap() as usize;
            let marks: Vec<usize> = (0..compiled.base.mm_token_type_ids.len())
                .filter(|&i| compiled.base.mm_token_type_ids[i] == 1)
                .collect();
            assert_eq!(marks, (start..end).collect::<Vec<_>>(), "{name}: mm marks");
            assert!(
                compiled.base.input_ids[start..end]
                    .iter()
                    .all(|&id| id == IMAGE_PAD),
                "{name}: mm marks off the pad run"
            );
        } else {
            assert!(
                compiled.base.mm_token_type_ids.iter().all(|&m| m == 0),
                "{name}: text state must not mark mm tokens"
            );
        }

        assert_eq!(
            compiled.logical_tokens,
            expected["logical_tokens"].as_u64().unwrap() as usize,
            "{name}: logical tokens"
        );
        assert_eq!(
            compiled.compute_tokens,
            expected["compute_tokens"].as_u64().unwrap() as usize,
            "{name}: compute tokens"
        );

        for (question, want) in compiled
            .questions
            .iter()
            .zip(expected["questions"].as_array().unwrap())
        {
            assert_eq!(question.qid, want["qid"].as_str().unwrap(), "{name}");
            let want_branches = want["branches"].as_array().unwrap();
            assert_eq!(
                question.branches.len(),
                want_branches.len(),
                "{name} {}",
                question.qid
            );
            for (branch, recorded) in question.branches.iter().zip(want_branches) {
                let suffix: Vec<u32> = recorded["suffix_ids"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u32)
                    .collect();
                assert_eq!(
                    &branch.tokens[..base_ids.len()],
                    &base_ids[..],
                    "{name} {}: branch loses the base",
                    question.qid
                );
                assert_eq!(
                    &branch.tokens[base_ids.len()..],
                    &suffix[..],
                    "{name} {}: suffix ids",
                    question.qid
                );
                let positions: Vec<usize> = recorded["candidate_positions"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect();
                assert_eq!(
                    branch.candidate_positions, positions,
                    "{name} {}",
                    question.qid
                );
                assert_eq!(
                    branch.decision_position,
                    recorded["decision_position"].as_u64().unwrap() as usize,
                    "{name} {}",
                    question.qid
                );
                assert_eq!(
                    branch.length,
                    recorded["length"].as_u64().unwrap() as usize,
                    "{name} {}",
                    question.qid
                );
                assert_eq!(branch.mm_token_type_ids.len(), branch.length);
                assert!(
                    branch.mm_token_type_ids[base_ids.len()..]
                        .iter()
                        .all(|&m| m == 0),
                    "{name} {}: suffix mm must be zero",
                    question.qid
                );
            }
        }
    }
}

#[test]
#[ignore]
fn token_budget_rejections_match_the_reference_bodies() {
    let base = std::env::var("VALEN_BASE").expect("set VALEN_BASE to the pinned Qwen3.5-2B base");
    let processor = omni_valen_native::Processor::load(std::path::Path::new(&base)).unwrap();
    let fixture = fixture();
    let compiler = processor.compiler();
    for entry in fixture["rejections"].as_array().unwrap() {
        let name = entry["name"].as_str().unwrap();
        if !name.starts_with("overflow_") {
            continue;
        }
        let raw = entry["body"].as_str().unwrap().as_bytes().to_vec();
        let body = omni_valen_native::wire::parse_body(&raw).unwrap();
        let request = protocol::parse_request(&body, &processor.special_tokens).unwrap();
        let prepared = preprocess::prepare(&request);
        let error = compiler.compile(&prepared).unwrap_err();
        assert_eq!(
            u64::from(error.status),
            entry["status"].as_u64().unwrap(),
            "{name}"
        );
        assert_eq!(error.message, entry["detail"].as_str().unwrap(), "{name}");
    }
}
