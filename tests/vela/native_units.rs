//! Unit tests for the native crate's state mapping and word splitter,
//! extracted from the modules per the repository's test-placement
//! rule (the private contract test lives in native_contract_unit.rs,
//! loaded by the `#[path]` wiring in contract.rs).

use omni_vela_native::scoring::{aggregate_windows, decode_spans, sigmoid};
use omni_vela_native::state::system_one_state;
use omni_vela_native::words::{split_words, unit_ids, word_first_tokens};
use serde_json::json;

// ------------------------------------------------------------- state

#[test]
fn a_plain_string_is_one_user_part() {
    let parts = system_one_state(&json!("hello")).expect("parts");
    assert_eq!(parts.parts.len(), 1);
    assert_eq!(parts.parts[0].r#type, "user");
    assert_eq!(parts.fields["request"].end, 5);
}

#[test]
fn typed_keys_group_into_ordered_parts_with_field_ranges() {
    let parts = system_one_state(&json!({
        "answer": "A", "request": "R", "source": "S", "extra": "E",
    }))
    .expect("parts");
    let roles: Vec<&str> = parts.parts.iter().map(|p| p.r#type.as_str()).collect();
    assert_eq!(roles, ["user", "context", "answer"]);
    let context = &parts.parts[1];
    assert_eq!(context.text, "source:\nS\n\nextra:\nE");
    assert_eq!(parts.fields["source"].start, "source:\n".chars().count());
    assert_eq!(
        parts.fields["extra"].start,
        "source:\nS\n\nextra:\n".chars().count()
    );
}

#[test]
fn an_empty_role_part_is_rejected() {
    let error = system_one_state(&json!({"request": "  "})).unwrap_err();
    assert_eq!(
        error.body["detail"][0]["msg"],
        serde_json::json!("the user part of the state is empty")
    );
}

#[test]
fn a_string_holding_a_json_object_is_read_as_that_object() {
    let parts = system_one_state(&json!("{\"request\": \"text\"}")).expect("parts");
    assert_eq!(parts.parts.len(), 1);
    assert_eq!(parts.fields["request"].end, 4);
}

#[test]
fn structured_float_state_uses_python_exponent_spelling() {
    let parts = system_one_state(&json!({
        "request": {"small": 1e-7, "large": 1e20},
    }))
    .expect("state maps");

    assert_eq!(
        parts.parts[0].text, r#"{"large":1e+20,"small":1e-07}"#,
        "state rendering must match Python json.dumps before tokenization"
    );
}

// ------------------------------------------------------------- words

#[test]
fn words_split_like_the_reference() {
    let text = "Hi, I'm Tom Baker (tom.baker@example.com).";
    let words: Vec<String> = split_words(text)
        .iter()
        .map(|(a, b)| text.chars().skip(*a).take(b - a).collect())
        .collect();
    assert_eq!(
        words,
        vec![
            "Hi",
            ",",
            "I",
            "'",
            "m",
            "Tom",
            "Baker",
            "(",
            "tom.baker@example.com",
            ")",
            "."
        ]
    );
}

#[test]
fn cjk_characters_are_one_unit_each() {
    let text = "追踪 parcel 仓库";
    let words: Vec<String> = split_words(text)
        .iter()
        .map(|(a, b)| text.chars().skip(*a).take(b - a).collect())
        .collect();
    assert_eq!(words, vec!["追", "踪", "parcel", "仓", "库"]);
    // Mixed content: each CJK char its own unit, the ASCII word one
    // unit, the space outside every unit.
    let chars: Vec<char> = "追踪 parcel".chars().collect();
    assert_eq!(unit_ids(&chars), vec![0, 1, -1, 2, 2, 2, 2, 2, 2]);
}

#[test]
fn first_tokens_skip_empty_tokens() {
    let text = "alpha beta";
    let offsets = vec![(0, 0), (0, 5), (6, 10)];
    let (word_offsets, first) = word_first_tokens(text, &offsets);
    assert_eq!(word_offsets, vec![(0, 5), (6, 10)]);
    assert_eq!(first, vec![1, 2]);
}

#[test]
fn a_rejected_word_ends_the_current_span() {
    for rejected in [0.1, 0.5] {
        let got = decode_spans(
            &[vec![0.9], vec![rejected], vec![0.9]],
            &[(0, 5), (6, 9), (10, 13)],
            &["PERSON".into()],
            "Alice met Bob",
            0.5,
        );
        assert_eq!(got.len(), 2, "rejected probability {rejected}");
        assert_eq!(got[0]["start"], 0);
        assert_eq!(got[0]["end"], 5);
        assert_eq!(got[1]["start"], 10);
        assert_eq!(got[1]["end"], 13);
        for span in got {
            assert_eq!(span["label"], "PERSON");
            assert_eq!(span["probability"], f64::from(0.9f32));
        }
    }
}

#[test]
fn empty_offsets_do_not_break_consecutive_labeled_words() {
    let got = decode_spans(
        &[vec![0.9], vec![0.1], vec![0.9]],
        &[(0, 5), (5, 5), (6, 9)],
        &["PERSON".into()],
        "Alice Bob",
        0.5,
    );
    assert_eq!(got.len(), 1);
    assert_eq!(got[0]["start"], 0);
    assert_eq!(got[0]["end"], 9);
    assert_eq!(got[0]["label"], "PERSON");
}

#[test]
fn window_aggregation_preserves_positive_spans_from_each_window() {
    let names = vec!["DELIVERY".to_owned()];
    let original_offsets = vec![(0, 8), (9, 11), (12, 17)];
    let window_logits = vec![vec![vec![8.0f32]], vec![vec![-8.0f32]], vec![vec![8.0f32]]];
    let window_offsets = vec![vec![(0, 8)], vec![(9, 11)], vec![(12, 17)]];

    let aggregated = aggregate_windows(&names, &original_offsets, &window_logits, &window_offsets);
    assert_eq!(aggregated.len(), 3);
    assert_eq!(aggregated[0][0], 8.0);
    assert_eq!(aggregated[1][0], -8.0);
    assert_eq!(aggregated[2][0], 8.0);

    let probabilities: Vec<Vec<f32>> = aggregated
        .iter()
        .map(|row| vec![sigmoid(row[0] as f64) as f32])
        .collect();
    let spans = decode_spans(
        &probabilities,
        &original_offsets,
        &names,
        "delivery an event",
        0.5,
    );
    assert_eq!(spans.len(), 2, "both window-backed words must remain spans");
    assert_eq!(spans[0]["start"], 0);
    assert_eq!(spans[0]["end"], 8);
    assert_eq!(spans[1]["start"], 12);
    assert_eq!(spans[1]["end"], 17);
}
