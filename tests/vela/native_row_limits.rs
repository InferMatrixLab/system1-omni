//! Weight-free regression tests through the real processor and renderer.
//! A tiny whitespace tokenizer keeps token costs deterministic and offline.

use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;
use crate::contract::RequestError;
use crate::render::{Block, RenderedRow};
use serde_json::json;

const CAP: usize = 16_384;

fn processor() -> Processor {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "vela-row-limits-{}-{}.json",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(
        &path,
        r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"WhitespaceSplit"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"word":1,"state":2},"unk_token":"[UNK]"}}"#,
    )
    .unwrap();
    let encoder = Encoder::load(&path).expect("small local tokenizer");
    std::fs::remove_file(path).unwrap();
    Processor {
        encoder,
        max_length: CAP,
        router_label_sets: Vec::new(),
        has_broad: true,
        halu_desc: "a claim not supported by the context".into(),
    }
}

fn question(kind: &str, words: usize) -> Value {
    let mut question = json!({"type": kind, "instructions": "word ".repeat(words)});
    let criteria = match kind {
        "span" => Some(json!({"PERSON": "person"})),
        "choice" | "set" => Some(json!({"a": "first", "b": "second"})),
        "score" => Some(json!(["low", "high"])),
        _ => None,
    };
    if let Some(criteria) = criteria {
        question["criteria"] = criteria;
    }
    question["over"] = json!("request");
    question
}

fn body(state_words: usize, questions: Value) -> Map<String, Value> {
    json!({
        "model": "vllm-sr/Vela-2.0-0.8B",
        "state": {"request": "word ".repeat(state_words)},
        "questions": questions,
    })
    .as_object()
    .unwrap()
    .clone()
}

fn planned(processor: &Processor, body: &Map<String, Value>) -> Planned {
    processor
        .plan(body)
        .unwrap_or_else(|error| panic!("unexpected rejection: {}", error.body))
}

fn rejected(result: Result<Planned, RequestError>) {
    let error = match result {
        Err(error) => error,
        Ok(plan) => panic!(
            "over-limit request was admitted: ordinary={:?}, windows={:?}",
            plan.rows.iter().map(|row| row.n_tokens).collect::<Vec<_>>(),
            plan.window_rows
                .iter()
                .map(|row| row.n_tokens)
                .collect::<Vec<_>>()
        ),
    };
    assert_eq!(error.status, 413);
    assert_eq!(
        error.body,
        json!({"detail": [{
            "loc": ["body", "questions"],
            "msg": "the questions alone do not fit the 16384-token window",
            "type": "too_long",
        }]})
    );
}

fn span_overhead(processor: &Processor) -> usize {
    planned(processor, &body(1, json!({"q": question("span", 1)}))).rows[0].n_tokens - 1
}

#[test]
fn span_rows_at_and_below_the_cap_are_admitted() {
    let processor = processor();
    let overhead = span_overhead(&processor);
    for length in [CAP - 1, CAP] {
        let plan = planned(
            &processor,
            &body(1, json!({"q": question("span", length - overhead)})),
        );
        assert_eq!(plan.rows.len(), 1);
        assert_eq!(plan.rows[0].n_tokens, length);
        assert!(plan.window_rows.is_empty());
    }
}

#[test]
fn a_span_row_one_token_over_the_cap_rejects_the_request() {
    let processor = processor();
    let words = CAP + 1 - span_overhead(&processor);
    rejected(processor.plan(&body(1, json!({"q": question("span", words)}))));
}

#[test]
fn a_span_header_larger_than_the_budget_returns_413_without_panicking() {
    let processor = processor();
    rejected(processor.plan(&body(1, json!({"q": question("span", 20_000)}))));
}

#[test]
fn non_span_schema_overflow_keeps_the_whole_request_413() {
    let processor = processor();
    for kind in ["noul", "choice", "score", "set"] {
        rejected(processor.plan(&body(1, json!({"q": question(kind, 20_000)}))));
    }
}

#[test]
fn mixed_rows_can_exceed_the_cap_in_total_and_keep_question_order() {
    let processor = processor();
    let plan = planned(
        &processor,
        &body(
            1,
            json!({
                "topic": question("noul", 1),
                "first": question("span", 9_000),
                "second": question("span", 9_000),
            }),
        ),
    );
    assert_eq!(plan.rows.len(), 2);
    assert!(plan.rows.iter().all(|row| row.n_tokens <= CAP));
    assert!(plan.rows.iter().map(|row| row.n_tokens).sum::<usize>() > CAP);
    let order: Vec<_> = plan
        .plan
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["qid"].as_str().unwrap())
        .collect();
    assert_eq!(order, ["topic", "first", "second"]);
    assert!(matches!(&plan.rows[0].blocks[0], Block::Question { qid, .. } if qid == "topic"));
    assert!(matches!(&plan.rows[1].blocks[0], Block::Span { .. }));
}

#[test]
fn an_oversized_later_row_rejects_the_whole_request() {
    let processor = processor();
    let words = CAP + 1 - span_overhead(&processor);
    rejected(processor.plan(&body(
        1,
        json!({
            "first": question("span", 1),
            "second": question("span", words),
        }),
    )));
}

#[test]
fn window_rows_can_exceed_the_cap_in_total() {
    let processor = processor();
    let plan = planned(&processor, &body(5_000, json!({"q": question("span", 1)})));
    assert_eq!(plan.rows.len(), 1);
    assert!(plan.window_rows.len() > 1);
    assert_eq!(plan.rows[0].trunc.protected_cut, 0);
    assert!(
        plan.rows
            .iter()
            .chain(&plan.window_rows)
            .all(|row| row.n_tokens <= CAP)
    );
    assert!(plan.window_rows.iter().all(|row| row.window));
    assert!(
        plan.rows
            .iter()
            .chain(&plan.window_rows)
            .map(|row| row.n_tokens)
            .sum::<usize>()
            > CAP
    );
}

#[test]
fn over_limit_raw_state_keeps_the_existing_clipping() {
    let processor = processor();
    let plan = planned(&processor, &body(20_000, json!({"q": question("noul", 1)})));
    assert_eq!(plan.rows.len(), 1);
    assert!(plan.rows[0].n_tokens <= CAP);
    assert!(plan.rows[0].trunc.protected_cut > 0);
    assert!(plan.rows[0].seg_lens["user"] < 20_000);
}

fn primary_row(processor: &Processor, body: &Map<String, Value>) -> RenderedRow {
    let state = system_one_state(&body["state"]).unwrap();
    let mapped = system_one_to_predict(state, body["questions"].as_object().unwrap()).unwrap();
    let generic: Vec<_> = mapped.request.questions.iter().map(to_generic).collect();
    let span_roles = ["user".to_string()].into_iter().collect();
    let parts = processor
        .prepared_parts(&mapped.request.parts, &span_roles)
        .unwrap();
    let context = RenderContext {
        encoder: &processor.encoder,
        max_length: CAP,
        router_label_sets: &processor.router_label_sets,
        has_broad: processor.has_broad,
        halu_desc: processor.halu_desc.clone(),
    };
    let (questions, dispatch) = context.prepare_questions(&generic).unwrap();
    context.render(&parts, &questions, &dispatch).unwrap()
}

#[test]
fn an_oversized_window_rejects_even_when_the_primary_row_fits() {
    let processor = processor();
    let body = body(2_176, json!({"q": question("span", CAP - 85)}));
    let primary = primary_row(&processor, &body);
    assert!(
        primary.n_tokens <= CAP,
        "primary row has {} tokens",
        primary.n_tokens
    );
    rejected(processor.plan(&body));
}
