//! Private-item tests for the truncation budget and the whole-request
//! 413, loaded by the `#[path]` wiring in
//! `src/models/vela/native/src/render.rs`. `assemble_budget` works on
//! plain token counts, so these mirror the reference
//! `schema_lib_r4.assemble` order — shrink floor 64 then 1, protected
//! straight to 1 — without a tokenizer or weights.

use std::collections::HashSet;

use super::*;

#[test]
fn an_under_budget_row_keeps_every_segment() {
    let mut segs = vec![("user".to_string(), 100), ("context".to_string(), 50)];
    let protected: HashSet<String> = ["context".to_string()].into_iter().collect();
    let (seg_lens, trunc) = assemble_budget(&[20], &[], &mut segs, 200, &protected, Some("user"))
        .expect("under budget");
    assert_eq!(seg_lens["user"], 100);
    assert_eq!(seg_lens["context"], 50);
    assert_eq!(trunc.shrink_cut, 0);
    assert_eq!(trunc.protected_cut, 0);
}

#[test]
fn the_shrink_part_cuts_to_64_then_to_1() {
    // fixed = 2 + 1 + 2 = 5, questions 10, segments 100 + 50 = 155,
    // budget 60: over 105 → keep min(100, max(64, 0)) = 64 (cut 36);
    // over 69 → keep min(64, max(1, 0)) = 1 (cut 63); still over 6, so
    // the protected "context" cuts to max(1, 50−6) = 44 → fits exactly.
    let mut segs = vec![("user".to_string(), 100), ("context".to_string(), 50)];
    let protected: HashSet<String> = ["context".to_string()].into_iter().collect();
    let (seg_lens, trunc) = assemble_budget(&[10], &[], &mut segs, 60, &protected, Some("user"))
        .expect("fits after both cuts");
    assert_eq!(seg_lens["user"], 1);
    assert_eq!(trunc.shrink_cut, 99);
    assert_eq!(seg_lens["context"], 44);
    assert_eq!(trunc.protected_cut, 6);
}

#[test]
fn a_protected_part_cuts_straight_to_one() {
    // No shrink part (the only segment is targeted): the protected
    // part goes toward the 1-token floor without the 64 stop.
    // fixed 4 + 10 + 100 = 114 vs 20 → keep max(1, 100−94) = 6, cut 94.
    let mut segs = vec![("user".to_string(), 100)];
    let protected: HashSet<String> = ["user".to_string()].into_iter().collect();
    let (seg_lens, trunc) = assemble_budget(&[10], &[], &mut segs, 20, &protected, None)
        .expect("fits after the protected cut");
    assert_eq!(seg_lens["user"], 6);
    assert_eq!(trunc.shrink_cut, 0);
    assert_eq!(trunc.protected_cut, 94);
}

#[test]
fn questions_alone_over_the_window_fail_the_budget() {
    // Even both segments cut to 1 cannot fit: the budget fails and the
    // caller answers the whole-request 413.
    let mut segs = vec![("user".to_string(), 100), ("context".to_string(), 50)];
    let protected: HashSet<String> = ["user".to_string(), "context".to_string()]
        .into_iter()
        .collect();
    assemble_budget(&[500], &[], &mut segs, 100, &protected, None).unwrap_err();
    let too_long = too_long(100);
    assert_eq!(too_long.status, 413);
    assert_eq!(
        too_long.body,
        serde_json::json!({"detail": [{"loc": ["body", "questions"],
            "msg": "the questions alone do not fit the 100-token window",
            "type": "too_long"}]})
    );
}
