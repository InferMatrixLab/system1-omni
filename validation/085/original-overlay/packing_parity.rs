//! Replay the same token manifest on the original native checkout and this one.
//! See packing_parity.md. Synthetic token cases test execution, not task accuracy.

use std::path::PathBuf;

use omni_open_jev_native::engine::Engine;
use omni_qwen3_5_native::prefix::PrefixMode;
use serde_json::{Value, json};

// Exercise the real grouping policy without making a production-private API public.
#[allow(dead_code)]
#[path = "../../src/models/open_jev/native/src/batching.rs"]
mod packing;

const MANIFEST: &str = include_str!("data/packing_cases.json");
const ORIGINAL: &str = "22b7717c9710de5ac5c6ccba51a6d6686dae90d9";

fn inputs(case: &Value) -> Vec<Vec<Vec<u32>>> {
    let candidates: Vec<Vec<u32>> = case["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|spec| {
            let length = spec[0].as_u64().unwrap() as usize;
            let seed = spec[1].as_u64().unwrap() as u32;
            // No shared prefix: each candidate starts with its own seed.
            (0..length)
                .map(|i| 1000 + (seed * 7919 + i as u32 * 37) % 30000)
                .collect()
        })
        .collect();
    let mut candidates = candidates.into_iter();
    let result = case["questions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| candidates.by_ref().take(n.as_u64().unwrap() as usize).collect())
        .collect();
    assert!(candidates.next().is_none());
    result
}

fn checked_groups(case: &Value, ids: &[Vec<Vec<u32>>]) -> Vec<usize> {
    let flat: Vec<&[u32]> = ids.iter().flatten().map(Vec::as_slice).collect();
    let actual: Vec<usize> = packing::ranges(&flat).iter().map(|r| r.len()).collect();
    assert_eq!(json!(actual), case["groups"], "{}", case["id"]);
    assert_eq!(flat.len(), case["candidates"].as_array().unwrap().len());
    actual
}

#[test]
fn manifest_exercises_real_mixed_packing_and_both_limits() {
    let manifest: Value = serde_json::from_str(MANIFEST).unwrap();
    let cases = manifest["cases"].as_array().unwrap();
    let mut mixed = 0;
    for case in cases {
        let ids = inputs(case);
        let groups = checked_groups(case, &ids);
        let lengths: Vec<usize> = ids.iter().flatten().map(Vec::len).collect();
        if groups.len() == 1 && groups[0] > 1 && lengths.windows(2).any(|v| v[0] != v[1]) {
            mixed += 1;
        }
    }
    assert!(mixed >= 6, "must exercise genuinely packed mixed lengths");
    assert!(cases.iter().any(|c| c["groups"] == json!([16, 1])));
    assert!(cases.iter().any(|c| c["id"] == "budget-4096"));
    assert!(cases.iter().any(|c| c["id"] == "budget-4097"));
}

async fn replay(original: bool) -> Value {
    let model = PathBuf::from(std::env::var_os("OPEN_JEV_MODEL").unwrap());
    let library = PathBuf::from(std::env::var_os("OPEN_JEV_CUDA_LIB").unwrap());
    let engine = Engine::load_with_prefix(&model, &library, PrefixMode::Off)
        .await
        .unwrap();
    assert_eq!(engine.checkpoint.alias, "open-jev-9b");
    assert_eq!(engine.checkpoint.pack_candidates, !original);
    let manifest: Value = serde_json::from_str(MANIFEST).unwrap();
    let mut records = Vec::new();
    for case in manifest["cases"].as_array().unwrap() {
        let ids = inputs(case);
        let groups = checked_groups(case, &ids);
        let lengths: Vec<usize> = ids.iter().flatten().map(Vec::len).collect();
        let (scores, used) = engine
            .executor
            .execute_with_prefix(&engine.scheduler, ids)
            .await
            .unwrap();
        assert!(!used.shared);
        assert_eq!(used.saved_tokens, 0);
        assert_eq!(scores.len(), case["questions"].as_array().unwrap().len());
        for (row, count) in scores.iter().zip(case["questions"].as_array().unwrap()) {
            assert_eq!(row.len(), count.as_u64().unwrap() as usize);
            assert!(row.iter().all(|x| x.is_finite()));
        }
        let bits: Vec<Vec<u32>> = scores.iter().map(|r| r.iter().map(|x| x.to_bits()).collect()).collect();
        let record = json!({"id":case["id"], "lengths":lengths,
            "policy_groups":groups, "packing_enabled":engine.checkpoint.pack_candidates,
            "scores":scores, "score_bits":bits});
        println!("PACKING_RECORD {record}");
        records.push(record);
    }
    json!({"original_revision":ORIGINAL, "manifest":manifest, "records":records})
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "run only on the frozen original checkout, pinned 9B and a reserved GPU"]
async fn record_original_native_reference() {
    // Deliberately separate from verification: ordinary test runs cannot rewrite the oracle.
    assert_ne!(std::env::var("CUA_S1_GRAPH").as_deref(), Ok("1"));
    let output = std::env::var_os("OPEN_JEV_ORACLE_OUTPUT").expect("explicit output required");
    let records = replay(true).await;
    std::fs::write(output, serde_json::to_vec_pretty(&records).unwrap()).unwrap();
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "needs pinned 9B, matching CUDA library and a reserved GPU; run eager and graph"]
async fn mixed_packing_matches_frozen_original_native() {
    let actual = replay(false).await;
    let reference: Value = serde_json::from_str(include_str!("data/packing_original.json")).unwrap();
    assert_eq!(reference["original_revision"], ORIGINAL);
    assert_eq!(actual["manifest"], reference["manifest"]);
    let expected = reference["records"].as_array().unwrap();
    let actual = actual["records"].as_array().unwrap();
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual["id"], expected["id"]);
        assert_eq!(actual["policy_groups"], expected["policy_groups"]);
        // Stronger than the public returned-probability tolerance: compare the
        // complete learned-head FP32 logits before response normalization/rounding.
        assert_eq!(actual["score_bits"], expected["score_bits"], "{}", actual["id"]);
    }
}
