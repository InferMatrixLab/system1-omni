//! Run with OPEN_JEV_PACK=0; this checks actual routing and the frozen original oracle.
use super::SchedulerKind;
use crate::engine::Engine;
use serde_json::Value;
use std::path::PathBuf;

#[tokio::test(flavor = "current_thread")]
#[ignore = "needs pinned 9B, OPEN_JEV_PACK=0, matching CUDA library and a reserved GPU"]
async fn rollback_uses_serial_candidates_and_matches_original_logits() {
    assert_eq!(std::env::var("OPEN_JEV_PACK").as_deref(), Ok("0"));
    let model = PathBuf::from(std::env::var_os("OPEN_JEV_MODEL").unwrap());
    let library = PathBuf::from(std::env::var_os("OPEN_JEV_CUDA_LIB").unwrap());
    let engine = Engine::load(&model, &library).await.unwrap();
    assert!(matches!(engine.scheduler.0, SchedulerKind::Serial(_)));
    assert!(!engine.executor.pack);
    assert!(!engine.executor.share_packed_prefix);
    assert!(!engine.executor.share_packed_inputs);
    assert!(!engine.executor.model.lock().unwrap().graph_stats().enabled);
    assert!(engine.is_available());
    let oracle: Value = serde_json::from_str(include_str!("data/packing_original.json")).unwrap();
    for (case, expected) in oracle["manifest"]["cases"]
        .as_array()
        .unwrap()
        .iter()
        .zip(oracle["records"].as_array().unwrap())
    {
        let candidates: Vec<Vec<u32>> = case["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                let length = c[0].as_u64().unwrap() as usize;
                let seed = c[1].as_u64().unwrap() as u32;
                (0..length)
                    .map(|i| 1000 + (seed * 7919 + i as u32 * 37) % 30000)
                    .collect()
            })
            .collect();
        let mut candidates = candidates.into_iter();
        let ids = case["questions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| {
                candidates
                    .by_ref()
                    .take(n.as_u64().unwrap() as usize)
                    .collect()
            })
            .collect();
        let (scores, used) = engine
            .executor
            .execute_with_prefix(&engine.scheduler, ids)
            .await
            .unwrap();
        assert!(!used.shared);
        let bits: Vec<Vec<u32>> = scores
            .iter()
            .map(|r| r.iter().map(|x| x.to_bits()).collect())
            .collect();
        assert_eq!(
            serde_json::json!(bits),
            expected["score_bits"],
            "{}",
            case["id"]
        );
    }
}
