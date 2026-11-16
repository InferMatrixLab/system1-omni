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
    let bits = replay_bits(&engine).await;
    let oracle: Value = serde_json::from_str(include_str!("data/packing_original.json")).unwrap();
    for (actual, expected) in bits.iter().zip(oracle["records"].as_array().unwrap()) {
        assert_eq!(*actual, expected["score_bits"], "{}", expected["id"]);
    }
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "needs pinned 9B, matching CUDA library and a reserved GPU; run eager and graph"]
async fn default_matches_explicit_serial_on_same_device() {
    use omni_qwen3_5_native::prefix::PrefixMode;
    let model = PathBuf::from(std::env::var_os("OPEN_JEV_MODEL").unwrap());
    let library = PathBuf::from(std::env::var_os("OPEN_JEV_CUDA_LIB").unwrap());
    let serial = Engine::load_with_packing(&model, &library, PrefixMode::Off, false)
        .await
        .unwrap();
    let expected = replay_bits(&serial).await;
    drop(serial);
    let default = Engine::load_with_prefix(&model, &library, PrefixMode::Off)
        .await
        .unwrap();
    eprintln!(
        "same-device parity: effective packing={}",
        default.executor.packing_enabled()
    );
    if !default.executor.packing_enabled() {
        assert!(matches!(default.scheduler.0, SchedulerKind::Serial(_)));
        assert!(!default.executor.share_packed_prefix);
        assert!(!default.executor.share_packed_inputs);
        assert!(!default.executor.model.lock().unwrap().graph_stats().enabled);
    }
    let actual = replay_bits(&default).await;
    assert_eq!(
        actual, expected,
        "default versus explicit serial on this device"
    );
}

async fn replay_bits(engine: &Engine) -> Vec<Value> {
    let mut records = Vec::new();
    let oracle: Value = serde_json::from_str(include_str!("data/packing_original.json")).unwrap();
    for case in oracle["manifest"]["cases"].as_array().unwrap().iter() {
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
        records.push(serde_json::json!(bits));
    }
    records
}
