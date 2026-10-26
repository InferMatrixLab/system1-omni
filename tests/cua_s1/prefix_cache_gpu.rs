//! The prefix cache on a merged-adapter text export: answers with a cold, warm or
//! evicting cache equal those of uncached fixed-GEMM passes, logit for logit.
use super::*;
use omni_runtime::SerialScheduler;

/// Three agent steps on new states, each asking the same two questions.
fn steps() -> Vec<Vec<u8>> {
    let goal = |task: &str| format!("{task} ").repeat(12);
    (0..3)
        .map(|step| {
            serde_json::to_vec(&serde_json::json!({
                "model": "cua-s1-4b-0.2",
                "state": format!("[{step}] button \"Search\"\n[{}] link \"Flights\"\n[{}] textbox \"From\"", step + 1, step + 2),
                "questions": {
                    "act": {"type": "choice", "instructions": goal("Find the cheapest nonstop flight from Zurich to London next Friday."),
                            "criteria": {"search": "Click Search", "flights": "Open Flights", "from": "Type in From"}},
                    "done": {"type": "choice", "instructions": goal("Decide whether the task is finished."),
                             "criteria": {"yes": "Finished", "no": "Not finished"}}
                }
            }))
            .unwrap()
        })
        .collect()
}

async fn answer(budget: usize) -> (Vec<Vec<u32>>, crate::prefix_cache::PrefixCacheStats) {
    let var = |name: &str| std::path::PathBuf::from(std::env::var_os(name).expect(name));
    let (model, library) = (var("CUA_S1_MODEL"), var("CUA_S1_CUDA_LIB"));
    let export = crate::provenance::text_export(&model).unwrap();
    assert_eq!(export, crate::provenance::TextExport::Merged);
    let mut processor = Processor::load(&model).unwrap();
    processor.checkpoints = true;
    let ids: Vec<u32> = LETTERS
        .chars()
        .map(|c| processor.tokenizer.token_to_id(&c.to_string()).unwrap())
        .collect();
    let executor = Executor::load(&model, &library, &ids, export, Some(budget))
        .await
        .unwrap();
    let scheduler = SerialScheduler::default();
    let mut rows = Vec::new();
    for raw in steps() {
        let prepared = processor.prepare(&raw).ok().unwrap();
        assert!(prepared.inputs.iter().all(|input| input.checkpoint >= 64));
        for row in executor.execute(&scheduler, prepared.inputs).await.unwrap() {
            rows.push(row.iter().map(|v| v.to_bits()).collect());
        }
    }
    (rows, executor.prefix_cache_stats().unwrap())
}

#[tokio::test]
#[ignore = "requires CUDA, CUA_S1_CUDA_LIB and a merged-adapter text export in CUA_S1_MODEL"]
async fn cached_blocks_answer_like_uncached_fixed_passes() {
    let (cached, stats) = answer(1 << 30).await;
    // each question's block is captured at the first step and reused at the next two
    assert_eq!((stats.misses, stats.hits, stats.entries), (2, 4, 2));
    assert!(stats.saved_tokens >= 4 * 64);
    // 80 MiB holds one of these blocks (~55 MiB each), so the two questions evict each
    // other at every step
    let (evicting, stats) = answer(80 << 20).await;
    assert_eq!((stats.hits, stats.misses, stats.evictions), (0, 6, 5));
    // cold, warm and evicted blocks run the same operations
    assert_eq!(cached, evicting);
    // one byte holds no block: every prompt runs as one forward_fixed pass, equal on GPUs
    // where the fixed GEMMs keep rows independent of the row count
    let (uncached, stats) = answer(1).await;
    assert_eq!((stats.hits, stats.misses, stats.uncached), (0, 0, 6));
    assert_eq!(cached, uncached);
}
