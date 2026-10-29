//! Full-checkpoint packing, sequence isolation and graph-shape regression check.
//! Run inside a GPU reservation with OPEN_JEV_MODEL and OPEN_JEV_CUDA_LIB set.

use std::path::PathBuf;

use omni_open_jev_native::engine::Engine;
use serde_json::json;

#[tokio::test(flavor = "current_thread")]
#[ignore = "needs the pinned Open-Jev export, CUDA library and a GPU reservation"]
async fn packed_candidates_preserve_isolation_order_and_graph_shapes() {
    let model = PathBuf::from(std::env::var_os("OPEN_JEV_MODEL").unwrap());
    let library = PathBuf::from(std::env::var_os("OPEN_JEV_CUDA_LIB").unwrap());
    let engine = Engine::load(&model, &library).await.unwrap();
    let request = json!({
        "state": "Refunds require a receipt. This customer has no receipt.",
        "questions": {
            "short": {"type": "noul", "instructions": "Is a refund permitted?"},
            "long": {"type": "noul", "instructions": format!("{}Is a refund permitted?", "Use only the stated policy. ".repeat(20))},
            "other": {"type": "noul", "instructions": "Does the customer have the required receipt?"}
        }
    });
    let prepared = engine
        .processor
        .prepare(&serde_json::to_vec(&request).unwrap())
        .unwrap();
    let inputs = prepared.inputs;
    assert_ne!(inputs[0][0].len(), inputs[1][0].len());
    let mut independent = Vec::new();
    for candidate in &inputs {
        independent.push(
            engine
                .executor
                .execute(&engine.scheduler, vec![candidate.clone()])
                .await
                .unwrap()[0][0],
        );
    }
    for order in [
        vec![0, 1, 2],
        vec![1, 0, 2],
        vec![2, 1, 0],
        vec![0, 1, 2],
        (0..17).map(|index| index % 3).collect(),
        vec![0, 1, 2],
        vec![0, 1, 2],
    ] {
        let grouped = order.iter().map(|&index| inputs[index].clone()).collect();
        let packed = engine
            .executor
            .execute(&engine.scheduler, grouped)
            .await
            .unwrap();
        assert_eq!(packed.len(), order.len());
        for (row, index) in packed.iter().zip(order) {
            assert_eq!(row.len(), 1);
            // Probability gates in the HTTP benchmark are stricter and cover
            // complete typed outputs. This logit check detects sequence leakage.
            assert!(
                (row[0] - independent[index]).abs() <= 0.01,
                "candidate {index}: packed={} independent={}",
                row[0],
                independent[index]
            );
        }
    }
    // A later singleton must not retain a packed sequence's recurrent state.
    let again = engine
        .executor
        .execute(&engine.scheduler, vec![inputs[0].clone()])
        .await
        .unwrap();
    assert_eq!(again[0][0], independent[0]);

    // Reuse both singleton and packed graph shapes with changed token IDs.
    // The new embeddings must replace the residual left by the previous replay.
    let mut changed = inputs.clone();
    changed[0][0].fill(42);
    let expected = engine
        .executor
        .execute(&engine.scheduler, vec![changed[0].clone()])
        .await
        .unwrap()[0][0];
    assert_ne!(expected, independent[0]);
    let packed = engine
        .executor
        .execute(&engine.scheduler, changed)
        .await
        .unwrap();
    assert_eq!(packed.len(), 3);
    for (row, expected) in packed
        .iter()
        .zip([expected, independent[1], independent[2]])
    {
        assert_eq!(row.len(), 1);
        assert!((row[0] - expected).abs() <= 0.01);
    }
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "needs the pinned Open-Jev export, CUDA library and a GPU reservation"]
async fn text_prefix_buffers_grow_and_shrink_without_reusing_prior_contents() {
    let model = PathBuf::from(std::env::var_os("OPEN_JEV_MODEL").unwrap());
    let library = PathBuf::from(std::env::var_os("OPEN_JEV_CUDA_LIB").unwrap());
    let engine = Engine::load(&model, &library).await.unwrap();
    let mut first = None;
    for (prefix, token) in [(2048, 42), (3072, 43), (2048, 44), (2048, 42)] {
        let inputs: Vec<Vec<u32>> = (0..3)
            .map(|index| {
                let mut ids = vec![token; prefix];
                ids.extend(vec![7 + index; 256]);
                ids
            })
            .collect();
        let mut expected = Vec::new();
        for ids in &inputs {
            expected.push(
                engine
                    .executor
                    .execute(&engine.scheduler, vec![vec![ids.clone()]])
                    .await
                    .unwrap()[0][0],
            );
        }
        let (scores, used) = engine
            .executor
            .execute_with_prefix(&engine.scheduler, vec![inputs])
            .await
            .unwrap();
        assert_eq!(
            used.saved_tokens,
            2 * prefix,
            "exercise actual prefix reuse"
        );
        assert_eq!(scores, vec![expected.clone()]);
        if token == 42 {
            if let Some(first) = &first {
                assert_eq!(&scores, first, "returning input after changed state");
            } else {
                first = Some(scores);
            }
        }
    }
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "needs the pinned Open-Jev-9B export, CUDA library and a GPU reservation"]
async fn nine_b_long_candidates_keep_prefix_off_and_independent() {
    use omni_qwen3_5_native::prefix::PrefixMode;

    let model = PathBuf::from(std::env::var_os("OPEN_JEV_MODEL").unwrap());
    let library = PathBuf::from(std::env::var_os("OPEN_JEV_CUDA_LIB").unwrap());
    let engine = Engine::load_with_prefix(&model, &library, PrefixMode::Off)
        .await
        .unwrap();
    assert_eq!(engine.checkpoint.alias, "open-jev-9b");
    let candidates: Vec<Vec<u32>> = (0..3)
        .map(|index| {
            let mut ids = vec![42; 2048];
            ids.extend(vec![7 + index; 64]);
            ids
        })
        .collect();
    let (packed, used) = engine
        .executor
        .execute_with_prefix(&engine.scheduler, vec![candidates.clone()])
        .await
        .unwrap();
    assert!(!used.shared, "9B batching must not turn on prefix reuse");
    assert_eq!(used.saved_tokens, 0);
    for (index, candidate) in candidates.into_iter().enumerate() {
        let independent = engine
            .executor
            .execute(&engine.scheduler, vec![vec![candidate]])
            .await
            .unwrap();
        assert_eq!(packed[0][index], independent[0][0]);
    }
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "needs pinned9B export, OPEN_JEV_EXACT_PREFIX=1, CUDA and a GPU reservation"]
async fn packed_prefix_matches_original_batches_across_budget_boundaries() {
    use omni_qwen3_5_native::prefix::PrefixMode;
    let model = PathBuf::from(std::env::var_os("OPEN_JEV_MODEL").unwrap());
    let library = PathBuf::from(std::env::var_os("OPEN_JEV_CUDA_LIB").unwrap());
    let scenarios = [
        (512, vec![64, 65, 70], 42),
        (768, vec![64, 65, 70, 72], 100),
        (1024, vec![32, 64, 65, 70], 42),
        (1984, vec![64, 65, 66], 100),
        (512, vec![1; 8], 42),
        (2048, vec![64; 3], 100),
        (512, vec![64, 65, 70], 42),
    ];
    let cases: Vec<Vec<Vec<u32>>> = scenarios
        .iter()
        .map(|(prefix, tails, token)| {
            tails
                .iter()
                .enumerate()
                .map(|(i, tail)| {
                    let mut ids = vec![*token; *prefix];
                    ids.extend(vec![7 + i as u32; *tail]);
                    ids
                })
                .collect()
        })
        .collect();
    let baseline = Engine::load_with_prefix(&model, &library, PrefixMode::Off)
        .await
        .unwrap();
    let mut expected = Vec::new();
    for inputs in &cases {
        expected.push(
            baseline
                .executor
                .execute(&baseline.scheduler, vec![inputs.clone()])
                .await
                .unwrap(),
        );
    }
    drop(baseline);
    let candidate = Engine::load(&model, &library).await.unwrap();
    let mut reused = 0;
    for ((inputs, expected), (prefix, _, _)) in cases.into_iter().zip(expected).zip(scenarios) {
        let count = inputs.len();
        let (actual, used) = candidate
            .executor
            .execute_with_prefix(&candidate.scheduler, vec![inputs])
            .await
            .unwrap();
        assert_eq!(actual, expected, "original packed projection arithmetic");
        if used.shared {
            assert_eq!(used.saved_tokens, (count - 1) * prefix);
            reused += 1;
        }
    }
    assert!(
        reused >= 3,
        "exercise actual midlength prefix reuse, not only fallback"
    );
}

#[test]
#[ignore = "needs pinned9B export, CUDA library and a GPU reservation"]
fn grouped_forward_preserves_original_projection_sizes_and_graph_keys() {
    use omni_qwen3_5_native::model::Model;
    let dir = PathBuf::from(std::env::var_os("OPEN_JEV_MODEL").unwrap());
    let lib = PathBuf::from(std::env::var_os("OPEN_JEV_CUDA_LIB").unwrap());
    let graph = std::env::var("CUA_S1_GRAPH").as_deref() == Ok("1");
    let mut model = Model::load_with_graph(&dir, &lib, graph).unwrap();
    model.enable_batched_gdn().unwrap();
    model.enable_packed_projections().unwrap();
    model.enable_equivalent_projection_tiles().unwrap();
    let lengths = [79, 81, 83, 255, 257, 513];
    for token in [42, 100, 42] {
        let ids: Vec<Vec<u32>> = lengths
            .iter()
            .enumerate()
            .map(|(i, &n)| vec![token + i as u32; n])
            .collect();
        let inputs: Vec<&[u32]> = ids.iter().map(Vec::as_slice).collect();
        for groups in [vec![3, 3], vec![2, 4], vec![1, 2, 3], vec![6], vec![3, 3]] {
            let mut expected = Vec::new();
            let mut start = 0;
            for &count in &groups {
                expected.extend(model.forward_batch(&inputs[start..start + count]).unwrap());
                start += count;
            }
            let actual = model
                .forward_batch_preserving_groups(&inputs, &groups)
                .unwrap();
            assert_eq!(
                actual, expected,
                "original groups={groups:?}, token={token}"
            );
        }
    }
    assert!(
        model
            .forward_batch_preserving_groups(&[&[42u32]], &[0, 1])
            .is_err()
    );
    assert!(
        model
            .forward_batch_preserving_groups(&[&[42u32]], &[2])
            .is_err()
    );
    if graph {
        assert!(model.graph_stats().replays > 0);
    }
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "needs pinned9B export, exact prefix enabled, CUDA and a GPU reservation"]
async fn concurrent_packed_and_prefix_requests_preserve_routing_and_state() {
    let dir = PathBuf::from(std::env::var_os("OPEN_JEV_MODEL").unwrap());
    let lib = PathBuf::from(std::env::var_os("OPEN_JEV_CUDA_LIB").unwrap());
    let engine = Engine::load(&dir, &lib).await.unwrap();
    let shared = |prefix: usize, tails: &[usize], token: u32| -> Vec<Vec<u32>> {
        tails
            .iter()
            .enumerate()
            .map(|(i, &tail)| {
                let mut ids = vec![token; prefix];
                ids.extend(vec![7 + i as u32; tail]);
                ids
            })
            .collect()
    };
    let requests = [
        vec![shared(37, &[90, 91, 89, 91], 42)],
        vec![vec![vec![45; 70]], vec![vec![46; 129], vec![47; 130]]],
        vec![shared(768, &[64, 65, 66, 67], 100)],
        vec![shared(1024, &[32, 32, 32, 32, 32], 42)],
        vec![vec![vec![48; 4096]]],
    ];
    let mut expected = Vec::new();
    for request in &requests {
        let (scores, used) = engine
            .executor
            .execute_with_prefix(&engine.scheduler, request.clone())
            .await
            .unwrap();
        expected.push((scores, used.shared, used.saved_tokens));
    }
    assert!(
        expected
            .iter()
            .any(|(_, shared, saved)| *shared && *saved > 0)
    );
    for order in [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [0, 1, 2, 3, 4]] {
        let (a, b, c, d, e) = tokio::join!(
            engine
                .executor
                .execute_with_prefix(&engine.scheduler, requests[order[0]].clone()),
            engine
                .executor
                .execute_with_prefix(&engine.scheduler, requests[order[1]].clone()),
            engine
                .executor
                .execute_with_prefix(&engine.scheduler, requests[order[2]].clone()),
            engine
                .executor
                .execute_with_prefix(&engine.scheduler, requests[order[3]].clone()),
            engine
                .executor
                .execute_with_prefix(&engine.scheduler, requests[order[4]].clone()),
        );
        for (actual, index) in [a, b, c, d, e].into_iter().zip(order) {
            let (scores, used) = actual.unwrap();
            assert_eq!((scores, used.shared, used.saved_tokens), expected[index]);
        }
    }
}

#[test]
#[ignore = "needs pinned9B export, retained projection CUDA symbols and a GPU reservation"]
fn pointwise_prefix_compaction_preserves_full_mixers_and_changed_inputs() {
    use omni_qwen3_5_native::model::Model;
    let dir = PathBuf::from(std::env::var_os("OPEN_JEV_MODEL").unwrap());
    let lib = PathBuf::from(std::env::var_os("OPEN_JEV_CUDA_LIB").unwrap());
    let mut original = Model::load(&dir, &lib).unwrap();
    let mut candidate = Model::load(&dir, &lib).unwrap();
    for model in [&mut original, &mut candidate] {
        model.enable_packed_projections().unwrap();
        model.enable_batched_gdn().unwrap();
        assert!(model.enable_equivalent_projection_tiles().unwrap());
    }
    let mut used = 0;
    for (lengths, prefix, seed) in [
        (vec![128, 126, 127, 128], 37, 42),
        (vec![128; 8], 63, 43),
        (vec![508, 510, 512], 433, 44),
        (vec![512; 8], 480, 45),
        (vec![128, 126, 127, 128], 37, 46),
        (vec![128, 126, 127, 128], 37, 42),
    ] {
        let ids: Vec<Vec<u32>> = lengths
            .iter()
            .enumerate()
            .map(|(i, &length)| {
                (0..length)
                    .map(|n| {
                        if n < prefix {
                            seed + n as u32 % 11
                        } else {
                            1000 + (i * 7919 + n * 37 + seed as usize) as u32 % 30000
                        }
                    })
                    .collect()
            })
            .collect();
        let inputs: Vec<&[u32]> = ids.iter().map(Vec::as_slice).collect();
        let expected = original.forward_batch(&inputs).unwrap();
        let actual = candidate
            .try_forward_prefix_compaction(&inputs, prefix)
            .unwrap();
        if let Some(actual) = actual {
            assert_eq!(
                actual, expected,
                "prefix={prefix} lengths={lengths:?} seed={seed}"
            );
            used += 1;
        }
        // A normal forward after compaction must see fresh embeddings and all rows.
        assert_eq!(candidate.forward_batch(&inputs).unwrap(), expected);
    }
    assert!(
        used >= 5,
        "exercise compact projections instead of only fallback"
    );
    // Different original split-K reductions must decline before modifying state.
    let a = vec![42; 80];
    let mut b = vec![42; 83];
    b[20] = 43;
    assert!(
        candidate
            .try_forward_prefix_compaction(&[&a, &b], 20)
            .unwrap()
            .is_none()
    );
    assert_eq!(
        candidate.forward_batch(&[&a, &b]).unwrap(),
        original.forward_batch(&[&a, &b]).unwrap()
    );
}
