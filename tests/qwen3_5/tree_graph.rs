//! Ordinary graph capture/replay remains exact around eager prefix-tree execution.
use super::*;
use std::path::PathBuf;

fn inputs(length: usize, prefix: usize, count: usize, seed: usize) -> Vec<Vec<u32>> {
    (0..count)
        .map(|branch| {
            (0..length)
                .map(|i| {
                    256 + ((i * 37 + seed * 17 + if i < prefix { 0 } else { branch * 701 }) % 30000)
                        as u32
                })
                .collect()
        })
        .collect()
}

#[test]
#[ignore = "needs pinned9B export allowing4224tokens, CUDA and GPU reservation"]
fn ordinary_graphs_preserve_live_inputs_around_eager_tree_and_growth() {
    let dir = PathBuf::from(std::env::var_os("OPEN_JEV_MODEL").unwrap());
    let lib = PathBuf::from(std::env::var_os("OPEN_JEV_CUDA_LIB").unwrap());
    let load = |enabled| {
        let mut model = Model::load_with_graph(&dir, &lib, enabled).unwrap();
        model.enable_packed_projections().unwrap();
        model.enable_batched_gdn().unwrap();
        assert!(model.enable_equivalent_projection_tiles().unwrap());
        model
    };
    let (mut eager, mut graph) = (load(false), load(true));
    for (index, prefix) in [64, 512, 4032, 4160, 64].into_iter().enumerate() {
        for seed in [index * 5 + 1, index * 5 + 2] {
            let ids = inputs(128, 0, 1, seed);
            let refs: Vec<_> = ids.iter().map(Vec::as_slice).collect();
            assert_eq!(
                eager.forward_batch(&refs).unwrap(),
                graph.forward_batch(&refs).unwrap()
            );
        }
        let before = graph.graph_stats();
        assert!(before.captures > 0 && before.replays > 0);
        let ids = inputs(prefix + 64, prefix, 3, index * 5 + 3);
        let refs: Vec<_> = ids.iter().map(Vec::as_slice).collect();
        let original = if ids.iter().map(Vec::len).sum::<usize>() <= 4096 {
            vec![(prefix + 64) * 3; 3]
        } else {
            vec![prefix + 64; 3]
        };
        let want = eager
            .try_forward_prefix_tree(&refs, prefix, &original)
            .unwrap()
            .unwrap();
        let got = graph
            .try_forward_prefix_tree(&refs, prefix, &original)
            .unwrap()
            .unwrap();
        assert_eq!(want.hidden, got.hidden);
        assert_eq!(want.saved_tokens, got.saved_tokens);
        let after = graph.graph_stats();
        assert_eq!(before.captures, after.captures, "tree must stay eager");
        assert_eq!(
            before.replays, after.replays,
            "tree must not replay ordinary graphs"
        );
    }
    assert!(
        graph.graph_stats().invalidations > 0,
        "exercise replacement of captured scratch"
    );
    assert_eq!(graph.graph_stats().fallbacks, 0);
    // Different candidate order and fresh IDs must retain independent causal rows.
    for seed in [101, 102] {
        let ids = inputs(128, 0, 2, seed);
        for order in [[0, 1], [1, 0]] {
            let refs: Vec<_> = order.iter().map(|&i| ids[i].as_slice()).collect();
            assert_eq!(
                eager.forward_batch(&refs).unwrap(),
                graph.forward_batch(&refs).unwrap()
            );
        }
    }
}

#[test]
#[ignore = "needs pinned9B export allowing4224tokens, CUDA and GPU reservation"]
fn compact_graphs_keep_maps_keys_and_fresh_inputs_after_tree_growth() {
    let dir = PathBuf::from(std::env::var_os("OPEN_JEV_MODEL").unwrap());
    let lib = PathBuf::from(std::env::var_os("OPEN_JEV_CUDA_LIB").unwrap());
    let load = |enabled| {
        let mut m = Model::load_with_graph(&dir, &lib, enabled).unwrap();
        m.enable_packed_projections().unwrap();
        m.enable_batched_gdn().unwrap();
        assert!(m.enable_equivalent_projection_tiles().unwrap());
        m
    };
    let (mut eager, mut graph) = (load(false), load(true));
    // An ordinary layout with the same candidate lengths must not alias compact keys.
    let ids = inputs(128, 37, 4, 1);
    let refs: Vec<_> = ids.iter().map(Vec::as_slice).collect();
    assert_eq!(
        eager.forward_batch(&refs).unwrap(),
        graph.forward_batch(&refs).unwrap()
    );
    let initial_invalidations = graph.graph_stats().invalidations;
    for (length, prefix, count) in [(128, 37, 4), (128, 63, 4), (512, 480, 8), (128, 37, 4)] {
        let before = graph.graph_stats().replays;
        for seed in [2, 3, 4] {
            let ids = inputs(length, prefix, count, seed);
            let refs: Vec<_> = ids.iter().map(Vec::as_slice).collect();
            assert_eq!(
                eager
                    .try_forward_prefix_compaction(&refs, prefix)
                    .unwrap()
                    .unwrap(),
                graph
                    .try_forward_prefix_compaction(&refs, prefix)
                    .unwrap()
                    .unwrap()
            );
        }
        assert!(
            graph.graph_stats().replays >= before + 2,
            "exercise live-map graph replay"
        );
    }
    assert_eq!(
        graph.graph_stats().invalidations,
        initial_invalidations,
        "first map allocation must preserve ordinary graphs"
    );
    let before = graph.graph_stats().invalidations;
    let ids = inputs(4224, 4160, 3, 5);
    let refs: Vec<_> = ids.iter().map(Vec::as_slice).collect();
    assert_eq!(
        eager
            .try_forward_prefix_tree(&refs, 4160, &[4224; 3])
            .unwrap()
            .unwrap()
            .hidden,
        graph
            .try_forward_prefix_tree(&refs, 4160, &[4224; 3])
            .unwrap()
            .unwrap()
            .hidden
    );
    assert!(graph.graph_stats().invalidations > before);
    let before = graph.graph_stats();
    for seed in [6, 7] {
        let ids = inputs(128, 37, 4, seed);
        let refs: Vec<_> = ids.iter().map(Vec::as_slice).collect();
        assert_eq!(
            eager
                .try_forward_prefix_compaction(&refs, 37)
                .unwrap()
                .unwrap(),
            graph
                .try_forward_prefix_compaction(&refs, 37)
                .unwrap()
                .unwrap()
        );
    }
    assert_eq!(graph.graph_stats().captures, before.captures + 1);
    assert_eq!(graph.graph_stats().replays, before.replays + 1);
    assert_eq!(graph.graph_stats().fallbacks, 0);
}
