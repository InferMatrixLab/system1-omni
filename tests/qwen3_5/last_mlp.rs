//! Full-checkpoint last-row arithmetic, changed inputs and scratch/graph lifetime.
use super::*;
use std::path::PathBuf;

#[test]
#[ignore = "needs pinned 27B export, retained GEMM CUDA symbols, SM90 and a GPU reservation"]
fn last_mlp_matches_complete_forward_at_boundaries_and_replay() {
    let model = PathBuf::from(std::env::var_os("OPEN_JEV_MODEL").unwrap());
    let library = PathBuf::from(std::env::var_os("OPEN_JEV_CUDA_LIB").unwrap());
    let mut original = Model::load(&model, &library).unwrap();
    let mut optimized = Model::load(&model, &library).unwrap();
    optimized.enable_packed_projections().unwrap();
    optimized.enable_batched_gdn().unwrap();
    assert!(
        optimized.last_mlp_reuse(1213).unwrap().is_some(),
        "must exercise one-row retained algorithms, not only fallback"
    );
    let lengths = [
        1usize, 2, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 1213,
        2047, 2048, 2049, 4095, 4096, 4097, 16383, 16384, 16385,
        // Return to earlier shapes after larger scratch invalidated their graphs.
        1213, 1, 4096, 1213,
    ];
    let mut compared = 0;
    for (visit, length) in lengths.into_iter().enumerate() {
        for seed in [20261022u32, 174723] {
            let ids: Vec<u32> = (0..length)
                .map(|index| {
                    let mut value = (index as u32) ^ seed ^ (visit as u32).wrapping_mul(713);
                    value ^= value >> 16;
                    value = value.wrapping_mul(0x7feb352d);
                    value ^= value >> 15;
                    256 + value % 30000
                })
                .collect();
            let expected = original.forward(&ids).unwrap();
            for replay in [false, true] {
                if replay && visit == 17 && seed == 20261022 {
                    let reuse = cuda::api().reuse_gemm().unwrap();
                    for rows in 1usize..=1024 {
                        // SAFETY: synchronous host-only plans for valid dimensions.
                        // Evict the 512 retained descriptors after the preceding
                        // forward captured its one-row MLP, before replaying it.
                        unsafe {
                            check(
                                (reuse.check)(optimized.gemm, &rows, 1, &rows, 1, 64, 64),
                                "last-row retained-plan eviction",
                            )
                            .unwrap();
                        }
                    }
                }
                let actual = optimized.forward(&ids).unwrap();
                assert_eq!(expected.len(), 5120);
                assert_eq!(actual.len(), expected.len());
                assert!(expected.iter().chain(&actual).all(|x| x.is_finite()));
                let different = expected
                    .iter()
                    .zip(&actual)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                assert_eq!(
                    different, 0,
                    "length={length} seed={seed} visit={visit} repeat={replay} graph={}",
                    optimized.graph_enabled
                );
                compared += 1;
            }
        }
    }
    original.synchronize().unwrap();
    optimized.synchronize().unwrap();
    assert_eq!(compared, 124);
}

#[test]
#[ignore = "needs pinned 27B export, older packed CUDA library without retained GEMM, and a GPU reservation"]
fn missing_last_mlp_symbols_preserve_complete_forward() {
    let model = PathBuf::from(std::env::var_os("OPEN_JEV_MODEL").unwrap());
    let library = PathBuf::from(std::env::var_os("OPEN_JEV_CUDA_LIB").unwrap());
    let mut original = Model::load(&model, &library).unwrap();
    let mut optimized = Model::load(&model, &library).unwrap();
    optimized.enable_packed_projections().unwrap();
    assert!(cuda::api().reuse_gemm().is_none());
    assert!(optimized.last_mlp_reuse(1213).unwrap().is_none());
    for (visit, length) in [1usize, 512, 1213, 4096, 1213].into_iter().enumerate() {
        for seed in [73usize, 871] {
            let ids: Vec<u32> = (0..length)
                .map(|index| (256 + (index * 23 + seed + visit * 127) % 30000) as u32)
                .collect();
            let expected = original.forward(&ids).unwrap();
            let actual = optimized.forward(&ids).unwrap();
            assert_eq!(expected.len(), 5120);
            assert_eq!(actual.len(), expected.len());
            assert!(expected.iter().chain(&actual).all(|x| x.is_finite()));
            let different = expected
                .iter()
                .zip(&actual)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(different, 0, "fallback length={length} seed={seed}");
        }
    }
    original.synchronize().unwrap();
    optimized.synchronize().unwrap();
}
