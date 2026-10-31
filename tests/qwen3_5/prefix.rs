//! GPU checks of `Model::forward_shared` against `forward_fixed` on each full prompt. They
//! need a GPU, CUA_S1_CUDA_LIB pointing at libqwen3_5_cuda.so and QWEN3_5_MODEL pointing at
//! a merged Qwen3.5/3.8 checkpoint, such as a Cua-S1 or Open-Jev export:
//!
//!     CUA_S1_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so QWEN3_5_MODEL=<export> \
//!       cargo test --release -p omni-qwen3-5-native --test prefix -- --ignored --test-threads=1
//!
//! With CUA_S1_GRAPH=1, repeated plain `forward` calls of one length replay a CUDA Graph,
//! also between shared runs.

use std::path::PathBuf;

use omni_qwen3_5_native::model::{Model, PromptGroup, SharedPrompts};
use omni_qwen3_5_native::prefix_cache::PrefixCache;

fn model() -> Model {
    let var = |name: &str| {
        std::env::var_os(name)
            .map(PathBuf::from)
            .unwrap_or_else(|| panic!("{name} must be set"))
    };
    Model::load(&var("QWEN3_5_MODEL"), &var("CUA_S1_CUDA_LIB")).unwrap()
}

/// Token ids in [1000, 100000) from a fixed seed, inside every Qwen3.5 vocabulary.
fn tokens(n: usize, seed: u64) -> Vec<u32> {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            1000 + (x >> 33) as u32 % 99_000
        })
        .collect()
}

struct Case {
    prefix: Vec<u32>,
    groups: Vec<(Vec<u32>, Vec<Vec<u32>>)>,
}

impl Case {
    /// A request prefix of `p` tokens and groups of (prefix length, branch lengths).
    fn new(p: usize, groups: &[(usize, &[usize])], seed: u64) -> Self {
        let mut seed = seed * 1000;
        let mut next = |n| {
            seed += 1;
            tokens(n, seed)
        };
        Self {
            prefix: next(p),
            groups: groups
                .iter()
                .map(|&(q, branches)| (next(q), branches.iter().map(|&b| next(b)).collect()))
                .collect(),
        }
    }

    fn prompts(&self) -> SharedPrompts<'_> {
        SharedPrompts {
            prefix: &self.prefix,
            groups: self
                .groups
                .iter()
                .map(|(prefix, branches)| PromptGroup {
                    prefix,
                    branches: branches.iter().map(Vec::as_slice).collect(),
                })
                .collect(),
        }
    }

    fn full(&self, g: usize, b: usize) -> Vec<u32> {
        let (prefix, branches) = &self.groups[g];
        [&self.prefix[..], prefix, &branches[b]].concat()
    }
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let diff: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
        .sum();
    let norm: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (diff / norm).sqrt()
}

fn bits(rows: &[Vec<Vec<f32>>]) -> Vec<u32> {
    rows.iter()
        .flatten()
        .flatten()
        .map(|v| v.to_bits())
        .collect()
}

#[test]
#[ignore = "needs a GPU, CUA_S1_CUDA_LIB and QWEN3_5_MODEL"]
fn shared_prefixes_match_full_prompts() {
    let mut model = model();
    // Request prefixes around and inside 64-token chunks; groups without a prefix of their
    // own, with one token, 70 and 128; branches of 1 to 200 tokens.
    let groups: &[(usize, &[usize])] =
        &[(0, &[1, 64]), (1, &[2, 65]), (70, &[1, 200]), (128, &[3])];
    for (i, p) in [0usize, 1, 63, 64, 65, 300, 1000].into_iter().enumerate() {
        let case = Case::new(p, groups, i as u64 + 1);
        let shared = model.forward_shared(&case.prompts()).unwrap();
        let mut to_forward = 0f64;
        for (g, rows) in shared.iter().enumerate() {
            for (b, row) in rows.iter().enumerate() {
                let full = case.full(g, b);
                let want = model.forward_fixed(&full).unwrap();
                assert!(row.iter().all(|v| v.is_finite()));
                assert!(
                    row.iter()
                        .map(|v| v.to_bits())
                        .eq(want.iter().map(|v| v.to_bits())),
                    "prefix {p}, group {g}, branch {b}"
                );
                // the GEMM algorithms differ from `forward`'s, not the math
                let plain = model.forward(&full).unwrap();
                to_forward = to_forward.max(rel_l2(row, &plain));
            }
        }
        eprintln!(
            "prefix {p}: bit-identical to forward_fixed; largest relative L2 difference from forward {to_forward:.2e}"
        );
        // A sanity bound for the change of GEMM algorithms, not a tolerance for reuse.
        assert!(to_forward <= 3e-2, "prefix {p}: {to_forward}");
    }
}

#[test]
#[ignore = "needs a GPU, CUA_S1_CUDA_LIB and QWEN3_5_MODEL"]
fn packing_options_leave_fixed_rows_unchanged() {
    let groups: &[(usize, &[usize])] =
        &[(0, &[1, 64]), (1, &[2, 65]), (70, &[1, 200]), (128, &[3])];
    let case = Case::new(300, groups, 21);
    let full: Vec<Vec<u32>> = groups
        .iter()
        .enumerate()
        .flat_map(|(g, (_, branches))| (0..branches.len()).map(move |b| (g, b)))
        .map(|(g, b)| case.full(g, b))
        .collect();
    let want: Vec<Vec<f32>> = {
        let mut plain = model();
        full.iter()
            .map(|ids| plain.forward_fixed(ids).unwrap())
            .collect()
    };
    // what Open-Jev selects at load for checkpoints that pack candidates (27B)
    let mut model = model();
    model.enable_batched_gdn().unwrap();
    model.enable_packed_projections().unwrap();
    model.check_fixed_rows().unwrap();
    let flat = |rows: Vec<Vec<Vec<f32>>>| rows.into_iter().flatten().collect::<Vec<_>>();
    let first = flat(model.forward_shared(&case.prompts()).unwrap());
    // the packed default path in between, as when `auto` declines a request
    let inputs: Vec<&[u32]> = full.iter().map(Vec::as_slice).collect();
    model.forward_batch(&inputs).unwrap();
    let again = flat(model.forward_shared(&case.prompts()).unwrap());
    for (i, ids) in full.iter().enumerate() {
        let fixed = model.forward_fixed(ids).unwrap();
        for (name, row) in [
            ("shared", &first[i]),
            ("shared again", &again[i]),
            ("fixed", &fixed),
        ] {
            assert!(
                row.iter()
                    .map(|v| v.to_bits())
                    .eq(want[i].iter().map(|v| v.to_bits())),
                "{name}, prompt {i}"
            );
        }
    }
}

#[test]
#[ignore = "needs a GPU, CUA_S1_CUDA_LIB and QWEN3_5_MODEL"]
fn shared_prefix_state_stays_inside_one_call() {
    let mut model = model();
    let a = Case::new(200, &[(10, &[3, 40, 1]), (0, &[64]), (65, &[2, 7])], 11);
    // longer than any prompt before it, so the buffers grow in between
    let b = Case::new(1100, &[(30, &[9, 100])], 12);
    let plain = model.forward(&a.full(0, 1)).unwrap();
    // with CUA_S1_GRAPH=1, the first call captured a graph and this one replays it
    assert_eq!(model.forward(&a.full(0, 1)).unwrap(), plain);
    let first = model.forward_shared(&a.prompts()).unwrap();
    // the same request again, and after a longer one
    let again = model.forward_shared(&a.prompts()).unwrap();
    assert_eq!(bits(&first), bits(&again));
    // the plain forward between shared runs, replayed with CUA_S1_GRAPH=1
    assert_eq!(model.forward(&a.full(0, 1)).unwrap(), plain);
    model.forward_shared(&b.prompts()).unwrap();
    let after = model.forward_shared(&a.prompts()).unwrap();
    assert_eq!(bits(&first), bits(&after));
    // groups and branches in reverse order: every branch keeps its result
    let reversed = Case {
        prefix: a.prefix.clone(),
        groups: a
            .groups
            .iter()
            .rev()
            .map(|(prefix, branches)| (prefix.clone(), branches.iter().rev().cloned().collect()))
            .collect(),
    };
    let back = model.forward_shared(&reversed.prompts()).unwrap();
    let mut back: Vec<Vec<Vec<f32>>> = back.into_iter().rev().collect();
    back.iter_mut().for_each(|branches| branches.reverse());
    assert_eq!(bits(&first), bits(&back));
    // the plain forward is unaffected (eager here: the longer request dropped the graphs)
    assert_eq!(model.forward(&a.full(0, 1)).unwrap(), plain);
    // Invalid requests fail without breaking the model. All of these are refused before
    // any GPU work is queued.
    let empty: &[u32] = &[];
    let outside: &[u32] = &[5, u32::MAX];
    let group = |prefix, branches| PromptGroup { prefix, branches };
    let bad = [
        SharedPrompts {
            prefix: &a.prefix,
            groups: vec![],
        },
        SharedPrompts {
            prefix: &a.prefix,
            groups: vec![group(empty, vec![])],
        },
        SharedPrompts {
            prefix: &a.prefix,
            groups: vec![group(empty, vec![empty])],
        },
        SharedPrompts {
            prefix: &a.prefix,
            groups: vec![group(empty, vec![outside])],
        },
        SharedPrompts {
            prefix: &a.prefix,
            groups: vec![group(outside, vec![&a.prefix])],
        },
        SharedPrompts {
            prefix: outside,
            groups: vec![group(empty, vec![&a.prefix])],
        },
    ];
    for prompts in &bad {
        assert!(model.forward_shared(prompts).is_err());
    }
    assert!(model.forward_fixed(outside).is_err());
    let last = model.forward_shared(&a.prompts()).unwrap();
    assert_eq!(bits(&first), bits(&last));
}

#[test]
#[ignore = "needs a GPU, CUA_S1_CUDA_LIB and QWEN3_5_MODEL"]
fn cached_prefixes_match_full_prompts_across_hits_misses_and_evictions() {
    let mut model = model();
    let block = |n, seed| tokens(n, seed);
    let (a, b, c) = (block(128, 21), block(128, 22), block(192, 23));
    let full = |head: &[u32], n, seed| [head, &tokens(n, seed)].concat();
    // room for two 128-token blocks, not for a 192-token block next to one
    let mut cache = PrefixCache::new(2 * model.prefix_bytes(128) + 1);
    let check = |model: &mut Model, cache: &mut PrefixCache, ids: &[u32], checkpoint| {
        let got = cache.forward(model, ids, checkpoint).unwrap();
        let want = model.forward_fixed(ids).unwrap();
        assert!(
            got.iter()
                .map(|v| v.to_bits())
                .eq(want.iter().map(|v| v.to_bits())),
            "{} tokens from {checkpoint}",
            ids.len()
        );
    };
    check(&mut model, &mut cache, &full(&a, 1, 31), 128); // miss
    check(&mut model, &mut cache, &full(&a, 300, 32), 128); // hit
    // a hit after the scratch buffers grow past every earlier prompt
    check(&mut model, &mut cache, &full(&a, 1500, 39), 128);
    check(&mut model, &mut cache, &full(&b, 40, 33), 128); // miss
    check(&mut model, &mut cache, &full(&a, 64, 34), 128); // hit
    let stats = cache.stats();
    assert_eq!((stats.hits, stats.misses, stats.entries), (3, 2, 2));
    assert_eq!(stats.saved_tokens, 384);
    // `c` drops both: `b`, least recently used, and then `a`
    check(&mut model, &mut cache, &full(&c, 5, 35), 192);
    let stats = cache.stats();
    assert_eq!((stats.misses, stats.evictions, stats.entries), (3, 2, 1));
    assert_eq!(stats.bytes, model.prefix_bytes(192));
    check(&mut model, &mut cache, &full(&a, 2, 36), 128); // a miss again
    // a block that ends at or past the prompt, or off a 64-token boundary, runs uncached
    let before = cache.stats();
    for (ids, checkpoint) in [
        (a.clone(), 128),
        (full(&a, 10, 37), 0),
        (full(&a, 10, 37), 100),
    ] {
        check(&mut model, &mut cache, &ids, checkpoint);
    }
    assert_eq!(
        cache.stats().hits + cache.stats().misses,
        before.hits + before.misses
    );
    // a block over the whole budget runs uncached and is not kept
    let mut tiny = PrefixCache::new(1);
    check(&mut model, &mut tiny, &full(&a, 3, 38), 128);
    assert_eq!((tiny.stats().uncached, tiny.stats().entries), (1, 0));
}
