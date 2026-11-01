use super::*;
use half::bf16;

#[test]
fn shared_multimodal_header_checks_tokens_positions_and_image_values() {
    let ids = vec![1; 129];
    let indices = [8];
    let features = [bf16::ZERO; 4];
    let p: Vec<_> = (0..129).map(i64::from).collect();
    let make = |ids, features, positions| MultimodalInput {
        token_ids: ids,
        image_token_indices: &indices,
        image_embeddings: features,
        position_ids: positions,
    };
    assert!(
        common_header(
            &[
                make(&ids, &features, [&p; 3]),
                make(&ids, &features, [&p; 3])
            ],
            None,
            128,
            4
        )
        .is_ok()
    );
    let mut changed = ids.clone();
    changed[12] = 2;
    assert!(
        common_header(
            &[
                make(&ids, &features, [&p; 3]),
                make(&changed, &features, [&p; 3])
            ],
            None,
            128,
            4
        )
        .is_err()
    );
    let different = [bf16::ONE; 4];
    assert!(
        common_header(
            &[
                make(&ids, &features, [&p; 3]),
                make(&ids, &different, [&p; 3])
            ],
            None,
            128,
            4
        )
        .is_err()
    );
    let signed_zero = [bf16::from_bits(0x8000); 4];
    assert!(
        common_header(
            &[
                make(&ids, &features, [&p; 3]),
                make(&ids, &signed_zero, [&p; 3])
            ],
            None,
            128,
            4
        )
        .is_err()
    );
    let mut axis = p.clone();
    axis[9] += 1;
    assert!(
        common_header(
            &[
                make(&ids, &features, [&p; 3]),
                make(&ids, &features, [&p, &axis, &p])
            ],
            None,
            128,
            4
        )
        .is_err()
    );
    assert!(
        common_header(
            &[
                make(&ids, &features, [&p; 3]),
                make(&ids, &features, [&p; 3])
            ],
            None,
            127,
            4
        )
        .is_err()
    );
}

#[test]
#[ignore = "requires CUDA and pinned CUA_S1_MODEL/CUA_S1_CUDA_LIB; run serially"]
fn multimodal_shared_suffix_batch_keeps_hidden_bits_after_pixel_changes() {
    use std::path::PathBuf;
    let dir = PathBuf::from(std::env::var_os("CUA_S1_MODEL").expect("CUA_S1_MODEL"));
    let lib = PathBuf::from(std::env::var_os("CUA_S1_CUDA_LIB").expect("CUA_S1_CUDA_LIB"));
    for graph in [false, true] {
        let mut model = Model::load_with_graph(&dir, &lib, graph).unwrap();
        model.enable_multimodal_batching().unwrap();
        for (prefix, seed) in [
            (64, 1),
            (128, 2),
            (576, 3),
            (576, 4),
            (8064, 6),
            (512, 5),
            (64, 1),
        ] {
            let lengths = if prefix == 8064 {
                // Per-prompt limit, oversized compact layout, then smaller replay.
                vec![8192; 16]
            } else if prefix == 576 {
                vec![1030; 8]
            } else if seed == 5 {
                vec![513; 16]
            } else {
                vec![192, 193, 192, 192]
            };
            let ids: Vec<Vec<_>> = lengths
                .into_iter()
                .enumerate()
                .map(|(i, n)| {
                    let mut ids: Vec<_> = (0..n)
                        .map(|j| {
                            (10 + (j * 17 + if j >= prefix { i * 31 + seed } else { 0 }) % 1000)
                                as u32
                        })
                        .collect();
                    ids[8..12].fill(model.cfg.image_token_id.unwrap());
                    ids
                })
                .collect();
            let features: Vec<_> = (0..4 * model.cfg.hidden)
                .map(|j| bf16::from_f32(((j * 7 + seed * 13) % 71) as f32 / 71. - 0.5))
                .collect();
            let positions: Vec<[Vec<i64>; 3]> = ids
                .iter()
                .map(|ids| {
                    std::array::from_fn(|axis| {
                        (0..ids.len()).map(|p| (p + axis * 3) as i64).collect()
                    })
                })
                .collect();
            let indices = [8, 9, 10, 11];
            let inputs: Vec<_> = ids
                .iter()
                .zip(&positions)
                .map(|(ids, p)| MultimodalInput {
                    token_ids: ids,
                    image_token_indices: &indices,
                    image_embeddings: &features,
                    position_ids: [&p[0], &p[1], &p[2]],
                })
                .collect();
            let expected: Vec<_> = inputs
                .iter()
                .map(|p| model.forward_multimodal(p).unwrap())
                .collect();
            for _ in 0..2 {
                let actual = model
                    .forward_multimodal_batch_with_prefix(&inputs, prefix)
                    .unwrap();
                for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
                    assert_eq!(
                        a.iter()
                            .zip(b)
                            .filter(|(a, b)| a.to_bits() != b.to_bits())
                            .count(),
                        0,
                        "question={i} prefix={prefix} seed={seed} graph={graph}"
                    );
                }
            }
        }
        if graph {
            let stats = model.graph_stats();
            assert!(stats.enabled);
            assert_eq!(stats.fallbacks, 0);
            assert!(
                model.mm_batch_windows.iter().any(|entry| matches!(
                    &entry.key, PassKey::Compact { lengths, .. } if lengths.len() == 16
                )),
                "qualify the compact path with seventeen internal segments"
            );
            assert!(
                model
                    .mm_batch_windows
                    .iter()
                    .any(|e| matches!(e.key, PassKey::Batch { .. } | PassKey::Compact { .. }))
            );
        }
    }
}

#[test]
fn compact_projection_rows_keep_prefix_once_and_bound_suffixes_and_scratch() {
    assert_eq!(compact_rows(&[1030; 8], 576), Some(4208));
    assert_eq!(compact_rows(&[320; 16], 64), Some(4160));
    for (lengths, prefix) in [
        (vec![129], 64),
        (vec![129; 17], 64),
        (vec![0, 129], 64),
        (vec![64; 2], 64),
        (vec![129; 2], 63),
        (vec![129; 2], 0),
        (vec![4096; 8], 64),
        (vec![8192; 2], 8128),
        (vec![usize::MAX; 2], 64),
    ] {
        assert_eq!(
            compact_rows(&lengths, prefix),
            None,
            "lengths={lengths:?} prefix={prefix}"
        );
    }
}
