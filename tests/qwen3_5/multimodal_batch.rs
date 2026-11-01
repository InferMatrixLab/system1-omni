use super::*;
use half::bf16;

fn input<'a>(
    ids: &'a [u32],
    indices: &'a [usize],
    features: &'a [bf16],
    positions: [&'a [i64]; 3],
) -> MultimodalInput<'a> {
    MultimodalInput {
        token_ids: ids,
        image_token_indices: indices,
        image_embeddings: features,
        position_ids: positions,
    }
}

#[test]
fn multimodal_batch_layout_bounds_prompt_count_and_total_tokens() {
    let ids = [1, 99, 2];
    let indices = [1];
    let features = [bf16::ZERO; 4];
    let positions = [0, 1, 2];
    let one = || input(&ids, &indices, &features, [&positions; 3]);
    assert!(batch_lengths(&[], 0).is_err());
    assert_eq!(batch_lengths(&[one(), one()], 0).unwrap(), [3, 3]);
    assert!(batch_lengths(&(0..17).map(|_| one()).collect::<Vec<_>>(), 0).is_err());
    let long = vec![1; MAX_MULTIMODAL_BATCH_TOKENS + 1];
    let too_big = input(&long, &[], &[], [&[]; 3]);
    assert!(batch_lengths(&[too_big], 0).is_err());
}

#[test]
#[ignore = "requires CUDA and pinned CUA_S1_MODEL/CUA_S1_CUDA_LIB; run serially"]
fn multimodal_batch_retains_serial_hidden_bits_and_current_inputs() {
    use std::path::PathBuf;
    let dir = PathBuf::from(std::env::var_os("CUA_S1_MODEL").expect("CUA_S1_MODEL"));
    let library = PathBuf::from(std::env::var_os("CUA_S1_CUDA_LIB").expect("CUA_S1_CUDA_LIB"));
    for graph in [false, true] {
        let mut model = Model::load_with_graph(&dir, &library, graph).unwrap();
        model.enable_multimodal_batching().unwrap();
        let image_token = model.cfg.image_token_id.unwrap();
        for (lengths, seed) in [
            (vec![65, 129, 257], 1),
            (vec![65, 129, 257], 2),
            (vec![512; 8], 3),
            (vec![65, 129, 257], 1),
        ] {
            let ids: Vec<Vec<u32>> = lengths
                .iter()
                .enumerate()
                .map(|(i, &n)| {
                    let mut ids: Vec<_> = (0..n)
                        .map(|j| (10 + (j * 17 + i * 31 + seed) % 1000) as u32)
                        .collect();
                    ids[8..12].fill(image_token);
                    ids
                })
                .collect();
            let features: Vec<Vec<bf16>> = lengths
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    (0..4 * model.cfg.hidden)
                        .map(|j| {
                            bf16::from_f32(((j * 13 + seed * 17 + i * 7) % 97) as f32 / 97. - 0.5)
                        })
                        .collect()
                })
                .collect();
            let positions: Vec<[Vec<i64>; 3]> = lengths
                .iter()
                .map(|&n| {
                    std::array::from_fn(|axis| (0..n).map(|p| (p + axis * 3) as i64).collect())
                })
                .collect();
            let indices = [8, 9, 10, 11];
            let inputs: Vec<_> = ids
                .iter()
                .zip(&features)
                .zip(&positions)
                .map(|((ids, features), p)| input(ids, &indices, features, [&p[0], &p[1], &p[2]]))
                .collect();
            let expected: Vec<_> = inputs
                .iter()
                .map(|p| model.forward_multimodal(p).unwrap())
                .collect();
            for _ in 0..2 {
                let actual = model.forward_multimodal_batch(&inputs).unwrap();
                assert_eq!(actual.len(), expected.len());
                for (a, b) in actual.iter().zip(&expected) {
                    let differences = a
                        .iter()
                        .zip(b)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    assert_eq!(
                        differences, 0,
                        "lengths={lengths:?} seed={seed} graph={graph}"
                    );
                }
            }
        }
        if graph {
            let stats = model.graph_stats();
            assert!(stats.enabled && stats.captures > 0 && stats.replays > 0);
            assert_eq!(stats.fallbacks, 0);
        }
    }
}

#[test]
fn shared_prefix_bounds_each_prompt_without_counting_duplicate_raw_rows_as_scratch() {
    let ids = vec![1; 1030];
    let inputs: Vec<_> = (0..8).map(|_| input(&ids, &[], &[], [&[]; 3])).collect();
    assert!(batch_lengths(&inputs, 0).is_err());
    assert_eq!(batch_lengths(&inputs, 576).unwrap(), vec![1030; 8]);
    let oversized = vec![1; MAX_MULTIMODAL_BATCH_TOKENS + 1];
    assert!(batch_lengths(&[input(&oversized, &[], &[], [&[]; 3])], 576).is_err());
}

#[test]
#[ignore = "requires CUDA and pinned CUA_S1_MODEL/CUA_S1_CUDA_LIB; run serially"]
fn repeated_device_features_still_validate_a_late_distinct_allocation_and_recover() {
    use std::path::PathBuf;
    let dir = PathBuf::from(std::env::var_os("CUA_S1_MODEL").unwrap());
    let lib = PathBuf::from(std::env::var_os("CUA_S1_CUDA_LIB").unwrap());
    let mut model = Model::load_with_graph(&dir, &lib, false).unwrap();
    model.enable_multimodal_batching().unwrap();
    let hidden = model.cfg.hidden;
    let mut ids = vec![1; 65];
    ids[8] = model.cfg.image_token_id.unwrap();
    let indices = [8];
    let positions: Vec<_> = (0..65).map(i64::from).collect();
    let features = vec![bf16::from_f32(0.25); hidden];
    let bytes: Vec<_> = features.iter().flat_map(|x| x.to_le_bytes()).collect();
    let first = cuda::DeviceBuffer::new(bytes.len()).unwrap();
    let second = cuda::DeviceBuffer::new(bytes.len()).unwrap();
    let mut bad_bytes = bytes.clone();
    bad_bytes[..2].copy_from_slice(&bf16::NAN.to_le_bytes());
    // SAFETY: exact allocation sizes, serialized uploads, no consumer is active.
    unsafe {
        cuda::upload(first.at(0), &bytes, model.stream).unwrap();
        cuda::upload(second.at(0), &bad_bytes, model.stream).unwrap();
    }
    model.synchronize().unwrap();
    let make = |image_embeddings| DeviceMultimodalInput {
        token_ids: &ids,
        image_token_indices: &indices,
        image_embeddings,
        position_ids: [&positions; 3],
    };
    {
        let a = first.bf16_matrix(1, hidden).unwrap();
        let b = second.bf16_matrix(1, hidden).unwrap();
        assert!(a.same_storage(&a));
        assert!(!a.same_storage(&b));
        assert!(!a.same_storage(&first.bf16_matrix(2, hidden / 2).unwrap()));
        let error = model
            .forward_multimodal_device_batch(&[make(a), make(a), make(b)])
            .unwrap_err();
        assert!(
            error.to_string().contains("non-finite image embedding"),
            "{error:#}"
        );
    }
    // SAFETY: the failed consumer drained; this producer allocation is reusable.
    unsafe { cuda::upload(second.at(0), &bytes, model.stream).unwrap() };
    model.synchronize().unwrap();
    let a = first.bf16_matrix(1, hidden).unwrap();
    let b = second.bf16_matrix(1, hidden).unwrap();
    let expected = model
        .forward_multimodal(&input(&ids, &indices, &features, [&positions; 3]))
        .unwrap();
    let actual = model
        .forward_multimodal_device_batch(&[make(a), make(a), make(b)])
        .unwrap();
    assert_eq!(actual.len(), 3);
    for row in actual {
        assert_eq!(
            row.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
    // SAFETY: the previous consumer drained. Revalidate this same allocation
    // after its producer changes it; no finite-check result survives a call.
    unsafe { cuda::upload(second.at(0), &bad_bytes, model.stream).unwrap() };
    model.synchronize().unwrap();
    let error = model
        .forward_multimodal_device_batch(&[
            make(first.bf16_matrix(1, hidden).unwrap()),
            make(second.bf16_matrix(1, hidden).unwrap()),
        ])
        .unwrap_err();
    assert!(error.to_string().contains("non-finite image embedding"));
}
