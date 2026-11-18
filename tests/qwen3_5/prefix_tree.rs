//! Whole hidden-vector equality across branching, growth and changed live inputs.
use super::*;
use std::path::PathBuf;

#[test]
#[ignore = "needs pinned9B export, shared-state CUDA and GPU reservation"]
fn prefix_tree_preserves_original_groups_and_every_hidden_component() {
    let dir = PathBuf::from(std::env::var_os("OPEN_JEV_MODEL").unwrap());
    let lib = PathBuf::from(std::env::var_os("OPEN_JEV_CUDA_LIB").unwrap());
    let mut model = Model::load(&dir, &lib).unwrap();
    model.enable_packed_projections().unwrap();
    model.enable_batched_gdn().unwrap();
    model.enable_equivalent_projection_tiles().unwrap();
    let mut executed = 0;
    for (prefix, tails, group, seed) in [
        (64, vec![64; 3], true, 40),
        (64, vec![64, 65, 63], true, 41),
        (448, vec![64; 8], true, 42),
        (512, vec![64; 3], true, 43),
        (1984, vec![64; 4], false, 44),
        (4032, vec![64; 3], false, 45),
        (4160, vec![64; 3], false, 46),
        (64, vec![1, 3, 65, 127], true, 47),
        (512, vec![64; 16], false, 48),
        (4032, vec![64; 16], false, 49),
    ] {
        let common: Vec<u32> = (0..prefix)
            .map(|i| 256 + ((i * 37 + seed) % 30000) as u32)
            .collect();
        let ids: Vec<Vec<u32>> = tails
            .iter()
            .enumerate()
            .map(|(j, &length)| {
                common
                    .iter()
                    .copied()
                    .chain((0..length).map(|i| 256 + ((i * 31 + seed + j * 701) % 30000) as u32))
                    .collect()
            })
            .collect();
        let inputs: Vec<&[u32]> = ids.iter().map(Vec::as_slice).collect();
        let original: Vec<_> = if group {
            vec![ids.iter().map(Vec::len).sum(); ids.len()]
        } else {
            ids.iter().map(Vec::len).collect()
        };
        let expected = if group {
            model.forward_batch(&inputs).unwrap()
        } else {
            inputs
                .iter()
                .map(|ids| model.forward_batch(&[*ids]).unwrap().remove(0))
                .collect()
        };
        let Some(output) = model
            .try_forward_prefix_tree(&inputs, prefix, &original)
            .unwrap()
        else {
            // Mixed original sizes can select incompatible algorithms. Declining
            // must leave the accepted full-forward result unchanged.
            assert!(
                tails.windows(2).any(|pair| pair[0] != pair[1]) || tails.len() == 16,
                "unexpected equal-size fallback prefix={prefix}"
            );
            let repeated = if group {
                model.forward_batch(&inputs).unwrap()
            } else {
                inputs
                    .iter()
                    .map(|ids| model.forward_batch(&[*ids]).unwrap().remove(0))
                    .collect()
            };
            assert_eq!(repeated, expected);
            eprintln!("safe fallback prefix={prefix} tails={tails:?}");
            continue;
        };
        executed += 1;
        eprintln!("tree executed prefix={prefix} tails={tails:?}");
        assert_eq!(output.saved_tokens, (inputs.len() - 1) * prefix);
        for (branch, (want, got)) in expected.iter().zip(output.hidden).enumerate() {
            let changed = want.iter().zip(&got).filter(|(a, b)| a != b).count();
            let max = want
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert_eq!(
                changed, 0,
                "prefix={prefix}, group={group}, branch={branch}, max={max}"
            );
        }
    }
    assert!(executed >= 6, "insufficient executed tree layouts");
}
