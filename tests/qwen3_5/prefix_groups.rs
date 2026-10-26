//! Shape compatibility changes execution order, never response order.
use super::compatible_prefix_groups;

#[test]
fn nonadjacent_compatible_candidates_keep_original_indices() {
    let algorithm = [1, 2, 1, 2, 1, 3];
    let groups = compatible_prefix_groups(6, |ids| {
        Ok(ids.iter().all(|&i| algorithm[i] == algorithm[ids[0]]))
    })
    .unwrap()
    .unwrap();
    assert_eq!(groups, [vec![0, 2, 4], vec![1, 3], vec![5]]);
    let mut restored = [0; 6];
    for group in groups {
        for index in group {
            restored[index] = 100 + index;
        }
    }
    assert_eq!(restored, [100, 101, 102, 103, 104, 105]);
}

#[test]
fn incompatible_windows_fall_back_and_backend_errors_are_not_hidden() {
    assert!(
        compatible_prefix_groups(4, |_| Ok(false))
            .unwrap()
            .is_none()
    );
    assert!(compatible_prefix_groups(4, |_| anyhow::bail!("backend failed")).is_err());
}

#[test]
fn fully_compatible_candidates_use_one_check() {
    let mut checks = 0;
    let groups = compatible_prefix_groups(16, |_| {
        checks += 1;
        Ok(true)
    })
    .unwrap()
    .unwrap();
    assert_eq!(checks, 1);
    assert_eq!(groups, vec![(0..16).collect::<Vec<_>>()]);
}

#[test]
fn suffix_segments_preserve_order_and_bound_only_suffix_rows() {
    // Complete candidates can be long even when their combined suffixes fit.
    let lengths = [4096, 4096, 4096, 16384, 4096, 4096];
    assert_eq!(
        super::suffix_segments(&[0, 2, 1, 3, 5, 4], &lengths, 2048),
        [vec![0, 2], vec![1], vec![3], vec![5, 4]]
    );
    assert_eq!(
        super::suffix_segments(&(0..33).collect::<Vec<_>>(), &[2049; 33], 2048)
            .iter()
            .map(Vec::len)
            .collect::<Vec<_>>(),
        [16, 16, 1]
    );
}
