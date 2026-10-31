use super::{ranges, shared_prefix};

#[test]
fn preserves_all_candidates_across_token_and_sequence_limits() {
    let prompts: Vec<Vec<u32>> = [2048, 2048, 1, 5000, 3]
        .into_iter()
        .chain(std::iter::repeat_n(4, 17))
        .map(|length| vec![0; length])
        .collect();
    let inputs: Vec<&[u32]> = prompts.iter().map(Vec::as_slice).collect();
    let batches = ranges(&inputs);
    assert_eq!(batches, vec![0..2, 2..3, 3..4, 4..20, 20..22]);
    assert_eq!(
        batches.into_iter().flatten().collect::<Vec<_>>(),
        (0..inputs.len()).collect::<Vec<_>>()
    );
}

#[test]
fn shared_prefix_is_exact_aligned_and_leaves_a_suffix() {
    let a = vec![1u32; 2100];
    let mut b = a.clone();
    b[2000] = 2;
    assert_eq!(shared_prefix(&[&a, &b]), 1984);
    b[100] = 3;
    assert_eq!(shared_prefix(&[&a, &b]), 0);
    assert_eq!(shared_prefix(&[&a, &a]), 2048);
    assert_eq!(shared_prefix(&[&a]), 0);
    assert_eq!(shared_prefix(&[&a, &a[..2048]]), 0);
    assert_eq!(shared_prefix(&vec![a.as_slice(); 17]), 0);
}

#[test]
fn packed_prefix_admits_midlength_without_changing_legacy_policy() {
    let a = vec![42u32; 801];
    let mut b = vec![42u32; 832];
    b[790] = 7;
    assert_eq!(shared_prefix(&[&a, &b]), 0);
    assert_eq!(super::packed_shared_prefix(&[&a, &b]), 768);
    assert_eq!(super::packed_shared_prefix(&[&a, &b[..511]]), 0);
    assert_eq!(super::packed_shared_prefix(&[&a, &b[..512]]), 0);
    assert_eq!(super::packed_shared_prefix(&[&a]), 0);
    assert_eq!(super::packed_shared_prefix(&vec![a.as_slice(); 17]), 0);
}

#[test]
fn projection_prefix_keeps_unaligned_rows_and_original_batch_bounds() {
    let a = vec![42; 128];
    let mut b = a.clone();
    b[43] = 7;
    assert_eq!(super::projection_prefix(&[&a, &b]), 43);
    assert_eq!(super::packed_shared_prefix(&[&a, &b]), 0);
    assert_eq!(super::projection_prefix(&[&a, &a]), 127);
    assert_eq!(super::projection_prefix(&[&a]), 0);
    assert_eq!(super::projection_prefix(&vec![a.as_slice(); 17]), 0);
    assert_eq!(super::projection_prefix(&[&a, &[]]), 0);
    let big = vec![42; 3000];
    assert_eq!(super::projection_prefix(&[&big, &big]), 0);
    b[15] = 8;
    assert_eq!(super::projection_prefix(&[&a, &b]), 0);
}
