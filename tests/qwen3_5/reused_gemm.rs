//! Retain complete-prompt projection arithmetic across prefix/suffix windows.
use super::*;

#[test]
#[ignore = "needs a reserved GPU and the optional retained GEMM CUDA symbols"]
fn retained_projection_windows_match_full_rows_and_graph_replay() {
    let stream = setup();
    let reuse = api().reuse_gemm().unwrap();
    // SAFETY: same workspace as the model; retired after all graph/readback work.
    let gemm = unsafe { (api().cs1_gemm_create)(32 << 20) };
    assert!(!gemm.is_null());
    let rows = 3776usize;
    let mut exercised = 0;
    for (n, k) in [(5120usize, 6144usize), (5120, 17408), (34816, 5120)] {
        let weights = to_device(&random(n * k, 71, 0.25), stream);
        let input = to_device(&random(rows * k, 72, 0.5), stream);
        let expected = DeviceBuffer::new(rows * n * 2).unwrap();
        let actual = DeviceBuffer::new(rows * n * 2).unwrap();
        for prefix in [512usize, 1984, 3072] {
            let windows = [prefix, rows - prefix];
            // SAFETY: capability check reads only these live host dimensions.
            let supported = unsafe {
                (reuse.check)(
                    gemm,
                    [rows].as_ptr(),
                    1,
                    windows.as_ptr(),
                    2,
                    n as i32,
                    k as i32,
                )
            };
            if supported == -1 {
                continue;
            }
            check(supported, "retained GEMM compatibility").unwrap();
            exercised += 1;
            let split = || unsafe {
                let mut offset = 0;
                for length in windows {
                    // SAFETY: both windows fit the same allocated row buffers.
                    check(
                        (reuse.run)(
                            gemm,
                            input.at(offset * k * 2),
                            weights.at(0),
                            actual.at(offset * n * 2),
                            length as i32,
                            n as i32,
                            k as i32,
                            rows as i32,
                            stream,
                        ),
                        "retained GEMM window",
                    )?;
                    offset += length;
                }
                Ok(())
            };
            split().unwrap();
            cuda::synchronize(stream).unwrap();
            let graph = cuda::Graph::capture(stream, split).unwrap();
            for replay in [false, true] {
                let values = random(rows * k, if replay { 74 } else { 73 }, 0.5);
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                unsafe {
                    // SAFETY: the original complete matrix has exactly rows*k BF16 values.
                    cuda::upload(input.at(0), &bytes, stream).unwrap();
                    check(
                        (api().cs1_gemm)(
                            gemm,
                            input.at(0),
                            weights.at(0),
                            expected.at(0),
                            rows as i32,
                            n as i32,
                            k as i32,
                            n as i32,
                            stream,
                        ),
                        "original rows",
                    )
                    .unwrap();
                }
                if replay {
                    if n == 5120 && k == 6144 && prefix == 512 {
                        evict_reused_plans(gemm);
                    }
                    graph.launch(stream).unwrap();
                } else {
                    split().unwrap();
                }
                let a = from_device(&expected, rows * n, stream);
                let b = from_device(&actual, rows * n, stream);
                let changes = a
                    .iter()
                    .zip(&b)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                assert_eq!(changes, 0, "n={n} k={k} prefix={prefix} graph={replay}");
            }
        }
    }
    assert!(
        exercised >= 3,
        "must exercise retained algorithms, not only fallback"
    );
    unsafe {
        // Invalid dimensions must be an error, distinct from unsupported=-1.
        let bad = (reuse.check)(
            gemm,
            [0usize].as_ptr(),
            1,
            [64usize].as_ptr(),
            1,
            5120,
            6144,
        );
        assert!(bad != 0 && bad != -1);
        (api().cs1_gemm_destroy)(gemm);
        (api().cs1_stream_destroy)(stream);
    }
}

fn evict_reused_plans(gemm: *mut std::ffi::c_void) {
    let reuse = api().reuse_gemm().unwrap();
    for rows in 1usize..=1024 {
        // SAFETY: host-only planning for identical original/target shapes. More
        // entries than the512-plan bound evict the older graph's descriptors.
        unsafe {
            check(
                (reuse.check)(gemm, [rows].as_ptr(), 1, [rows].as_ptr(), 1, 64, 64),
                "retained-plan eviction filler",
            )
            .unwrap();
        }
    }
}
