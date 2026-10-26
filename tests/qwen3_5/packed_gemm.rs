//! Differential checks for optional projections with retained cuBLASLt algorithms.
use super::*;

#[test]
#[ignore = "needs a GPU and CUA_S1_CUDA_LIB with packed GEMMs"]
fn packed_projections_match_individual_shapes_and_changed_input_replay() {
    let stream = setup();
    let packed = api().packed_gemm().unwrap();
    // SAFETY: model-equivalent workspace, released after the final readback.
    let gemm = unsafe { (api().cs1_gemm_create)(32 << 20) };
    assert!(!gemm.is_null());
    for (n, k) in [(5120usize, 6144usize), (5120, 17408)] {
        let weights = to_device(&random(n * k, 91, 0.25), stream);
        for lengths in [
            vec![78, 78, 82, 80],
            vec![64; 4],
            vec![1; 3],
            vec![129, 63, 130],
            vec![1024; 2],
        ] {
            let rows: usize = lengths.iter().sum();
            let input = to_device(&random(rows * k, 92, 0.5), stream);
            let expected = DeviceBuffer::new(rows * n * 2).unwrap();
            let actual = DeviceBuffer::new(rows * n * 2).unwrap();
            let run_packed = || unsafe {
                // SAFETY: all rows and weights have the dimensions passed here;
                // the host lengths remain alive through synchronous plan lookup.
                check(
                    packed(
                        gemm,
                        input.at(0),
                        weights.at(0),
                        actual.at(0),
                        lengths.as_ptr(),
                        lengths.len(),
                        n as i32,
                        k as i32,
                        stream,
                    ),
                    "packed test",
                )
            };
            run_packed().unwrap(); // warm plan before capture
            cuda::synchronize(stream).unwrap();
            let graph = cuda::Graph::capture(stream, run_packed).unwrap();
            for replay in [false, true] {
                let values = random(rows * k, if replay { 94 } else { 93 }, 0.5);
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                unsafe {
                    // SAFETY: input owns precisely rows*k BF16 values.
                    cuda::upload(input.at(0), &bytes, stream).unwrap();
                    let mut offset = 0;
                    for &length in &lengths {
                        check(
                            (api().cs1_gemm)(
                                gemm,
                                input.at(offset * k * 2),
                                weights.at(0),
                                expected.at(offset * n * 2),
                                length as i32,
                                n as i32,
                                k as i32,
                                n as i32,
                                stream,
                            ),
                            "individual test",
                        )
                        .unwrap();
                        offset += length;
                    }
                }
                if replay {
                    if n == 5120 && k == 6144 && lengths == [78, 78, 82, 80] {
                        evict_packed_plans(gemm, stream);
                    }
                    graph.launch(stream).unwrap();
                } else {
                    run_packed().unwrap();
                }
                let a = from_device(&expected, rows * n, stream);
                let b = from_device(&actual, rows * n, stream);
                let changed = a
                    .iter()
                    .zip(&b)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                assert_eq!(changed, 0, "n={n} k={k} lengths={lengths:?} graph={replay}");
            }
        }
    }
    // Invalid row counts must fail before reading any device address.
    unsafe {
        let empty = [0usize];
        assert_ne!(
            packed(
                gemm,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
                empty.as_ptr(),
                1,
                5120,
                6144,
                stream
            ),
            0
        );
        let oversized = [i32::MAX as usize, 1];
        assert_ne!(
            packed(
                gemm,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
                oversized.as_ptr(),
                2,
                5120,
                6144,
                stream
            ),
            0
        );
        (api().cs1_gemm_destroy)(gemm);
        (api().cs1_stream_destroy)(stream);
    }
}

/// More distinct length vectors than the bounded cache, while an older graph
/// remains alive. Replaying it afterwards must use its original launch arguments.
fn evict_packed_plans(gemm: *mut std::ffi::c_void, stream: Stream) {
    let packed = api().packed_gemm().unwrap();
    let input = to_device(&random(32 * 64, 101, 0.5), stream);
    let weight = to_device(&random(64 * 64, 102, 0.25), stream);
    let output = DeviceBuffer::new(32 * 64 * 2).unwrap();
    for shape in 0usize..1024 {
        let lengths: Vec<usize> = (0..16).map(|i| 1 + ((shape >> i) & 1)).collect();
        // SAFETY: sixteen positive lengths sum to at most32; buffers match64columns.
        unsafe {
            check(
                packed(
                    gemm,
                    input.at(0),
                    weight.at(0),
                    output.at(0),
                    lengths.as_ptr(),
                    lengths.len(),
                    64,
                    64,
                    stream,
                ),
                "eviction filler",
            )
            .unwrap();
        }
    }
    cuda::synchronize(stream).unwrap();
}
