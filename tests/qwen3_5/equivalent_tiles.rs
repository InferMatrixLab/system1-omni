//! Opt-in output-tile compatibility, exact arithmetic and unchanged fallback.
use super::*;

#[test]
#[ignore = "needs SM90, cuBLASLt 13.1.1, the optional tile capability and a GPU reservation"]
fn equivalent_tiles_preserve_full_rows_packing_and_graph_replay() {
    let st = setup();
    let enable = api().equivalent_tiles().expect("new optional tile symbol");
    let reuse = api().reuse_gemm().unwrap();
    let packed = api().packed_gemm().unwrap();
    // SAFETY: handles own independent plans/workspaces and are retired below.
    let original = unsafe { (api().cs1_gemm_create)(32 << 20) };
    let optimized = unsafe { (api().cs1_gemm_create)(32 << 20) };
    assert!(!original.is_null() && !optimized.is_null());
    // SAFETY: no plans or graphs have been created on the ordinary handle.
    assert_eq!(
        unsafe { enable(optimized) },
        1,
        "exercise supported profile"
    );
    let compatible = |handle, rows: &[usize], n, k| unsafe {
        // SAFETY: capability check only reads the live row arrays.
        (reuse.check)(
            handle,
            rows.as_ptr(),
            rows.len(),
            [512usize].as_ptr(),
            1,
            n,
            k,
        )
    };
    assert_eq!(compatible(original, &[2048, 3782], 4096, 4096), -1);
    assert_eq!(compatible(optimized, &[2048, 3782], 4096, 4096), 0);
    for (n, k) in [
        (4096, 4096),
        (4096, 12288),
        (12352, 4096),
        (10240, 4096),
        (24576, 4096),
    ] {
        assert_eq!(compatible(optimized, &[3776, 3779, 3780, 3782], n, k), 0);
    }
    // Different split-K reductions must not become compatible.
    assert_eq!(compatible(optimized, &[80, 90], 4096, 12288), -1);
    // A capability change after plans exist would invalidate cached grouping.
    assert!(unsafe { enable(optimized) } < 0);

    let rows = 4096usize;
    let mut checked = 0;
    for (n, k) in [
        (4096usize, 4096usize),
        (4096, 12288),
        (12352, 4096),
        (10240, 4096),
        (24576, 4096),
    ] {
        let weights = to_device(&random(n * k, 71, 0.25), st);
        let input = to_device(&random(rows * k, 72, 0.5), st);
        let expected = DeviceBuffer::new(rows * n * 2).unwrap();
        let actual = DeviceBuffer::new(rows * n * 2).unwrap();
        let prefix = || unsafe {
            // SAFETY: input/output own at least 512 rows; weights are [n,k].
            check(
                (reuse.run)(
                    optimized,
                    input.at(0),
                    weights.at(0),
                    actual.at(0),
                    512,
                    n as i32,
                    k as i32,
                    2048,
                    st,
                ),
                "equivalent prefix",
            )
        };
        prefix().unwrap();
        cuda::synchronize(st).unwrap();
        let graph = cuda::Graph::capture(st, prefix).unwrap();
        for seed in [73, 74, 75] {
            let bytes: Vec<u8> = random(rows * k, seed, if seed == 75 { 32.0 } else { 0.5 })
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect();
            // SAFETY: input owns exactly rows*k BF16 elements.
            unsafe { cuda::upload(input.at(0), &bytes, st).unwrap() };
            for m in [
                512usize, 513, 576, 577, 582, 768, 832, 833, 838, 840, 1024, 1056, 1088, 1089,
                1094, 1673, 1735, 2048, 2112, 2113, 2240, 2241, 2299, 2300, 2367, 2496, 3233, 3343,
                3419, 3483, 3591, 3776, 3779, 3780, 3782, 4096,
            ] {
                let supported = compatible(optimized, &[2048, m], n as i32, k as i32);
                if supported == -1 {
                    continue;
                }
                check(supported, "tile compatibility").unwrap();
                // SAFETY: complete original rows fit both allocations.
                unsafe {
                    check(
                        (api().cs1_gemm)(
                            original,
                            input.at(0),
                            weights.at(0),
                            expected.at(0),
                            m as i32,
                            n as i32,
                            k as i32,
                            n as i32,
                            st,
                        ),
                        "complete original projection",
                    )
                    .unwrap()
                };
                if seed == 74 {
                    graph.launch(st).unwrap();
                } else {
                    prefix().unwrap();
                }
                assert_eq!(
                    from_device(&expected, 512 * n, st),
                    from_device(&actual, 512 * n, st),
                    "M={m} N={n} K={k} seed={seed}"
                );
                checked += 1;
            }
            let lengths = [512usize, 1024, 2048];
            let mut offset = 0;
            for m in lengths {
                // SAFETY: each original submatrix is within the packed row buffers.
                unsafe {
                    check(
                        (api().cs1_gemm)(
                            original,
                            input.at(offset * k * 2),
                            weights.at(0),
                            expected.at(offset * n * 2),
                            m as i32,
                            n as i32,
                            k as i32,
                            n as i32,
                            st,
                        ),
                        "independent packed reference",
                    )
                    .unwrap()
                };
                offset += m;
            }
            // SAFETY: all packed rows and dimensions fit the same allocations.
            unsafe {
                check(
                    packed(
                        optimized,
                        input.at(0),
                        weights.at(0),
                        actual.at(0),
                        lengths.as_ptr(),
                        lengths.len(),
                        n as i32,
                        k as i32,
                        st,
                    ),
                    "equivalent packed projection",
                )
                .unwrap()
            };
            assert_eq!(
                from_device(&expected, offset * n, st),
                from_device(&actual, offset * n, st)
            );
        }
    }
    assert!(
        checked >= 120,
        "exercise actual equivalence across projection shapes"
    );
    cuda::synchronize(st).unwrap();
    // SAFETY: downloads and synchronization retired all work and captured graphs.
    unsafe {
        (api().cs1_gemm_destroy)(original);
        (api().cs1_gemm_destroy)(optimized);
    }
}

#[test]
#[ignore = "needs SM90, cuBLASLt13.1.1, expanded profiles and a GPU reservation"]
fn short_reuse_profiles_preserve_full_rows_and_ordinary_packing() {
    let st = setup();
    let enable = api().equivalent_tiles().unwrap();
    let reuse = api().reuse_gemm().unwrap();
    let packed = api().packed_gemm().unwrap();
    // SAFETY: independent handles own all plans and are retired after synchronization.
    let original = unsafe { (api().cs1_gemm_create)(32 << 20) };
    let optimized = unsafe { (api().cs1_gemm_create)(32 << 20) };
    assert!(!original.is_null() && !optimized.is_null());
    assert_eq!(unsafe { enable(optimized) }, 1);
    let mut checked = 0;
    for (n, k) in [
        (4096usize, 4096usize),
        (4096, 12288),
        (12352, 4096),
        (10240, 4096),
        (24576, 4096),
    ] {
        let weights = to_device(&random(n * k, 88, 0.25), st);
        let input = DeviceBuffer::new(4096 * k * 2).unwrap();
        let expected = DeviceBuffer::new(4096 * n * 2).unwrap();
        let actual = DeviceBuffer::new(4096 * n * 2).unwrap();
        let lengths = [
            63usize, 64, 79, 80, 83, 127, 128, 160, 192, 254, 256, 332, 380, 506,
        ];
        let run_packed = || unsafe {
            // SAFETY: summed ragged lengths fit the 4096-row input/output allocations.
            check(
                packed(
                    optimized,
                    input.at(0),
                    weights.at(0),
                    actual.at(0),
                    lengths.as_ptr(),
                    lengths.len(),
                    n as i32,
                    k as i32,
                    st,
                ),
                "short packed projection",
            )
        };
        let mut graph: Option<cuda::Graph> = None;
        for (seed, scale) in [(89, 0.5), (90, 32.0)] {
            let bytes: Vec<u8> = random(4096 * k, seed, scale)
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect();
            // SAFETY: input owns exactly these 4096*k BF16 elements.
            unsafe {
                cuda::upload(input.at(0), &bytes, st).unwrap();
            }
            for m in [
                64usize, 79, 80, 83, 127, 128, 160, 192, 254, 256, 332, 380, 506, 512,
            ] {
                // SAFETY: shape-only check reads live original and target arrays.
                let supported = unsafe {
                    (reuse.check)(
                        optimized,
                        [2048usize, m].as_ptr(),
                        2,
                        [64usize].as_ptr(),
                        1,
                        n as i32,
                        k as i32,
                    )
                };
                if supported == -1 {
                    continue;
                } // Retain split-K/unsupported profiles.
                check(supported, "short profile compatibility").unwrap();
                unsafe {
                    // SAFETY: original complete rows and selected 64 rows fit allocations.
                    check(
                        (api().cs1_gemm)(
                            original,
                            input.at(0),
                            weights.at(0),
                            expected.at(0),
                            m as i32,
                            n as i32,
                            k as i32,
                            n as i32,
                            st,
                        ),
                        "short complete original",
                    )
                    .unwrap();
                    check(
                        (reuse.run)(
                            optimized,
                            input.at(0),
                            weights.at(0),
                            actual.at(0),
                            64,
                            n as i32,
                            k as i32,
                            2048,
                            st,
                        ),
                        "short equivalent rows",
                    )
                    .unwrap();
                }
                assert_eq!(
                    from_device(&expected, 64 * n, st),
                    from_device(&actual, 64 * n, st),
                    "M={m} N={n} K={k} seed={seed}"
                );
                checked += 1;
            }
            let mut offset = 0;
            for m in lengths {
                unsafe {
                    // SAFETY: each independent reference occupies its original slice.
                    check(
                        (api().cs1_gemm)(
                            original,
                            input.at(offset * k * 2),
                            weights.at(0),
                            expected.at(offset * n * 2),
                            m as i32,
                            n as i32,
                            k as i32,
                            n as i32,
                            st,
                        ),
                        "ragged original",
                    )
                    .unwrap();
                }
                offset += m;
            }
            if let Some(graph) = &graph {
                graph.launch(st).unwrap();
            } else {
                run_packed().unwrap();
                cuda::synchronize(st).unwrap();
                graph = Some(cuda::Graph::capture(st, run_packed).unwrap());
            }
            assert_eq!(
                from_device(&expected, offset * n, st),
                from_device(&actual, offset * n, st),
                "ragged N={n} K={k} seed={seed}"
            );
        }
    }
    assert!(
        checked >= 90,
        "exercise short profiles rather than only falling back"
    );
    cuda::synchronize(st).unwrap();
    // SAFETY: all graphs and pending device work have retired.
    unsafe {
        (api().cs1_gemm_destroy)(original);
        (api().cs1_gemm_destroy)(optimized);
    }
}
