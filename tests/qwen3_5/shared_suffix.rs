//! Compact suffix operators must match the original full-layout windows exactly.
use super::*;

#[test]
#[ignore = "needs a GPU reservation and compact shared-prefix CUDA symbols"]
fn shared_suffix_convolution_matches_history_rows_and_changed_graph_replay() {
    let st = setup();
    let ops = api().shared_suffix().unwrap();
    for (kd, vd) in [(128usize, 128usize), (2048, 6144)] {
        let ld = 2 * kd + vd + 8;
        let weights = to_device(&random((2 * kd + vd) * 4, 311, 0.25), st);
        for t in [1usize, 3, 63, 64, 65, 129] {
            let input = to_device(&random((t + 3) * ld, 312, 0.5), st);
            let expected: Vec<_> = [kd, kd, vd]
                .map(|d| DeviceBuffer::new((t + 3) * d * 2).unwrap())
                .into();
            let actual: Vec<_> = [kd, kd, vd]
                .map(|d| DeviceBuffer::new(t * d * 2).unwrap())
                .into();
            let run = || unsafe {
                // SAFETY: the first three input rows are immutable history; suffix
                // and output buffers have the dimensions supplied to the operator.
                check(
                    (ops.conv)(
                        input.at(3 * ld * 2),
                        input.at(0),
                        ld as i32,
                        weights.at(0),
                        actual[0].at(0),
                        actual[1].at(0),
                        actual[2].at(0),
                        t as i32,
                        kd as i32,
                        vd as i32,
                        st,
                    ),
                    "shared suffix conv",
                )
            };
            run().unwrap();
            cuda::synchronize(st).unwrap();
            let graph = cuda::Graph::capture(st, run).unwrap();
            for replay in [false, true] {
                let changed = random((t + 3) * ld, if replay { 314 } else { 313 }, 0.5);
                let bytes: Vec<_> = changed.iter().flat_map(|x| x.to_le_bytes()).collect();
                unsafe {
                    // SAFETY: replace both history and suffix, then run the original
                    // convolution over their contiguous concatenation as the oracle.
                    cuda::upload(input.at(0), &bytes, st).unwrap();
                    check(
                        (api().cs1_gdn_conv)(
                            input.at(0),
                            ld as i32,
                            weights.at(0),
                            expected[0].at(0),
                            expected[1].at(0),
                            expected[2].at(0),
                            (t + 3) as i32,
                            kd as i32,
                            vd as i32,
                            st,
                        ),
                        "original history conv",
                    )
                    .unwrap();
                }
                if replay {
                    graph.launch(st).unwrap();
                } else {
                    run().unwrap();
                }
                for (i, d) in [kd, kd, vd].into_iter().enumerate() {
                    let reference = from_device(&expected[i], (t + 3) * d, st);
                    assert_eq!(
                        from_device(&actual[i], t * d, st),
                        reference[3 * d..],
                        "kd={kd}, vd={vd}, t={t}, output={i}, graph={replay}"
                    );
                }
            }
        }
    }
    unsafe {
        (api().cs1_stream_destroy)(st);
    }
}

#[test]
#[ignore = "needs a GPU reservation and compact shared-prefix CUDA symbols"]
fn shared_suffix_attention_matches_full_layout_and_changed_prefix_replay() {
    let st = setup();
    let ops = api().shared_suffix().unwrap();
    let d = 256;
    for (hq, hk) in [(8usize, 2usize), (24, 4)] {
        let ld = hk * d + 8;
        for (prefix, suffix) in [
            (64usize, 1usize),
            (64, 33),
            (64, 65),
            (512, 64),
            (2048, 129),
        ] {
            let total = prefix + suffix;
            let q = to_device(&random(total * hq * d, 321, 0.5), st);
            let k = to_device(&random(total * hk * d, 322, 0.5), st);
            let values = random(total * ld, 323, 0.5);
            let v = to_device(&values, st);
            let pv: Vec<_> = values[..prefix * ld]
                .chunks(ld)
                .flat_map(|row| row[..hk * d].iter().copied())
                .collect();
            let prefix_v = to_device(&pv, st);
            let gate = to_device(&random(total * hq * d, 324, 0.5), st);
            let expected = DeviceBuffer::new(total * hq * d * 2).unwrap();
            let actual = DeviceBuffer::new(suffix * hq * d * 2).unwrap();
            let run = || unsafe {
                // SAFETY: compact Q/K/V/gate start at the first suffix row; prefix
                // KV owns contiguous rows and output covers only suffix queries.
                check(
                    (ops.attention)(
                        q.at(prefix * hq * d * 2),
                        k.at(prefix * hk * d * 2),
                        v.at(prefix * ld * 2),
                        ld as i32,
                        gate.at(prefix * hq * d * 2),
                        actual.at(0),
                        k.at(0),
                        prefix_v.at(0),
                        suffix as i32,
                        prefix as i32,
                        hq as i32,
                        hk as i32,
                        d as i32,
                        (d as f32).powf(-0.5),
                        st,
                    ),
                    "shared suffix attention",
                )
            };
            run().unwrap();
            cuda::synchronize(st).unwrap();
            let graph = cuda::Graph::capture(st, run).unwrap();
            for replay in [false, true] {
                let changed = random(total * hk * d, if replay { 326 } else { 325 }, 0.5);
                let bytes: Vec<_> = changed.iter().flat_map(|x| x.to_le_bytes()).collect();
                unsafe {
                    // SAFETY: changing prefix and suffix K exercises live graph inputs.
                    cuda::upload(k.at(0), &bytes, st).unwrap();
                    check(
                        (api().cs1_attention_gated_prefix)(
                            q.at(0),
                            k.at(0),
                            v.at(0),
                            ld as i32,
                            gate.at(0),
                            expected.at(0),
                            total as i32,
                            hq as i32,
                            hk as i32,
                            d as i32,
                            (d as f32).powf(-0.5),
                            prefix as i32,
                            st,
                        ),
                        "original suffix attention",
                    )
                    .unwrap();
                }
                if replay {
                    graph.launch(st).unwrap();
                } else {
                    run().unwrap();
                }
                let reference = from_device(&expected, total * hq * d, st);
                assert_eq!(
                    from_device(&actual, suffix * hq * d, st),
                    reference[prefix * hq * d..],
                    "hq={hq}, prefix={prefix}, suffix={suffix}, graph={replay}"
                );
            }
            unsafe {
                // Invalid alignment must fail before dereferencing null pointers.
                assert_ne!(
                    (ops.attention)(
                        std::ptr::null(),
                        std::ptr::null(),
                        std::ptr::null(),
                        ld as i32,
                        std::ptr::null(),
                        std::ptr::null_mut(),
                        std::ptr::null(),
                        std::ptr::null(),
                        suffix as i32,
                        63,
                        hq as i32,
                        hk as i32,
                        d as i32,
                        1.0,
                        st
                    ),
                    0
                );
            }
        }
    }
    unsafe {
        (api().cs1_stream_destroy)(st);
    }
}
