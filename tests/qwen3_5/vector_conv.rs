//! The opt-in vector path and its layout fallbacks retain scalar BF16 results.
use super::*;

#[test]
#[ignore = "needs a GPU reservation and the optional vector convolution symbol"]
fn vector_convolution_matches_scalar_layouts_and_changed_graph_replay() {
    let st = setup();
    let vector = api().vector_conv().unwrap();
    for (kd, vd, extra, tokens) in [
        (
            128usize,
            128usize,
            8usize,
            vec![1usize, 3, 63, 64, 65, 16384],
        ),
        (2048, 6144, 96, vec![64, 129, 1213, 4096]),
        (130, 126, 1, vec![64, 65]),
    ] {
        let ld = 2 * kd + vd + extra;
        let weights = to_device(&random((2 * kd + vd) * 4, 451, 0.25), st);
        for t in tokens {
            let input = to_device(&random(t * ld, 452, 0.5), st);
            let expected = [kd, kd, vd].map(|d| DeviceBuffer::new(t * d * 2).unwrap());
            let actual = [kd, kd, vd].map(|d| DeviceBuffer::new(t * d * 2).unwrap());
            let run = || unsafe {
                // SAFETY: each allocation owns all supplied rows and channels.
                check(
                    vector(
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
                    "vector convolution",
                )
            };
            run().unwrap();
            cuda::synchronize(st).unwrap();
            let graph = cuda::Graph::capture(st, run).unwrap();
            for replay in [false, true] {
                let changed = random(t * ld, if replay { 454 } else { 453 }, 0.5);
                let bytes: Vec<_> = changed.iter().flat_map(|x| x.to_le_bytes()).collect();
                unsafe {
                    // SAFETY: replace the entire live input before either dispatch.
                    cuda::upload(input.at(0), &bytes, st).unwrap();
                    check(
                        (api().cs1_gdn_conv)(
                            input.at(0),
                            ld as i32,
                            weights.at(0),
                            expected[0].at(0),
                            expected[1].at(0),
                            expected[2].at(0),
                            t as i32,
                            kd as i32,
                            vd as i32,
                            st,
                        ),
                        "scalar convolution",
                    )
                    .unwrap();
                }
                if replay {
                    graph.launch(st).unwrap();
                } else {
                    run().unwrap();
                }
                for (i, d) in [kd, kd, vd].into_iter().enumerate() {
                    assert_eq!(
                        from_device(&actual[i], t * d, st),
                        from_device(&expected[i], t * d, st),
                        "kd={kd}, vd={vd}, t={t}, extra={extra}, graph={replay}, output={i}"
                    );
                }
            }
        }
    }
    unsafe {
        // Invalid layout must be rejected before dereferencing the pointers.
        assert_ne!(
            vector(
                std::ptr::null(),
                1,
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                64,
                128,
                128,
                st
            ),
            0
        );
        (api().cs1_stream_destroy)(st);
    }
}
