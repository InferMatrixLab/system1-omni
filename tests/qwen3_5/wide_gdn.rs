//! Wider preparation and output tiles retain BF16 bits across dispatch boundaries and graph replay.
use super::*;

#[test]
#[ignore = "needs a GPU reservation and the optional wide GDN symbol"]
fn wide_prefill_matches_legacy_and_changed_graph_replay() {
    let st = setup();
    let wide = api().wide_gdn().unwrap();
    for (h, hk, lengths) in [
        (
            48usize,
            16usize,
            vec![
                1usize, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256, 257, 383, 384, 385,
                487, 511, 512, 513, 936, 1213, 2047, 2048, 2049, 4095, 4096, 4097, 16383, 16384,
                16385,
            ],
        ),
        (4, 2, vec![1, 128, 129, 1213]),
    ] {
        for t in lengths {
            let sizes = [t * hk * 128, t * hk * 128, t * h * 128, t * h];
            let input = sizes.map(|n| to_device(&random(n, 715, 0.1), st));
            let gates: Vec<_> = random(t * h, 716, 0.5)
                .iter()
                .map(|x| (x.to_f32() - 0.5) * 0.01)
                .collect();
            let g = f32_to_device(&gates, st);
            let n = t * h * 128;
            let output = [
                DeviceBuffer::new(n * 2).unwrap(),
                DeviceBuffer::new(n * 2).unwrap(),
            ];
            // SAFETY: the workspace size is a pure function of positive dimensions.
            let floats = unsafe { (api().cs1_gdn_workspace_floats)(t as i32, h as i32) };
            let ws = [
                DeviceBuffer::new(floats * 4).unwrap(),
                DeviceBuffer::new(floats * 4).unwrap(),
            ];
            let run = |f: cuda::GdnPrefill, arm: usize| unsafe {
                // SAFETY: separate outputs/workspaces own all rows for these dimensions.
                check(
                    f(
                        input[0].at(0),
                        input[1].at(0),
                        input[2].at(0),
                        g.at(0).cast(),
                        input[3].at(0),
                        output[arm].at(0),
                        ws[arm].at(0).cast(),
                        t as i32,
                        h as i32,
                        hk as i32,
                        128f32.powf(-0.5),
                        st,
                    ),
                    "wide GDN prefill",
                )
            };
            run(wide, 1).unwrap();
            cuda::synchronize(st).unwrap();
            let graph = cuda::Graph::capture(st, || run(wide, 1)).unwrap();
            for replay in [false, true] {
                for (i, size) in sizes.into_iter().enumerate() {
                    let seed = 721 + i as u64 + u64::from(replay) * 17;
                    let values: Vec<_> = random(size, seed, if i == 2 { 1.0 } else { 0.5 })
                        .into_iter()
                        .map(|x| {
                            if i == 3 {
                                bf16::from_f32(x.to_f32() + 0.5)
                            } else {
                                x
                            }
                        })
                        .collect();
                    let bytes: Vec<_> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
                    // SAFETY: every upload replaces the whole live allocation.
                    unsafe {
                        cuda::upload(input[i].at(0), &bytes, st).unwrap();
                    }
                }
                let changed_gates: Vec<_> = random(t * h, if replay { 741 } else { 740 }, 0.5)
                    .iter()
                    .flat_map(|x| ((x.to_f32() - 0.5) * 0.01).to_le_bytes())
                    .collect();
                // SAFETY: the FP32 gates allocation owns these bytes.
                unsafe {
                    cuda::upload(g.at(0), &changed_gates, st).unwrap();
                }
                run(api().cs1_gdn_prefill, 0).unwrap();
                if replay {
                    graph.launch(st).unwrap();
                } else {
                    run(wide, 1).unwrap();
                }
                let expected = from_device(&output[0], n, st);
                let actual = from_device(&output[1], n, st);
                assert!(
                    actual.iter().all(|x| x.is_finite()),
                    "non-finite t={t},h={h}"
                );
                assert_eq!(
                    actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    "t={t},h={h},hk={hk},graph={replay}"
                );
            }
        }
    }
    // SAFETY: zero tokens and invalid dimensions must return before using pointers.
    unsafe {
        for (t, h, hk) in [(0, 48, 16), (-1, 48, 16), (1, 48, 0), (1, 48, 17)] {
            let result = wide(
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                t,
                h,
                hk,
                1.0,
                st,
            );
            assert_eq!(result == 0, t == 0);
        }
        (api().cs1_stream_destroy)(st);
    }
}
