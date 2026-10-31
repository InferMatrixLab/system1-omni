//! Independent suffix queries retain their exact per-sequence arithmetic.
use super::*;

#[test]
#[ignore = "needs CUDA, a GPU reservation, and the batched shared-attention symbol"]
fn batched_shared_attention_matches_serial_bits_and_live_graph_inputs() {
    let st = setup();
    let serial = api().shared_suffix().unwrap().attention;
    let batch = api().shared_batched_attention().unwrap();
    for (hq, hk) in [(8usize, 2usize), (16, 4), (24, 4)] {
        let d = 256;
        let ld = hk * d + 8;
        for (prefix, lengths) in [
            (64usize, vec![1usize]),
            (64, vec![1, 65]),
            (512, vec![31, 63, 64, 65, 127, 128, 129, 3]),
            (2048, vec![17, 129, 33]),
            (4096, (0..16).map(|i| 1 + i * 9).collect()),
        ] {
            let rows: usize = lengths.iter().sum();
            let q = to_device(&random(rows * hq * d, 451, 0.5), st);
            let k = to_device(&random(rows * hk * d, 452, 0.5), st);
            let v = to_device(&random(rows * ld, 453, 0.5), st);
            let gate = to_device(&random(rows * hq * d, 454, 0.5), st);
            let pk = to_device(&random(prefix * hk * d, 455, 0.5), st);
            let pv = to_device(&random(prefix * hk * d, 456, 0.5), st);
            let n = rows * hq * d;
            let sentinel = vec![bf16::from_f32(-7.0); n + 16];
            let expected = to_device(&sentinel, st);
            let actual = to_device(&sentinel, st);
            let run = || unsafe {
                // SAFETY: offsets cover disjoint suffix rows; both KV allocations
                // cover prefix rows. The host descriptor is consumed by value.
                check(
                    batch(
                        q.at(0),
                        k.at(0),
                        v.at(0),
                        ld as i32,
                        gate.at(0),
                        actual.at(16),
                        pk.at(0),
                        pv.at(0),
                        lengths.as_ptr(),
                        lengths.len(),
                        prefix as i32,
                        hq as i32,
                        hk as i32,
                        d as i32,
                        (d as f32).powf(-0.5),
                        st,
                    ),
                    "batched shared attention",
                )
            };
            run().unwrap();
            cuda::synchronize(st).unwrap();
            let graph = cuda::Graph::capture(st, run).unwrap();
            for seed in [457, 458, 459] {
                let changed = random(prefix * hk * d, seed, 0.5);
                let bytes: Vec<_> = changed.iter().flat_map(|x| x.to_le_bytes()).collect();
                unsafe {
                    // SAFETY: overwrite live prefix values before replay; keep
                    // all inputs alive until downloads retire the stream.
                    cuda::upload(pk.at(0), &bytes, st).unwrap();
                    let mut offset = 0;
                    for &length in &lengths {
                        check(
                            serial(
                                q.at(offset * hq * d * 2),
                                k.at(offset * hk * d * 2),
                                v.at(offset * ld * 2),
                                ld as i32,
                                gate.at(offset * hq * d * 2),
                                expected.at(16 + offset * hq * d * 2),
                                pk.at(0),
                                pv.at(0),
                                length as i32,
                                prefix as i32,
                                hq as i32,
                                hk as i32,
                                d as i32,
                                (d as f32).powf(-0.5),
                                st,
                            ),
                            "serial shared attention",
                        )
                        .unwrap();
                        offset += length;
                    }
                }
                if seed == 457 {
                    run().unwrap();
                } else {
                    graph.launch(st).unwrap();
                }
                let mut a = vec![0; (n + 16) * 2];
                let mut b = a.clone();
                unsafe {
                    // SAFETY: downloads cover allocated outputs and both guards.
                    cuda::download(&mut a, expected.at(0), st).unwrap();
                    cuda::download(&mut b, actual.at(0), st).unwrap();
                }
                assert_eq!(
                    a, b,
                    "hq={hq} prefix={prefix} lengths={lengths:?} seed={seed}"
                );
                let guard: Vec<_> = sentinel[..8].iter().flat_map(|x| x.to_le_bytes()).collect();
                assert_eq!(&b[..16], guard);
                assert_eq!(&b[b.len() - 16..], guard);
            }
            for invalid in [vec![], vec![0], vec![usize::MAX], vec![1; 17]] {
                let rc = unsafe {
                    // SAFETY: invalid lengths must be rejected before launching.
                    batch(
                        q.at(0),
                        k.at(0),
                        v.at(0),
                        ld as i32,
                        gate.at(0),
                        actual.at(16),
                        pk.at(0),
                        pv.at(0),
                        invalid.as_ptr(),
                        invalid.len(),
                        prefix as i32,
                        hq as i32,
                        hk as i32,
                        d as i32,
                        (d as f32).powf(-0.5),
                        st,
                    )
                };
                assert_ne!(rc, 0);
            }
        }
    }
    unsafe {
        (api().cs1_stream_destroy)(st);
    }
}

#[test]
#[ignore = "needs CUDA, a GPU reservation, and the batched causal-attention symbol"]
fn batched_causal_attention_matches_serial_bits_and_live_graph_inputs() {
    let st = setup();
    let batch = api().batched_attention().unwrap();
    for (hq, hk) in [(8usize, 2usize), (16, 4), (24, 4)] {
        let d = 256;
        let ld = hk * d + 8;
        for lengths in [
            vec![1usize],
            vec![1, 65],
            vec![31, 63, 64, 65, 127, 128, 129, 3],
            vec![512, 1024, 2048],
            vec![64; 16],
            vec![4096, 65],
        ] {
            let rows: usize = lengths.iter().sum();
            let q = to_device(&random(rows * hq * d, 481, 0.5), st);
            let k = to_device(&random(rows * hk * d, 482, 0.5), st);
            let v = to_device(&random(rows * ld, 483, 0.5), st);
            let gate = to_device(&random(rows * hq * d, 484, 0.5), st);
            let n = rows * hq * d;
            let sentinel = vec![bf16::from_f32(-7.0); n + 16];
            let expected = to_device(&sentinel, st);
            let actual = to_device(&sentinel, st);
            let run = || unsafe {
                // SAFETY: concatenated input/output rows have independent boundaries;
                // the host lengths are copied into the launch before return.
                check(
                    batch(
                        q.at(0),
                        k.at(0),
                        v.at(0),
                        ld as i32,
                        gate.at(0),
                        actual.at(16),
                        lengths.as_ptr(),
                        lengths.len(),
                        hq as i32,
                        hk as i32,
                        d as i32,
                        (d as f32).powf(-0.5),
                        st,
                    ),
                    "batched causal attention",
                )
            };
            run().unwrap();
            cuda::synchronize(st).unwrap();
            let graph = cuda::Graph::capture(st, run).unwrap();
            for seed in [485, 486, 487] {
                let changed = random(rows * hk * d, seed, 0.5);
                let bytes: Vec<_> = changed.iter().flat_map(|x| x.to_le_bytes()).collect();
                unsafe {
                    // SAFETY: replace all sequence keys; serial calls use exactly the
                    // corresponding independent ranges, including uneven final blocks.
                    cuda::upload(k.at(0), &bytes, st).unwrap();
                    let mut offset = 0;
                    for &length in &lengths {
                        check(
                            (api().cs1_attention_gated)(
                                q.at(offset * hq * d * 2),
                                k.at(offset * hk * d * 2),
                                v.at(offset * ld * 2),
                                ld as i32,
                                gate.at(offset * hq * d * 2),
                                expected.at(16 + offset * hq * d * 2),
                                length as i32,
                                hq as i32,
                                hk as i32,
                                d as i32,
                                (d as f32).powf(-0.5),
                                st,
                            ),
                            "serial causal attention",
                        )
                        .unwrap();
                        offset += length;
                    }
                }
                if seed == 485 {
                    run().unwrap();
                } else {
                    graph.launch(st).unwrap();
                }
                let mut a = vec![0; (n + 16) * 2];
                let mut b = a.clone();
                unsafe {
                    // SAFETY: downloads include both allocated output guards.
                    cuda::download(&mut a, expected.at(0), st).unwrap();
                    cuda::download(&mut b, actual.at(0), st).unwrap();
                }
                assert_eq!(a, b, "hq={hq} lengths={lengths:?} seed={seed}");
                let guard: Vec<_> = sentinel[..8].iter().flat_map(|x| x.to_le_bytes()).collect();
                assert_eq!(&b[..16], guard);
                assert_eq!(&b[b.len() - 16..], guard);
            }
            for invalid in [vec![], vec![0], vec![usize::MAX], vec![1; 17]] {
                let rc = unsafe {
                    // SAFETY: invalid dimensions must be rejected before any launch.
                    batch(
                        q.at(0),
                        k.at(0),
                        v.at(0),
                        ld as i32,
                        gate.at(0),
                        actual.at(16),
                        invalid.as_ptr(),
                        invalid.len(),
                        hq as i32,
                        hk as i32,
                        d as i32,
                        (d as f32).powf(-0.5),
                        st,
                    )
                };
                assert_ne!(rc, 0);
            }
        }
    }
    unsafe {
        (api().cs1_stream_destroy)(st);
    }
}
