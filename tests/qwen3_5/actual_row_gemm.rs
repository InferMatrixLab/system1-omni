//! Opted-in target-row heuristics preserve retained projection arithmetic.
use super::*;

#[test]
#[ignore = "needs SM90, cuBLASLt 13.1.1 and a reserved GPU"]
fn actual_row_heuristics_match_retained_algorithms_and_graph_replay() {
    let st = setup();
    let reuse = api().reuse_gemm().unwrap();
    // SAFETY: independent handles and workspaces remain live through every readback.
    let original = unsafe { (api().cs1_gemm_create)(32 << 20) };
    let optimized = unsafe { (api().cs1_gemm_create)(32 << 20) };
    assert!(!original.is_null() && !optimized.is_null());
    assert_eq!(unsafe { api().equivalent_tiles().unwrap()(optimized) }, 1);
    let mut checked = 0;
    for (n, k) in [
        (4096usize, 4096usize),
        (4096, 12288),
        (12352, 4096),
        (10240, 4096),
        (24576, 4096),
    ] {
        let weights = to_device(&random(n * k, 521, 0.25), st);
        for (full, rows) in [
            (31usize, 64usize),
            (128, 65),
            (128, 193),
            (256, 127),
            (512, 129),
            (1024, 255),
            (2048, 513),
            (4096, 2049),
            (2048, 4095),
        ] {
            let capability = |handle| unsafe {
                // SAFETY: shape checks consume only these live dimension arrays.
                (reuse.check)(handle, &full, 1, &rows, 1, n as i32, k as i32)
            };
            let old = capability(original);
            assert_eq!(old, capability(optimized), "eligibility changed");
            if old == -1 {
                continue;
            }
            check(old, "retained shape").unwrap();
            let input = DeviceBuffer::new(rows * k * 2).unwrap();
            let sentinel = vec![bf16::from_f32(-7.0); rows * n + 16];
            let expected = to_device(&sentinel, st);
            let actual = to_device(&sentinel, st);
            let run = |handle, out: &DeviceBuffer| unsafe {
                // SAFETY: complete target rows and weights fit; guards surround output.
                check(
                    (reuse.run)(
                        handle,
                        input.at(0),
                        weights.at(0),
                        out.at(16),
                        rows as i32,
                        n as i32,
                        k as i32,
                        full as i32,
                        st,
                    ),
                    "actual-row retained projection",
                )
            };
            let mut graph = None;
            for (seed, scale) in [(522, 0.5), (523, 32.0)] {
                let bytes: Vec<_> = random(rows * k, seed, scale)
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect();
                // SAFETY: replace all live inputs before eager execution or replay.
                unsafe { cuda::upload(input.at(0), &bytes, st).unwrap() };
                run(original, &expected).unwrap();
                if let Some(g) = &graph {
                    cuda::Graph::launch(g, st).unwrap();
                } else {
                    run(optimized, &actual).unwrap();
                    cuda::synchronize(st).unwrap();
                    graph = Some(cuda::Graph::capture(st, || run(optimized, &actual)).unwrap());
                }
                let mut a = vec![0u8; sentinel.len() * 2];
                let mut b = vec![0u8; sentinel.len() * 2];
                unsafe {
                    // SAFETY: read complete output buffers, including both guard regions.
                    cuda::download(&mut a, expected.at(0), st).unwrap();
                    cuda::download(&mut b, actual.at(0), st).unwrap();
                }
                assert_eq!(
                    a, b,
                    "N={n} K={k} original={full} actual={rows} seed={seed}"
                );
                let guard: Vec<_> = sentinel[..8].iter().flat_map(|x| x.to_le_bytes()).collect();
                assert_eq!(&b[..16], &guard);
                assert_eq!(&b[b.len() - 16..], &guard);
            }
            checked += 1;
        }
    }
    assert!(checked >= 35, "exercise actual shapes, not only fallbacks");
    unsafe {
        // SAFETY: graph owners dropped in their iterations; readbacks retired all work.
        (api().cs1_gemm_destroy)(optimized);
        (api().cs1_gemm_destroy)(original);
        (api().cs1_stream_destroy)(st);
    }
}
