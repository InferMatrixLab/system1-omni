//! Exact sequence isolation, chunk padding, and changed-input graph replay.
use super::*;

#[test]
#[ignore = "needs a GPU reservation and the optional batched GDN CUDA symbols"]
fn batched_gdn_matches_serial_with_partial_chunks_and_graph_replay() {
    check_batched(false);
}

#[test]
#[ignore = "needs a GPU reservation and shared-state batched GDN CUDA symbol"]
fn batched_gdn_shared_state_matches_serial_and_changed_state_replay() {
    check_batched(true);
}

fn check_batched(shared: bool) {
    let stream = setup();
    let batched = api().batched_gdn().unwrap();
    let dimensions = if shared {
        [(4usize, 2usize), (32, 16)]
    } else {
        [(4, 2), (48, 16)]
    };
    for (heads, key_heads) in dimensions {
        for lengths in [
            vec![1usize, 63, 64, 65, 129],
            vec![1; 16],
            vec![78, 82, 78, 79, 78, 82],
            vec![2048, 2048],
            vec![4096, 1],
        ] {
            let rows: usize = lengths.iter().sum();
            let keys = rows * key_heads * 128;
            let values = rows * heads * 128;
            let q = to_device(&random(keys, 81, 0.5), stream);
            let k = to_device(&random(keys, 82, 0.5), stream);
            let v = to_device(&random(values, 83, 0.5), stream);
            let gates: Vec<f32> = random(rows * heads, 84, 0.1)
                .iter()
                .map(|v| -0.2 + v.to_f32())
                .collect();
            let g = f32_to_device(&gates, stream);
            let beta = to_device(
                &random(rows * heads, 85, 0.25)
                    .iter()
                    .map(|v| bf16::from_f32(0.5 + v.to_f32()))
                    .collect::<Vec<_>>(),
                stream,
            );
            let expected = DeviceBuffer::new(values * 2).unwrap();
            let actual = DeviceBuffer::new(values * 2).unwrap();
            // SAFETY: dimensions are positive and bounded by this test's allocations.
            let capacity = unsafe {
                (batched.workspace_floats)(rows as i32, heads as i32, lengths.len() as i32)
            };
            let scratch = DeviceBuffer::new(capacity * 4).unwrap();
            let serial_size = unsafe {
                (api().cs1_gdn_workspace_floats)(
                    *lengths.iter().max().unwrap() as i32,
                    heads as i32,
                )
            };
            let serial_scratch = DeviceBuffer::new(serial_size * 4).unwrap();
            let state = f32_to_device(&vec![0.0; heads * 128 * 128], stream);
            let scale = (128f32).powf(-0.5);
            let batch = || unsafe {
                // SAFETY: packed buffers cover every length; scratch uses the API's bound.
                let rc = if shared {
                    (api().shared_batched_gdn().unwrap())(
                        q.at(0),
                        k.at(0),
                        v.at(0),
                        g.at(0).cast(),
                        beta.at(0),
                        actual.at(0),
                        scratch.at(0).cast(),
                        capacity * 4,
                        lengths.as_ptr(),
                        lengths.len(),
                        heads as i32,
                        key_heads as i32,
                        scale,
                        state.at(0),
                        stream,
                    )
                } else {
                    (batched.run)(
                        q.at(0),
                        k.at(0),
                        v.at(0),
                        g.at(0).cast(),
                        beta.at(0),
                        actual.at(0),
                        scratch.at(0).cast(),
                        capacity * 4,
                        lengths.as_ptr(),
                        lengths.len(),
                        heads as i32,
                        key_heads as i32,
                        scale,
                        stream,
                    )
                };
                check(rc, "batch GDN")
            };
            batch().unwrap();
            cuda::synchronize(stream).unwrap();
            let graph = cuda::Graph::capture(stream, batch).unwrap();
            for replay in [false, true] {
                let changed = random(values, if replay { 87 } else { 86 }, 0.5);
                let bytes: Vec<u8> = changed.iter().flat_map(|v| v.to_le_bytes()).collect();
                let state_bytes: Vec<u8> =
                    random(heads * 128 * 128, if replay { 89 } else { 88 }, 0.01)
                        .iter()
                        .flat_map(|v| v.to_f32().to_le_bytes())
                        .collect();
                unsafe {
                    // SAFETY: replace all value rows, then compare independent sequence runs.
                    cuda::upload(v.at(0), &bytes, stream).unwrap();
                    cuda::upload(state.at(0), &state_bytes, stream).unwrap();
                    let mut offset = 0;
                    for &length in &lengths {
                        if shared {
                            check(
                                (api().cs1_gdn_prefill_x)(
                                    q.at(offset * key_heads * 128 * 2),
                                    k.at(offset * key_heads * 128 * 2),
                                    v.at(offset * heads * 128 * 2),
                                    g.at(offset * heads * 4).cast(),
                                    beta.at(offset * heads * 2),
                                    expected.at(offset * heads * 128 * 2),
                                    serial_scratch.at(0).cast(),
                                    length as i32,
                                    heads as i32,
                                    key_heads as i32,
                                    scale,
                                    state.at(0),
                                    std::ptr::null_mut(),
                                    stream,
                                ),
                                "serial GDN",
                            )
                            .unwrap();
                        } else {
                            check(
                                (api().cs1_gdn_prefill)(
                                    q.at(offset * key_heads * 128 * 2),
                                    k.at(offset * key_heads * 128 * 2),
                                    v.at(offset * heads * 128 * 2),
                                    g.at(offset * heads * 4).cast(),
                                    beta.at(offset * heads * 2),
                                    expected.at(offset * heads * 128 * 2),
                                    serial_scratch.at(0).cast(),
                                    length as i32,
                                    heads as i32,
                                    key_heads as i32,
                                    scale,
                                    stream,
                                ),
                                "serial GDN",
                            )
                            .unwrap();
                        }
                        offset += length;
                    }
                }
                if replay {
                    graph.launch(stream).unwrap();
                } else {
                    batch().unwrap();
                }
                let mut after = vec![0u8; state_bytes.len()];
                // SAFETY: the full allocation is downloaded after all consumers.
                unsafe {
                    cuda::download(&mut after, state.at(0), stream).unwrap();
                }
                assert_eq!(after, state_bytes, "shared input state was modified");
                let a = from_device(&expected, values, stream);
                let b = from_device(&actual, values, stream);
                let changes = a
                    .iter()
                    .zip(&b)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                assert_eq!(
                    changes, 0,
                    "heads={heads} lengths={lengths:?} graph={replay} shared={shared}"
                );
            }
            unsafe {
                // An undersized workspace is rejected before any kernel launch.
                assert_ne!(
                    (batched.run)(
                        q.at(0),
                        k.at(0),
                        v.at(0),
                        g.at(0).cast(),
                        beta.at(0),
                        actual.at(0),
                        scratch.at(0).cast(),
                        0,
                        lengths.as_ptr(),
                        lengths.len(),
                        heads as i32,
                        key_heads as i32,
                        scale,
                        stream
                    ),
                    0
                );
            }
        }
    }
    unsafe {
        (api().cs1_stream_destroy)(stream);
    }
}
