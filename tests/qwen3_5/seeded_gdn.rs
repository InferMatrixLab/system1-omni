//! Seed each independent suffix from the same readonly FP32 prefix state.
use super::*;

#[test]
#[ignore = "needs a GPU and the optional seeded GDN batch symbol"]
fn seeded_gdn_batch_matches_independent_outputs_and_fp32_states_after_replay() {
    let st = setup();
    let seeded = api().seeded_gdn_batch().expect("seeded GDN batch symbol");
    let batch = api().batched_gdn().unwrap();
    for (h, hk, lengths) in [
        (32usize, 16usize, vec![65usize]),
        (32, 16, vec![65, 129, 257]),
        (
            32,
            16,
            vec![
                1, 63, 64, 65, 127, 128, 129, 257, 1, 63, 64, 65, 127, 128, 129, 257,
            ],
        ),
        (48, 16, vec![63, 65]),
    ] {
        let total: usize = lengths.iter().sum();
        let sizes = [
            total * hk * 128,
            total * hk * 128,
            total * h * 128,
            total * h,
        ];
        let inputs = sizes.map(|n| to_device(&random(n, 1801, 0.25), st));
        let gate_values: Vec<_> = random(total * h, 1802, 0.5)
            .iter()
            .map(|x| (x.to_f32() - 0.5) * 0.01)
            .collect();
        let gates = f32_to_device(&gate_values, st);
        let state_values: Vec<_> = random(h * 128 * 128, 1803, 0.5)
            .iter()
            .map(|x| x.to_f32() * 0.01731 + 0.00017)
            .collect();
        assert!(
            state_values
                .iter()
                .any(|&x| bf16::from_f32(x).to_f32().to_bits() != x.to_bits())
        );
        let initial = f32_to_device(&state_values, st);
        let output_elements = total * h * 128;
        let outputs = [
            DeviceBuffer::new(output_elements * 2).unwrap(),
            DeviceBuffer::new(output_elements * 2).unwrap(),
        ];
        let state_bytes = lengths.len() * h * 128 * 128 * 4;
        let states = [
            DeviceBuffer::new(state_bytes).unwrap(),
            DeviceBuffer::new(state_bytes).unwrap(),
        ];
        // SAFETY: positive bounded dimensions; byte counts mirror the kernel layouts.
        let serial_floats = unsafe {
            (api().cs1_gdn_workspace_floats)(*lengths.iter().max().unwrap() as i32, h as i32)
        };
        let batch_floats =
            unsafe { (batch.workspace_floats)(total as i32, h as i32, lengths.len() as i32) };
        let serial_ws = DeviceBuffer::new(serial_floats * 4).unwrap();
        let batch_ws = DeviceBuffer::new(batch_floats * 4).unwrap();
        let run_batch = || unsafe {
            // SAFETY: all sequences have disjoint workspaces, outputs and final states;
            // the shared FP32 initial allocation is immutable throughout this launch.
            check(
                seeded(
                    inputs[0].at(0),
                    inputs[1].at(0),
                    inputs[2].at(0),
                    gates.at(0).cast(),
                    inputs[3].at(0),
                    outputs[1].at(0),
                    batch_ws.at(0).cast(),
                    batch_floats * 4,
                    lengths.as_ptr(),
                    lengths.len(),
                    h as i32,
                    hk as i32,
                    128f32.powf(-0.5),
                    initial.at(0),
                    states[1].at(0),
                    st,
                ),
                "seeded GDN batch",
            )
        };
        run_batch().unwrap();
        cuda::synchronize(st).unwrap();
        let graph = cuda::Graph::capture(st, run_batch).unwrap();
        for visit in 0..2 {
            for (index, size) in sizes.into_iter().enumerate() {
                let values: Vec<_> = random(size, 1810 + index as u64 + visit * 13, 0.5)
                    .into_iter()
                    .map(|x| {
                        if index == 3 {
                            bf16::from_f32(x.to_f32() + 0.5)
                        } else {
                            x
                        }
                    })
                    .collect();
                let bytes: Vec<_> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
                // SAFETY: replace complete live input allocations before consumers.
                unsafe { cuda::upload(inputs[index].at(0), &bytes, st).unwrap() };
            }
            let values: Vec<_> = random(total * h, 1830 + visit, 0.5)
                .iter()
                .flat_map(|x| ((x.to_f32() - 0.5) * 0.01).to_le_bytes())
                .collect();
            let fresh_state: Vec<_> = random(h * 128 * 128, 1840 + visit, 0.5)
                .iter()
                .flat_map(|x| (x.to_f32() * 0.01731 + 0.00017).to_le_bytes())
                .collect();
            unsafe {
                // SAFETY: exact FP32 allocations; consumers follow on this same stream.
                cuda::upload(gates.at(0), &values, st).unwrap();
                cuda::upload(initial.at(0), &fresh_state, st).unwrap();
            }
            let mut offset = 0;
            for (index, &length) in lengths.iter().enumerate() {
                unsafe {
                    // SAFETY: sequence-local row offsets and separate final FP32 states.
                    check(
                        (api().cs1_gdn_prefill_x)(
                            inputs[0].at(offset * hk * 128 * 2),
                            inputs[1].at(offset * hk * 128 * 2),
                            inputs[2].at(offset * h * 128 * 2),
                            gates.at(offset * h * 4).cast(),
                            inputs[3].at(offset * h * 2),
                            outputs[0].at(offset * h * 128 * 2),
                            serial_ws.at(0).cast(),
                            length as i32,
                            h as i32,
                            hk as i32,
                            128f32.powf(-0.5),
                            initial.at(0),
                            states[0].at(index * h * 128 * 128 * 4),
                            st,
                        ),
                        "serial seeded GDN",
                    )
                    .unwrap();
                }
                offset += length;
            }
            if visit == 0 {
                run_batch().unwrap()
            } else {
                graph.launch(st).unwrap()
            }
            let expected: Vec<_> = from_device(&outputs[0], output_elements, st)
                .iter()
                .map(|x| x.to_bits())
                .collect();
            let actual: Vec<_> = from_device(&outputs[1], output_elements, st)
                .iter()
                .map(|x| x.to_bits())
                .collect();
            assert_eq!(
                actual, expected,
                "BF16 h={h} lengths={lengths:?} visit={visit}"
            );
            let download = |buffer: &DeviceBuffer, bytes: usize| {
                let mut out = vec![0u8; bytes];
                // SAFETY: buffer owns the requested bytes; download drains the stream.
                unsafe { cuda::download(&mut out, buffer.at(0), st).unwrap() };
                out
            };
            assert_eq!(
                download(&states[1], state_bytes),
                download(&states[0], state_bytes),
                "FP32 state h={h} lengths={lengths:?} visit={visit}"
            );
            assert_eq!(
                download(&initial, fresh_state.len()),
                fresh_state,
                "initial state was mutated"
            );
        }
    }
    // SAFETY: every consumer and download completed; the stream belongs to this test.
    unsafe { (api().cs1_stream_destroy)(st) };
}
