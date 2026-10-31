//! A state-only capture must retain every byte consumed by later suffixes.
use super::*;
use std::path::PathBuf;

#[test]
#[ignore = "needs pinned9B export, CUDA retained GEMMs and a GPU reservation"]
fn state_only_capture_matches_complete_prefix_state_bytes() {
    let dir = PathBuf::from(std::env::var_os("OPEN_JEV_MODEL").unwrap());
    let lib = PathBuf::from(std::env::var_os("OPEN_JEV_CUDA_LIB").unwrap());
    let mut model = Model::load(&dir, &lib).unwrap();
    model.enable_packed_projections().unwrap();
    model.enable_batched_gdn().unwrap();
    assert_eq!(model.cfg.hidden, 4096);
    assert!(matches!(model.layers.last().unwrap().mixer, Mixer::Full(_)));
    let reuse = cuda::api().reuse_gemm().unwrap();
    for (prefix, seed) in [(512, 42), (2048, 43), (4096, 44), (512, 45)] {
        let ids: Vec<u32> = (0..prefix)
            .map(|n| 256 + (n * 37 + seed) as u32 % 30000)
            .collect();
        let mut complete = model.alloc_prefix(prefix).unwrap();
        let mut state_only = model.alloc_prefix(prefix).unwrap();
        model.prepare_scratch(prefix).unwrap();
        let s = model.scratch.as_ref().unwrap();
        for (state, readout) in [
            (&mut complete, WindowReadout::Hidden),
            (&mut state_only, WindowReadout::StateOnly),
        ] {
            model.embed_tokens(s, &ids).unwrap();
            model
                .run_window(
                    s,
                    0,
                    prefix,
                    false,
                    Some(state),
                    None,
                    model.gemm,
                    Some((reuse, prefix)),
                    readout,
                )
                .unwrap();
            model.synchronize().unwrap();
        }
        let compare = |a: &DeviceBuffer, b: &DeviceBuffer, bytes: usize, label: &str| {
            let mut want = vec![0u8; bytes];
            let mut got = vec![0u8; bytes];
            // SAFETY: sizes match the prefix-state allocations for the pinned model.
            unsafe {
                cuda::download(&mut want, a.at(0), model.stream).unwrap();
                cuda::download(&mut got, b.at(0), model.stream).unwrap();
            }
            assert!(want == got, "prefix={prefix} seed={seed} buffer={label}");
        };
        for ((ka, va), (kb, vb)) in complete.attn_kv.iter().zip(&state_only.attn_kv) {
            let bytes = prefix * model.cfg.kv_heads * model.cfg.head_dim * BF16;
            compare(ka, kb, bytes, "K");
            compare(va, vb, bytes, "V");
        }
        for (a, b) in complete.gdn_state.iter().zip(&state_only.gdn_state) {
            compare(
                a,
                b,
                model.cfg.lin_v_heads * model.cfg.lin_k_dim * model.cfg.lin_v_dim * F32,
                "GDN state",
            );
        }
        for (a, b) in complete.conv_tail.iter().zip(&state_only.conv_tail) {
            compare(
                a,
                b,
                3 * Widths::of(&model.cfg).gdn_in * BF16,
                "convolution tail",
            );
        }
    }
}
