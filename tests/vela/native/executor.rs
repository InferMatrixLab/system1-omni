//! Device-gated encoder execution: runs only when a checkpoint bundle
//! and the regenerated CUDA library are provided, on the hardware the
//! bundle was built for.
//!
//!   VELA_NATIVE_CHECKPOINT=<dir with config.json + model.safetensors>
//!   VELA_NATIVE_LIBRARY=<path to liblaya_cuda.so>

use omni_vela_native::executor::Encoder;
use std::path::PathBuf;

fn gated(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).map(PathBuf::from)
}

#[test]
fn encoder_runs_and_produces_finite_hidden() {
    let (Some(checkpoint), Some(library)) = (
        gated("VELA_NATIVE_CHECKPOINT"),
        gated("VELA_NATIVE_LIBRARY"),
    ) else {
        eprintln!("skipping: set VELA_NATIVE_CHECKPOINT and VELA_NATIVE_LIBRARY");
        return;
    };
    let encoder = unsafe { Encoder::load(&checkpoint, &library) }.unwrap();
    assert_eq!(encoder.structure.hidden_size, 768);
    let ids: Vec<i64> = (1..=97).collect();
    let hidden = encoder.encode(&ids, &[0, 47, 96]).unwrap();
    assert_eq!(hidden.len(), 3);
    for row in &hidden {
        assert_eq!(row.len(), 768);
        assert!(row.iter().all(|v| v.is_finite()), "non-finite hidden");
    }
}
