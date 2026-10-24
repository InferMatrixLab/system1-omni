//! The shared fake-CUDA boundary helper for the wired-in LoRA test suites
//! (`lora.rs`, `lora_valen.rs`): the implementation lives here, under root
//! `tests/`, and `src/models/qwen3_5/native/src/lora.rs` keeps only the
//! `#[cfg(test)]` module wiring and re-exports, per CONTRIBUTING.

/// The fake-CUDA tests route device calls through the process-wide API
/// (first `cuda::load` wins) while reading counters through their own
/// `dlopen`, so every such test must build and open the ONE shared library
/// (same path, same instance) and no two may run concurrently.
pub(crate) static FAKE_CUDA_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Compile the fake CUDA boundary once per process; every fake-lib test
/// dlopens this same path so its counters observe the instance that serves
/// the global API, whichever test loaded it.
pub(crate) fn fake_cuda_library() -> std::path::PathBuf {
    use std::sync::Mutex;
    static BUILT: Mutex<Option<std::path::PathBuf>> = Mutex::new(None);
    let mut built = BUILT
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(path) = built.as_ref() {
        return path.clone();
    }
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory =
        std::env::temp_dir().join(format!("qwen-lora-fake-{}-{unique}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let source = directory.join("fake.c");
    let library = directory.join(if cfg!(target_os = "macos") {
        "fake.dylib"
    } else {
        "fake.so"
    });
    let mut code = include_str!("lora_fake.c").to_owned();
    let implemented = [
        "cs1_abi_version",
        "cs1_get_device",
        "cs1_error_string",
        "cs1_malloc",
        "cs1_free",
        "cs1_stream_create",
        "cs1_stream_sync",
        "cs1_stream_destroy",
        "cs1_upload",
        "cs1_download",
        "cs1_copy2d",
        "cs1_vision_to_float",
        "cs1_gemm_f32",
        "cs1_gemm",
        "cs1_vision_lora_add",
    ];
    // Required but unused functions return an error, so an unexpected dispatch fails.
    let cuda_source = include_str!("../../src/models/qwen3_5/native/src/cuda.rs");
    let abi = cuda_source
        .lines()
        .find(|line| line.starts_with("const ABI_VERSION:"))
        .unwrap()
        .split('=')
        .nth(1)
        .unwrap()
        .trim()
        .trim_end_matches(';');
    let api = cuda_source
        .split("api! {")
        .nth(1)
        .unwrap()
        .split("\n}\n")
        .next()
        .unwrap();
    for line in api.lines() {
        let line = line.trim();
        if !line.starts_with("cs1_") {
            continue;
        }
        let name = line.split('(').next().unwrap();
        if !implemented.contains(&name) {
            code.push_str(&format!("\nint {name}(void) {{ return 77; }}\n"));
        }
    }
    std::fs::write(&source, code).unwrap();
    let output = std::process::Command::new("cc")
        .args(["-shared", "-fPIC", "-ffp-contract=off"])
        .arg(format!("-DLORA_FAKE_ABI={abi}"))
        .arg(&source)
        .arg("-o")
        .arg(&library)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    *built = Some(library.clone());
    library
}
