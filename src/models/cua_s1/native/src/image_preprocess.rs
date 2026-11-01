//! Cua-S1 retains its original RGB8 processor limits through the shared implementation.
pub use omni_qwen3_5_native::image_preprocess::{ProcessedImage, preprocess_rgb8};

/// Cua-only explicit accelerated CPU preparation. Other model adapters retain
/// their existing shared entry points and limits.
pub fn preprocess_rgb8_workers(
    width: usize,
    height: usize,
    rgb: &[u8],
    workers: usize,
) -> anyhow::Result<ProcessedImage> {
    omni_qwen3_5_native::image_preprocess::preprocess_rgb8_with_workers(
        width,
        height,
        rgb,
        &omni_qwen3_5_native::image_preprocess::ImageLimits::cua(),
        workers,
    )
}
