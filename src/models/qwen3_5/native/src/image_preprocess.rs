//! Shared CPU RGB8 preprocessing for Qwen3.5/3.8 vision encoders.
//!
//! Input is already decoded, interleaved RGB8. No image codecs, GPU, model
//! weights, or HTTP handling are involved. The output is row-major
//! `[patches, 1536]`, ready for a separate vision encoder.
//!
//! The resize algorithm is a Rust adaptation of PyTorch's CPU uint8 bicubic
//! antialias implementation (which credits Pillow). Smart resize and packing
//! follow Transformers' Qwen2VLImageProcessor. See `../THIRD_PARTY_NOTICES.md`.

use std::borrow::Cow;

#[cfg(target_arch = "x86_64")]
mod x86;

use anyhow::{Result, ensure};

const PATCH_SIZE: usize = 16;
const MERGE_SIZE: usize = 2;
const FACTOR: usize = PATCH_SIZE * MERGE_SIZE;
const PATCH_VALUES: usize = 3 * 2 * PATCH_SIZE * PATCH_SIZE;
// Preserve the reference's float32 subtraction and division for every RGB8 value.
// Packing only reorders bytes, so each normalized value can be reused exactly.
const RGB8_NORMALIZED: [f32; 256] = {
    let mut values = [0.0; 256];
    let mut pixel = 0;
    while pixel < values.len() {
        values[pixel] = (pixel as f32 - 127.5) / 127.5;
        pixel += 1;
    }
    values
};
/// Source validation and upstream smart-resize limits. Prompt budgeting belongs to callers.
///
/// Shared by the cua-s1 and jemm consumers, so the limit surface stays
/// single-sourced through [`preprocess_rgb8`].
#[derive(Clone, Copy, Debug)]
pub struct ImageLimits {
    pub max_source_side: usize,
    pub max_source_pixels: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub max_patches: usize,
}
impl ImageLimits {
    pub const fn cua() -> Self {
        Self {
            max_source_side: 2048,
            max_source_pixels: 1_048_576,
            min_pixels: 65_536,
            max_pixels: 16_777_216,
            max_patches: 4608,
        }
    }
}

/// Normalized image patches and the spatial metadata used by the vision model.
#[derive(Debug)]
pub struct ProcessedImage {
    /// Contiguous row-major `[patches, 1536]` float32 values.
    pub pixel_values: Vec<f32>,
    /// `[1, resized_height / 16, resized_width / 16]`.
    pub image_grid_thw: [usize; 3],
    pub resized_width: usize,
    pub resized_height: usize,
}

impl ProcessedImage {
    /// Number of image tokens after the model's 2-by-2 spatial merge.
    pub fn image_tokens(&self) -> usize {
        self.pixel_values.len() / PATCH_VALUES / (MERGE_SIZE * MERGE_SIZE)
    }
}

/// Preprocess a decoded RGB8 image with CUA source and patch limits.
///
/// Rejects empty dimensions, sides over 2048, area over 1,048,576 pixels,
/// aspect ratios over 200, and buffers whose length is not `width * height * 3`.
/// Geometry and buffer arithmetic are checked before allocating image buffers.
pub fn preprocess_rgb8(width: usize, height: usize, rgb: &[u8]) -> Result<ProcessedImage> {
    preprocess_rgb8_with_limits(width, height, rgb, &ImageLimits::cua())
}

/// Uses the same RGB8 bicubic resize and block-major patch packing with explicit limits.
pub fn preprocess_rgb8_with_limits(
    width: usize,
    height: usize,
    rgb: &[u8],
    limits: &ImageLimits,
) -> Result<ProcessedImage> {
    preprocess_impl(width, height, rgb, limits, 0)
}

/// Explicit RGB convolution/row scheduling opt-in. Existing entry points retain
/// their channel-wise serial resize. Uses at most four active CPU workers and
/// preserves every per-channel integer multiply/add, shift and saturation.
pub fn preprocess_rgb8_with_workers(
    width: usize,
    height: usize,
    rgb: &[u8],
    limits: &ImageLimits,
    workers: usize,
) -> Result<ProcessedImage> {
    ensure!((1..=4).contains(&workers), "resize workers must be 1..=4");
    preprocess_impl(width, height, rgb, limits, workers)
}

fn preprocess_impl(
    width: usize,
    height: usize,
    rgb: &[u8],
    limits: &ImageLimits,
    workers: usize,
) -> Result<ProcessedImage> {
    ensure!(
        limits.min_pixels >= FACTOR * FACTOR
            && limits.max_pixels >= limits.min_pixels
            && limits.max_pixels <= 16_777_216
            && limits.max_patches > 0
            && limits.max_patches <= 65536
            && limits.max_source_side <= 32768
            && limits.max_source_pixels <= 16_777_216,
        "invalid image processing limits"
    );
    ensure!(width > 0 && height > 0, "image dimensions must be nonzero");
    ensure!(
        width <= limits.max_source_side && height <= limits.max_source_side,
        "image sides must not exceed {}",
        limits.max_source_side
    );
    let area = width
        .checked_mul(height)
        .ok_or_else(|| anyhow::anyhow!("image area overflow"))?;
    ensure!(
        area <= limits.max_source_pixels,
        "image area must not exceed {} pixels",
        limits.max_source_pixels
    );
    ensure!(
        width.max(height) <= width.min(height) * 200,
        "image aspect ratio must not exceed 200"
    );
    let input_len = area
        .checked_mul(3)
        .ok_or_else(|| anyhow::anyhow!("RGB buffer length overflow"))?;
    ensure!(
        rgb.len() == input_len,
        "RGB buffer length must be {input_len}, got {}",
        rgb.len()
    );

    let (resized_width, resized_height) = smart_resize(width, height, limits);
    let resized_area = resized_width
        .checked_mul(resized_height)
        .ok_or_else(|| anyhow::anyhow!("resized area overflow"))?;
    ensure!(
        resized_area / (PATCH_SIZE * PATCH_SIZE) <= limits.max_patches,
        "resized image exceeds patch budget"
    );
    let resized_len = resized_area
        .checked_mul(3)
        .ok_or_else(|| anyhow::anyhow!("resized buffer length overflow"))?;
    let horizontal_len = resized_width
        .checked_mul(height)
        .and_then(|area| area.checked_mul(3))
        .ok_or_else(|| anyhow::anyhow!("horizontal buffer length overflow"))?;
    let output_len = resized_area
        .checked_mul(6)
        .ok_or_else(|| anyhow::anyhow!("patch buffer length overflow"))?;
    output_len
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| anyhow::anyhow!("patch buffer byte length overflow"))?;

    let mut resized = Cow::Borrowed(rgb);
    if resized_width != width {
        let axis = AxisWeights::new(width, resized_width);
        let mut horizontal = vec![0; horizontal_len];
        if workers == 0 {
            for y in 0..height {
                for (x, kernel) in axis.kernels.iter().enumerate() {
                    for channel in 0..3 {
                        horizontal[(y * resized_width + x) * 3 + channel] = axis
                            .apply(kernel, |source_x| rgb[(y * width + source_x) * 3 + channel]);
                    }
                }
            }
        } else {
            resize_rows(&mut horizontal, resized_width, workers, |y, row| {
                #[cfg(target_arch = "x86_64")]
                if std::arch::is_x86_feature_detected!("sse4.1") {
                    // SAFETY: checked feature, complete source row and axis/output extents.
                    unsafe {
                        x86::horizontal(&rgb[y * width * 3..(y + 1) * width * 3], row, &axis);
                    }
                    return;
                }
                for (pixel, kernel) in row.as_chunks_mut::<3>().0.iter_mut().zip(&axis.kernels) {
                    pixel.copy_from_slice(&axis.apply_rgb(kernel, |source_x| {
                        let i = (y * width + source_x) * 3;
                        [rgb[i], rgb[i + 1], rgb[i + 2]]
                    }));
                }
            });
        }
        resized = Cow::Owned(horizontal);
    }
    if resized_height != height {
        let axis = AxisWeights::new(height, resized_height);
        let mut vertical = vec![0; resized_len];
        if workers == 0 {
            for (y, kernel) in axis.kernels.iter().enumerate() {
                for x in 0..resized_width {
                    for channel in 0..3 {
                        vertical[(y * resized_width + x) * 3 + channel] = axis
                            .apply(kernel, |source_y| {
                                resized[(source_y * resized_width + x) * 3 + channel]
                            });
                    }
                }
            }
        } else {
            resize_rows(&mut vertical, resized_width, workers, |y, row| {
                let kernel = &axis.kernels[y];
                #[cfg(target_arch = "x86_64")]
                if std::arch::is_x86_feature_detected!("avx2") {
                    // SAFETY: checked feature, each source row matches row.len(),
                    // and the axis builder bounds every source row in this kernel.
                    unsafe {
                        x86::vertical(&resized, row, kernel, axis.precision);
                    }
                    return;
                }
                for (x, pixel) in row.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                    pixel.copy_from_slice(&axis.apply_rgb(kernel, |source_y| {
                        let i = (source_y * resized_width + x) * 3;
                        [resized[i], resized[i + 1], resized[i + 2]]
                    }));
                }
            });
        }
        resized = Cow::Owned(vertical);
    }

    let mut pixel_values = Vec::with_capacity(output_len);
    for block_y in 0..resized_height / FACTOR {
        for block_x in 0..resized_width / FACTOR {
            for merge_y in 0..MERGE_SIZE {
                for merge_x in 0..MERGE_SIZE {
                    for channel in 0..3 {
                        let mut plane = [0.0; PATCH_SIZE * PATCH_SIZE];
                        for patch_y in 0..PATCH_SIZE {
                            let y = block_y * FACTOR + merge_y * PATCH_SIZE + patch_y;
                            let x = block_x * FACTOR + merge_x * PATCH_SIZE;
                            let start = (y * resized_width + x) * 3 + channel;
                            for patch_x in 0..PATCH_SIZE {
                                let pixel = resized[start + patch_x * 3];
                                plane[patch_y * PATCH_SIZE + patch_x] =
                                    RGB8_NORMALIZED[usize::from(pixel)];
                            }
                        }
                        // Still images repeat the same patch in both temporal slots.
                        pixel_values.extend_from_slice(&plane);
                        pixel_values.extend_from_slice(&plane);
                    }
                }
            }
        }
    }
    Ok(ProcessedImage {
        pixel_values,
        image_grid_thw: [1, resized_height / PATCH_SIZE, resized_width / PATCH_SIZE],
        resized_width,
        resized_height,
    })
}

fn smart_resize(width: usize, height: usize, limits: &ImageLimits) -> (usize, usize) {
    // Python round uses ties-to-even; Rust's ordinary round does not.
    let mut w = (width as f64 / FACTOR as f64).round_ties_even() as usize * FACTOR;
    let mut h = (height as f64 / FACTOR as f64).round_ties_even() as usize * FACTOR;
    if w * h > limits.max_pixels {
        let beta = ((width * height) as f64 / limits.max_pixels as f64).sqrt();
        w = ((width as f64 / beta / FACTOR as f64).floor() as usize * FACTOR).max(FACTOR);
        h = ((height as f64 / beta / FACTOR as f64).floor() as usize * FACTOR).max(FACTOR);
    } else if w * h < limits.min_pixels {
        let beta = (limits.min_pixels as f64 / (width * height) as f64).sqrt();
        w = (width as f64 * beta / FACTOR as f64).ceil() as usize * FACTOR;
        h = (height as f64 * beta / FACTOR as f64).ceil() as usize * FACTOR;
    }
    (w, h)
}

struct Kernel {
    start: usize,
    weights: Vec<i16>,
}

struct AxisWeights {
    kernels: Vec<Kernel>,
    precision: u32,
}

impl AxisWeights {
    fn new(input: usize, output: usize) -> Self {
        let scale = input as f64 / output as f64;
        let support = 2.0 * scale.max(1.0);
        let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
        let max_size = support.ceil() as usize * 2 + 1;
        let mut maximum = 0.0_f64;
        let mut floating = Vec::with_capacity(output);
        for index in 0..output {
            let center = scale * (index as f64 + 0.5);
            // C++ conversion truncates toward zero before clamping the bounds.
            let start = ((center - support + 0.5) as isize).max(0) as usize;
            let end = ((center + support + 0.5) as usize).min(input);
            let count = end.saturating_sub(start).min(max_size);
            let mut weights: Vec<f64> = (0..count)
                .map(|j| cubic((j as f64 + start as f64 - center + 0.5) * invscale))
                .collect();
            let total: f64 = weights.iter().sum();
            if total != 0.0 {
                for weight in &mut weights {
                    *weight /= total;
                    maximum = maximum.max(*weight);
                }
            }
            floating.push((start, weights));
        }
        // One precision for the whole axis, as in PyTorch's int16 path.
        let mut precision = 0;
        while precision < 22 {
            if (0.5 + maximum * f64::from(1 << (precision + 1))) as i32 >= (1 << 15) {
                break;
            }
            precision += 1;
        }
        let multiplier = f64::from(1 << precision);
        let kernels = floating
            .into_iter()
            .map(|(start, weights)| Kernel {
                start,
                weights: weights
                    .into_iter()
                    .map(|weight| {
                        let value = weight * multiplier;
                        (value + if value < 0.0 { -0.5 } else { 0.5 }) as i16
                    })
                    .collect(),
            })
            .collect();
        Self { kernels, precision }
    }

    fn apply_rgb(&self, kernel: &Kernel, pixel: impl Fn(usize) -> [u8; 3]) -> [u8; 3] {
        let mut accumulators = [1_i32 << (self.precision - 1); 3];
        for (offset, &weight) in kernel.weights.iter().enumerate() {
            let rgb = pixel(kernel.start + offset);
            let weight = i32::from(weight);
            for (sum, value) in accumulators.iter_mut().zip(rgb) {
                *sum += i32::from(value) * weight;
            }
        }
        accumulators.map(|sum| (sum >> self.precision).clamp(0, 255) as u8)
    }

    fn apply(&self, kernel: &Kernel, pixel: impl Fn(usize) -> u8) -> u8 {
        let mut accumulator = 1_i32 << (self.precision - 1);
        for (offset, &weight) in kernel.weights.iter().enumerate() {
            accumulator += i32::from(pixel(kernel.start + offset)) * i32::from(weight);
        }
        (accumulator >> self.precision).clamp(0, 255) as u8
    }
}

/// Disjoint output rows: coefficient and source reads are immutable. The caller
/// processes the first chunk while at most workers-1 scoped threads process the
/// rest; both passes join before their intermediate buffer can be reused.
fn resize_rows(
    output: &mut [u8],
    width: usize,
    workers: usize,
    operation: impl Fn(usize, &mut [u8]) + Sync,
) {
    let row_bytes = width * 3;
    let height = output.len() / row_bytes;
    let workers = workers.min(height);
    if workers <= 1 || output.len() < 256 * 1024 {
        for (y, row) in output.chunks_exact_mut(row_bytes).enumerate() {
            operation(y, row);
        }
        return;
    }
    let chunk_rows = height.div_ceil(workers);
    std::thread::scope(|scope| {
        let mut chunks = output.chunks_mut(chunk_rows * row_bytes);
        let first = chunks.next().unwrap();
        let op = &operation;
        for (index, chunk) in chunks.enumerate() {
            scope.spawn(move || {
                for (y, row) in chunk.chunks_exact_mut(row_bytes).enumerate() {
                    op((index + 1) * chunk_rows + y, row);
                }
            });
        }
        for (y, row) in first.chunks_exact_mut(row_bytes).enumerate() {
            operation(y, row);
        }
    });
}

fn cubic(x: f64) -> f64 {
    let x = x.abs();
    const A: f64 = -0.5;
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A
    } else {
        0.0
    }
}

#[cfg(test)]
#[path = "../../../../../tests/qwen3_5/image_preprocess.rs"]
mod tests;
