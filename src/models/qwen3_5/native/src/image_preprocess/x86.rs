//! Explicit resize path only: preserve independent integer channel accumulators.
use super::{AxisWeights, Kernel};
use std::arch::x86_64::*;

/// # Safety
/// The caller checks SSE4.1. Source spans and output RGB rows match `axis`.
#[target_feature(enable = "sse4.1")]
pub(super) unsafe fn horizontal(source: &[u8], output: &mut [u8], axis: &AxisWeights) {
    // Intrinsics operate on registers under the checked target feature.
    // RGB reads/stores use checked slices and never read a fourth byte.
    let zero = _mm_setzero_si128();
    let maximum = _mm_set1_epi32(255);
    let initial = _mm_set1_epi32(1 << (axis.precision - 1));
    let shift = _mm_cvtsi32_si128(axis.precision as i32);
    for (pixel, kernel) in output.as_chunks_mut::<3>().0.iter_mut().zip(&axis.kernels) {
        let mut sum = initial;
        for (offset, &weight) in kernel.weights.iter().enumerate() {
            let i = (kernel.start + offset) * 3;
            let rgb = u32::from_le_bytes([source[i], source[i + 1], source[i + 2], 0]);
            let values = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(rgb as i32));
            sum = _mm_add_epi32(
                sum,
                _mm_mullo_epi32(values, _mm_set1_epi32(i32::from(weight))),
            );
        }
        let shifted = _mm_min_epi32(_mm_sra_epi32(sum, shift), maximum);
        let words = _mm_packus_epi32(shifted, zero);
        let bytes = (_mm_cvtsi128_si32(_mm_packus_epi16(words, zero)) as u32).to_le_bytes();
        pixel.copy_from_slice(&bytes[..3]);
    }
}

/// # Safety
/// The caller checks AVX2. Each source row has output.len() bytes, and every
/// kernel row lies inside source. Source and output are distinct allocations.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn vertical(source: &[u8], output: &mut [u8], kernel: &Kernel, precision: u32) {
    let stride = output.len();
    let mut lane = 0;
    // SAFETY: lane+8<=stride bounds every 8-byte load/store within its checked
    // row. The axis builder bounds source rows. Each lane follows scalar tap order.
    unsafe {
        let initial = _mm256_set1_epi32(1 << (precision - 1));
        let shift = _mm256_set1_epi32(precision as i32);
        let maximum = _mm256_set1_epi32(255);
        let zero = _mm_setzero_si128();
        while lane + 8 <= stride {
            let mut sum = initial;
            for (offset, &weight) in kernel.weights.iter().enumerate() {
                let i = (kernel.start + offset) * stride + lane;
                let values = _mm256_cvtepu8_epi32(_mm_loadl_epi64(source.as_ptr().add(i).cast()));
                sum = _mm256_add_epi32(
                    sum,
                    _mm256_mullo_epi32(values, _mm256_set1_epi32(i32::from(weight))),
                );
            }
            let shifted = _mm256_min_epi32(_mm256_srav_epi32(sum, shift), maximum);
            let words = _mm_packus_epi32(
                _mm256_castsi256_si128(shifted),
                _mm256_extracti128_si256(shifted, 1),
            );
            _mm_storel_epi64(
                output.as_mut_ptr().add(lane).cast(),
                _mm_packus_epi16(words, zero),
            );
            lane += 8;
        }
    }
    for (index, pixel) in output.iter_mut().enumerate().skip(lane) {
        let mut sum = 1_i32 << (precision - 1);
        for (offset, &weight) in kernel.weights.iter().enumerate() {
            sum += i32::from(source[(kernel.start + offset) * stride + index]) * i32::from(weight);
        }
        *pixel = (sum >> precision).clamp(0, 255) as u8;
    }
}
