use super::*;

#[test]
fn rgb8_lookup_preserves_float32_subtract_then_divide_for_every_byte() {
    for pixel in 0..=255u8 {
        // Keep both operations and their order: reciprocal multiplication can
        // differ in the low bits consumed by the BF16 vision encoder.
        let expected = (f32::from(pixel) - 127.5) / 127.5;
        assert_eq!(
            RGB8_NORMALIZED[usize::from(pixel)].to_bits(),
            expected.to_bits()
        );
    }
}

#[test]
fn spatial_merge_channel_and_duplicate_temporal_planes_match_the_tensor_layout() {
    let limits = ImageLimits {
        min_pixels: 1024,
        ..ImageLimits::cua()
    };
    for (width, height) in [(32, 32), (64, 32), (32, 64), (256, 256)] {
        let rgb: Vec<u8> = (0..width * height * 3)
            .map(|i| ((i * 17 + i / 251) % 256) as u8)
            .collect();
        let image = preprocess_rgb8_with_limits(width, height, &rgb, &limits).unwrap();
        assert_eq!(image.image_grid_thw, [1, height / 16, width / 16]);
        assert_eq!(image.pixel_values.len(), width * height * 6);
        // Independently index the output tensor as [block_y, block_x,
        // merge_y, merge_x, channel, temporal, patch_y, patch_x].
        for y in 0..height {
            for x in 0..width {
                let patch = ((y / 32 * (width / 32) + x / 32) * 2 + y % 32 / 16) * 2 + x % 32 / 16;
                for channel in 0..3 {
                    let expected = (f32::from(rgb[(y * width + x) * 3 + channel]) - 127.5) / 127.5;
                    for temporal in 0..2 {
                        let index =
                            ((patch * 3 + channel) * 2 + temporal) * 256 + y % 16 * 16 + x % 16;
                        assert_eq!(image.pixel_values[index].to_bits(), expected.to_bits());
                    }
                }
            }
        }
    }
}

#[test]
fn explicit_resize_workers_match_default_pixels_and_metadata() {
    for (width, height) in [(17, 19), (33, 49), (97, 65), (512, 500), (1001, 733)] {
        let rgb: Vec<_> = (0..width * height * 3)
            .map(|j| ((j * 17 + j / 251 + 31) % 256) as u8)
            .collect();
        let expected = preprocess_rgb8(width, height, &rgb).unwrap();
        for workers in [1, 2, 4] {
            let actual =
                preprocess_rgb8_with_workers(width, height, &rgb, &ImageLimits::cua(), workers)
                    .unwrap();
            assert_eq!(
                (
                    actual.resized_width,
                    actual.resized_height,
                    actual.image_grid_thw
                ),
                (
                    expected.resized_width,
                    expected.resized_height,
                    expected.image_grid_thw
                )
            );
            assert_eq!(actual.pixel_values.len(), expected.pixel_values.len());
            assert!(
                actual
                    .pixel_values
                    .iter()
                    .zip(&expected.pixel_values)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{width}x{height} workers={workers}"
            );
        }
    }
    for workers in [0, 5, usize::MAX] {
        assert!(
            preprocess_rgb8_with_workers(1, 1, &[0, 0, 0], &ImageLimits::cua(), workers).is_err()
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn simd_resize_rows_match_scalar_at_short_and_unaligned_edges() {
    if std::arch::is_x86_feature_detected!("sse4.1") {
        for source_width in [1, 2, 7, 17] {
            for width in [1, 2, 3, 7, 8, 9, 17, 33] {
                let source: Vec<_> = (0..source_width * 3)
                    .map(|j| ((j * 193 + 11) % 256) as u8)
                    .collect();
                let axis = AxisWeights::new(source_width, width);
                let mut expected = Vec::new();
                for kernel in &axis.kernels {
                    for channel in 0..3 {
                        expected.push(axis.apply(kernel, |i| source[i * 3 + channel]));
                    }
                }
                let mut actual = vec![0; width * 3];
                // SAFETY: runtime feature check and axis/source/output extents match.
                unsafe {
                    x86::horizontal(&source, &mut actual, &axis);
                }
                assert_eq!(actual, expected, "horizontal {source_width}->{width}");
            }
        }
    }
    if std::arch::is_x86_feature_detected!("avx2") {
        for width in [1, 2, 3, 7, 8, 9, 17, 33] {
            for height in [1, 2, 7] {
                for target in [1, 3, 9] {
                    let source: Vec<_> = (0..width * height * 3)
                        .map(|j| ((j * 193 + 11) % 256) as u8)
                        .collect();
                    let axis = AxisWeights::new(height, target);
                    for kernel in &axis.kernels {
                        let expected: Vec<_> = (0..width * 3)
                            .map(|x| axis.apply(kernel, |y| source[y * width * 3 + x]))
                            .collect();
                        let mut actual = vec![0; width * 3];
                        // SAFETY: runtime feature check and every source row has width*3 bytes.
                        unsafe {
                            x86::vertical(&source, &mut actual, kernel, axis.precision);
                        }
                        assert_eq!(actual, expected, "vertical {width}x{height}->{target}");
                    }
                }
            }
        }
    }
}
