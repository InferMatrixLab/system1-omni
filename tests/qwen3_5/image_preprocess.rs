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
