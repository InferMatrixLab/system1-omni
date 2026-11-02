//! Device boundary smoke tests need a GPU/library, but no model weights.
use super::*;
#[test]
#[ignore = "requires CUDA and CUA_S1_CUDA_LIB; run serially"]
fn device_finite_scan_rejects_all_nonfinite_encodings_and_recovers() {
    let library = std::path::PathBuf::from(std::env::var_os("CUA_S1_CUDA_LIB").unwrap());
    load(&library).unwrap();
    set_device(0).unwrap();
    let stream = new_stream().unwrap();
    let buffer = DeviceBuffer::new(514).unwrap();
    let mut values = vec![half::bf16::from_f32(1.); 257];
    for bad in [0x7f80, 0xff80, 0x7fc0, 0xffc1] {
        values[256] = half::bf16::from_bits(bad);
        let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        unsafe {
            upload(buffer.at(0), &bytes, stream).unwrap();
        }
        let view = buffer.bf16_matrix(257, 1).unwrap();
        assert!(view.validate_finite(stream).is_err());
        values[256] = half::bf16::from_f32(-0.);
        let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        unsafe {
            upload(buffer.at(0), &bytes, stream).unwrap();
        }
        let view = buffer.bf16_matrix(257, 1).unwrap();
        view.validate_finite(stream).unwrap();
        assert_eq!(view.download().unwrap(), values);
    }
    synchronize(stream).unwrap();
    unsafe {
        (api().cs1_stream_destroy)(stream);
    }
}

#[test]
#[ignore = "requires CUDA and CUA_S1_CUDA_LIB; run serially"]
fn device_row_subviews_check_extent_offset_and_only_scan_selected_rows() {
    load(&std::path::PathBuf::from(
        std::env::var_os("CUA_S1_CUDA_LIB").unwrap(),
    ))
    .unwrap();
    set_device(0).unwrap();
    let stream = new_stream().unwrap();
    let buffer = DeviceBuffer::new(24).unwrap();
    let values: Vec<_> = (0..12).map(|i| half::bf16::from_f32(i as f32)).collect();
    let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    unsafe {
        upload(buffer.at(0), &bytes, stream).unwrap();
    }
    synchronize(stream).unwrap();
    let full = buffer.bf16_matrix(4, 3).unwrap();
    let middle = full.slice_rows(1..3).unwrap().unwrap();
    assert_eq!(middle.download().unwrap(), values[3..9]);
    assert_eq!(
        middle
            .slice_rows(1..2)
            .unwrap()
            .unwrap()
            .download()
            .unwrap(),
        values[6..9]
    );
    assert!(full.slice_rows(4..4).unwrap().is_none());
    let reversed = std::ops::Range { start: 3, end: 2 };
    for range in [reversed, 0..5, 5..5] {
        assert!(full.slice_rows(range).is_err());
    }
    unsafe {
        upload(buffer.at(0), &half::bf16::NAN.to_le_bytes(), stream).unwrap();
    }
    assert!(full.validate_finite(stream).is_err());
    middle.validate_finite(stream).unwrap();
    synchronize(stream).unwrap();
    unsafe {
        (api().cs1_stream_destroy)(stream);
    }
}
