//! Private-item tests for the native crate's head loader, loaded by the
//! `#[path]` wiring in `src/models/vela/native/src/heads.rs`.

use super::{linear, scalar_tensor, tensor};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};

fn f32_tensor(name: &str, shape: Vec<usize>, values: &[f32]) -> Vec<u8> {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let view = TensorView::new(Dtype::F32, shape, &bytes).expect("tensor view");
    safetensors::tensor::serialize([(name, view)], None).expect("serialize")
}

#[test]
fn tensor_rejects_a_shape_that_only_matches_when_flattened() {
    let raw = f32_tensor("head.weight", vec![4], &[0.0; 4]);
    let file = SafeTensors::deserialize(&raw).expect("safetensors");
    let error = match linear(&file, "head", false, 2, 2) {
        Ok(_) => panic!("wrong rank must fail"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("head.weight has shape"));
}

#[test]
fn tensor_rejects_non_finite_values() {
    let raw = f32_tensor("head.weight", vec![1], &[f32::NAN]);
    let file = SafeTensors::deserialize(&raw).expect("safetensors");
    let error = tensor(&file, "head.weight", &[1]).expect_err("non-finite tensor must fail");
    assert!(error.to_string().contains("non-finite"));
}

#[test]
fn scalar_tensor_accepts_the_checkpoint_rank_zero_shape() {
    let raw = f32_tensor("set_bias", vec![], &[0.25]);
    let file = SafeTensors::deserialize(&raw).expect("safetensors");
    let values = scalar_tensor(&file, "set_bias").expect("rank-0 scalar");
    assert_eq!(values, [0.25]);
}
