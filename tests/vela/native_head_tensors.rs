//! Private head-loader checks, registered from `heads.rs`.

use super::{Linear, tensor};
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
    let error =
        match Linear::from_weight(tensor(&file, "head.weight").expect("tensor"), "head.weight") {
            Ok(_) => panic!("wrong rank must fail"),
            Err(error) => error,
        };
    assert!(error.to_string().contains("head.weight must be 2-D"));
}

#[test]
fn tensor_rejects_non_finite_values() {
    let raw = f32_tensor("head.weight", vec![1], &[f32::NAN]);
    let file = SafeTensors::deserialize(&raw).expect("safetensors");
    let error = match tensor(&file, "head.weight") {
        Ok(_) => panic!("non-finite tensor must fail"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("non-finite"));
}

#[test]
fn scalar_tensor_accepts_the_checkpoint_rank_zero_shape() {
    let raw = f32_tensor("set_bias", vec![], &[0.25]);
    let file = SafeTensors::deserialize(&raw).expect("safetensors");
    let value = tensor(&file, "set_bias").expect("rank-0 scalar");
    assert!(value.shape.is_empty());
    assert_eq!(value.data, [0.25]);
}
