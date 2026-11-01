use omni_kev_native::{head::PointerHead, postprocess::probabilities};
use safetensors::{
    Dtype,
    tensor::{TensorView, serialize},
};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

static SERIAL: AtomicU64 = AtomicU64::new(0);

struct Fixture(PathBuf);
impl Fixture {
    fn new(weights: &[f32]) -> Self {
        let path = std::env::temp_dir().join(format!(
            "kev-head-{}-{}.safetensors",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        let data: Vec<u8> = weights.iter().flat_map(|v| v.to_le_bytes()).collect();
        let bias: Vec<u8> = [0.5f32, -0.25]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let tensors = [
            (
                "q.weight",
                TensorView::new(Dtype::F32, vec![2, 2], &data).unwrap(),
            ),
            (
                "k.weight",
                TensorView::new(Dtype::F32, vec![2, 2], &data).unwrap(),
            ),
            (
                "q.bias",
                TensorView::new(Dtype::F32, vec![2], &bias).unwrap(),
            ),
            (
                "k.bias",
                TensorView::new(Dtype::F32, vec![2], &bias).unwrap(),
            ),
        ];
        std::fs::write(&path, serialize(tensors, None).unwrap()).unwrap();
        Self(path)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn trained_biases_scaling_calibration_and_candidate_order() {
    let fixture = Fixture::new(&[1., 0., 0., 1.]);
    let head = PointerHead::load(&fixture.0).unwrap();
    let logits = head
        .logits(&[1., 2.], &[&[3., 4.], &[1., 0.], &[3., 4.]])
        .unwrap();
    let expected = [11.8125f32, 1.8125, 11.8125]; // dot after both learned biases
    for (actual, expected) in logits.iter().zip(expected) {
        assert!((actual - expected / 2f32.sqrt()).abs() < 1e-6);
    }
    let probs = probabilities(&logits, 2.4).unwrap();
    assert_eq!(probs[0], probs[2]);
    assert!(probs[0] > probs[1]);
    assert!((probs.iter().sum::<f32>() - 1.).abs() < 1e-6);
    assert_eq!(probabilities(&[123.], 2.4).unwrap(), vec![1.]);
    assert!(head.logits(&[1., 2.], &vec![&[0., 1.][..]; 255]).is_ok());
}

#[test]
fn rejects_invalid_weights_inputs_and_calibration() {
    let fixture = Fixture::new(&[1., 0., 0., 1.]);
    let head = PointerHead::load(&fixture.0).unwrap();
    assert!(head.logits(&[1.], &[&[1., 2.]]).is_err());
    assert!(head.logits(&[1., 2.], &[&[f32::NAN, 2.]]).is_err());
    assert!(head.logits(&[1., 2.], &[]).is_err());
    assert!(head.logits(&[1., 2.], &vec![&[0., 1.][..]; 256]).is_err());
    assert!(PointerHead::load(&Fixture::new(&[f32::NAN; 4]).0).is_err());
    std::fs::write(&fixture.0, b"invalid safetensors").unwrap();
    assert!(PointerHead::load(&fixture.0).is_err());
    for temperature in [0., -1., f64::NAN, f64::INFINITY, 1e-100, 1e100] {
        assert!(probabilities(&[1., 2.], temperature).is_err());
    }
    assert!(probabilities(&[f32::INFINITY], 1.).is_err());
    assert!(probabilities(&[], 1.).is_err());
}

fn hidden(row: usize) -> Vec<f32> {
    (0..2560)
        .map(|i| (((i * 37 + row * 19) % 101) as f32 - 50.) / 64.)
        .collect()
}

#[test]
#[ignore = "needs KEV_HEAD pointing to the exported pinned head.safetensors; CPU only"]
fn released_head_matches_pinned_upstream_class() {
    let path = PathBuf::from(std::env::var_os("KEV_HEAD").expect("set KEV_HEAD"));
    let head = PointerHead::load(&path).unwrap();
    let fixture: serde_json::Value = serde_json::from_str(include_str!("data/head.json")).unwrap();
    let mut max_logit_error = 0f64;
    let mut max_probability_error = 0f64;
    for case in fixture["cases"].as_array().unwrap() {
        let options: Vec<_> = (1..=case["options"].as_u64().unwrap() as usize)
            .map(hidden)
            .collect();
        let refs: Vec<_> = options.iter().map(Vec::as_slice).collect();
        let logits = head.logits(&hidden(0), &refs).unwrap();
        let probs = probabilities(&logits, fixture["temperature"].as_f64().unwrap()).unwrap();
        for (&actual, expected) in logits.iter().zip(case["logits"].as_array().unwrap()) {
            let expected = expected.as_f64().unwrap();
            let error = (actual as f64 - expected).abs();
            max_logit_error = max_logit_error.max(error);
            assert!(error <= 1e-4 + 1e-5 * expected.abs(), "logit error {error}");
        }
        for (&actual, expected) in probs.iter().zip(case["probabilities"].as_array().unwrap()) {
            let error = (actual as f64 - expected.as_f64().unwrap()).abs();
            max_probability_error = max_probability_error.max(error);
            assert!(error <= 1e-5, "probability error {error}");
        }
    }
    eprintln!(
        "max logit error={max_logit_error:.8e}; max probability error={max_probability_error:.8e}"
    );
}
