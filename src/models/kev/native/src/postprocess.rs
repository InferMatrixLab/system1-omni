//! Kev checkpoint-temperature calibration; wire answers remain a later slice.
use anyhow::{Result, ensure};

pub fn probabilities(logits: &[f32], temperature: f64) -> Result<Vec<f32>> {
    ensure!((1..=255).contains(&logits.len()), "expected 1..=255 logits");
    let temperature = temperature as f32;
    ensure!(
        temperature.is_finite() && temperature > 0.0,
        "invalid temperature"
    );
    let scaled: Vec<f32> = logits.iter().map(|v| v / temperature).collect();
    ensure!(
        scaled.iter().all(|v| v.is_finite()),
        "non-finite calibrated logits"
    );
    let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut values: Vec<f32> = scaled.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = values.iter().sum();
    values.iter_mut().for_each(|v| *v /= sum);
    Ok(values)
}
