//! Restricted FP32 softmax and the released FP64 temperature calibration.

use anyhow::{Context, Result, ensure};
use serde_json::{Map, Value, json};

use crate::processing::{Field, Kind, score_level};

pub const TEMPERATURE: f64 = 1.99241824;
pub const MODEL_NAME: &str = "Intern-Decision-4B";
pub const BACKEND: &str = "rust-cuda";

pub fn probabilities(logits: &[f32]) -> Result<(Vec<f64>, Vec<f64>)> {
    ensure!(
        !logits.is_empty() && logits.iter().all(|v| v.is_finite()),
        "invalid logits"
    );
    let max = logits.iter().copied().reduce(f32::max).unwrap();
    let weights: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = weights.iter().sum();
    let raw: Vec<f64> = weights.into_iter().map(|w| f64::from(w / sum)).collect();
    let logs: Vec<f64> = raw.iter().map(|p| p.ln()).collect();
    let max = logs.iter().copied().reduce(f64::max).unwrap();
    let weights: Vec<f64> = logs
        .into_iter()
        .map(|p| ((p - max) / TEMPERATURE).exp())
        .collect();
    let sum: f64 = weights.iter().sum();
    Ok((raw, weights.into_iter().map(|w| w / sum).collect()))
}

pub fn finish(
    fields: &[Field],
    rows: &[Vec<f32>],
    input_tokens: usize,
    inference_ms: f64,
) -> Result<Value> {
    ensure!(
        fields.len() == rows.len(),
        "field and readout counts differ"
    );
    ensure!(
        inference_ms.is_finite() && inference_ms >= 0.0,
        "invalid inference timing"
    );
    let mut answers = Map::new();
    for (field, row) in fields.iter().zip(rows) {
        let logits = row
            .get(..field.options.len())
            .context("missing legal logits")?;
        let (raw, calibrated) = probabilities(logits)?;
        let decision = |probabilities: &[f64]| {
            field
                .options
                .iter()
                .zip(probabilities)
                .enumerate()
                .min_by(|(_, ((left, _), lp)), (_, ((right, _), rp))| {
                    rp.total_cmp(lp).then_with(|| left.cmp(right))
                })
                .map(|(index, _)| index)
                .unwrap()
        };
        let best = decision(&calibrated);
        ensure!(
            best == decision(&raw),
            "temperature calibration changed the decision"
        );
        let value = &field.options[best].0;
        let probabilities: Map<String, Value> = field
            .options
            .iter()
            .zip(&calibrated)
            .map(|((label, _), &p)| (label.clone(), json!(p)))
            .collect();
        let mut answer = Map::new();
        let kind = match field.kind {
            Kind::Choice => "choice",
            Kind::Noul => "noul",
            Kind::Score => "score",
        };
        answer.insert("type".into(), json!(kind));
        answer.insert("probabilities".into(), Value::Object(probabilities));
        answer.insert("confidence".into(), json!(calibrated[best]));
        match field.kind {
            Kind::Choice => {
                answer.insert("choice".into(), json!(value));
            }
            Kind::Noul => {
                answer.insert("noul".into(), json!(calibrated[1]));
            }
            Kind::Score => {
                let levels = field
                    .options
                    .iter()
                    .map(|(label, _)| score_level(label).context("malformed Score level"))
                    .collect::<Result<Vec<_>>>()?;
                let legend: Map<_, _> = field
                    .options
                    .iter()
                    .map(|(label, description)| (label.clone(), json!(description)))
                    .collect();
                answer.insert("score".into(), json!(expectation(&levels, &calibrated)));
                answer.insert("legend".into(), Value::Object(legend));
            }
        }
        answer.insert("source".into(), json!("local"));
        answer.insert("decision".into(), json!(value));
        answers.insert(field.name.clone(), Value::Object(answer));
    }
    Ok(json!({
        "answers": answers,
        "usage": {"input_tokens": input_tokens, "output_tokens": fields.len(), "decision_count": fields.len()},
        "timing": {"inference_ms": (inference_ms * 100.0).round() / 100.0},
        "calibration": {"method": "temperature-scaling", "temperature": TEMPERATURE},
        "model": MODEL_NAME,
        "backend": BACKEND
    }))
}

fn expectation(levels: &[f64], probabilities: &[f64]) -> f64 {
    if let [level] = levels {
        return *level;
    }
    let min = levels.iter().copied().reduce(f64::min).unwrap();
    let max = levels.iter().copied().reduce(f64::max).unwrap();
    let ordered: f64 = levels
        .iter()
        .zip(probabilities)
        .map(|(level, p)| level * p)
        .sum();
    if ordered.is_finite() {
        return ordered.clamp(min, max);
    }
    let scale = min.abs().max(max.abs());
    let scaled: f64 = levels
        .iter()
        .zip(probabilities)
        .map(|(level, p)| (level / scale) * p)
        .sum();
    scaled.clamp(min / scale, max / scale) * scale
}
