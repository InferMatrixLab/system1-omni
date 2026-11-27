//! Kev checkpoint-temperature calibration and official response interpretation.
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

use crate::contract::{Kind, Question};
use serde_json::{Map, Value, json};

/// Python round(float(x), 4), including ties to even, before wire serialization.
fn rounded(value: f64) -> f64 {
    format!("{value:.4}")
        .parse()
        .expect("finite rounded scalar")
}

pub fn answers(questions: &[Question], probabilities: &[Vec<f64>]) -> Result<Value> {
    ensure!(
        questions.len() == probabilities.len(),
        "question probability count mismatch"
    );
    let mut answers = Map::new();
    for (q, p) in questions.iter().zip(probabilities) {
        ensure!(
            p.len() == q.keys.len()
                && !p.is_empty()
                && p.iter().all(|x| x.is_finite() && *x >= 0.0 && *x <= 1.0),
            "invalid probabilities"
        );
        let total: f64 = p.iter().sum();
        ensure!((total - 1.0).abs() < 1e-4, "probabilities must sum to one");
        let mode = (1..p.len()).fold(0, |best, i| if p[i] > p[best] { i } else { best });
        let dist: Map<String, Value> = q
            .keys
            .iter()
            .cloned()
            .zip(p.iter().map(|&v| json!(rounded(v))))
            .collect();
        // Construct in the official insertion order: billing tokenization observes it.
        let mut answer = Map::new();
        match q.kind {
            Kind::Noul => {
                answer.insert("type".into(), json!("noul"));
                answer.insert("noul".into(), json!(rounded(p[1])));
            }
            Kind::Choice => {
                let n = p.len() as f64;
                let confidence = if p.len() == 1 {
                    1.0
                } else {
                    (p[mode] / total - 1.0 / n) / (1.0 - 1.0 / n)
                };
                answer.insert("type".into(), json!("choice"));
                answer.insert("choice".into(), json!(q.keys[mode]));
                answer.insert("confidence".into(), json!(rounded(confidence)));
                answer.insert("probabilities".into(), Value::Object(dist));
            }
            Kind::Score => {
                let n = p.len() as f64;
                let score: f64 = p.iter().enumerate().map(|(i, &p)| i as f64 * p).sum();
                let confidence = if p.len() == 1 {
                    1.0
                } else {
                    let d = (0..p.len())
                        .map(|i| (i as f64 - (n - 1.0) / 2.0).abs())
                        .sum::<f64>()
                        / n;
                    let distance = p
                        .iter()
                        .enumerate()
                        .map(|(i, p)| i.abs_diff(mode) as f64 * p / total)
                        .sum::<f64>();
                    (1.0 - distance / d).max(0.0)
                };
                answer.insert("type".into(), json!("score"));
                answer.insert("score".into(), json!(rounded(score)));
                answer.insert("legend".into(), q.legend.clone().unwrap_or(Value::Null));
                answer.insert("probabilities".into(), Value::Object(dist));
                answer.insert("confidence".into(), json!(rounded(confidence)));
            }
        }
        answers.insert(q.id.clone(), Value::Object(answer));
    }
    Ok(Value::Object(answers))
}

/// json.dumps(answers), ensure_ascii=True, as used by Kev's output_tokens.
pub fn billing_json(value: &Value) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    for c in omni_qwen3_5_native::json::dumps(value).chars() {
        if c.is_ascii() && c != '\u{7f}' {
            out.push(c);
        } else {
            for unit in c.encode_utf16(&mut [0; 2]) {
                write!(out, "\\u{unit:04x}").unwrap();
            }
        }
    }
    out
}
