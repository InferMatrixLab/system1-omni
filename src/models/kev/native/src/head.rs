//! Bilinear pointer readout from jaredpalmer/kev at 2074cc44256200bb37540103fa4572901faa10f3.
//! Learned projection stays separate from calibration and response interpretation.
use anyhow::{Context, Result, ensure};
use safetensors::{Dtype, SafeTensors};
use std::path::Path;

pub struct PointerHead {
    hidden: usize,
    pointer: usize,
    q: Vec<f32>,
    k: Vec<f32>,
    q_bias: Vec<f32>,
    k_bias: Vec<f32>,
}

fn tensor(tensors: &SafeTensors<'_>, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let view = tensors
        .tensor(name)
        .with_context(|| format!("missing {name}"))?;
    ensure!(
        view.dtype() == Dtype::F32 && view.shape() == shape,
        "invalid {name} shape or dtype"
    );
    let values: Vec<f32> = view
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| f32::from_le_bytes(b))
        .collect();
    ensure!(values.iter().all(|v| v.is_finite()), "non-finite {name}");
    Ok(values)
}

fn project(weights: &[f32], bias: &[f32], input: &[f32]) -> Vec<f32> {
    weights
        .chunks_exact(input.len())
        .zip(bias)
        .map(|(row, &bias)| row.iter().zip(input).map(|(&w, &x)| w * x).sum::<f32>() + bias)
        .collect()
}

impl PointerHead {
    /// Load exactly q/k matrices and biases, preserving their FP32 representation.
    /// Dimensions come from the head; Kev-4B exports use 2560 hidden / 256 pointer.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        ensure!(
            tensors.names().len() == 4,
            "expected exactly four pointer-head tensors"
        );
        let q = tensors.tensor("q.weight")?;
        ensure!(
            q.shape().len() == 2 && q.shape().iter().all(|&d| d > 0),
            "invalid q.weight shape"
        );
        let (pointer, hidden) = (q.shape()[0], q.shape()[1]);
        Ok(Self {
            hidden,
            pointer,
            q: tensor(&tensors, "q.weight", &[pointer, hidden])?,
            k: tensor(&tensors, "k.weight", &[pointer, hidden])?,
            q_bias: tensor(&tensors, "q.bias", &[pointer])?,
            k_bias: tensor(&tensors, "k.bias", &[pointer])?,
        })
    }

    /// h_decide is the final-normalized state at <|fim_suffix|>; each h_opt is
    /// at an option's closing <|box_end|> in the SAME complete question row.
    /// Returns one raw FP32 logit per option, in caller order.
    pub fn logits(&self, h_decide: &[f32], h_opts: &[&[f32]]) -> Result<Vec<f32>> {
        ensure!(
            (1..=255).contains(&h_opts.len()),
            "expected 1..=255 options"
        );
        ensure!(
            std::iter::once(h_decide)
                .chain(h_opts.iter().copied())
                .all(|row| row.len() == self.hidden && row.iter().all(|v| v.is_finite())),
            "invalid hidden-state shape or values"
        );
        let query = project(&self.q, &self.q_bias, h_decide);
        let scale = (self.pointer as f64).sqrt().recip() as f32;
        let logits: Vec<f32> = h_opts
            .iter()
            .map(|row| {
                project(&self.k, &self.k_bias, row)
                    .iter()
                    .zip(&query)
                    .map(|(&k, &q)| k * q)
                    .sum::<f32>()
                    * scale
            })
            .collect();
        ensure!(
            logits.iter().all(|v| v.is_finite()),
            "non-finite pointer logits"
        );
        Ok(logits)
    }
}
