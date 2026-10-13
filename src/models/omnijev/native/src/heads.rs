//! OmniJev's trained FP32 heads on the CPU and the backbone log-probability features
//! they take, following OmniJev @ 14dbec4 (`mso/head.py`, `mso/branch.py`).

use std::fs;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use safetensors::{Dtype, SafeTensors};

pub const HIDDEN: usize = 2560;
const SCORER_WIDTH: usize = 1024;
const ORDINAL_WIDTH: usize = 512;
pub const FEATURES: usize = 6;

struct Linear {
    weight: Vec<f32>,
    bias: Vec<f32>,
    inputs: usize,
}

impl Linear {
    fn load(tensors: &SafeTensors, name: &str, outputs: usize, inputs: usize) -> Result<Self> {
        Ok(Self {
            weight: tensor(tensors, &format!("{name}.weight"), &[outputs, inputs])?,
            bias: tensor(tensors, &format!("{name}.bias"), &[outputs])?,
            inputs,
        })
    }

    fn apply(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.inputs);
        self.weight
            .chunks_exact(self.inputs)
            .zip(&self.bias)
            .map(|(row, &b)| dot(row, x) + b)
            .collect()
    }
}

/// Linear, exact GELU, Linear.
struct Mlp(Linear, Linear);

impl Mlp {
    fn load(
        tensors: &SafeTensors,
        name: &str,
        inputs: usize,
        width: usize,
        outputs: usize,
    ) -> Result<Self> {
        Ok(Self(
            Linear::load(tensors, &format!("{name}.0"), width, inputs)?,
            Linear::load(tensors, &format!("{name}.2"), outputs, width)?,
        ))
    }

    fn apply(&self, x: &[f32]) -> Vec<f32> {
        let hidden: Vec<f32> = self.0.apply(x).into_iter().map(gelu).collect();
        self.1.apply(&hidden)
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    // Eight partial sums, which the compiler vectorizes.
    let mut acc = [0.0f32; 8];
    let (a8, a_rest) = a.as_chunks::<8>();
    let (b8, b_rest) = b.as_chunks::<8>();
    for (x, y) in a8.iter().zip(b8) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let tail: f32 = a_rest.iter().zip(b_rest).map(|(x, y)| x * y).sum();
    acc.iter().sum::<f32>() + tail
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2))
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// PyTorch's softplus with beta 1 and threshold 20.
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

fn tensor(tensors: &SafeTensors, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let view = tensors
        .tensor(name)
        .with_context(|| format!("missing {name}"))?;
    ensure!(
        view.dtype() == Dtype::F32 && view.shape() == shape,
        "{name} must be F32 {shape:?}"
    );
    Ok(view
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| f32::from_le_bytes(b))
        .collect())
}

/// `OptionScorer` with softmax normalization, and `OrdinalScoreHead`.
pub struct Heads {
    option: Mlp,
    question: Mlp,
    score: Linear,
    log_tau: Vec<f32>,
    abstain: Linear,
    feature: Linear,
    feature_logit: Linear,
    ordinal_z: Mlp,
    ordinal_cut: Mlp,
}

impl Heads {
    /// Load `head.*` and `ord.*` from the export's FP32 `heads.safetensors`.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let t = SafeTensors::deserialize(&bytes)?;
        Ok(Self {
            option: Mlp::load(&t, "head.opt", HIDDEN, SCORER_WIDTH, SCORER_WIDTH)?,
            question: Mlp::load(&t, "head.que", HIDDEN, SCORER_WIDTH, SCORER_WIDTH)?,
            score: Linear::load(&t, "head.score", 1, SCORER_WIDTH)?,
            log_tau: tensor(&t, "head.log_tau", &[3])?,
            abstain: Linear::load(&t, "head.abstain", 1, SCORER_WIDTH)?,
            feature: Linear::load(&t, "head.feat", SCORER_WIDTH, FEATURES)?,
            feature_logit: Linear::load(&t, "head.feat_lin", 1, FEATURES)?,
            ordinal_z: Mlp::load(&t, "ord.z", HIDDEN, ORDINAL_WIDTH, 1)?,
            ordinal_cut: Mlp::load(&t, "ord.cut", HIDDEN, ORDINAL_WIDTH, 1)?,
        })
    }

    /// `OptionScorer.logits`: one logit per option, then the abstain logit for Choice
    /// and Score or zero for Noul. `options` holds one hidden row per option.
    pub fn option_logits(
        &self,
        options: &[Vec<f32>],
        question: &[f32],
        type_id: usize,
        features: &[[f32; FEATURES]],
    ) -> Result<Vec<f32>> {
        ensure!(
            !options.is_empty()
                && options.len() == features.len()
                && options.iter().all(|u| u.len() == HIDDEN)
                && question.len() == HIDDEN
                && type_id < 3,
            "head inputs must be one {HIDDEN}-wide row and feature row per option"
        );
        let gate: Vec<f32> = self
            .question
            .apply(question)
            .into_iter()
            .map(f32::tanh)
            .collect();
        let tau = self.log_tau[type_id].exp();
        let mut logits: Vec<f32> = options
            .iter()
            .zip(features)
            .map(|(u, f)| {
                let mut h: Vec<f32> = self
                    .option
                    .apply(u)
                    .iter()
                    .zip(&gate)
                    .map(|(a, g)| a * g)
                    .collect();
                for (h, f) in h.iter_mut().zip(self.feature.apply(f)) {
                    *h += f;
                }
                (self.score.apply(&h)[0] + self.feature_logit.apply(f)[0]) / tau
            })
            .collect();
        logits.push(if type_id == 0 {
            0.0
        } else {
            self.abstain.apply(&gate)[0] / tau
        });
        Ok(logits)
    }

    /// `OptionScorer.forward`: the options' probabilities; their shortfall from one is
    /// the abstain probability. Noul is a sigmoid of its one logit.
    pub fn option_probabilities(
        &self,
        options: &[Vec<f32>],
        question: &[f32],
        type_id: usize,
        features: &[[f32; FEATURES]],
    ) -> Result<Vec<f32>> {
        let logits = self.option_logits(options, question, type_id, features)?;
        let k = logits.len() - 1;
        if type_id == 0 {
            return Ok(logits[..k].iter().map(|&x| sigmoid(x)).collect());
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
        let total: f32 = exp.iter().sum();
        Ok(exp[..k].iter().map(|e| e / total).collect())
    }

    /// `OrdinalScoreHead`: cumulative-link probabilities of the levels in caller order.
    pub fn ordinal(&self, question: &[f32], levels: &[Vec<f32>]) -> Result<Vec<f32>> {
        ensure!(
            !levels.is_empty()
                && levels.iter().all(|u| u.len() == HIDDEN)
                && question.len() == HIDDEN,
            "ordinal inputs must be {HIDDEN}-wide rows"
        );
        let z = self.ordinal_z.apply(question)[0];
        let raw: Vec<f32> = levels
            .iter()
            .map(|u| self.ordinal_cut.apply(u)[0])
            .collect();
        let mut theta = Vec::with_capacity(raw.len());
        let mut sum = 0.0f32;
        for (i, &r) in raw.iter().enumerate() {
            if i > 0 {
                sum += softplus(r);
            }
            theta.push(raw[0] + sum);
        }
        let mut cdf: Vec<f32> = theta.iter().map(|&t| sigmoid(t - z)).collect();
        *cdf.last_mut().expect("at least one level") = 1.0;
        Ok((0..cdf.len())
            .map(|i| if i == 0 { cdf[0] } else { cdf[i] - cdf[i - 1] }.max(1e-8))
            .collect())
    }
}

/// The six backbone features of every option of one question: the summed, mean and
/// counted log-probabilities of its text tokens, their log-softmax across the
/// question's options, the first token's log-probability from `zq`, and validity.
/// `picked[i]` holds option `i`'s text-token log-probabilities, `first[i]` the first
/// one's from `zq`.
pub fn lm_features(picked: &[Vec<f32>], first: &[Option<f32>]) -> Vec<[f32; FEATURES]> {
    let sums: Vec<f32> = picked
        .iter()
        .map(|p| p.iter().fold(0.0f32, |a, &x| a + x))
        .collect();
    let masked: Vec<f32> = picked
        .iter()
        .zip(&sums)
        .map(|(p, &s)| if p.is_empty() { -1e4 } else { s })
        .collect();
    let max = masked.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_total = masked.iter().map(|&m| (m - max).exp()).sum::<f32>().ln();
    picked
        .iter()
        .zip(&sums)
        .zip(&masked)
        .zip(first)
        .map(|(((p, &sum), &m), &first)| {
            let count = p.len() as f32;
            let valid = !p.is_empty();
            [
                sum / 10.0,
                if valid { sum / count } else { 0.0 },
                count / 10.0,
                m - max - log_total,
                first.unwrap_or(0.0),
                if valid { 1.0 } else { 0.0 },
            ]
        })
        .collect()
}
