//! The Vela-owned F32 heads, mirrored from the checkpoint's
//! `CandidateHead` and `SpanHeadV2` (the reference computes both with
//! autocast disabled, in float32 — deterministic math). Tensors load
//! from the exported bundle's `heads.safetensors` (37 F32 tensors:
//! CandidateHead, `set_bias`, the router `span2.*` and the side-loaded
//! `span_broad.*`).

use std::path::Path;

use anyhow::{Context, Result, ensure};
use safetensors::SafeTensors;

const HIDDEN: usize = 1024;
const HEAD_DIM: usize = 256;

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x / std::f32::consts::SQRT_2))
}

/// One `Linear`: weights row-major `[out, in]`, optional bias.
struct Linear {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    out: usize,
    input: usize,
}

impl Linear {
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.input);
        let mut out = vec![0.0; self.out];
        for (row, slot) in out.iter_mut().enumerate() {
            let mut sum = self.bias.as_ref().map_or(0.0, |b| b[row]);
            let base = row * self.input;
            for (index, value) in x.iter().enumerate() {
                sum += self.weight[base + index] * value;
            }
            *slot = sum;
        }
        out
    }
}

/// One `LayerNorm` over the hidden vector (PyTorch defaults, eps 1e-5).
struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl LayerNorm {
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        let n = x.len() as f32;
        let mean = x.iter().sum::<f32>() / n;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv = 1.0 / (var + 1e-5).sqrt();
        x.iter()
            .enumerate()
            .map(|(i, v)| (v - mean) * inv * self.weight[i] + self.bias[i])
            .collect()
    }
}

/// `SpanHeadV2` (router `span2.*` or broad `span_broad.*`).
pub struct SpanHead {
    word_norm: LayerNorm,
    label_norm: LayerNorm,
    slot: Vec<f32>, // [slots, hidden]
    slots: usize,
    d: usize,
    k: Linear,
    q: Linear,
    m: Linear,
    n: Linear,
    v: Linear,
}

impl SpanHead {
    /// score (W, L): words (W, H) raw states, labels (L, H) block means.
    pub fn forward(&self, words: &[Vec<f32>], labels: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut w: Vec<Vec<f32>> = words.iter().map(|row| self.word_norm.apply(row)).collect();
        // Per-column mean over the words, subtracted.
        if !w.is_empty() {
            let mut mean = vec![0.0f32; HIDDEN];
            for row in &w {
                for (i, value) in row.iter().enumerate() {
                    mean[i] += value;
                }
            }
            for value in mean.iter_mut() {
                *value /= w.len() as f32;
            }
            for row in w.iter_mut() {
                for (i, value) in row.iter_mut().enumerate() {
                    *value -= mean[i];
                }
            }
        }
        // Label states: normalized + slot embedding, mean-centered when
        // two or more labels.
        let mut e: Vec<Vec<f32>> = labels
            .iter()
            .enumerate()
            .map(|(index, row)| {
                let mut state = self.label_norm.apply(row);
                let slot = index.min(self.slots - 1);
                let base = slot * HIDDEN;
                for (i, value) in state.iter_mut().enumerate() {
                    *value += self.slot[base + i];
                }
                state
            })
            .collect();
        if e.len() >= 2 {
            let mut mean = vec![0.0f32; HIDDEN];
            for row in &e {
                for (i, value) in row.iter().enumerate() {
                    mean[i] += value;
                }
            }
            for value in mean.iter_mut() {
                *value /= e.len() as f32;
            }
            for row in e.iter_mut() {
                for (i, value) in row.iter_mut().enumerate() {
                    *value -= mean[i];
                }
            }
        }
        let scale = 1.0 / (self.d as f32).sqrt();
        let k: Vec<Vec<f32>> = w.iter().map(|row| self.k.apply(row)).collect();
        let q: Vec<Vec<f32>> = e.iter().map(|row| self.q.apply(row)).collect();
        let m_rows: Vec<Vec<f32>> = w.iter().map(|row| self.m.apply(row)).collect();
        let n_rows: Vec<Vec<f32>> = e.iter().map(|row| self.n.apply(row)).collect();
        w.iter()
            .enumerate()
            .map(|(wi, _)| {
                (0..labels.len())
                    .map(|li| {
                        let mut bil = 0.0f32;
                        for (a, b) in k[wi].iter().zip(&q[li]) {
                            bil += a * b;
                        }
                        bil *= scale;
                        let mut summed = vec![0.0f32; HEAD_DIM];
                        for (index, value) in summed.iter_mut().enumerate() {
                            *value = gelu(m_rows[wi][index] + n_rows[li][index]);
                        }
                        let mut score = 0.0f32;
                        for (index, value) in summed.iter().enumerate() {
                            score += self.v.weight[index] * value;
                        }
                        bil + score
                    })
                    .collect()
            })
            .collect()
    }
}

/// The two heads plus `set_bias`, loaded from one safetensors file.
pub struct Heads {
    candidate_norm: LayerNorm,
    query_norm: LayerNorm,
    key: Linear,
    query: Linear,
    candidate_mlp: Linear,
    query_mlp: Linear,
    scalar: Linear,
    pub set_bias: f32,
    pub router: SpanHead,
    pub broad: SpanHead,
}

fn tensor<'a>(file: &'a SafeTensors<'a>, name: &str) -> Result<Vec<f32>> {
    let view = file
        .tensor(name)
        .with_context(|| format!("heads.safetensors is missing {name}"))?;
    ensure!(
        view.dtype() == safetensors::Dtype::F32,
        "{name} must be F32"
    );
    Ok(view
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

fn linear(file: &SafeTensors, prefix: &str, bias: bool, input: usize) -> Result<Linear> {
    let weight = tensor(file, &format!("{prefix}.weight"))?;
    let out = weight.len() / input;
    ensure!(
        weight.len() % input == 0 && out > 0,
        "{prefix}.weight shape"
    );
    Ok(Linear {
        weight,
        bias: if bias {
            Some(tensor(file, &format!("{prefix}.bias"))?)
        } else {
            None
        },
        out,
        input,
    })
}

fn norm(file: &SafeTensors, prefix: &str) -> Result<LayerNorm> {
    Ok(LayerNorm {
        weight: tensor(file, &format!("{prefix}.weight"))?,
        bias: tensor(file, &format!("{prefix}.bias"))?,
    })
}

fn span_head(file: &SafeTensors, prefix: &str, d: usize, slots: usize) -> Result<SpanHead> {
    let slot = tensor(file, &format!("{prefix}.slot.weight"))?;
    Ok(SpanHead {
        word_norm: norm(file, &format!("{prefix}.wn"))?,
        label_norm: norm(file, &format!("{prefix}.ln"))?,
        slots,
        slot,
        d,
        k: linear(file, &format!("{prefix}.K"), false, HIDDEN)?,
        q: linear(file, &format!("{prefix}.Q"), false, HIDDEN)?,
        m: linear(file, &format!("{prefix}.M"), true, HIDDEN)?,
        n: linear(file, &format!("{prefix}.N"), false, HIDDEN)?,
        v: linear(file, &format!("{prefix}.v"), false, HEAD_DIM)?,
    })
}

impl Heads {
    pub fn load(bundle: &Path) -> Result<Self> {
        let raw = std::fs::read(bundle.join("heads.safetensors")).context("heads.safetensors")?;
        let file = SafeTensors::deserialize(&raw).context("heads.safetensors must parse")?;
        let set_bias = tensor(&file, "set_bias")?;
        ensure!(set_bias.len() == 1, "set_bias is a scalar");
        // The span-head geometry ships in the manifest's calibration
        // (d 256, slots 64 for this member).
        Ok(Self {
            candidate_norm: norm(&file, "head.candidate_norm")?,
            query_norm: norm(&file, "head.query_norm")?,
            key: linear(&file, "head.key", false, HIDDEN)?,
            query: linear(&file, "head.query", false, HIDDEN)?,
            candidate_mlp: linear(&file, "head.candidate_mlp", true, HIDDEN)?,
            query_mlp: linear(&file, "head.query_mlp", false, HIDDEN)?,
            scalar: linear(&file, "head.scalar", false, HEAD_DIM)?,
            set_bias: set_bias[0],
            router: span_head(&file, "span2", HEAD_DIM, 64)?,
            broad: span_head(&file, "span_broad", HEAD_DIM, 64)?,
        })
    }

    /// CandidateHead: option-end hiddens (W, H) and the block's query
    /// hidden (H) -> one logit per option (plus `[ABS]` last when the
    /// row carries it — the caller slices).
    pub fn forward_candidate(&self, candidates: &[Vec<f32>], query: &[f32]) -> Vec<f32> {
        let global_query = self.query_norm.apply(query);
        let q_proj = self.query.apply(&global_query);
        let q_mlp = self.query_mlp.apply(&global_query);
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        candidates
            .iter()
            .map(|row| {
                let candidate = self.candidate_norm.apply(row);
                let key = self.key.apply(&candidate);
                let mut bilinear = 0.0f32;
                for (a, b) in key.iter().zip(&q_proj) {
                    bilinear += a * b;
                }
                bilinear *= scale;
                let mlp = self.candidate_mlp.apply(&candidate);
                let mut summed = vec![0.0f32; HEAD_DIM];
                for (index, value) in summed.iter_mut().enumerate() {
                    *value = gelu(mlp[index] + q_mlp[index]);
                }
                let mut nonlinear = 0.0f32;
                for (index, value) in summed.iter().enumerate() {
                    nonlinear += self.scalar.weight[index] * value;
                }
                bilinear + nonlinear
            })
            .collect()
    }
}
