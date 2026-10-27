//! The Vela-owned F32 heads, mirrored from the checkpoint's
//! `CandidateHead` and `SpanHeadV2` (the reference computes both with
//! autocast disabled, in float32 — deterministic math). Tensors load
//! from the exported bundle's `heads.safetensors` (37 F32 tensors:
//! CandidateHead, `set_bias`, the router `span2.*` and the side-loaded
//! `span_broad.*`). Every width is derived from the tensors' own
//! shapes — the norms carry the backbone hidden size, the bilinear
//! projections carry the head dim, the slot table carries its slot
//! count — so the same code serves every decoder member (hidden 1024
//! on 0.8B, 2560 on 4B, 4096 on 9B); the only constants left are
//! cross-checks between tensors, not member sizes.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use safetensors::SafeTensors;

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x / std::f32::consts::SQRT_2))
}

/// One F32 tensor: flat row-major data plus its shape.
struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Tensor {
    fn ndim1(&self, name: &str) -> Result<usize> {
        ensure!(
            self.shape.len() == 1,
            "{name} must be 1-D, got {:?}",
            self.shape
        );
        Ok(self.shape[0])
    }

    fn ndim2(&self, name: &str) -> Result<(usize, usize)> {
        ensure!(
            self.shape.len() == 2,
            "{name} must be 2-D, got {:?}",
            self.shape
        );
        Ok((self.shape[0], self.shape[1]))
    }
}

/// One `Linear`: weights row-major `[out, in]`, optional bias.
struct Linear {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    out: usize,
    input: usize,
}

impl Linear {
    fn from_weight(tensor: Tensor, name: &str) -> Result<Self> {
        let (out, input) = tensor.ndim2(name)?;
        ensure!(out > 0 && input > 0, "{name} has a zero dimension");
        Ok(Self {
            weight: tensor.data,
            bias: None,
            out,
            input,
        })
    }

    fn with_bias(self, bias: Tensor, name: &str) -> Result<Self> {
        let len = bias.ndim1(name)?;
        ensure!(
            len == self.out,
            "{name}.bias is {len}; the weight says {}",
            self.out
        );
        Ok(Self {
            bias: Some(bias.data),
            ..self
        })
    }

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
    width: usize,
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
    hidden: usize,
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
            let mut mean = vec![0.0f32; self.hidden];
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
                let base = slot * self.hidden;
                for (i, value) in state.iter_mut().enumerate() {
                    *value += self.slot[base + i];
                }
                state
            })
            .collect();
        if e.len() >= 2 {
            let mut mean = vec![0.0f32; self.hidden];
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
                        let mut summed = vec![0.0f32; self.d];
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
    /// The bilinear head dimension (the `d` of `span_head` in
    /// config.json; 256 across the family, derived here from tensors).
    d: usize,
    /// The backbone hidden width this member's heads read.
    hidden: usize,
    pub set_bias: f32,
    pub router: SpanHead,
    pub broad: SpanHead,
}

fn tensor<'a>(file: &'a SafeTensors<'a>, name: &str) -> Result<Tensor> {
    let view = file
        .tensor(name)
        .with_context(|| format!("heads.safetensors is missing {name}"))?;
    ensure!(
        view.dtype() == safetensors::Dtype::F32,
        "{name} must be F32"
    );
    Ok(Tensor {
        data: view
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect(),
        shape: view.shape().to_vec(),
    })
}

fn norm(file: &SafeTensors, prefix: &str) -> Result<LayerNorm> {
    let weight = tensor(file, &format!("{prefix}.weight"))?;
    let bias = tensor(file, &format!("{prefix}.bias"))?;
    let width = weight.ndim1(&format!("{prefix}.weight"))?;
    ensure!(
        bias.ndim1(&format!("{prefix}.bias"))? == width,
        "{prefix} norm weight and bias widths differ"
    );
    Ok(LayerNorm {
        weight: weight.data,
        bias: bias.data,
        width,
    })
}

fn span_head(file: &SafeTensors, prefix: &str) -> Result<SpanHead> {
    // The geometry from the tensors themselves: the norms carry the
    // hidden width, the slot table [slots, hidden] its slot count, the
    // K projection [d, hidden] the bilinear dimension; everything else
    // is validated against those.
    let word_norm = norm(file, &format!("{prefix}.wn"))?;
    let label_norm = norm(file, &format!("{prefix}.ln"))?;
    let hidden = word_norm.width;
    ensure!(
        label_norm.width == hidden,
        "{prefix} label norm is {} wide; the word norm says {hidden}",
        label_norm.width
    );
    let slot = tensor(file, &format!("{prefix}.slot.weight"))?;
    let (slots, slot_hidden) = slot.ndim2(&format!("{prefix}.slot.weight"))?;
    ensure!(
        slot_hidden == hidden,
        "{prefix}.slot.weight is [ {slots}, {slot_hidden} ]; hidden is {hidden}"
    );
    let k = Linear::from_weight(
        tensor(file, &format!("{prefix}.K.weight"))?,
        &format!("{prefix}.K"),
    )?;
    let q = Linear::from_weight(
        tensor(file, &format!("{prefix}.Q.weight"))?,
        &format!("{prefix}.Q"),
    )?;
    let n = Linear::from_weight(
        tensor(file, &format!("{prefix}.N.weight"))?,
        &format!("{prefix}.N"),
    )?;
    let d = k.out;
    for t in [&q, &n] {
        ensure!(
            t.out == d && t.input == hidden,
            "{prefix} bilinear projection is [ {}, {} ]; K says [ {d}, {hidden} ]",
            t.out,
            t.input
        );
    }
    let m = Linear::from_weight(
        tensor(file, &format!("{prefix}.M.weight"))?,
        &format!("{prefix}.M"),
    )?
    .with_bias(
        tensor(file, &format!("{prefix}.M.bias"))?,
        &format!("{prefix}.M"),
    )?;
    ensure!(
        m.out == d && m.input == hidden,
        "{prefix}.M is [ {}, {} ]; K says [ {d}, {hidden} ]",
        m.out,
        m.input
    );
    let v = Linear::from_weight(
        tensor(file, &format!("{prefix}.v.weight"))?,
        &format!("{prefix}.v"),
    )?;
    ensure!(
        v.out == 1 && v.input == d,
        "{prefix}.v is [ {}, {} ]; expected [ 1, {d} ]",
        v.out,
        v.input
    );
    Ok(SpanHead {
        word_norm,
        label_norm,
        slots,
        slot: slot.data,
        hidden,
        d,
        k,
        q,
        m,
        n,
        v,
    })
}

impl Heads {
    pub fn load(bundle: &Path) -> Result<Self> {
        let raw = std::fs::read(bundle.join("heads.safetensors")).context("heads.safetensors")?;
        let file = SafeTensors::deserialize(&raw).context("heads.safetensors must parse")?;
        let set_bias = tensor(&file, "set_bias")?;
        // The members store the scalar either 0-D or as [1]; both are
        // one float.
        ensure!(
            set_bias.shape.is_empty() || (set_bias.shape.len() == 1 && set_bias.shape[0] == 1),
            "set_bias is a scalar, got {:?}",
            set_bias.shape
        );
        let candidate_norm = norm(&file, "head.candidate_norm")?;
        let query_norm = norm(&file, "head.query_norm")?;
        let hidden = candidate_norm.width;
        ensure!(
            query_norm.width == hidden,
            "head norms differ in width: {} vs {}",
            hidden,
            query_norm.width
        );
        let key = Linear::from_weight(tensor(&file, "head.key.weight")?, "head.key")?;
        let d = key.out;
        ensure!(key.input == hidden, "head.key input is {}", key.input);
        let query = Linear::from_weight(tensor(&file, "head.query.weight")?, "head.query")?;
        let candidate_mlp = Linear::from_weight(
            tensor(&file, "head.candidate_mlp.weight")?,
            "head.candidate_mlp",
        )?
        .with_bias(
            tensor(&file, "head.candidate_mlp.bias")?,
            "head.candidate_mlp",
        )?;
        let query_mlp =
            Linear::from_weight(tensor(&file, "head.query_mlp.weight")?, "head.query_mlp")?;
        for t in [&query, &candidate_mlp, &query_mlp] {
            ensure!(
                t.out == d && t.input == hidden,
                "a CandidateHead projection is [ {}, {} ]; key says [ {d}, {hidden} ]",
                t.out,
                t.input
            );
        }
        let scalar = Linear::from_weight(tensor(&file, "head.scalar.weight")?, "head.scalar")?;
        ensure!(
            scalar.out == 1 && scalar.input == d,
            "head.scalar is [ {}, {} ]; expected [ 1, {d} ]",
            scalar.out,
            scalar.input
        );
        Ok(Self {
            candidate_norm,
            query_norm,
            key,
            query,
            candidate_mlp,
            query_mlp,
            scalar,
            d,
            hidden,
            set_bias: set_bias.data[0],
            router: span_head(&file, "span2")?,
            broad: span_head(&file, "span_broad")?,
        })
    }

    /// The backbone hidden width these heads read (the executor
    /// cross-checks it against the loaded backbone).
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// CandidateHead: option-end hiddens (W, H) and the block's query
    /// hidden (H) -> one logit per option (plus `[ABS]` last when the
    /// row carries it — the caller slices).
    pub fn forward_candidate(&self, candidates: &[Vec<f32>], query: &[f32]) -> Vec<f32> {
        let global_query = self.query_norm.apply(query);
        let q_proj = self.query.apply(&global_query);
        let q_mlp = self.query_mlp.apply(&global_query);
        let scale = 1.0 / (self.d as f32).sqrt();
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
                let mut summed = vec![0.0f32; self.d];
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
