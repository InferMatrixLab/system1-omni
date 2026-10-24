//! The learned decision head on CPU FP32.
//!
//! `head.safetensors` from the five-file export carries two FP32 `(256, 2048)`
//! projections, `head.candidate.weight` and `head.decision.weight`, applied
//! without bias: `d = W_dec · h_dec`, `c_k = W_cand · h_cand_k`, and the
//! candidate logit is `(c_k · d) / sqrt(256)` — every product FP32, as the
//! pinned reference computes them (`valen.modeling.model.DecisionHead`).
//! The head runs on the CPU (the spec's decision); only the reduction order
//! differs from the reference's device FP32 GEMMs, at the FP32 rounding scale.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use safetensors::SafeTensors;

use crate::text::hash_file;

/// `head.candidate.weight` and `head.decision.weight`, row-major `[256, 2048]`.
pub struct Head {
    candidate: Vec<f32>,
    decision: Vec<f32>,
    projection: usize,
    hidden: usize,
}

const CANDIDATE: &str = "head.candidate.weight";
const DECISION: &str = "head.decision.weight";

impl Head {
    /// Load from the export directory, checking the manifest's record and the
    /// file's hash first, exactly as the other export consumers do.
    pub fn load(export: &Path) -> Result<Self> {
        let manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(export.join("valen_export.json"))
                .context("valen_export.json is missing")?,
        )
        .context("invalid Valen export manifest")?;
        let head = &manifest["head"];
        ensure!(
            head["file"].as_str() == Some("head.safetensors")
                && head["dtype"].as_str() == Some("float32")
                && head["bias"] == false,
            "the export must serve a bias-free FP32 head as head.safetensors"
        );
        let projection = head["projection_dim"]
            .as_u64()
            .context("head.projection_dim")? as usize;
        let hidden = head["hidden_size"].as_u64().context("head.hidden_size")? as usize;
        ensure!(
            projection == 256 && hidden == 2048,
            "the pinned Preview head is projection 256 over hidden 2048"
        );
        let expected = &manifest["files"]["head.safetensors"];
        let path = export.join("head.safetensors");
        ensure!(
            Some(std::fs::metadata(&path).context("head.safetensors")?.len())
                == expected["size"].as_u64(),
            "head.safetensors size mismatch"
        );
        ensure!(
            Some(hash_file(&path)?.as_str()) == expected["sha256"].as_str(),
            "head.safetensors checksum mismatch"
        );
        let bytes = std::fs::read(&path).context("head.safetensors")?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        let read = |name: &str| -> Result<Vec<f32>> {
            let tensor = tensors.tensor(name).with_context(|| name.to_owned())?;
            ensure!(
                tensor.dtype() == safetensors::Dtype::F32 && tensor.shape() == [projection, hidden],
                "{name} must be float32 [{projection}, {hidden}]"
            );
            Ok(tensor
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        };
        let head = Self {
            candidate: read(CANDIDATE)?,
            decision: read(DECISION)?,
            projection,
            hidden,
        };
        ensure!(
            head.candidate
                .iter()
                .chain(&head.decision)
                .all(|w| w.is_finite()),
            "non-finite decision-head weight"
        );
        Ok(head)
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    /// A head over caller-supplied matrices (row-major `[projection, hidden]`),
    /// for the deterministic synthetic fixtures in the CPU suite. Serving
    /// always loads the export's own file instead.
    pub fn from_parts(
        candidate: Vec<f32>,
        decision: Vec<f32>,
        projection: usize,
        hidden: usize,
    ) -> Self {
        debug_assert_eq!(candidate.len(), projection * hidden);
        debug_assert_eq!(decision.len(), projection * hidden);
        Self {
            candidate,
            decision,
            projection,
            hidden,
        }
    }

    fn project(&self, weights: &[f32], h: &[f32]) -> Vec<f32> {
        (0..self.projection)
            .map(|i| {
                let row = &weights[i * self.hidden..(i + 1) * self.hidden];
                row.iter().zip(h).map(|(&w, &x)| w * x).sum()
            })
            .collect()
    }

    /// One logit per candidate hidden state against the decision hidden state,
    /// both already FP32 upcasts of the backbone's BF16 output.
    pub fn logits(&self, candidates: &[Vec<f32>], decision: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            decision.len() == self.hidden && candidates.iter().all(|h| h.len() == self.hidden),
            "hidden states must be {} wide",
            self.hidden
        );
        let scale = (self.projection as f32).sqrt();
        let d = self.project(&self.decision, decision);
        let mut logits = Vec::with_capacity(candidates.len());
        for h in candidates {
            let c = self.project(&self.candidate, h);
            let dot: f32 = c.iter().zip(&d).map(|(&a, &b)| a * b).sum();
            let logit = dot / scale;
            ensure!(logit.is_finite(), "non-finite decision logit");
            logits.push(logit);
        }
        Ok(logits)
    }
}
