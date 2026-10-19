use anyhow::{Context, Result, ensure};
use memmap2::Mmap;
use omni_modernbert::structure::ModernBertStructure;
use safetensors::{Dtype, SafeTensors};
use std::{collections::BTreeMap, fs::File, path::Path};

/// Storage class per tensor name, mirroring the bundle's serving
/// conventions: the embedding uploads as f16, every norm stays f32, and
/// the GEMM-class projections narrow to bf16 at upload. The pinned
/// checkpoint stores everything f32; the narrowing is a declared
/// numerics choice carried in the profile identity.
pub fn storage_dtype(name: &str) -> &'static str {
    if name == "encoder.embeddings.tok_embeddings.weight" {
        "f16"
    } else if name.contains("norm") {
        "f32"
    } else {
        "bf16"
    }
}

/// The head-side tensors of the pinned checkpoint, recorded from its
/// safetensors header. They belong to the task heads (the readout
/// projections, the classifier MLP, the calibration scalars) and load
/// in that slice; the checkpoint accounting here covers them so every
/// tensor in the file is named exactly once.
pub fn head_tensors() -> Vec<(String, Vec<usize>)> {
    let h = 768;
    let mut tensors = vec![
        ("cls_mlp.0.weight".to_string(), vec![1536, h]),
        ("cls_mlp.0.bias".to_string(), vec![1536]),
        ("cls_mlp.3.weight".to_string(), vec![1, 1536]),
        ("cls_mlp.3.bias".to_string(), vec![1]),
        ("norm.weight".to_string(), vec![h]),
        ("norm.bias".to_string(), vec![h]),
    ];
    for name in ["w_e", "w_o", "w_p", "w_q", "w_t"] {
        tensors.push((format!("{name}.weight"), vec![256, h]));
        tensors.push((format!("{name}.bias"), vec![256]));
    }
    tensors.push(("log_tau".to_string(), vec![]));
    tensors.push(("log_tau_span".to_string(), vec![]));
    tensors
}

pub struct Weights {
    data: Mmap,
}

impl Weights {
    /// The checkpoint must remain immutable while the mapping exists.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: model files are read-only inputs; no mutable mapping is created.
        let data = unsafe { Mmap::map(&file)? };
        SafeTensors::deserialize(&data)?;
        Ok(Self { data })
    }

    /// Validate the whole checkpoint against the admitted structure:
    /// every encoder tensor must appear with the derived shape and f32
    /// storage, the head inventory must cover the rest exactly, and any
    /// missing, extra or duplicate name fails listing both sides.
    pub fn validate(&self, structure: &ModernBertStructure) -> Result<usize> {
        let tensors = SafeTensors::deserialize(&self.data)?;
        let mut expected: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (name, shape) in omni_modernbert::geometry::encoder_tensors(structure) {
            ensure!(expected.insert(name.clone(), shape).is_none(), "duplicate {name}");
        }
        for (name, shape) in head_tensors() {
            ensure!(expected.insert(name.clone(), shape).is_none(), "duplicate {name}");
        }
        let actual: std::collections::BTreeSet<String> = tensors
            .names()
            .into_iter()
            .map(String::from)
            .collect();
        let expected_names: std::collections::BTreeSet<String> =
            expected.keys().cloned().collect();
        if actual != expected_names {
            let missing: Vec<_> = expected_names.difference(&actual).collect();
            let extra: Vec<_> = actual.difference(&expected_names).collect();
            anyhow::bail!(
                "checkpoint inventory does not match the profile: missing {missing:?}, unexpected {extra:?}"
            );
        }
        for (name, shape) in &expected {
            let view = tensors.tensor(name)?;
            ensure!(
                view.shape() == shape,
                "{name}: expected {shape:?}, got {:?}",
                view.shape()
            );
            ensure!(
                view.dtype() == Dtype::F32,
                "{name} is {:?}, the pinned checkpoint stores f32",
                view.dtype()
            );
        }
        Ok(expected.len())
    }

    /// Read one f32 tensor with its expected shape. Upload-time storage
    /// conversion (f16/bf16 narrowing per [`storage_dtype`]) happens in
    /// the executor slice.
    pub fn f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let tensors = SafeTensors::deserialize(&self.data)?;
        let t = tensors
            .tensor(name)
            .with_context(|| format!("{name} is missing"))?;
        ensure!(
            t.shape() == shape,
            "{name}: expected {shape:?}, got {:?}",
            t.shape()
        );
        ensure!(t.dtype() == Dtype::F32, "{name} is not f32");
        Ok(t.data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }
}
