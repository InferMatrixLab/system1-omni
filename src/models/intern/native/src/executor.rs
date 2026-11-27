//! One full causal Qwen prefill, then the checkpoint's tied BF16 vocabulary head.

use std::{ffi::c_void, fs::File, path::Path, ptr::NonNull};

use anyhow::{Context, Result, ensure};
use half::bf16;
use omni_qwen3_5_native::{
    cuda::{self, DeviceBuffer, Stream},
    model::Model,
};
use serde_json::Value;

const EMBED: &str = "model.language_model.embed_tokens.weight";
const HIDDEN: usize = 2560;
const VOCAB: usize = 248_320;
const MAX_FIELDS: usize = 16;

pub struct Executor {
    model: Model,
    head: VocabularyHead,
}

impl Executor {
    pub fn load(checkpoint: &Path, library: &Path) -> Result<Self> {
        let mut model = Model::load_with_graph(checkpoint, library, false)?;
        ensure!(
            model.cfg.image_token_id == Some(248056),
            "checkpoint does not match Intern-Decision-4B"
        );
        model.enable_framework_text_4b()?;
        Ok(Self {
            model,
            head: VocabularyHead::load(checkpoint)?,
        })
    }

    pub fn hidden(&mut self, ids: &[u32], rows: &[usize]) -> Result<Vec<Vec<f32>>> {
        self.model.forward_framework_text_readout(ids, rows)
    }

    pub fn project(&mut self, hidden: &[Vec<f32>], symbol_ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        self.head.project(hidden, symbol_ids)
    }
}

struct Gemm(NonNull<c_void>);

// SAFETY: the handle moves between threads with its owner and is never used
// concurrently. Its work is completed before the owner releases admission.
unsafe impl Send for Gemm {}

impl Drop for Gemm {
    fn drop(&mut self) {
        // SAFETY: this handle belongs exclusively to the head.
        unsafe { (cuda::api().cs1_gemm_destroy)(self.0.as_ptr()) }
    }
}

struct VocabularyHead {
    gemm: Gemm,
    stream: HeadStream,
    weight: DeviceBuffer,
    hidden: DeviceBuffer,
    output: DeviceBuffer,
}

struct HeadStream(Stream);

impl Drop for HeadStream {
    fn drop(&mut self) {
        // SAFETY: the head owns this idle stream.
        unsafe { (cuda::api().cs1_stream_destroy)(self.0) };
    }
}

impl VocabularyHead {
    fn load(checkpoint: &Path) -> Result<Self> {
        let config: Value = serde_json::from_reader(File::open(checkpoint.join("config.json"))?)?;
        ensure!(
            config["tie_word_embeddings"] == true
                && config["text_config"]["tie_word_embeddings"] == true
                && config["text_config"]["vocab_size"] == VOCAB,
            "expected the tied Intern vocabulary head"
        );
        let index: Value =
            serde_json::from_reader(File::open(checkpoint.join("model.safetensors.index.json"))?)?;
        let weights = index["weight_map"]
            .as_object()
            .context("missing checkpoint weight map")?;
        ensure!(
            !weights.keys().any(|name| name.ends_with("lm_head.weight")),
            "unexpected independent vocabulary head"
        );
        let shard = weights.get(EMBED).and_then(Value::as_str).context(EMBED)?;
        let file = File::open(checkpoint.join(shard))?;
        // SAFETY: the checkpoint is immutable for the duration of loading.
        let map = unsafe { memmap2::Mmap::map(&file)? };
        let tensors = safetensors::SafeTensors::deserialize(&map)?;
        let embedding = tensors.tensor(EMBED)?;
        ensure!(
            embedding.dtype() == safetensors::Dtype::BF16 && embedding.shape() == [VOCAB, HIDDEN],
            "expected BF16 embedding [{VOCAB}, {HIDDEN}]"
        );
        let stream = HeadStream(cuda::new_stream()?);
        let weight = DeviceBuffer::new(embedding.data().len())?;
        // SAFETY: the exact-size buffer remains live, and upload synchronizes the source.
        unsafe { cuda::upload(weight.at(0), embedding.data(), stream.0)? };
        let reference = cuda::api().language_reference()?;
        // SAFETY: the new handle and workspace are private to this head.
        let gemm = Gemm(
            NonNull::new(unsafe { (reference.gemm_create)(32 << 20) })
                .context("reference vocabulary GEMM allocation failed")?,
        );
        Ok(Self {
            gemm,
            stream,
            weight,
            hidden: DeviceBuffer::new(MAX_FIELDS * HIDDEN * 2)?,
            output: DeviceBuffer::new(MAX_FIELDS * VOCAB * 2)?,
        })
    }

    fn project(&mut self, hidden: &[Vec<f32>], symbol_ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        ensure!(
            (1..=MAX_FIELDS).contains(&hidden.len()),
            "invalid field count"
        );
        ensure!(
            hidden.iter().all(|row| row.len() == HIDDEN),
            "invalid hidden width"
        );
        ensure!(
            symbol_ids.iter().all(|&id| (id as usize) < VOCAB),
            "symbol ID outside vocabulary"
        );
        cuda::set_device(0)?;
        let inputs: Vec<u8> = hidden
            .iter()
            .flatten()
            .flat_map(|&value| bf16::from_f32(value).to_le_bytes())
            .collect();
        let mut logits = vec![0; hidden.len() * VOCAB * 2];
        let stream = self.stream.0;
        // SAFETY: all matrix sizes fit the model-owned allocations and share the stream.
        let result = unsafe {
            cuda::upload(self.hidden.at(0), &inputs, stream)?;
            cuda::check(
                (cuda::api().cs1_gemm)(
                    self.gemm.0.as_ptr(),
                    self.hidden.at(0),
                    self.weight.at(0),
                    self.output.at(0),
                    hidden.len() as i32,
                    VOCAB as i32,
                    HIDDEN as i32,
                    VOCAB as i32,
                    stream,
                ),
                "Intern vocabulary GEMM",
            )
            .and_then(|()| cuda::download(&mut logits, self.output.at(0), stream))
        };
        if result.is_err() {
            let _ = cuda::synchronize(stream);
        }
        result?;
        Ok(logits
            .as_chunks::<{ VOCAB * 2 }>()
            .0
            .iter()
            .map(|row| {
                symbol_ids
                    .iter()
                    .map(|&id| {
                        let index = 2 * id as usize;
                        bf16::from_le_bytes([row[index], row[index + 1]]).to_f32()
                    })
                    .collect()
            })
            .collect())
    }
}
