//! Pinned little-endian F32 safetensors artifact inspection, without a model load.

use crate::{PAD_TOKEN_ID, VOCAB_SIZE};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{self, Read, Seek, SeekFrom},
    path::Path,
};
use tokenizers::Tokenizer;

pub const FORMAT: &str = "omni-lfm2-f32-v1";
pub const WEIGHTS: &str = "weights.f32.safetensors";
const SOURCE_WEIGHTS: &str = "model.safetensors";
const PINNED_PROFILE: &str = include_str!("../profile.json");
const WEIGHTS_SHA256: &str = include_str!("../weights.sha256");
const MAX_HEADER_BYTES: usize = 32_768;
const TOKENIZER_VOCAB_SIZE: usize = 64_402;

pub const HIDDEN_SIZE: usize = 1024;
pub const HEAD_DIM: usize = 64;
pub const KV_HEADS: usize = 8;
pub const INTERMEDIATE_SIZE: usize = 4608;
pub const CONV_WIDTH: usize = 3;
pub const ATTENTION_LAYERS: [usize; 6] = [2, 5, 8, 10, 12, 14];

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub format: String,
    pub profile: Value,
    pub files: BTreeMap<String, FileIdentity>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileIdentity {
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TensorHeader {
    pub(crate) dtype: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) data_offsets: [usize; 2],
}

#[derive(Debug, Serialize)]
pub struct Inspection {
    pub manifest: Manifest,
    pub tensors: usize,
    pub parameters: usize,
    pub tensor_data_bytes: u64,
}

fn pinned_profile() -> Value {
    serde_json::from_str(PINNED_PROFILE).expect("checked-in LFM profile")
}

pub fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(
        format!("{:x}", Sha256::digest(&bytes)) == pinned_profile()["checkpoint"]["tokenizer.json"],
        "tokenizer.json differs from the pinned checkpoint"
    );
    let mut tokenizer = Tokenizer::from_bytes(bytes)
        .map_err(|error| anyhow::anyhow!("load pinned tokenizer: {error}"))?;
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(None)
        .map_err(|error| anyhow::anyhow!("disable tokenizer truncation: {error}"))?;
    ensure!(
        tokenizer.get_vocab_size(true) == TOKENIZER_VOCAB_SIZE,
        "unexpected tokenizer vocabulary size"
    );
    ensure!(
        tokenizer.token_to_id("<|pad|>") == Some(PAD_TOKEN_ID),
        "unexpected padding ID"
    );
    Ok(tokenizer)
}

pub fn inspect(root: &Path) -> Result<Inspection> {
    let profile = pinned_profile();
    let path = root.join("manifest.json");
    let manifest_value =
        crate::json::parse(&fs::read(&path).with_context(|| format!("read {}", path.display()))?)
            .map_err(anyhow::Error::msg)?;
    let manifest: Manifest =
        serde_json::from_value(manifest_value).context("invalid artifact manifest")?;
    validate_manifest(&manifest, &profile)?;
    let mut parameters = 0;
    let mut tensors = 0;
    let mut tensor_data_bytes = 0;
    for (name, expected) in &manifest.files {
        let path = root.join(name);
        ensure!(
            fs::symlink_metadata(&path)?.is_file(),
            "{name} is not a regular file"
        );
        let mut file = File::open(path).with_context(|| format!("open {name}"))?;
        let bytes = file.metadata()?.len();
        ensure!(bytes == expected.bytes, "{name} file size mismatch");
        if name == WEIGHTS {
            (tensors, parameters, tensor_data_bytes) = inspect_weights(&mut file, bytes)?;
            file.seek(SeekFrom::Start(0))?;
        }
        let mut digest = Sha256::new();
        io::copy(&mut file, &mut digest)?;
        ensure!(
            format!("{:x}", digest.finalize()) == expected.sha256,
            "{name} SHA-256 mismatch"
        );
    }
    Ok(Inspection {
        manifest,
        tensors,
        parameters,
        tensor_data_bytes,
    })
}

pub(crate) fn validate_manifest(manifest: &Manifest, profile: &Value) -> Result<()> {
    ensure!(manifest.format == FORMAT, "unsupported artifact format");
    ensure!(
        manifest.profile == *profile,
        "artifact profile differs from the pinned reference"
    );
    let checkpoint = profile["checkpoint"]
        .as_object()
        .expect("pinned checkpoint file map");
    ensure!(
        manifest.files.len() == checkpoint.len(),
        "artifact file inventory mismatch"
    );
    for (name, digest) in checkpoint {
        let (name, digest) = if name == SOURCE_WEIGHTS {
            (WEIGHTS, WEIGHTS_SHA256.trim())
        } else {
            (name.as_str(), digest.as_str().expect("pinned source hash"))
        };
        ensure!(
            manifest
                .files
                .get(name)
                .is_some_and(|entry| entry.bytes > 0 && entry.sha256 == digest),
            "{name} missing or digest differs from the pinned artifact"
        );
    }
    Ok(())
}

fn inspect_weights(file: &mut File, bytes: u64) -> Result<(usize, usize, u64)> {
    let mut length = [0u8; 8];
    file.read_exact(&mut length)?;
    let length = u64::from_le_bytes(length);
    ensure!(
        (1..=MAX_HEADER_BYTES as u64).contains(&length)
            && length.is_multiple_of(8)
            && length + 8 <= bytes,
        "invalid F32 safetensors header size"
    );
    let mut raw = vec![0; length as usize];
    file.read_exact(&mut raw)?;
    let mut header = crate::json::parse(&raw).map_err(anyhow::Error::msg)?;
    let object = header
        .as_object_mut()
        .context("safetensors header must be an object")?;
    if let Some(metadata) = object.shift_remove("__metadata__") {
        ensure!(
            metadata == serde_json::json!({"format": "pt"}),
            "invalid safetensors metadata"
        );
    }
    let header: BTreeMap<String, TensorHeader> = serde_json::from_value(header)?;
    let data_bytes = bytes - length - 8;
    let parameters =
        validate_tensor_layout(&header, &expected_shapes(), usize::try_from(data_bytes)?)?;
    Ok((header.len(), parameters, data_bytes))
}

pub(crate) fn validate_tensor_layout(
    header: &BTreeMap<String, TensorHeader>,
    expected: &BTreeMap<String, Vec<usize>>,
    data_bytes: usize,
) -> Result<usize> {
    ensure!(header.len() == expected.len(), "tensor inventory mismatch");
    let mut spans = Vec::with_capacity(header.len());
    for (name, shape) in expected {
        let tensor = header
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        ensure!(tensor.dtype == "F32", "{name} must be F32");
        ensure!(&tensor.shape == shape, "{name} shape mismatch");
        let [start, end] = tensor.data_offsets;
        let expected_bytes = shape.iter().product::<usize>() * 4;
        ensure!(
            end.checked_sub(start) == Some(expected_bytes),
            "{name} byte length mismatch"
        );
        spans.push((start, end));
    }
    spans.sort_unstable();
    let mut cursor = 0;
    for (start, end) in spans {
        ensure!(start == cursor, "tensor offsets overlap or leave a gap");
        cursor = end;
    }
    ensure!(cursor == data_bytes, "tensor data length mismatch");
    Ok(cursor / 4)
}

pub fn expected_shapes() -> BTreeMap<String, Vec<usize>> {
    let mut shapes = BTreeMap::from([
        (
            "model.embed_tokens.weight".to_owned(),
            vec![VOCAB_SIZE, HIDDEN_SIZE],
        ),
        ("model.embedding_norm.weight".to_owned(), vec![HIDDEN_SIZE]),
    ]);
    for layer in 0..16 {
        let mut add = |name, shape| {
            shapes.insert(format!("model.layers.{layer}.{name}.weight"), shape);
        };
        add("operator_norm", vec![HIDDEN_SIZE]);
        add("ffn_norm", vec![HIDDEN_SIZE]);
        add("feed_forward.w1", vec![INTERMEDIATE_SIZE, HIDDEN_SIZE]);
        add("feed_forward.w2", vec![HIDDEN_SIZE, INTERMEDIATE_SIZE]);
        add("feed_forward.w3", vec![INTERMEDIATE_SIZE, HIDDEN_SIZE]);
        if ATTENTION_LAYERS.contains(&layer) {
            add("self_attn.q_proj", vec![HIDDEN_SIZE, HIDDEN_SIZE]);
            add("self_attn.k_proj", vec![KV_HEADS * HEAD_DIM, HIDDEN_SIZE]);
            add("self_attn.v_proj", vec![KV_HEADS * HEAD_DIM, HIDDEN_SIZE]);
            add("self_attn.out_proj", vec![HIDDEN_SIZE, HIDDEN_SIZE]);
            add("self_attn.q_layernorm", vec![HEAD_DIM]);
            add("self_attn.k_layernorm", vec![HEAD_DIM]);
        } else {
            add("conv.conv", vec![HIDDEN_SIZE, 1, CONV_WIDTH]);
            add("conv.in_proj", vec![3 * HIDDEN_SIZE, HIDDEN_SIZE]);
            add("conv.out_proj", vec![HIDDEN_SIZE, HIDDEN_SIZE]);
        }
    }
    shapes
}
