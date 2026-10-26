//! The Qwen3.5/3.8 text backbone, prefill only: one
//! forward pass over a prompt, returning the final-norm hidden state of the last
//! position. The layer loop and buffers live here; the operations are the CUDA
//! kernels in `src/backends/cuda/qwen3_5`.
//!
//! The order of operations follows `modeling_qwen3_5.py`, and so do the points where
//! it rounds to bfloat16, except inside attention and the Gated DeltaNet prefill (see
//! their kernels). Text prompts use one position per
//! token. The explicit multimodal boundary inserts adapted image rows and supplies
//! the interleaved temporal/height/width rotary positions to the same layer loop.
//!
//! `Model::forward_shared` runs prompts that share token prefixes, running each prefix
//! once and every branch from the state after it (see `SharedPrompts`).

mod matched;

use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use serde_json::Value as Json;

use crate::cuda::{self, DeviceBuffer, Stream, check};
use crate::inputs::{DeviceMultimodalInput, MultimodalInput, rotary_tables};
use crate::lora::{self, Group};

const ALIGN: usize = 256;
const BF16: usize = 2;
const F32: usize = 4;
const GEMM_WORKSPACE: usize = 32 << 20;
/// Rows at which `forward_shared` fixes each weight shape's GEMM algorithm (see
/// cs1_gemm_create_fixed); its branches run tens of tokens.
const FIXED_GEMM_M: i32 = 64;

/// Partition by backend compatibility while retaining original output indices.
/// The all-compatible path needs one check; singleton groups use complete forwards.
fn compatible_prefix_groups(
    count: usize,
    mut compatible: impl FnMut(&[usize]) -> Result<bool>,
) -> Result<Option<Vec<Vec<usize>>>> {
    let all: Vec<usize> = (0..count).collect();
    if compatible(&all)? {
        return Ok(Some(vec![all]));
    }
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for index in 0..count {
        let mut placed = false;
        for group in &mut groups {
            group.push(index);
            if compatible(group)? {
                placed = true;
                break;
            }
            group.pop();
        }
        if !placed {
            groups.push(vec![index]);
        }
    }
    Ok(groups.iter().any(|g| g.len() > 1).then_some(groups))
}

/// Each nonempty compatible group computes its prefix once. Singletons save nothing.
fn shared_prefix_savings(groups: &[Vec<usize>], prefix_len: usize) -> usize {
    groups
        .iter()
        .map(|group| (group.len() - 1) * prefix_len)
        .sum()
}

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden: usize,
    pub intermediate: usize,
    pub eps: f32,
    pub full_attention: Vec<bool>,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Half the number of rotary dims (rotate_half pairs dim i with dim i + half).
    pub rotary_half: usize,
    pub rope_theta: f64,
    pub mrope_section: [usize; 3],
    pub max_positions: usize,
    pub image_token_id: Option<u32>,
    pub lin_k_heads: usize,
    pub lin_v_heads: usize,
    pub lin_k_dim: usize,
    pub lin_v_dim: usize,
}

impl Config {
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let root: Json = serde_json::from_str(
            &std::fs::read_to_string(&path).with_context(|| format!("{}", path.display()))?,
        )?;
        let c = root.get("text_config").unwrap_or(&root);
        let int = |k: &str| {
            c[k].as_u64()
                .map(|v| v as usize)
                .with_context(|| format!("config.json: `{k}` is missing"))
        };
        let rope = &c["rope_parameters"];
        let partial = rope["partial_rotary_factor"]
            .as_f64()
            .or(c["partial_rotary_factor"].as_f64())
            .unwrap_or(1.0);
        let head_dim = int("head_dim")?;
        let full_attention = c["layer_types"]
            .as_array()
            .context("config.json: `layer_types` is missing")?
            .iter()
            .map(|t| match t.as_str() {
                Some("full_attention") => Ok(true),
                Some("linear_attention") => Ok(false),
                other => bail!("unknown layer type {other:?}"),
            })
            .collect::<Result<Vec<_>>>()?;
        let sections = rope
            .get("mrope_section")
            .cloned()
            .unwrap_or(serde_json::json!([11, 11, 10]));
        let sections = sections
            .as_array()
            .context("mrope_section must be an array")?
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .context("mrope_section must contain integers")
            })
            .collect::<Result<Vec<_>>>()?;
        let mrope_section: [usize; 3] = sections
            .try_into()
            .map_err(|_| anyhow::anyhow!("mrope_section must have three entries"))?;
        let image_token_id = root
            .get("image_token_id")
            .map(|v| {
                v.as_u64()
                    .and_then(|id| u32::try_from(id).ok())
                    .context("image_token_id must be a u32")
            })
            .transpose()?;
        let cfg = Config {
            hidden: int("hidden_size")?,
            intermediate: int("intermediate_size")?,
            eps: c["rms_norm_eps"].as_f64().context("rms_norm_eps")? as f32,
            heads: int("num_attention_heads")?,
            kv_heads: int("num_key_value_heads")?,
            head_dim,
            rotary_half: (head_dim as f64 * partial) as usize / 2,
            rope_theta: rope["rope_theta"]
                .as_f64()
                .or(c["rope_theta"].as_f64())
                .context("rope_theta")?,
            mrope_section,
            max_positions: int("max_position_embeddings")?,
            image_token_id,
            lin_k_heads: int("linear_num_key_heads")?,
            lin_v_heads: int("linear_num_value_heads")?,
            lin_k_dim: int("linear_key_head_dim")?,
            lin_v_dim: int("linear_value_head_dim")?,
            full_attention,
        };
        // What the kernels implement.
        ensure!(
            cfg.full_attention.len() == int("num_hidden_layers")?,
            "layer_types does not match num_hidden_layers"
        );
        ensure!(c["hidden_act"] == "silu", "hidden_act is not silu");
        ensure!(
            c["attn_output_gate"].as_bool().unwrap_or(true),
            "attention without the output gate"
        );
        ensure!(
            c["attention_bias"].as_bool() != Some(true),
            "attention with bias"
        );
        ensure!(
            rope["rope_type"].as_str().unwrap_or("default") == "default",
            "rope type {}",
            rope["rope_type"]
        );
        ensure!(int("linear_conv_kernel_dim")? == 4, "conv kernel is not 4");
        ensure!(
            cfg.head_dim == 256 && cfg.lin_k_dim == 128 && cfg.lin_v_dim == 128,
            "head dims {} / {} / {}",
            cfg.head_dim,
            cfg.lin_k_dim,
            cfg.lin_v_dim
        );
        ensure!(cfg.rotary_half == 32, "{} rotary dims", 2 * cfg.rotary_half);
        ensure!(
            cfg.rope_theta.is_finite() && cfg.rope_theta > 0.0,
            "invalid rope_theta"
        );
        ensure!(
            rope["mrope_interleaved"].as_bool() != Some(false),
            "non-interleaved mrope is unsupported"
        );
        ensure!(
            cfg.mrope_section
                .iter()
                .try_fold(0usize, |sum, &x| sum.checked_add(x))
                == Some(cfg.rotary_half),
            "mrope sections must sum to rotary_half"
        );
        ensure!(
            cfg.max_positions > 0 && cfg.max_positions <= (1 << 24),
            "unsupported max_position_embeddings"
        );
        ensure!(
            cfg.kv_heads > 0 && cfg.heads.is_multiple_of(cfg.kv_heads),
            "attention heads"
        );
        ensure!(
            cfg.lin_k_heads > 0 && cfg.lin_v_heads.is_multiple_of(cfg.lin_k_heads),
            "linear attention heads"
        );
        ensure!(cfg.hidden.is_multiple_of(8), "hidden size");
        Ok(cfg)
    }

    fn key_dim(&self) -> usize {
        self.lin_k_heads * self.lin_k_dim
    }

    fn value_dim(&self) -> usize {
        self.lin_v_heads * self.lin_v_dim
    }
}

/// A weight in the device arena.
#[derive(Clone)]
struct Tensor {
    ptr: *const c_void,
    shape: Vec<usize>,
}

impl Tensor {
    fn bytes(&self) -> usize {
        self.shape.iter().product::<usize>() * BF16
    }
}

/// Projections that run as one GEMM, in the order their rows are stacked.
const GROUPS: &[&str] = &[
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.in_proj_a.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
];

/// Upload order: by layer, and inside a layer the GROUPS members first and in order,
/// so each group's matrices sit back to back and form one [sum N, K] matrix.
/// HF Qwen3.5 folders store these two GDN parameters in FP32. Every other
/// language tensor is BF16. The kernels read both as BF16.
fn fp32_gdn_param(name: &str) -> bool {
    name.ends_with("linear_attn.A_log") || name.ends_with("linear_attn.norm.weight")
}

fn bf16_payload_len(name: &str, dtype: safetensors::Dtype, nbytes: usize) -> Result<usize> {
    match dtype {
        safetensors::Dtype::BF16 => Ok(nbytes),
        safetensors::Dtype::F32 if fp32_gdn_param(name) => {
            ensure!(nbytes.is_multiple_of(4), "{name} has a short FP32 payload");
            Ok(nbytes / 2)
        }
        other => bail!("{name} is {other:?}, not bfloat16"),
    }
}

fn round_f32_param_to_bf16(data: &[u8]) -> Result<Vec<u8>> {
    ensure!(data.len().is_multiple_of(4), "FP32 parameter length");
    let (chunks, _) = data.as_chunks::<4>();
    let mut out = Vec::with_capacity(chunks.len() * 2);
    for chunk in chunks {
        let value = f32::from_le_bytes(*chunk);
        ensure!(value.is_finite(), "non-finite FP32 GDN parameter");
        out.extend_from_slice(&half::bf16::from_f32(value).to_le_bytes());
    }
    Ok(out)
}

fn upload_order(name: &str) -> (usize, usize, String) {
    if let Some(tail) = name.strip_prefix("layers.")
        && let Some((layer, rest)) = tail.split_once('.')
        && let Ok(layer) = layer.parse::<usize>()
    {
        let rank = GROUPS
            .iter()
            .position(|g| *g == rest)
            .unwrap_or(GROUPS.len());
        return (layer, rank, rest.to_string());
    }
    (usize::MAX, 0, name.to_string())
}

struct Weights {
    _arena: DeviceBuffer,
    tensors: HashMap<String, Tensor>,
    prefix: String,
}

impl Weights {
    /// Upload every bfloat16 tensor of the language model into one allocation.
    fn load(dir: &Path, stream: Stream) -> Result<Self> {
        let index = dir.join("model.safetensors.index.json");
        let mut files: Vec<String> = if index.exists() {
            let index: Json = serde_json::from_str(&std::fs::read_to_string(&index)?)?;
            index["weight_map"]
                .as_object()
                .context("weight_map")?
                .values()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        } else {
            vec!["model.safetensors".to_string()]
        };
        files.sort();
        files.dedup();
        let maps = files
            .iter()
            .map(|f| {
                let file = std::fs::File::open(dir.join(f)).with_context(|| f.clone())?;
                // SAFETY: the checkpoint is not modified while it is loaded.
                Ok(unsafe { memmap2::Mmap::map(&file)? })
            })
            .collect::<Result<Vec<_>>>()?;
        let sts = maps
            .iter()
            .map(|m| safetensors::SafeTensors::deserialize(m).map_err(anyhow::Error::from))
            .collect::<Result<Vec<_>>>()?;
        let names: Vec<(usize, String)> = sts
            .iter()
            .enumerate()
            .flat_map(|(i, st)| st.names().into_iter().map(move |n| (i, n.to_string())))
            .collect();
        let prefix = ["model.language_model.", "model.", ""]
            .into_iter()
            .find(|p| {
                names
                    .iter()
                    .any(|(_, n)| *n == format!("{p}embed_tokens.weight"))
            })
            .context("no embed_tokens.weight in the checkpoint")?
            .to_string();
        let mut ours: Vec<(usize, String)> = names
            .into_iter()
            .filter(|(_, n)| n.starts_with(&prefix))
            .collect();
        ours.sort_by_key(|(_, n)| upload_order(&n[prefix.len()..]));
        let mut plan = Vec::new();
        let mut total = 0usize;
        for (i, name) in ours {
            let view = sts[i].tensor(&name)?;
            let bytes = bf16_payload_len(&name, view.dtype(), view.data().len())?;
            plan.push((i, name, total));
            total = (total + bytes).next_multiple_of(ALIGN);
        }
        let arena = DeviceBuffer::new(total)?;
        let mut tensors = HashMap::new();
        for (i, name, offset) in plan {
            let view = sts[i].tensor(&name)?;
            // A_log and the GDN norm are FP32 in current HF Qwen3.5 folders.
            // The kernels consume BF16, so round each finite value once at load.
            let converted = if view.dtype() == safetensors::Dtype::F32 {
                Some(round_f32_param_to_bf16(view.data())?)
            } else {
                None
            };
            let bytes = converted.as_deref().unwrap_or_else(|| view.data());
            // SAFETY: the arena has room for every planned tensor at its offset.
            unsafe { cuda::upload(arena.at(offset), bytes, stream)? };
            tensors.insert(
                name[prefix.len()..].to_string(),
                Tensor {
                    ptr: arena.at(offset),
                    shape: view.shape().to_vec(),
                },
            );
        }
        Ok(Self {
            _arena: arena,
            tensors,
            prefix,
        })
    }

    fn get(&self, name: &str, shape: &[usize]) -> Result<Tensor> {
        let t = self
            .tensors
            .get(name)
            .with_context(|| format!("{}{name} is missing", self.prefix))?;
        ensure!(
            t.shape == shape,
            "{}{name}: shape {:?}, expected {:?}",
            self.prefix,
            t.shape,
            shape
        );
        Ok(t.clone())
    }

    /// The row-stacked matrix of tensors that were uploaded back to back.
    fn stacked(&self, parts: &[Tensor]) -> Result<Tensor> {
        let k = parts[0].shape[1];
        let mut rows = 0;
        for (i, p) in parts.iter().enumerate() {
            ensure!(
                p.shape.len() == 2 && p.shape[1] == k,
                "stacked shapes differ"
            );
            if i > 0 {
                let prev = &parts[i - 1];
                ensure!(
                    p.ptr == prev.ptr.wrapping_byte_add(prev.bytes()),
                    "stacked weights are not contiguous"
                );
            }
            rows += p.shape[0];
        }
        Ok(Tensor {
            ptr: parts[0].ptr,
            shape: vec![rows, k],
        })
    }
}

struct LinearAttention {
    /// in_proj_qkv | in_proj_z | in_proj_b | in_proj_a
    in_proj: Tensor,
    conv: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    norm: Tensor,
    out: Tensor,
}

struct FullAttention {
    /// q_proj (query and gate per head) | k_proj | v_proj
    qkv: Tensor,
    o: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
}

enum Mixer {
    Linear(LinearAttention),
    Full(FullAttention),
}

struct Layer {
    input_norm: Tensor,
    post_norm: Tensor,
    mixer: Mixer,
    /// gate_proj | up_proj
    gate_up: Tensor,
    down: Tensor,
}

/// Row widths of the stacked projection outputs.
struct Widths {
    conv: usize,
    gdn_in: usize,
    attn_q: usize,
    attn_in: usize,
}

impl Widths {
    fn of(cfg: &Config) -> Self {
        let conv = 2 * cfg.key_dim() + cfg.value_dim();
        let attn_q = cfg.heads * cfg.head_dim * 2;
        Self {
            conv,
            gdn_in: conv + cfg.value_dim() + 2 * cfg.lin_v_heads,
            attn_q,
            attn_in: attn_q + 2 * cfg.kv_heads * cfg.head_dim,
        }
    }
}

/// Per-request buffers for up to `cap` tokens, as byte offsets into one allocation,
/// plus the rotary tables for positions below `cap`.
struct Scratch {
    cap: usize,
    buf: DeviceBuffer,
    ids: usize,
    res: usize,
    x: usize,
    delta: usize,
    gdn_in: usize,
    beta: usize,
    g: usize,
    lq: usize,
    lk: usize,
    lv: usize,
    lo: usize,
    ln: usize,
    workspace: usize,
    workspace_bytes: usize,
    attn_in: usize,
    aq: usize,
    agate: usize,
    ak: usize,
    ao: usize,
    gate_up: usize,
    act: usize,
    cos: usize,
    sin: usize,
    custom_cos: usize,
    custom_sin: usize,
    lora: Option<lora::Work>,
}

impl Scratch {
    fn new(
        cfg: &Config,
        cap: usize,
        stream: Stream,
        lora: Option<&lora::Lora>,
        reference: Option<&cuda::Reference>,
        fp32_gdn: Option<&cuda::Fp32Gdn>,
        batched_gdn: Option<&cuda::BatchedGdn>,
    ) -> Result<Self> {
        let (h, kd, vd, hv) = (cfg.hidden, cfg.key_dim(), cfg.value_dim(), cfg.lin_v_heads);
        let (hq, hk, hd) = (cfg.heads, cfg.kv_heads, cfg.head_dim);
        let w = Widths::of(cfg);
        let mut next = 0usize;
        let mut take = |bytes: usize| {
            let off = next;
            next = (off + bytes).next_multiple_of(ALIGN);
            off
        };
        // SAFETY: pure function of its arguments.
        let ws_floats = if let Some(reference) = reference {
            // SAFETY: capacity and heads derive from validated architecture/request limits.
            unsafe {
                let gdn = (reference.gdn_workspace_floats)(cap as i32, hv as i32);
                let attention =
                    (reference.language_attention_workspace_floats)(cap as i32, hq as i32);
                ensure!(attention > 0, "cannot size reference attention workspace");
                gdn.max(attention)
            }
        } else {
            unsafe { (cuda::api().cs1_gdn_workspace_floats)(cap as i32, hv as i32) }
        };
        let ws_floats = if let Some(gdn) = fp32_gdn {
            ws_floats.max(gdn.workspace_floats(cap, hv)?)
        } else {
            ws_floats
        };
        let ws_floats = if let Some(gdn) = batched_gdn {
            // SAFETY: validated model dimensions and a bounded sequence count.
            let batched = unsafe { (gdn.workspace_floats)(cap as i32, hv as i32, 16) };
            ensure!(batched > 0, "cannot size batched GDN workspace");
            ws_floats.max(batched)
        } else {
            ws_floats
        };
        let offsets = [
            take(cap * 4),
            take(cap * h * BF16),
            take(cap * h * BF16),
            take(cap * h * BF16),
            take(cap * w.gdn_in * BF16),
            take(cap * hv * BF16),
            take(cap * hv * F32),
            take(cap * kd * BF16),
            take(cap * kd * BF16),
            take(cap * vd * BF16),
            take(cap * vd * BF16),
            take(cap * vd * BF16),
            take(ws_floats * F32),
            take(cap * w.attn_in * BF16),
            take(cap * hq * hd * BF16),
            take(cap * hq * hd * BF16),
            take(cap * hk * hd * BF16),
            take(cap * hq * hd * BF16),
            take(cap * 2 * cfg.intermediate * BF16),
            take(cap * cfg.intermediate * BF16),
            take(cap * cfg.rotary_half * BF16),
            take(cap * cfg.rotary_half * BF16),
            take(cap * cfg.rotary_half * BF16),
            take(cap * cfg.rotary_half * BF16),
        ];
        let buf = DeviceBuffer::new(next)?;
        let [
            ids,
            res,
            x,
            delta,
            gdn_in,
            beta,
            g,
            lq,
            lk,
            lv,
            lo,
            ln,
            workspace,
            attn_in,
            aq,
            agate,
            ak,
            ao,
            gate_up,
            act,
            cos,
            sin,
            custom_cos,
            custom_sin,
        ] = offsets;
        let positions: Vec<i64> = (0..cap as i64).collect();
        let (cos_t, sin_t) = rotary_tables(
            [&positions; 3],
            cfg.rotary_half,
            cfg.rope_theta,
            cfg.mrope_section,
        );
        // SAFETY: both tables were laid out for cap * rotary_half bfloat16 values.
        unsafe {
            cuda::upload(buf.at(cos), &cos_t, stream)?;
            cuda::upload(buf.at(sin), &sin_t, stream)?;
        }
        Ok(Self {
            cap,
            buf,
            ids,
            res,
            x,
            delta,
            gdn_in,
            beta,
            g,
            lq,
            lk,
            lv,
            lo,
            ln,
            workspace,
            workspace_bytes: ws_floats * F32,
            attn_in,
            aq,
            agate,
            ak,
            ao,
            gate_up,
            act,
            cos,
            sin,
            custom_cos,
            custom_sin,
            lora: lora
                .map(|adapter| adapter.work(cap.min(cfg.max_positions)))
                .transpose()?,
        })
    }

    fn at(&self, offset: usize) -> *mut c_void {
        self.buf.at(offset)
    }
}

/// Cumulative backbone capture/replay evidence, distinct from the decision response.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct GraphStats {
    pub requested: bool,
    pub enabled: bool,
    pub captures: u64,
    pub replays: u64,
    pub fallbacks: u64,
    pub invalidations: u64,
    pub cached_shapes: usize,
}

/// The cached device-side state of one token prefix (rows `[0, len)`) for the R2d
/// hybrid path: per full-attention layer the post-prep K rows plus the pre-GEMM V
/// rows, per Gated DeltaNet layer the float32 recurrent state and the three
/// pre-conv projection columns feeding the conv window. States belong to the
/// prefix itself; a continuation seeds its own scratch from them, read-only.
pub struct PrefixState {
    owner: Arc<()>,
    initialized: bool,
    /// Tokens this prefix covers; always a multiple of 64 (the GDN chunk length),
    /// which also aligns the flash-attention key tiles of the one-shot pass.
    len: usize,
    /// Tokens the buffers hold; `forward_shared` reuses a state for shorter prefixes.
    cap: usize,
    /// Post-prep K [len, Hk*Dh] and raw V [len, Hk*Dh] rows per full-attention layer.
    attn_kv: Vec<(DeviceBuffer, DeviceBuffer)>,
    /// Float32 [H, K, V] recurrent states, one per Gated DeltaNet layer.
    gdn_state: Vec<DeviceBuffer>,
    /// Three pre-conv projection columns [3, gdn_in width], one per Gated DeltaNet layer.
    conv_tail: Vec<DeviceBuffer>,
    /// Total device bytes, for cache-budget accounting by the caller.
    bytes: usize,
}

/// A packed suffix batch uses one freshly captured prefix and original algorithm.
struct SuffixBatch<'a> {
    prefix: &'a PrefixState,
    ops: cuda::SharedSuffix,
    gemm: cuda::ReuseGemm,
    original_rows: usize,
}

/// Bound only the suffix work; oversized suffixes retain individual windows.
fn suffix_segments(indices: &[usize], lengths: &[usize], prefix: usize) -> Vec<Vec<usize>> {
    let mut groups = Vec::new();
    let mut group = Vec::new();
    let mut tokens = 0;
    for &index in indices {
        let suffix = lengths[index] - prefix;
        if !group.is_empty() && (group.len() == 16 || tokens + suffix > 4096) {
            groups.push(std::mem::take(&mut group));
            tokens = 0;
        }
        group.push(index);
        tokens += suffix;
    }
    if !group.is_empty() {
        groups.push(group);
    }
    groups
}

impl PrefixState {
    /// Number of tokens covered by this nonempty prefix.
    pub fn token_count(&self) -> usize {
        self.len
    }

    /// Device allocation size used by cache-budget accounting.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Tokens the buffers hold; a capture by `Model::readout_window` may cover fewer.
    pub fn capacity(&self) -> usize {
        self.cap
    }
}

/// Rows of logits computed at once by `forward_multimodal_readout`; with the fixed GEMM
/// algorithms, a row's logits do not depend on how many share the call.
const VOCAB_ROWS: usize = 64;

/// What `Model::forward_multimodal_readout` reads from one prefill.
#[derive(Debug, Clone)]
pub struct Readout {
    /// Final-norm hidden states at the requested rows, in order.
    pub hidden: Vec<Vec<f32>>,
    /// For each requested `(row, token)`, the log-probability of `token` after `row`.
    pub logprobs: Vec<f32>,
}

/// log-softmax of one row of bfloat16 logits over its first `vocab` entries, at `token`:
/// the logits as float32, their maximum, and a float64 sum of the exponentials.
pub(crate) fn token_logprob(logits: &[u8], vocab: usize, token: u32) -> f32 {
    let value = |i: usize| half::bf16::from_le_bytes([logits[2 * i], logits[2 * i + 1]]).to_f32();
    let max = (0..vocab).map(value).fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = (0..vocab).map(|i| (value(i) as f64 - max).exp()).sum();
    (value(token as usize) as f64 - max - sum.ln()) as f32
}

/// Gated DeltaNet chunk length. `forward_shared` ends prefixes at its multiples, where
/// a continuation repeats the one-shot arithmetic (see cs1_gdn_prefill_x).
const GDN_CHUNK: usize = 64;

/// Prompts that share token prefixes, for `Model::forward_shared`. Branch `b` of group `g`
/// is the prompt `prefix`, then `groups[g].prefix`, then `groups[g].branches[b]`. How the
/// prompts are grouped does not change any result where the fixed GEMM algorithms keep
/// rows independent of M (see `forward_shared`).
pub struct SharedPrompts<'a> {
    pub prefix: &'a [u32],
    pub groups: Vec<PromptGroup<'a>>,
}

pub struct PromptGroup<'a> {
    pub prefix: &'a [u32],
    pub branches: Vec<&'a [u32]>,
}

/// A successful compatible-prefix execution, with results in original input order.
pub struct SharedPrefixOutput {
    pub hidden: Vec<Vec<f32>>,
    /// Prefix tokens not recomputed, accounting for separate groups and singletons.
    pub saved_tokens: usize,
}

/// The token runs `forward_shared` executes: the request prefix, then per group its own
/// prefix and branches. Each prefix ends at the last multiple of `GDN_CHUNK` within it;
/// its remaining tokens run with every group or branch after it.
struct Runs {
    prefix: Vec<u32>,
    groups: Vec<GroupRuns>,
}

struct GroupRuns {
    prefix: Vec<u32>,
    branches: Vec<Vec<u32>>,
}

fn runs(prompts: &SharedPrompts) -> Runs {
    let end = |n: usize| n - n % GDN_CHUNK;
    let p = end(prompts.prefix.len());
    let groups = prompts
        .groups
        .iter()
        .map(|g| {
            let shared = [prompts.prefix, g.prefix].concat();
            let q = end(shared.len());
            GroupRuns {
                prefix: shared[p..q].to_vec(),
                branches: g
                    .branches
                    .iter()
                    .map(|b| [&shared[q..], b].concat())
                    .collect(),
            }
        })
        .collect();
    Runs {
        prefix: prompts.prefix[..p].to_vec(),
        groups,
    }
}

pub struct Model {
    pub cfg: Config,
    prefix_owner: Arc<()>,
    /// Allocation capacity only: state is invalidated and fully recaptured per request.
    text_prefix_workspace: Option<(usize, PrefixState)>,
    _weights: Weights,
    embed: Tensor,
    final_norm: Tensor,
    layers: Vec<Layer>,
    lora: Option<lora::Lora>,
    stream: Stream,
    gemm: *mut c_void,
    /// One GEMM algorithm per weight shape for `forward_shared`; created on first use.
    fixed_gemm: *mut c_void,
    matched_ref: usize,
    matched_handles: HashMap<usize, *mut c_void>,
    reference: Option<&'static cuda::Reference>,
    fp32_gdn: Option<cuda::Fp32Gdn>,
    batched_gdn: Option<cuda::BatchedGdn>,
    packed_gemm: Option<cuda::PackedGemm>,
    /// Buffers for the largest packed token count so far; grows as needed.
    scratch: Option<Scratch>,
    /// The request and group prefix states of `forward_shared`; grow as needed.
    shared: [Option<PrefixState>; 2],
    /// Hidden rows and logits for `forward_multimodal_readout`'s vocabulary pass, and
    /// the host copy of the logits; allocated on first use.
    vocab_buffers: Option<(DeviceBuffer, DeviceBuffer, Vec<u8>)>,
    /// Opt-in replay keyed by sequence lengths; bounded separately for each mode.
    graph_enabled: bool,
    graph_stats: GraphStats,
    graphs: VecDeque<(Vec<usize>, cuda::Graph)>,
    mm_graphs: VecDeque<(Vec<usize>, cuda::Graph)>,
    // The stream retires after every model allocation, including adapter resources.
    _stream_owner: ModelStream,
}

// SAFETY: the raw pointers are device addresses and cuBLASLt handles owned by the
// model; the engine runs one forward pass at a time behind a mutex.
unsafe impl Send for Model {}

struct ModelStream(Stream);
impl Drop for ModelStream {
    fn drop(&mut self) {
        // SAFETY: the Model owns this stream, and has retired all allocations.
        unsafe { (cuda::api().cs1_stream_destroy)(self.0) };
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        let _ = cuda::set_device(0);
        let _ = cuda::synchronize(self.stream);
        self.graphs.clear();
        self.mm_graphs.clear();
        self.scratch = None;
        self.shared = [None, None];
        // Retire the FP32 cuBLAS context while its last stream is still valid.
        self.fp32_gdn = None;
        // SAFETY: both GEMM handles are owned by this model and destroyed once.
        unsafe {
            for (_, h) in self.matched_handles.drain() {
                (cuda::api().cs1_gemm_destroy)(h);
            }
            (cuda::api().cs1_gemm_destroy)(self.gemm);
            if !self.fixed_gemm.is_null() {
                (cuda::api().cs1_gemm_destroy)(self.fixed_gemm);
            }
        }
    }
}

impl Model {
    /// Load the CUDA library and the weights.
    pub fn load(dir: &Path, library: &Path) -> Result<Self> {
        Self::load_with_graph(
            dir,
            library,
            std::env::var("CUA_S1_GRAPH").as_deref() == Ok("1"),
        )
    }

    /// Explicit mode for model adapters; does not read another worker's switch.
    pub fn load_with_graph(dir: &Path, library: &Path, graph: bool) -> Result<Self> {
        let cfg = Config::load(dir)?;
        Self::load_checkpoint(dir, library, cfg, None, graph)
    }

    /// Load an immutable FP32 PEFT language adapter without merging it into BF16 weights.
    /// `adapter_path` names one safetensors file; partial paired targets and ranks 1..=128
    /// are supported. Validate the pinned full inventory separately when required.
    pub fn load_with_lora(
        dir: &Path,
        library: &Path,
        adapter_path: &Path,
        scale: f32,
    ) -> Result<Self> {
        let cfg = Config::load(dir)?;
        let adapter = lora::Checkpoint::load(&cfg, adapter_path, scale)?;
        Self::load_checkpoint(
            dir,
            library,
            cfg,
            Some(adapter),
            std::env::var("CUA_S1_GRAPH").as_deref() == Ok("1"),
        )
    }

    fn load_checkpoint(
        dir: &Path,
        library: &Path,
        cfg: Config,
        adapter: Option<lora::Checkpoint>,
        graph: bool,
    ) -> Result<Self> {
        cuda::load(library)?;
        cuda::set_device(0)?;
        let stream = cuda::new_stream()?;
        let stream_owner = ModelStream(stream);
        let weights = Weights::load(dir, stream)?;
        let lora = adapter
            .map(|source| source.upload(cfg.full_attention.len(), stream))
            .transpose()?;
        let (h, kd, vd) = (cfg.hidden, cfg.key_dim(), cfg.value_dim());
        let embed = weights
            .tensors
            .get("embed_tokens.weight")
            .context("embed_tokens.weight is missing")?
            .clone();
        ensure!(
            embed.shape.len() == 2 && embed.shape[1] == h,
            "embed_tokens.weight shape {:?}",
            embed.shape
        );
        let final_norm = weights.get("norm.weight", &[h])?;
        let mut layers = Vec::with_capacity(cfg.full_attention.len());
        for (i, &full) in cfg.full_attention.iter().enumerate() {
            let w = |n: &str, s: &[usize]| weights.get(&format!("layers.{i}.{n}"), s);
            let mixer = if full {
                let (hq, hk, hd) = (cfg.heads, cfg.kv_heads, cfg.head_dim);
                Mixer::Full(FullAttention {
                    qkv: weights.stacked(&[
                        w("self_attn.q_proj.weight", &[hq * hd * 2, h])?,
                        w("self_attn.k_proj.weight", &[hk * hd, h])?,
                        w("self_attn.v_proj.weight", &[hk * hd, h])?,
                    ])?,
                    o: w("self_attn.o_proj.weight", &[h, hq * hd])?,
                    q_norm: w("self_attn.q_norm.weight", &[hd])?,
                    k_norm: w("self_attn.k_norm.weight", &[hd])?,
                })
            } else {
                let hv = cfg.lin_v_heads;
                Mixer::Linear(LinearAttention {
                    in_proj: weights.stacked(&[
                        w("linear_attn.in_proj_qkv.weight", &[2 * kd + vd, h])?,
                        w("linear_attn.in_proj_z.weight", &[vd, h])?,
                        w("linear_attn.in_proj_b.weight", &[hv, h])?,
                        w("linear_attn.in_proj_a.weight", &[hv, h])?,
                    ])?,
                    conv: w("linear_attn.conv1d.weight", &[2 * kd + vd, 1, 4])?,
                    a_log: w("linear_attn.A_log", &[hv])?,
                    dt_bias: w("linear_attn.dt_bias", &[hv])?,
                    norm: w("linear_attn.norm.weight", &[cfg.lin_v_dim])?,
                    out: w("linear_attn.out_proj.weight", &[h, vd])?,
                })
            };
            layers.push(Layer {
                input_norm: w("input_layernorm.weight", &[h])?,
                post_norm: w("post_attention_layernorm.weight", &[h])?,
                mixer,
                gate_up: weights.stacked(&[
                    w("mlp.gate_proj.weight", &[cfg.intermediate, h])?,
                    w("mlp.up_proj.weight", &[cfg.intermediate, h])?,
                ])?,
                down: w("mlp.down_proj.weight", &[h, cfg.intermediate])?,
            });
        }
        // SAFETY: plain allocation; checked for null below.
        let gemm = unsafe { (cuda::api().cs1_gemm_create)(GEMM_WORKSPACE) };
        ensure!(!gemm.is_null(), "cuBLASLt setup failed");
        let model = Self {
            cfg,
            prefix_owner: Arc::new(()),
            text_prefix_workspace: None,
            _weights: weights,
            embed,
            final_norm,
            layers,
            lora,
            stream,
            gemm,
            fixed_gemm: std::ptr::null_mut(),
            matched_ref: 0,
            matched_handles: HashMap::new(),
            reference: None,
            fp32_gdn: None,
            batched_gdn: None,
            packed_gemm: None,
            scratch: None,
            shared: [None, None],
            vocab_buffers: None,
            graph_enabled: graph,
            graph_stats: GraphStats {
                requested: graph,
                ..GraphStats::default()
            },
            graphs: VecDeque::new(),
            _stream_owner: stream_owner,
            mm_graphs: VecDeque::new(),
        };
        Ok(model)
    }

    pub fn graph_stats(&self) -> GraphStats {
        GraphStats {
            enabled: self.graph_enabled,
            cached_shapes: self.graphs.len() + self.mm_graphs.len(),
            ..self.graph_stats
        }
    }

    /// Opt into packing output/down projections when the original algorithms agree.
    /// Select before the first forward; other model consumers retain separate GEMMs.
    pub fn enable_packed_projections(&mut self) -> Result<()> {
        ensure!(
            self.scratch.is_none() && self.reference.is_none() && self.lora.is_none(),
            "select packed projections on a merged model before the first forward"
        );
        self.packed_gemm = Some(cuda::api().packed_gemm()?);
        Ok(())
    }

    /// Opt into independent GDN batching before allocating any request buffers.
    pub fn enable_batched_gdn(&mut self) -> Result<()> {
        ensure!(
            self.scratch.is_none()
                && self.reference.is_none()
                && self.fp32_gdn.is_none()
                && self.lora.is_none(),
            "select batched GDN on a merged model before the first forward"
        );
        self.batched_gdn = Some(cuda::api().batched_gdn()?);
        Ok(())
    }

    /// Cua-S1's unmerged text path: framework norm/GEMM/attention plus FP32 GDN.
    pub fn enable_reference_text_numerics(&mut self) -> Result<()> {
        ensure!(
            self.cfg.hidden == 2560 && self.lora.is_some(),
            "reference text numerics require an unmerged 4B language model"
        );
        ensure!(
            self.scratch.is_none() && self.reference.is_none() && self.fp32_gdn.is_none(),
            "select reference text numerics before the first forward"
        );
        cuda::set_device(0)?;
        let gdn = cuda::api().fp32_gdn()?;
        let reference = cuda::api().language_reference()?;
        cuda::synchronize(self.stream)?;
        // SAFETY: both handles are model-owned; no queued work or captures exist.
        let gemm = unsafe { (reference.gemm_create)(GEMM_WORKSPACE) };
        ensure!(!gemm.is_null(), "reference text cuBLASLt setup failed");
        unsafe { (cuda::api().cs1_gemm_destroy)(self.gemm) };
        self.gemm = gemm;
        self.reference = Some(reference);
        self.fp32_gdn = Some(gdn);
        Ok(())
    }

    /// Select JEMM's pinned framework arithmetic before the first forward.
    /// Prefix caching is outside this execution contract.
    pub fn enable_reference_numerics(&mut self) -> Result<()> {
        ensure!(
            self.fp32_gdn.is_none(),
            "cannot combine JEMM reference numerics with FP32 GDN"
        );
        ensure!(
            self.cfg.hidden == 5120 && self.cfg.head_dim == 256 && self.lora.is_some(),
            "reference numerics require the unmerged JEMM 27B model"
        );
        ensure!(
            self.scratch.is_none() && self.graphs.is_empty() && self.mm_graphs.is_empty(),
            "select reference numerics before the first forward"
        );
        if self.reference.is_some() {
            return Ok(());
        }
        cuda::set_device(0)?;
        let reference = cuda::api().reference()?;
        cuda::synchronize(self.stream)?;
        // SAFETY: the model owns both handles, with no queued work or captured graphs.
        let gemm = unsafe { (reference.gemm_create)(GEMM_WORKSPACE) };
        ensure!(!gemm.is_null(), "reference cuBLASLt setup failed");
        unsafe { (cuda::api().cs1_gemm_destroy)(self.gemm) };
        self.gemm = gemm;
        self.reference = Some(reference);
        self.prefix_owner = Arc::new(());
        Ok(())
    }

    fn gemm(&self, s: &Scratch, x: usize, w: &Tensor, y: usize, m: usize) -> Result<()> {
        self.gemm_with(self.gemm, s, x, w, y, m)
    }

    /// `gemm` with the cuBLASLt handle `handle`.
    fn gemm_with(
        &self,
        handle: *mut c_void,
        s: &Scratch,
        x: usize,
        w: &Tensor,
        y: usize,
        m: usize,
    ) -> Result<()> {
        let (n, k) = (w.shape[0] as i32, w.shape[1] as i32);
        // SAFETY: x and y are scratch buffers sized for m rows of w's shape.
        check(
            unsafe {
                (cuda::api().cs1_gemm)(
                    handle,
                    s.at(x),
                    w.ptr,
                    s.at(y),
                    m as i32,
                    n,
                    k,
                    n,
                    self.stream,
                )
            },
            "gemm",
        )
    }

    #[allow(clippy::too_many_arguments)] // Base projection coordinates and sequence shape.
    fn gemm_group(
        &self,
        s: &Scratch,
        x: usize,
        w: &Tensor,
        y: usize,
        m: usize,
        lengths: &[usize],
        group: Group,
    ) -> Result<()> {
        self.gemm_group_with(self.gemm, s, x, w, y, m, lengths, group)
    }

    /// `gemm_group` with the cuBLASLt handle `handle`.
    #[allow(clippy::too_many_arguments)] // Base projection coordinates and sequence shape.
    fn gemm_group_with(
        &self,
        handle: *mut c_void,
        s: &Scratch,
        x: usize,
        w: &Tensor,
        y: usize,
        m: usize,
        lengths: &[usize],
        group: Group,
    ) -> Result<()> {
        if self.lora.is_none() {
            return self.gemm_with(handle, s, x, w, y, m);
        }
        let components = lora::component_rows(&self.cfg, group)?;
        ensure!(
            components.iter().sum::<usize>() == w.shape[0],
            "base component weight shape"
        );
        let work = s.lora.as_ref().context("missing base component work")?;
        // SAFETY: stacked() validated contiguous component weights; scratch holds the
        // complete packed input/output rows, and work belongs to this serialized Model.
        unsafe {
            lora::project_components(
                s.at(x),
                w.ptr,
                s.at(y),
                lengths,
                w.shape[1],
                &components,
                work,
                handle,
                self.stream,
            )
        }
    }

    #[allow(clippy::too_many_arguments)] // Layer projection coordinates in shared scratch.
    fn apply_lora(
        &self,
        layer: usize,
        group: Group,
        s: &Scratch,
        input: usize,
        output: usize,
        lengths: &[usize],
    ) -> Result<()> {
        let Some(adapter) = &self.lora else {
            return Ok(());
        };
        let work = s.lora.as_ref().context("missing language LoRA work")?;
        // SAFETY: caller passes the validated projection's input/output ranges in scratch;
        // per-sequence row counts are bounded by config, and work/weights belong to this Model.
        unsafe {
            adapter.apply(
                layer,
                group,
                s.at(input),
                s.at(output),
                lengths,
                work,
                self.gemm,
                self.stream,
            )
        }
    }

    /// Keep per-prompt output/down projections, or opt into packing with their algorithms.
    fn gemm_sequences(
        &self,
        s: &Scratch,
        x: usize,
        w: &Tensor,
        y: usize,
        lengths: &[usize],
    ) -> Result<()> {
        let (n, k) = (w.shape[0], w.shape[1]);
        if let Some(packed) = self.packed_gemm
            && self.reference.is_none()
            && self.lora.is_none()
        {
            // SAFETY: x/y contain all validated packed rows, w is [n,k], and
            // lengths is read synchronously by the host planner. Device work
            // remains on this serialized model's stream and workspace.
            return unsafe {
                check(
                    packed(
                        self.gemm,
                        s.at(x),
                        w.ptr,
                        s.at(y),
                        lengths.as_ptr(),
                        lengths.len(),
                        n as i32,
                        k as i32,
                        self.stream,
                    ),
                    "packed projection",
                )
            };
        }
        let mut offset = 0;
        for &length in lengths {
            self.gemm(s, x + offset * k * BF16, w, y + offset * n * BF16, length)?;
            offset += length;
        }
        Ok(())
    }

    fn gdn_prefill(&self) -> cuda::GdnPrefill {
        let api = cuda::api();
        if let Some(reference) = self.reference {
            reference.gdn_prefill
        } else if self.packed_gemm.is_some() {
            api.wide_gdn().unwrap_or(api.cs1_gdn_prefill)
        } else {
            api.cs1_gdn_prefill
        }
    }

    fn convolution(&self) -> cuda::GdnConv {
        let api = cuda::api();
        if self.packed_gemm.is_some() {
            api.vector_conv().unwrap_or(api.cs1_gdn_conv)
        } else {
            api.cs1_gdn_conv
        }
    }

    /// Finish queued work before releasing external execution admission.
    pub fn synchronize(&self) -> Result<()> {
        cuda::set_device(0)?;
        cuda::synchronize(self.stream)
    }

    fn prepare_scratch(&mut self, t: usize) -> Result<()> {
        cuda::set_device(0)?;
        // Graphs retain scratch addresses. Reserve the bounded packing window on
        // the first opt-in text execution so later shapes within it do not discard
        // already-captured graphs. Larger supported inputs still grow normally.
        let capacity = if self.graph_enabled && self.packed_gemm.is_some() {
            t.max(4096)
        } else {
            t
        }
        .next_multiple_of(1024);
        if self.scratch.as_ref().is_none_or(|s| capacity > s.cap) {
            cuda::synchronize(self.stream)?;
            if !self.graphs.is_empty() || !self.mm_graphs.is_empty() {
                self.graph_stats.invalidations += 1;
            }
            self.graphs.clear();
            self.mm_graphs.clear();
            self.scratch = None;
            self.scratch = Some(Scratch::new(
                &self.cfg,
                capacity,
                self.stream,
                self.lora.as_ref(),
                self.reference,
                self.fp32_gdn.as_ref(),
                self.batched_gdn.as_ref(),
            )?);
        }
        Ok(())
    }

    /// The final-norm hidden state at the last position, as float32.
    pub fn forward(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        Ok(self.forward_batch(&[ids])?.pop().unwrap())
    }

    /// One prompt whose full-attention layers read the whole prompt in both
    /// directions. Linear attention stays causal. CUDA graphs, LoRA and the
    /// reference numerics are refused: those paths are causal-only.
    pub fn forward_prompt_bidirectional(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        ensure!(
            !self.graph_enabled && self.reference.is_none() && self.lora.is_none(),
            "prompt-bidirectional prefill is eager, merged and uses the default kernels"
        );
        ensure!(
            !ids.is_empty() && ids.len() <= self.cfg.max_positions,
            "empty or oversized prompt"
        );
        ensure!(
            ids.iter().all(|&id| (id as usize) < self.embed.shape[0]),
            "token id outside the vocabulary"
        );
        self.prepare_scratch(ids.len())?;
        let s = self.scratch.as_ref().unwrap();
        self.embed_tokens(s, ids)?;
        self.run(s, &[ids.len()], false, true)?;
        self.last_hidden(s, ids.len())
    }

    /// Fail at load when this library cannot run an Instant prompt.
    pub fn require_prompt_bidirectional(&self) -> Result<()> {
        ensure!(
            !self.graph_enabled && self.reference.is_none() && self.lora.is_none(),
            "prompt-bidirectional prefill is eager, merged and uses the default kernels"
        );
        cuda::api().attention_gated_bidirectional()?;
        Ok(())
    }

    /// Pack independent text prompts for input and gate/up GEMMs.
    /// Mixers reset at each boundary; final-position hidden states retain input order.
    pub fn forward_batch(&mut self, inputs: &[&[u32]]) -> Result<Vec<Vec<f32>>> {
        ensure!(!inputs.is_empty(), "empty batch");
        ensure!(
            inputs
                .iter()
                .all(|ids| !ids.is_empty() && ids.len() <= self.cfg.max_positions),
            "empty or oversized prompt"
        );
        ensure!(
            inputs
                .iter()
                .flat_map(|ids| ids.iter())
                .all(|&id| (id as usize) < self.embed.shape[0]),
            "token id outside the vocabulary"
        );
        let lengths: Vec<usize> = inputs.iter().map(|ids| ids.len()).collect();
        let t = lengths
            .iter()
            .try_fold(0usize, |total, &length| total.checked_add(length))
            .context("packed token count overflow")?;
        ensure!(
            t <= i32::MAX as usize,
            "packed token count exceeds the CUDA layout"
        );
        self.prepare_scratch(t)?;
        let s = self.scratch.as_ref().unwrap();
        let ids: Vec<u32> = inputs.iter().flat_map(|ids| ids.iter().copied()).collect();
        self.embed_tokens(s, &ids)?;
        if self.graph_enabled {
            if let Some((_, graph)) = self.graphs.iter().find(|(shape, _)| *shape == lengths) {
                graph.launch(self.stream)?;
                self.graph_stats.replays += 1;
                if std::env::var("CUA_S1_GRAPH_TRACE").as_deref() == Ok("1") {
                    eprintln!("CUA_S1_GRAPH mode=text event=replay tokens={t}");
                }
            } else {
                // Warm plans and keep the eager result: run() advances the residual
                // in place, so replaying on this cache miss would advance it twice.
                self.run(s, &lengths, false, false)?;
                cuda::synchronize(self.stream)?;
                let captured =
                    cuda::Graph::capture(self.stream, || self.run(s, &lengths, false, false));
                self.cache_graph(&lengths, false, captured)?;
            }
        } else {
            self.run(s, &lengths, false, false)?;
        }
        let s = self.scratch.as_ref().unwrap();
        let mut end = 0;
        lengths
            .iter()
            .map(|&length| {
                end += length;
                self.last_hidden(s, end)
            })
            .collect()
    }

    /// Evaluate candidates sharing an exact, chunk-aligned token prefix once.
    /// Retain each complete forward's algorithms, sharing only within compatible
    /// candidate groups. Return None before computation when no group can share.
    /// Every request recaptures all numeric state; only allocation capacity persists.
    pub fn try_forward_shared_prefix(
        &mut self,
        inputs: &[&[u32]],
        prefix_len: usize,
    ) -> Result<Option<SharedPrefixOutput>> {
        ensure!(
            self.lora.is_none(),
            "shared text prefix requires merged weights"
        );
        ensure!(
            self.reference.is_none() && self.fp32_gdn.is_none(),
            "shared text prefix requires standard BF16 numerics"
        );
        ensure!(
            prefix_len >= 64 && prefix_len.is_multiple_of(64),
            "shared prefix must be a positive chunk multiple"
        );
        ensure!(inputs.len() >= 2, "shared prefix needs multiple candidates");
        ensure!(
            inputs
                .iter()
                .all(|ids| ids.len() > prefix_len && ids.len() <= self.cfg.max_positions),
            "empty suffix or oversized prompt"
        );
        ensure!(
            inputs
                .iter()
                .flat_map(|ids| ids.iter())
                .all(|&id| (id as usize) < self.embed.shape[0]),
            "token id outside the vocabulary"
        );
        ensure!(
            inputs
                .iter()
                .all(|ids| ids[..prefix_len] == inputs[0][..prefix_len]),
            "candidate prefixes differ"
        );
        cuda::set_device(0)?;
        let Some(reuse) = cuda::api().reuse_gemm() else {
            return Ok(None);
        };
        let original_rows: Vec<usize> = inputs.iter().map(|ids| ids.len()).collect();
        let mut shapes = std::collections::BTreeSet::new();
        for layer in &self.layers {
            let (input, output) = match &layer.mixer {
                Mixer::Linear(la) => (&la.in_proj, &la.out),
                Mixer::Full(fa) => (&fa.qkv, &fa.o),
            };
            for weight in [input, output, &layer.gate_up, &layer.down] {
                shapes.insert((weight.shape[0], weight.shape[1]));
            }
        }
        let gemm = self.gemm;
        let Some(groups) = compatible_prefix_groups(inputs.len(), |indices| {
            let originals: Vec<usize> = indices.iter().map(|&i| original_rows[i]).collect();
            let mut windows: Vec<usize> = originals.iter().map(|&n| n - prefix_len).collect();
            windows.push(prefix_len);
            for &(n, k) in &shapes {
                // SAFETY: host arrays stay alive for the synchronous capability
                // check. No device output or prefix state is produced here.
                let rc = unsafe {
                    (reuse.check)(
                        gemm,
                        originals.as_ptr(),
                        originals.len(),
                        windows.as_ptr(),
                        windows.len(),
                        n as i32,
                        k as i32,
                    )
                };
                if rc == -1 {
                    return Ok(false);
                }
                check(rc, "prefix projection compatibility")?;
            }
            Ok(true)
        })?
        else {
            return Ok(None);
        };
        let (capacity, mut state) = match self.text_prefix_workspace.take() {
            Some((capacity, state)) if capacity >= prefix_len => (capacity, state),
            old => {
                // Release the smaller allocation before growth, bounding peak memory.
                drop(old);
                (prefix_len, self.alloc_prefix(prefix_len)?)
            }
        };
        state.len = prefix_len;
        state.initialized = false;
        let result = (|| {
            self.prepare_scratch(inputs.iter().map(|ids| ids.len()).max().unwrap())?;
            let mut hidden = vec![Vec::new(); inputs.len()];
            let saved_tokens = shared_prefix_savings(&groups, prefix_len);
            for group in groups {
                if group.len() == 1 {
                    hidden[group[0]] = self.forward(inputs[group[0]])?;
                    continue;
                }
                let ops = cuda::api().shared_suffix();
                let mut segments = Vec::new();
                let mut required = inputs.iter().map(|ids| ids.len()).max().unwrap();
                for indices in suffix_segments(&group, &original_rows, prefix_len) {
                    let rows: usize = indices.iter().map(|&i| original_rows[i] - prefix_len).sum();
                    let original = [original_rows[indices[0]]];
                    let target = [rows];
                    let mut packed = ops.is_some() && indices.len() > 1;
                    if packed {
                        for &(n, k) in &shapes {
                            // SAFETY: synchronous shape-only check, before prefix capture.
                            let rc = unsafe {
                                (reuse.check)(
                                    gemm,
                                    original.as_ptr(),
                                    1,
                                    target.as_ptr(),
                                    1,
                                    n as i32,
                                    k as i32,
                                )
                            };
                            if rc == -1 {
                                packed = false;
                                break;
                            }
                            check(rc, "packed suffix compatibility")?;
                        }
                    }
                    if packed {
                        required = required.max(rows);
                    }
                    segments.push((indices, packed));
                }
                self.prepare_scratch(required)?;
                // Different groups can have different original algorithms. Recompute
                // every captured value; buffers never make a later request's state valid.
                state.initialized = false;
                let projection = Some((reuse, original_rows[group[0]]));
                let s = self.scratch.as_ref().unwrap();
                self.embed_tokens(s, &inputs[group[0]][..prefix_len])?;
                self.run_window(
                    s,
                    0,
                    prefix_len,
                    false,
                    Some(&mut state),
                    None,
                    self.gemm,
                    projection,
                )?;
                self.synchronize()?;
                state.initialized = true;
                for (indices, packed) in segments {
                    if packed {
                        let lengths: Vec<usize> = indices
                            .iter()
                            .map(|&i| inputs[i].len() - prefix_len)
                            .collect();
                        let ids: Vec<u32> = indices
                            .iter()
                            .flat_map(|&i| inputs[i][prefix_len..].iter().copied())
                            .collect();
                        self.embed_tokens(s, &ids)?;
                        self.run_with_suffix(
                            s,
                            &lengths,
                            false,
                            false,
                            Some(&SuffixBatch {
                                prefix: &state,
                                ops: ops.unwrap(),
                                gemm: reuse,
                                original_rows: original_rows[indices[0]],
                            }),
                        )?;
                        let mut end = 0;
                        for (index, length) in indices.into_iter().zip(lengths) {
                            end += length;
                            hidden[index] = self.last_hidden(s, end)?;
                        }
                    } else {
                        for index in indices {
                            let ids = inputs[index];
                            self.embed_tokens_at(s, &ids[prefix_len..], prefix_len)?;
                            self.run_window(
                                s,
                                prefix_len,
                                ids.len(),
                                false,
                                None,
                                Some(&state),
                                self.gemm,
                                projection,
                            )?;
                            hidden[index] = self.last_hidden(s, ids.len())?;
                        }
                    }
                }
            }
            Ok(SharedPrefixOutput {
                hidden,
                saved_tokens,
            })
        })();
        // A failed forward may have queued work. Finish it before retaining buffers;
        // on a synchronization error, drop them instead of recycling their state.
        self.synchronize()?;
        state.initialized = false;
        self.text_prefix_workspace = Some((capacity, state));
        result.map(Some)
    }

    /// Prefill one unpadded prompt with already-adapted BF16 image embeddings and
    /// explicit `[3, 1, sequence]` T/H/W positions. No vision tower runs here.
    /// Load a checkpoint with the matching multimodal language adapter merged.
    pub fn forward_multimodal(&mut self, input: &MultimodalInput<'_>) -> Result<Vec<f32>> {
        let image_token = self
            .cfg
            .image_token_id
            .context("checkpoint has no image_token_id")?;
        input.validate(
            self.cfg.hidden,
            self.embed.shape[0],
            image_token,
            self.cfg.max_positions,
        )?;
        let t = input.token_ids.len();
        self.prepare_scratch(t)?;
        let s = self.scratch.as_ref().unwrap();
        self.upload_positions(s, input.position_ids)?;
        self.embed_tokens(s, input.token_ids)?;
        self.overwrite_image_rows(s, input, 0)?;
        if self.graph_enabled {
            if let Some((_, graph)) = self
                .mm_graphs
                .iter()
                .find(|(shape, _)| shape.as_slice() == [t])
            {
                graph.launch(self.stream)?;
                self.graph_stats.replays += 1;
                if std::env::var("CUA_S1_GRAPH_TRACE").as_deref() == Ok("1") {
                    eprintln!("CUA_S1_GRAPH mode=multimodal event=replay tokens={t}");
                }
            } else {
                // Uploads and embedding are outside capture. Keep the warmed eager
                // result; launching now would advance the residual a second time.
                self.run(s, &[t], true, false)?;
                cuda::synchronize(self.stream)?;
                let captured = cuda::Graph::capture(self.stream, || self.run(s, &[t], true, false));
                self.cache_graph(&[t], true, captured)?;
            }
        } else {
            self.run(s, &[t], true, false)?;
        }
        self.last_hidden(self.scratch.as_ref().unwrap(), t)
    }

    /// Both modes retain the eager miss result; capture records without executing.
    fn cache_graph(
        &mut self,
        lengths: &[usize],
        multimodal: bool,
        captured: Result<cuda::Graph>,
    ) -> Result<()> {
        let mode = if multimodal { "multimodal" } else { "text" };
        match captured {
            Ok(graph) => {
                self.graph_stats.captures += 1;
                // Candidate packing produces many length vectors. Keep a bounded
                // larger text cache for opt-in packed consumers; other modes retain
                // their existing budget. Captures store operations, never input data.
                let capacity = if !multimodal && self.packed_gemm.is_some() {
                    256
                } else {
                    64
                };
                let cache = if multimodal {
                    &mut self.mm_graphs
                } else {
                    &mut self.graphs
                };
                if cache.len() == capacity {
                    cuda::synchronize(self.stream)?;
                    cache.pop_front();
                }
                cache.push_back((lengths.to_vec(), graph));
                if std::env::var("CUA_S1_GRAPH_TRACE").as_deref() == Ok("1") {
                    let t: usize = lengths.iter().sum();
                    eprintln!("CUA_S1_GRAPH mode={mode} event=capture tokens={t}");
                }
            }
            Err(error) => {
                eprintln!(
                    "CUDA Graph capture failed in {mode} mode; using eager execution: {error:#}"
                );
                self.graph_stats.fallbacks += 1;
                self.graph_enabled = false;
                self.graphs.clear();
                self.mm_graphs.clear();
            }
        }
        Ok(())
    }

    /// `forward_multimodal` with the fixed-algorithm GEMMs of `forward_fixed`, reading the
    /// final-norm hidden state at each of `rows` and, for each `(row, token)` in `targets`,
    /// the log-probability of `token` after `row` under the tied output embedding:
    /// bfloat16 logits over the vocabulary, converted to float32, with a float64 sum in
    /// the log-softmax. Each result is the same however many rows or targets the call
    /// reads. `readout_window` with neither a prefix nor a capture.
    pub fn forward_multimodal_readout(
        &mut self,
        input: &MultimodalInput<'_>,
        rows: &[usize],
        targets: &[(usize, u32)],
    ) -> Result<Readout> {
        self.readout_window(input, None, None, rows, targets)
    }

    /// One window of a prompt with the fixed-algorithm GEMMs, continuing `prefix` (the
    /// prompt's rows `[0, prefix.len)`, or nothing) and optionally capturing the state
    /// after it into `capture`, which must hold the prompt so far. `window` holds the
    /// window's own rows: their token ids, the indices of their image placeholders
    /// within the window, those rows' image embeddings and the rows' rotary positions.
    /// `rows` and `targets` index the whole prompt and must lie in the window; they are
    /// read as by `forward_multimodal_readout`. When the prefix was itself captured here,
    /// and every boundary is a multiple of 64 tokens, every result equals that of one
    /// `forward_multimodal_readout` pass over the whole prompt.
    ///
    /// Work is queued on the model's stream; `capture` is ready for continuations queued
    /// after this call on the same model.
    pub fn readout_window(
        &mut self,
        window: &MultimodalInput<'_>,
        prefix: Option<&PrefixState>,
        capture: Option<&mut PrefixState>,
        rows: &[usize],
        targets: &[(usize, u32)],
    ) -> Result<Readout> {
        self.ensure_fixed_profile()?;
        let result = self.run_readout_window(window, prefix, capture, rows, targets);
        if result.is_err() {
            // Work already queued may still use the buffers.
            let _ = cuda::synchronize(self.stream);
        }
        result
    }

    fn run_readout_window(
        &mut self,
        window: &MultimodalInput<'_>,
        prefix: Option<&PrefixState>,
        mut capture: Option<&mut PrefixState>,
        rows: &[usize],
        targets: &[(usize, u32)],
    ) -> Result<Readout> {
        let image_token = self
            .cfg
            .image_token_id
            .context("checkpoint has no image_token_id")?;
        let vocab = self.embed.shape[0];
        window.validate(self.cfg.hidden, vocab, image_token, self.cfg.max_positions)?;
        if let Some(pre) = prefix {
            ensure!(
                Arc::ptr_eq(&self.prefix_owner, &pre.owner),
                "prefix state belongs to a different model instance"
            );
            ensure!(pre.initialized, "prefix state has not completed capture");
        }
        let qb = prefix.map_or(0, |pre| pre.len);
        let tend = qb + window.token_ids.len();
        ensure!(
            tend <= self.cfg.max_positions,
            "prompt exceeds the configured maximum length"
        );
        if let Some(cap) = capture.as_deref_mut() {
            ensure!(
                Arc::ptr_eq(&self.prefix_owner, &cap.owner),
                "prefix state belongs to a different model instance"
            );
            ensure!(
                tend.is_multiple_of(64) && tend <= cap.cap,
                "a capture must end on a multiple of 64 within its state"
            );
        }
        ensure!(
            rows.iter().all(|&r| (qb..tend).contains(&r))
                && targets
                    .iter()
                    .all(|&(r, token)| (qb..tend).contains(&r) && (token as usize) < vocab),
            "readout row or token outside the window"
        );
        self.prepare_scratch(tend)?;
        let gemm = self.fixed_gemm()?;
        let s = self.scratch.as_ref().unwrap();
        self.upload_positions(s, window.position_ids)?;
        self.embed_tokens_at(s, window.token_ids, qb)?;
        self.overwrite_image_rows(s, window, qb)?;
        if let Some(cap) = capture.as_deref_mut() {
            cap.initialized = false;
            cap.len = tend;
        }
        self.run_window(
            s,
            qb,
            tend,
            true,
            capture.as_deref_mut(),
            prefix,
            gemm,
            None,
        )?;
        if let Some(cap) = capture {
            cap.initialized = true;
        }
        let hidden = rows
            .iter()
            .map(|&r| self.hidden_row(s, r))
            .collect::<Result<Vec<_>>>()?;
        let logprobs = self.vocabulary_logprobs(targets, gemm)?;
        Ok(Readout { hidden, logprobs })
    }

    /// The log-probabilities of `targets` from the final-norm hidden states in the
    /// scratch, in chunks of `VOCAB_ROWS` rows.
    fn vocabulary_logprobs(
        &mut self,
        targets: &[(usize, u32)],
        gemm: *mut c_void,
    ) -> Result<Vec<f32>> {
        let mut logprobs = Vec::with_capacity(targets.len());
        if targets.is_empty() {
            return Ok(logprobs);
        }
        let vocab = self.embed.shape[0];
        let hb = self.cfg.hidden * BF16;
        // padded so that each logits row starts on a 16-byte boundary
        let ld = vocab.next_multiple_of(8);
        let row = ld * BF16;
        if self.vocab_buffers.is_none() {
            self.vocab_buffers = Some((
                DeviceBuffer::new(VOCAB_ROWS * hb)?,
                DeviceBuffer::new(VOCAB_ROWS * row)?,
                vec![0u8; VOCAB_ROWS * row],
            ));
        }
        let s = self.scratch.as_ref().unwrap();
        let (picked, logits, host) = self.vocab_buffers.as_mut().unwrap();
        for chunk in targets.chunks(VOCAB_ROWS) {
            let m = chunk.len();
            // SAFETY: rows lie inside the scratch's final hidden states; picked holds
            // VOCAB_ROWS >= m rows, logits VOCAB_ROWS rows of ld, and the tied embedding
            // is [vocab, hidden].
            unsafe {
                // one copy per run of consecutive rows
                let mut i = 0;
                while i < m {
                    let mut j = i + 1;
                    while j < m && chunk[j].0 == chunk[j - 1].0 + 1 {
                        j += 1;
                    }
                    cuda::copy_dd(
                        picked.at(i * hb),
                        s.at(s.x + chunk[i].0 * hb),
                        (j - i) * hb,
                        self.stream,
                    )?;
                    i = j;
                }
                check(
                    (cuda::api().cs1_gemm)(
                        gemm,
                        picked.at(0),
                        self.embed.ptr,
                        logits.at(0),
                        m as i32,
                        vocab as i32,
                        self.cfg.hidden as i32,
                        ld as i32,
                        self.stream,
                    ),
                    "vocabulary gemm",
                )?;
                cuda::download(&mut host[..m * row], logits.at(0), self.stream)?;
            }
            logprobs.extend(chunk.iter().enumerate().map(|(i, &(_, token))| {
                token_logprob(&host[i * row..(i + 1) * row], vocab, token)
            }));
        }
        Ok(logprobs)
    }

    /// Eager multimodal execution using borrowed device-resident image rows.
    /// The producer must finish writes before entry. All queued consumer work is
    /// completed on every exit path, before the external allocation can be reused.
    pub fn forward_multimodal_device(
        &mut self,
        input: &DeviceMultimodalInput<'_>,
    ) -> Result<Vec<f32>> {
        let result = (|| {
            cuda::set_device(0)?;
            let image_token = self
                .cfg
                .image_token_id
                .context("checkpoint has no image_token_id")?;
            input.validate(
                self.cfg.hidden,
                self.embed.shape[0],
                image_token,
                self.cfg.max_positions,
                0,
            )?;
            input.image_embeddings.validate_finite(self.stream)?;
            let t = input.token_ids.len();
            self.prepare_scratch(t)?;
            let s = self.scratch.as_ref().unwrap();
            self.upload_positions(s, input.position_ids)?;
            self.embed_tokens(s, input.token_ids)?;
            let indices = input.image_token_indices;
            let row_bytes = self.cfg.hidden * BF16;
            let mut begin = 0;
            while begin < indices.len() {
                let mut end = begin + 1;
                while end < indices.len() && indices[end] == indices[end - 1] + 1 {
                    end += 1;
                }
                // SAFETY: metadata validates source rows, device and allocation
                // extent and indices exactly match placeholders inside s.res.
                unsafe {
                    cuda::copy_dd(
                        s.at(s.res + indices[begin] * row_bytes),
                        input.image_embeddings.at(begin * row_bytes),
                        (end - begin) * row_bytes,
                        self.stream,
                    )?;
                }
                begin = end;
            }
            self.run(s, &[t], true, false)?;
            self.last_hidden(s, t)
        })();
        let synced = self.synchronize();
        synced?;
        result
    }

    fn upload_positions(&self, s: &Scratch, positions: [&[i64]; 3]) -> Result<()> {
        let (cos, sin) = rotary_tables(
            positions,
            self.cfg.rotary_half,
            self.cfg.rope_theta,
            self.cfg.mrope_section,
        );
        // SAFETY: tables contain at most s.cap rows of rotary_half BF16 values.
        unsafe {
            cuda::upload(s.at(s.custom_cos), &cos, self.stream)?;
            cuda::upload(s.at(s.custom_sin), &sin, self.stream)?;
        }
        Ok(())
    }

    /// Overwrite the residual rows of validated image placeholders with the adapted
    /// image embeddings. `qb` is the absolute row base of the passed slice inside
    /// the residual buffer (0 for a full prompt, the cached prefix length for a
    /// continuation — indices are local to the slice in both cases).
    fn overwrite_image_rows(
        &self,
        s: &Scratch,
        input: &MultimodalInput<'_>,
        qb: usize,
    ) -> Result<()> {
        let bytes: Vec<u8> = input
            .image_embeddings
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        // Coalesce adjacent placeholders. Text rows remain those of embed_tokens.
        let indices = input.image_token_indices;
        let mut begin = 0;
        while begin < indices.len() {
            let mut end = begin + 1;
            while end < indices.len() && indices[end] == indices[end - 1] + 1 {
                end += 1;
            }
            let row_bytes = self.cfg.hidden * BF16;
            // SAFETY: validated indices lie in the residual buffer, and features
            // contain exactly one hidden-size BF16 row per placeholder.
            unsafe {
                cuda::upload(
                    s.at(s.res + (qb + indices[begin]) * row_bytes),
                    &bytes[begin * row_bytes..end * row_bytes],
                    self.stream,
                )?;
            }
            begin = end;
        }
        Ok(())
    }

    /// Allocate an uninitialized prefix of `len` tokens (a multiple of 64).
    /// Capture must finish successfully on this model before continuation.
    pub fn alloc_prefix(&self, len: usize) -> Result<PrefixState> {
        ensure!(
            self.fp32_gdn.is_none(),
            "FP32 GDN does not support prefix caching"
        );
        ensure!(
            self.reference.is_none(),
            "JEMM reference numerics do not support prefix caching"
        );
        ensure!(
            len >= 64 && len.is_multiple_of(64),
            "cached prefix length must be a positive multiple of 64"
        );
        ensure!(
            len <= self.cfg.max_positions,
            "cached prefix exceeds the configured maximum length"
        );
        let kvrow = self.cfg.kv_heads * self.cfg.head_dim * BF16;
        let w = Widths::of(&self.cfg);
        let nfull = self.cfg.full_attention.iter().filter(|&&f| f).count();
        let ngdn = self.cfg.full_attention.len() - nfull;
        let mut bytes = 0usize;
        let mut attn_kv = Vec::with_capacity(nfull);
        for _ in 0..nfull {
            let k = DeviceBuffer::new(len * kvrow)?;
            let v = DeviceBuffer::new(len * kvrow)?;
            bytes += 2 * len * kvrow;
            attn_kv.push((k, v));
        }
        let state_floats = self.cfg.lin_v_heads * self.cfg.lin_k_dim * self.cfg.lin_v_dim;
        let tail_bytes = 3 * w.gdn_in * BF16;
        let mut gdn_state = Vec::with_capacity(ngdn);
        let mut conv_tail = Vec::with_capacity(ngdn);
        for _ in 0..ngdn {
            gdn_state.push(DeviceBuffer::new(state_floats * F32)?);
            conv_tail.push(DeviceBuffer::new(tail_bytes)?);
            bytes += state_floats * F32 + tail_bytes;
        }
        Ok(PrefixState {
            owner: Arc::clone(&self.prefix_owner),
            initialized: false,
            len,
            cap: len,
            attn_kv,
            gdn_state,
            conv_tail,
            bytes,
        })
    }

    /// Run the rows `[0, state.len)` of one prompt, collecting the per-layer
    /// prefix state (post-prep K/V columns, float32 GDN states, conv tails) into
    /// `state`. Synchronize before marking the state ready for continuation.
    /// The hidden states of these rows are computed but not returned.
    pub fn forward_multimodal_capture(
        &mut self,
        input: &MultimodalInput<'_>,
        state: &mut PrefixState,
    ) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.prefix_owner, &state.owner),
            "prefix state belongs to a different model instance"
        );
        let t = input.token_ids.len();
        ensure!(
            t == state.len,
            "capture input must cover the cached prefix exactly"
        );
        state.initialized = false;
        let image_token = self
            .cfg
            .image_token_id
            .context("checkpoint has no image_token_id")?;
        input.validate(
            self.cfg.hidden,
            self.embed.shape[0],
            image_token,
            self.cfg.max_positions,
        )?;
        self.prepare_scratch(t)?;
        let s = self.scratch.as_ref().unwrap();
        self.upload_positions(s, input.position_ids)?;
        self.embed_tokens(s, input.token_ids)?;
        self.overwrite_image_rows(s, input, 0)?;
        self.run_window(s, 0, t, true, Some(state), None, self.gemm, None)?;
        self.synchronize()?;
        state.initialized = true;
        Ok(())
    }

    /// Run rows `[prefix.len, prefix.len + input.len)` of one prompt seeded from a
    /// captured prefix; returns the final-norm hidden state at the last position.
    pub fn forward_multimodal_continue(
        &mut self,
        input: &MultimodalInput<'_>,
        prefix: &PrefixState,
    ) -> Result<Vec<f32>> {
        ensure!(
            Arc::ptr_eq(&self.prefix_owner, &prefix.owner),
            "prefix state belongs to a different model instance"
        );
        ensure!(prefix.initialized, "prefix state has not completed capture");
        let image_token = self
            .cfg
            .image_token_id
            .context("checkpoint has no image_token_id")?;
        input.validate(
            self.cfg.hidden,
            self.embed.shape[0],
            image_token,
            self.cfg.max_positions,
        )?;
        let rows = input.token_ids.len();
        let tend = prefix
            .len
            .checked_add(rows)
            .context("cached prompt length overflow")?;
        ensure!(
            tend <= self.cfg.max_positions,
            "cached prompt exceeds the configured maximum length"
        );
        self.prepare_scratch(tend)?;
        let s = self.scratch.as_ref().unwrap();
        self.upload_positions(s, input.position_ids)?;
        self.embed_tokens_at(s, input.token_ids, prefix.len)?;
        self.overwrite_image_rows(s, input, prefix.len)?;
        self.run_window(
            s,
            prefix.len,
            tend,
            true,
            None,
            Some(prefix),
            self.gemm,
            None,
        )?;
        self.last_hidden(s, tend)
    }

    /// `forward` with the GEMM algorithms of `forward_shared`, one per weight shape, run
    /// eagerly: the result that `forward_shared` reproduces for this prompt.
    pub fn forward_fixed(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        self.ensure_fixed_profile()?;
        let result = self.run_fixed(ids);
        if result.is_err() {
            // Work already queued may still use the buffers.
            let _ = cuda::synchronize(self.stream);
        }
        result
    }

    fn run_fixed(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        let t = ids.len();
        ensure!(
            t > 0 && t <= self.cfg.max_positions,
            "empty or oversized prompt"
        );
        ensure!(
            ids.iter().all(|&i| (i as usize) < self.embed.shape[0]),
            "token id outside the vocabulary"
        );
        self.prepare_scratch(t)?;
        let gemm = self.fixed_gemm()?;
        let s = self.scratch.as_ref().unwrap();
        self.embed_tokens(s, ids)?;
        self.run_window(s, 0, t, false, None, None, gemm, None)?;
        self.last_hidden(s, t)
    }

    /// The final-norm hidden state at the last position of every branch, grouped and
    /// ordered as in `prompts`. Each prefix runs once, up to its last multiple of 64
    /// tokens, capturing its state as `forward_multimodal_capture` does, and each branch
    /// continues from the state after its prefixes, with its positions and causal mask
    /// continuing theirs. A prefix's remaining tokens run with every group or branch
    /// after it.
    ///
    /// On a GPU where the fixed GEMM algorithms keep rows independent of M (see
    /// cs1_gemm_create_fixed), each result equals `forward_fixed` on the full prompt bit
    /// for bit, however the prompts are grouped. It differs from `forward`, which picks
    /// GEMM algorithms per row count, by rounding.
    ///
    /// Runs eagerly. The prefix states are overwritten by the next call, and the stream
    /// is synchronized before this returns, also on error.
    pub fn forward_shared(&mut self, prompts: &SharedPrompts) -> Result<Vec<Vec<Vec<f32>>>> {
        self.ensure_fixed_profile()?;
        let result = self.run_shared(prompts);
        if result.is_err() {
            // Work already queued may still use the buffers.
            let _ = cuda::synchronize(self.stream);
        }
        result
    }

    fn run_shared(&mut self, prompts: &SharedPrompts) -> Result<Vec<Vec<Vec<f32>>>> {
        let vocab = self.embed.shape[0];
        ensure!(!prompts.groups.is_empty(), "no prompts");
        let mut longest = 0;
        for g in &prompts.groups {
            ensure!(!g.branches.is_empty(), "a prompt group without branches");
            for b in &g.branches {
                ensure!(!b.is_empty(), "an empty branch");
                longest = longest.max(prompts.prefix.len() + g.prefix.len() + b.len());
            }
        }
        ensure!(longest <= self.cfg.max_positions, "oversized prompt");
        let all = prompts
            .groups
            .iter()
            .flat_map(|g| g.prefix.iter().chain(g.branches.iter().copied().flatten()));
        ensure!(
            prompts
                .prefix
                .iter()
                .chain(all)
                .all(|&i| (i as usize) < vocab),
            "token id outside the vocabulary"
        );
        let runs = runs(prompts);
        let p = runs.prefix.len();
        // the longest group prefix that runs on its own, if any
        let q = runs
            .groups
            .iter()
            .filter(|g| !g.prefix.is_empty())
            .map(|g| p + g.prefix.len())
            .max()
            .unwrap_or(0);
        self.prepare_scratch(longest)?;
        let gemm = self.fixed_gemm()?;
        let [mut request, mut group] = std::mem::take(&mut self.shared);
        let result = (|| {
            let request = self.shared_state(&mut request, p)?;
            let group = self.shared_state(&mut group, q)?;
            let s = self.scratch.as_ref().unwrap();
            let request = match request {
                Some(state) => {
                    self.embed_tokens(s, &runs.prefix)?;
                    self.run_window(s, 0, p, false, Some(&mut *state), None, gemm, None)?;
                    Some(&*state)
                }
                None => None,
            };
            let mut group = group;
            let mut hidden = Vec::with_capacity(runs.groups.len());
            for run in &runs.groups {
                let start = p + run.prefix.len();
                let from = if run.prefix.is_empty() {
                    request
                } else {
                    let state = group.as_deref_mut().unwrap();
                    state.len = start;
                    self.embed_tokens_at(s, &run.prefix, p)?;
                    self.run_window(s, p, start, false, Some(&mut *state), request, gemm, None)?;
                    Some(&*state)
                };
                let rows = run
                    .branches
                    .iter()
                    .map(|b| {
                        let end = start + b.len();
                        self.embed_tokens_at(s, b, start)?;
                        self.run_window(s, start, end, false, None, from, gemm, None)?;
                        self.last_hidden(s, end)
                    })
                    .collect::<Result<Vec<_>>>()?;
                hidden.push(rows);
            }
            Ok(hidden)
        })();
        self.shared = [request, group];
        result
    }

    /// `slot`'s state resized to hold `len` tokens, none for an empty prefix.
    fn shared_state<'s>(
        &self,
        slot: &'s mut Option<PrefixState>,
        len: usize,
    ) -> Result<Option<&'s mut PrefixState>> {
        if len == 0 {
            return Ok(None);
        }
        if slot.as_ref().is_none_or(|state| state.cap < len) {
            *slot = None;
            let cap = len
                .next_multiple_of(1024)
                .min(self.cfg.max_positions - self.cfg.max_positions % GDN_CHUNK);
            *slot = Some(self.alloc_prefix(cap.max(len))?);
        }
        let state = slot.as_mut().unwrap();
        state.len = len;
        Ok(Some(state))
    }

    /// `forward_fixed`, `forward_shared` and `readout_window` cover merged weights with
    /// the default numerics. Refuse the other profiles before any allocation or CUDA
    /// call.
    fn ensure_fixed_profile(&self) -> Result<()> {
        ensure!(
            self.lora.is_none() && self.reference.is_none() && self.fp32_gdn.is_none(),
            "fixed-algorithm GEMMs and shared prefixes support only merged weights with the default numerics"
        );
        Ok(())
    }

    /// The GEMM handle with one algorithm per weight shape, created on first use.
    fn fixed_gemm(&mut self) -> Result<*mut c_void> {
        if self.matched_ref > 0 {
            return Ok(self.matched_handles[&self.matched_ref]);
        }

        if self.fixed_gemm.is_null() {
            // SAFETY: plain allocation; checked for null below.
            self.fixed_gemm =
                unsafe { (cuda::api().cs1_gemm_create_fixed)(GEMM_WORKSPACE, FIXED_GEMM_M) };
            ensure!(!self.fixed_gemm.is_null(), "cuBLASLt setup failed");
        }
        Ok(self.fixed_gemm)
    }

    fn embed_tokens(&self, s: &Scratch, ids: &[u32]) -> Result<()> {
        self.embed_tokens_at(s, ids, 0)
    }

    /// Embed `ids` into the residual buffer rows starting at absolute row `qb`.
    fn embed_tokens_at(&self, s: &Scratch, ids: &[u32], qb: usize) -> Result<()> {
        let ids32: Vec<u8> = ids.iter().flat_map(|&i| (i as i32).to_le_bytes()).collect();
        // SAFETY: IDs were checked against the vocabulary; scratch holds qb + t rows.
        unsafe {
            cuda::upload(s.at(s.ids), &ids32, self.stream)?;
            check(
                (cuda::api().cs1_embed)(
                    s.at(s.ids).cast(),
                    self.embed.ptr,
                    s.at(s.res + qb * self.cfg.hidden * BF16),
                    ids.len() as i32,
                    self.cfg.hidden as i32,
                    self.stream,
                ),
                "embed",
            )?;
        }
        Ok(())
    }

    fn last_hidden(&self, s: &Scratch, t: usize) -> Result<Vec<f32>> {
        self.hidden_row(s, t - 1)
    }

    /// The final-norm hidden state at `row`.
    fn hidden_row(&self, s: &Scratch, row: usize) -> Result<Vec<f32>> {
        let mut last = vec![0u8; self.cfg.hidden * BF16];
        // SAFETY: x holds more than `row` rows of the hidden size.
        unsafe {
            cuda::download(
                &mut last,
                s.at(s.x + row * self.cfg.hidden * BF16),
                self.stream,
            )?;
        }
        let (pairs, _) = last.as_chunks::<2>();
        Ok(pairs
            .iter()
            .map(|&b| half::bf16::from_le_bytes(b).to_f32())
            .collect())
    }

    /// Keep the complete prompt's algorithms when its final MLP can use one row.
    fn last_mlp_reuse(&self, t: usize) -> Result<Option<cuda::ReuseGemm>> {
        if self.packed_gemm.is_none()
            || self.reference.is_some()
            || self.lora.is_some()
            || self.fp32_gdn.is_some()
            || self.cfg.hidden != 5120
            || self.cfg.intermediate != 17408
            || self.layers.len() != 64
            || !(512..=16384).contains(&t)
        {
            return Ok(None);
        }
        let Some(reuse) = cuda::api().reuse_gemm() else {
            return Ok(None);
        };
        let layer = self.layers.last().expect("validated nonempty model");
        for weight in [&layer.gate_up, &layer.down] {
            // SAFETY: both host row arrays remain live for this synchronous check.
            // Check both projections before any final-layer output is changed.
            let rc = unsafe {
                (reuse.check)(
                    self.gemm,
                    &t,
                    1,
                    &1usize,
                    1,
                    weight.shape[0] as i32,
                    weight.shape[1] as i32,
                )
            };
            if rc == -1 {
                return Ok(None);
            }
            check(rc, "last-row MLP compatibility")?;
        }
        Ok(Some(reuse))
    }

    /// Only last_hidden() consumes this single prompt's final layer outputs.
    /// Earlier rows need no final MLP; the next call embeds a fresh residual.
    fn run_last_mlp(
        &self,
        s: &Scratch,
        layer: &Layer,
        t: usize,
        reuse: cuda::ReuseGemm,
    ) -> Result<()> {
        let cfg = &self.cfg;
        let row = t - 1;
        let hidden_offset = row * cfg.hidden * BF16;
        let gate_offset = row * 2 * cfg.intermediate * BF16;
        let act_offset = row * cfg.intermediate * BF16;
        let p = |offset| s.at(offset);
        // SAFETY: the complete prompt already ran its mixer, scratch holds t rows,
        // and the retained-algorithm check validated both one-row projections.
        // These row-local operations keep the original BF16 rounding boundaries.
        unsafe {
            check(
                (cuda::api().cs1_add_rms_norm)(
                    p(s.res + hidden_offset),
                    p(s.delta + hidden_offset),
                    layer.post_norm.ptr,
                    p(s.x + hidden_offset),
                    1,
                    cfg.hidden as i32,
                    cfg.eps,
                    self.stream,
                ),
                "last-row post-attention norm",
            )?;
            check(
                (reuse.run)(
                    self.gemm,
                    p(s.x + hidden_offset),
                    layer.gate_up.ptr,
                    p(s.gate_up + gate_offset),
                    1,
                    (2 * cfg.intermediate) as i32,
                    cfg.hidden as i32,
                    t as i32,
                    self.stream,
                ),
                "last-row gate/up projection",
            )?;
            check(
                (cuda::api().cs1_silu_mul)(
                    p(s.gate_up + gate_offset),
                    (2 * cfg.intermediate) as i32,
                    p(s.act + act_offset),
                    1,
                    cfg.intermediate as i32,
                    self.stream,
                ),
                "last-row silu mul",
            )?;
            check(
                (reuse.run)(
                    self.gemm,
                    p(s.act + act_offset),
                    layer.down.ptr,
                    p(s.delta + hidden_offset),
                    1,
                    cfg.hidden as i32,
                    cfg.intermediate as i32,
                    t as i32,
                    self.stream,
                ),
                "last-row down projection",
            )?;
            check(
                (cuda::api().cs1_add_rms_norm)(
                    p(s.res + hidden_offset),
                    p(s.delta + hidden_offset),
                    self.final_norm.ptr,
                    p(s.x + hidden_offset),
                    1,
                    cfg.hidden as i32,
                    cfg.eps,
                    self.stream,
                ),
                "last-row final norm",
            )?;
        }
        Ok(())
    }

    /// Queue language layers over packed embeddings, resetting sequence positions.
    /// Final-norm hidden states end up in `s.x`. Bidirectional attention
    /// changes only the full-attention mask; Gated DeltaNet stays causal.
    fn run(
        &self,
        s: &Scratch,
        lengths: &[usize],
        custom_positions: bool,
        bidirectional: bool,
    ) -> Result<()> {
        self.run_with_suffix(s, lengths, custom_positions, bidirectional, None)
    }

    fn run_with_suffix(
        &self,
        s: &Scratch,
        lengths: &[usize],
        custom_positions: bool,
        bidirectional: bool,
        suffix: Option<&SuffixBatch<'_>>,
    ) -> Result<()> {
        let bidir = if bidirectional {
            ensure!(
                self.reference.is_none() && self.lora.is_none() && suffix.is_none(),
                "prompt-bidirectional prefill does not use reference numerics or an adapter"
            );
            Some(cuda::api().attention_gated_bidirectional()?)
        } else {
            None
        };
        let t: usize = lengths.iter().sum();
        let mut start = 0;
        let sequences: Vec<(usize, i32)> = lengths
            .iter()
            .map(|&length| {
                let offset = start;
                start += length;
                (offset, length as i32)
            })
            .collect();
        let cfg = &self.cfg;
        let st = self.stream;
        let (ti, hi, eps) = (t as i32, cfg.hidden as i32, cfg.eps);
        let (kd, vd, hv) = (cfg.key_dim(), cfg.value_dim(), cfg.lin_v_heads);
        let (hq, hk, hd) = (cfg.heads as i32, cfg.kv_heads as i32, cfg.head_dim as i32);
        let w = Widths::of(cfg);
        let p = |off: usize| s.at(off);
        let (cos, sin) = if custom_positions {
            (s.custom_cos, s.custom_sin)
        } else {
            let offset = suffix.map_or(0, |x| x.prefix.len * cfg.rotary_half * BF16);
            (s.cos + offset, s.sin + offset)
        };
        let project = |x: usize, weight: &Tensor, y: usize, group: Option<Group>| {
            if let Some(suffix) = suffix {
                // SAFETY: combined rows were checked against the original algorithm;
                // scratch owns the compact batch and all work uses the model stream.
                unsafe {
                    check(
                        (suffix.gemm.run)(
                            self.gemm,
                            p(x),
                            weight.ptr,
                            p(y),
                            t as i32,
                            weight.shape[0] as i32,
                            weight.shape[1] as i32,
                            suffix.original_rows as i32,
                            st,
                        ),
                        "packed suffix projection",
                    )
                }
            } else if let Some(group) = group {
                self.gemm_group(s, x, weight, y, t, lengths, group)
            } else {
                self.gemm_sequences(s, x, weight, y, lengths)
            }
        };
        let mut linear_index = 0;
        let mut attention_index = 0;
        // SAFETY (every kernel call below): the pointers are weights in the arena or
        // scratch buffers laid out for at least t tokens with the widths used here.
        unsafe {
            check(
                (self
                    .reference
                    .map_or(cuda::api().cs1_rms_norm, |r| r.rms_norm))(
                    p(s.res),
                    self.layers[0].input_norm.ptr,
                    p(s.x),
                    ti,
                    hi,
                    eps,
                    st,
                ),
                "input norm",
            )?;
        }
        let last_mlp =
            if !bidirectional && !custom_positions && suffix.is_none() && lengths.len() == 1 {
                self.last_mlp_reuse(t)?
            } else {
                None
            };
        let convolution = self.convolution();
        let gdn_prefill = self.gdn_prefill();
        for (i, layer) in self.layers.iter().enumerate() {
            match &layer.mixer {
                Mixer::Linear(la) => {
                    project(s.x, &la.in_proj, s.gdn_in, Some(Group::LinearInput))?;
                    self.apply_lora(i, Group::LinearInput, s, s.x, s.gdn_in, lengths)?;
                    let ld = w.gdn_in as i32;
                    let z = s.gdn_in + w.conv * BF16;
                    let b = z + vd * BF16;
                    let a = b + hv * BF16;
                    unsafe {
                        for &(offset, length) in &sequences {
                            check(
                                if let Some(suffix) = suffix {
                                    (suffix.ops.conv)(
                                        p(s.gdn_in + offset * w.gdn_in * BF16),
                                        suffix.prefix.conv_tail[linear_index].at(0),
                                        ld,
                                        la.conv.ptr,
                                        p(s.lq + offset * kd * BF16),
                                        p(s.lk + offset * kd * BF16),
                                        p(s.lv + offset * vd * BF16),
                                        length,
                                        kd as i32,
                                        vd as i32,
                                        st,
                                    )
                                } else {
                                    (convolution)(
                                        p(s.gdn_in + offset * w.gdn_in * BF16),
                                        ld,
                                        la.conv.ptr,
                                        p(s.lq + offset * kd * BF16),
                                        p(s.lk + offset * kd * BF16),
                                        p(s.lv + offset * vd * BF16),
                                        length,
                                        kd as i32,
                                        vd as i32,
                                        st,
                                    )
                                },
                                "gdn conv",
                            )?;
                        }
                        check(
                            (cuda::api().cs1_gdn_gates)(
                                p(b),
                                p(a),
                                ld,
                                la.a_log.ptr,
                                la.dt_bias.ptr,
                                p(s.beta),
                                p(s.g).cast(),
                                ti,
                                hv as i32,
                                st,
                            ),
                            "gdn gates",
                        )?;
                        if let Some(suffix) = suffix {
                            for &(offset, length) in &sequences {
                                check(
                                    (cuda::api().cs1_gdn_prefill_x)(
                                        p(s.lq + offset * kd * BF16),
                                        p(s.lk + offset * kd * BF16),
                                        p(s.lv + offset * vd * BF16),
                                        p(s.g + offset * hv * F32).cast(),
                                        p(s.beta + offset * hv * BF16),
                                        p(s.lo + offset * vd * BF16),
                                        p(s.workspace).cast(),
                                        length,
                                        hv as i32,
                                        cfg.lin_k_heads as i32,
                                        (cfg.lin_k_dim as f32).powf(-0.5),
                                        suffix.prefix.gdn_state[linear_index].at(0),
                                        std::ptr::null_mut(),
                                        st,
                                    ),
                                    "suffix GDN prefill",
                                )?;
                            }
                        } else if let Some(gdn) = &self.batched_gdn
                            && (2..=16).contains(&lengths.len())
                        {
                            check(
                                (gdn.run)(
                                    p(s.lq),
                                    p(s.lk),
                                    p(s.lv),
                                    p(s.g).cast(),
                                    p(s.beta),
                                    p(s.lo),
                                    p(s.workspace).cast(),
                                    s.workspace_bytes,
                                    lengths.as_ptr(),
                                    lengths.len(),
                                    hv as i32,
                                    cfg.lin_k_heads as i32,
                                    (cfg.lin_k_dim as f32).powf(-0.5),
                                    st,
                                ),
                                "batched GDN prefill",
                            )?;
                        } else {
                            for &(offset, length) in &sequences {
                                check(
                                    if let Some(gdn) = &self.fp32_gdn {
                                        gdn.run(
                                            p(s.lq + offset * kd * BF16),
                                            p(s.lk + offset * kd * BF16),
                                            p(s.lv + offset * vd * BF16),
                                            p(s.g + offset * hv * F32).cast(),
                                            p(s.beta + offset * hv * BF16),
                                            p(s.lo + offset * vd * BF16),
                                            p(s.workspace).cast(),
                                            length,
                                            hv as i32,
                                            cfg.lin_k_heads as i32,
                                            (cfg.lin_k_dim as f32).powf(-0.5),
                                            st,
                                        )
                                    } else {
                                        gdn_prefill(
                                            p(s.lq + offset * kd * BF16),
                                            p(s.lk + offset * kd * BF16),
                                            p(s.lv + offset * vd * BF16),
                                            p(s.g + offset * hv * F32).cast(),
                                            p(s.beta + offset * hv * BF16),
                                            p(s.lo + offset * vd * BF16),
                                            p(s.workspace).cast(),
                                            length,
                                            hv as i32,
                                            cfg.lin_k_heads as i32,
                                            (cfg.lin_k_dim as f32).powf(-0.5),
                                            st,
                                        )
                                    },
                                    "gdn prefill",
                                )?;
                            }
                        }
                        check(
                            (cuda::api().cs1_gated_rms_norm)(
                                p(s.lo),
                                p(z),
                                ld,
                                la.norm.ptr,
                                p(s.ln),
                                ti,
                                hv as i32,
                                cfg.lin_v_dim as i32,
                                eps,
                                st,
                            ),
                            "gated norm",
                        )?;
                    }
                    project(s.ln, &la.out, s.delta, None)?;
                    self.apply_lora(i, Group::LinearOutput, s, s.ln, s.delta, lengths)?;
                    linear_index += 1;
                }
                Mixer::Full(fa) => {
                    project(s.x, &fa.qkv, s.attn_in, Some(Group::AttentionInput))?;
                    self.apply_lora(i, Group::AttentionInput, s, s.x, s.attn_in, lengths)?;
                    let ld = w.attn_in as i32;
                    let k = s.attn_in + w.attn_q * BF16;
                    let v = k + cfg.kv_heads * cfg.head_dim * BF16;
                    unsafe {
                        for &(offset, length) in &sequences {
                            check(
                                (self
                                    .reference
                                    .map_or(cuda::api().cs1_attn_prep, |r| r.attn_prep))(
                                    p(s.attn_in + offset * w.attn_in * BF16),
                                    p(k + offset * w.attn_in * BF16),
                                    ld,
                                    fa.q_norm.ptr,
                                    fa.k_norm.ptr,
                                    p(cos),
                                    p(sin),
                                    p(s.aq + offset * cfg.heads * cfg.head_dim * BF16),
                                    p(s.agate + offset * cfg.heads * cfg.head_dim * BF16),
                                    p(s.ak + offset * cfg.kv_heads * cfg.head_dim * BF16),
                                    length,
                                    hq,
                                    hk,
                                    hd,
                                    cfg.rotary_half as i32,
                                    eps,
                                    st,
                                ),
                                "attention prep",
                            )?;
                            let scale = (cfg.head_dim as f32).powf(-0.5);
                            check(
                                if let Some(suffix) = suffix {
                                    (suffix.ops.attention)(
                                        p(s.aq + offset * cfg.heads * cfg.head_dim * BF16),
                                        p(s.ak + offset * cfg.kv_heads * cfg.head_dim * BF16),
                                        p(v + offset * w.attn_in * BF16),
                                        ld,
                                        p(s.agate + offset * cfg.heads * cfg.head_dim * BF16),
                                        p(s.ao + offset * cfg.heads * cfg.head_dim * BF16),
                                        suffix.prefix.attn_kv[attention_index].0.at(0),
                                        suffix.prefix.attn_kv[attention_index].1.at(0),
                                        length,
                                        suffix.prefix.len as i32,
                                        hq,
                                        hk,
                                        hd,
                                        (cfg.head_dim as f32).powf(-0.5),
                                        st,
                                    )
                                } else if let Some(bidir) = bidir {
                                    (bidir)(
                                        p(s.aq + offset * cfg.heads * cfg.head_dim * BF16),
                                        p(s.ak + offset * cfg.kv_heads * cfg.head_dim * BF16),
                                        p(v + offset * w.attn_in * BF16),
                                        ld,
                                        p(s.agate + offset * cfg.heads * cfg.head_dim * BF16),
                                        p(s.ao + offset * cfg.heads * cfg.head_dim * BF16),
                                        length,
                                        hq,
                                        hk,
                                        hd,
                                        scale,
                                        st,
                                    )
                                } else if let Some(reference) = self.reference {
                                    (reference.attention)(
                                        p(s.aq + offset * cfg.heads * cfg.head_dim * BF16),
                                        p(s.ak + offset * cfg.kv_heads * cfg.head_dim * BF16),
                                        p(v + offset * w.attn_in * BF16),
                                        ld,
                                        p(s.agate + offset * cfg.heads * cfg.head_dim * BF16),
                                        p(s.ao + offset * cfg.heads * cfg.head_dim * BF16),
                                        length,
                                        hq,
                                        hk,
                                        scale,
                                        p(s.workspace),
                                        st,
                                    )
                                } else {
                                    (cuda::api().cs1_attention_gated)(
                                        p(s.aq + offset * cfg.heads * cfg.head_dim * BF16),
                                        p(s.ak + offset * cfg.kv_heads * cfg.head_dim * BF16),
                                        p(v + offset * w.attn_in * BF16),
                                        ld,
                                        p(s.agate + offset * cfg.heads * cfg.head_dim * BF16),
                                        p(s.ao + offset * cfg.heads * cfg.head_dim * BF16),
                                        length,
                                        hq,
                                        hk,
                                        hd,
                                        scale,
                                        st,
                                    )
                                },
                                "gated attention",
                            )?;
                        }
                    }
                    project(s.ao, &fa.o, s.delta, None)?;
                    self.apply_lora(i, Group::AttentionOutput, s, s.ao, s.delta, lengths)?;
                    attention_index += 1;
                }
            }
            if i + 1 == self.layers.len()
                && let Some(reuse) = last_mlp
            {
                return self.run_last_mlp(s, layer, t, reuse);
            }
            unsafe {
                check(
                    (self
                        .reference
                        .map_or(cuda::api().cs1_add_rms_norm, |r| r.add_rms_norm))(
                        p(s.res),
                        p(s.delta),
                        layer.post_norm.ptr,
                        p(s.x),
                        ti,
                        hi,
                        eps,
                        st,
                    ),
                    "post-attention norm",
                )?;
            }
            project(s.x, &layer.gate_up, s.gate_up, Some(Group::GateUp))?;
            self.apply_lora(i, Group::GateUp, s, s.x, s.gate_up, lengths)?;
            unsafe {
                check(
                    (cuda::api().cs1_silu_mul)(
                        p(s.gate_up),
                        (2 * cfg.intermediate) as i32,
                        p(s.act),
                        ti,
                        cfg.intermediate as i32,
                        st,
                    ),
                    "silu mul",
                )?;
            }
            project(s.act, &layer.down, s.delta, None)?;
            self.apply_lora(i, Group::Down, s, s.act, s.delta, lengths)?;
            let next = self
                .layers
                .get(i + 1)
                .map_or(&self.final_norm, |l| &l.input_norm);
            unsafe {
                check(
                    (self
                        .reference
                        .map_or(cuda::api().cs1_add_rms_norm, |r| r.add_rms_norm))(
                        p(s.res),
                        p(s.delta),
                        next.ptr,
                        p(s.x),
                        ti,
                        hi,
                        eps,
                        st,
                    ),
                    "input norm",
                )?;
            }
        }
        Ok(())
    }

    /// The R2d prefix-cache pass: run rows `[qb, tend)` of one prompt with
    /// explicit positions, touching per-layer buffers only for those rows.
    ///
    /// `capture` (`tend` rows): collect post-prep K and pre-GEMM V columns of each
    /// full-attention layer plus the float32 GDN state and the three pre-conv
    /// projection columns of each Gated DeltaNet layer, for rows `[0, tend)`; with a
    /// `prefix` too, the state after it continued. `prefix` (qb = prefix.len):
    /// seed the window from a captured entry — the conv tail columns are copied
    /// back into place before the convolution, and the prefix K/V columns before
    /// the attention. Everything else is `run()` with shifted row bases, so a
    /// captured+continued pass retains the sequence-local kernel arithmetic.
    /// Fixed-profile callers use their fixed-algorithm handle. Open-Jev text
    /// reuse instead validates and retains each complete prompt's algorithm.
    /// Custom position tables contain window rows; text tables use absolute rows.
    #[allow(clippy::too_many_arguments)] // Window, prefix states, GEMM handle and optional original shape.
    fn run_window(
        &self,
        s: &Scratch,
        qb: usize,
        tend: usize,
        custom_positions: bool,
        capture: Option<&mut PrefixState>,
        prefix: Option<&PrefixState>,
        gemm: *mut c_void,
        reuse: Option<(cuda::ReuseGemm, usize)>,
    ) -> Result<()> {
        ensure!(
            self.fp32_gdn.is_none(),
            "FP32 GDN does not support prefix continuation"
        );
        ensure!(
            qb < tend
                && prefix.map_or(qb == 0, |pre| pre.len == qb)
                && capture.as_deref().is_none_or(|cap| cap.len == tend),
            "prefix window shape"
        );
        let cfg = &self.cfg;
        let st = self.stream;
        let rows = tend - qb;
        let project = |x, weight: &Tensor, y, group| -> Result<()> {
            if let Some((api, original_rows)) = reuse {
                // SAFETY: compatibility was checked before prefix capture. Scratch
                // owns rows at the offsets passed by the unchanged layer layout.
                unsafe {
                    check(
                        (api.run)(
                            gemm,
                            s.at(x),
                            weight.ptr,
                            s.at(y),
                            rows as i32,
                            weight.shape[0] as i32,
                            weight.shape[1] as i32,
                            original_rows as i32,
                            self.stream,
                        ),
                        "retained prefix projection",
                    )
                }
            } else if let Some(group) = group {
                self.gemm_group_with(gemm, s, x, weight, y, rows, &[rows], group)
            } else {
                self.gemm_with(gemm, s, x, weight, y, rows)
            }
        };
        let (ti, hi, eps) = (rows as i32, cfg.hidden as i32, cfg.eps);
        let (kd, vd, hv) = (cfg.key_dim(), cfg.value_dim(), cfg.lin_v_heads);
        let (hq, hk, hd) = (cfg.heads as i32, cfg.kv_heads as i32, cfg.head_dim as i32);
        let w = Widths::of(cfg);
        let p = |off: usize| s.at(off);
        let (cos, sin) = if custom_positions {
            (s.custom_cos, s.custom_sin)
        } else {
            let rope = qb * cfg.rotary_half * BF16;
            (s.cos + rope, s.sin + rope)
        };
        let hb = cfg.hidden * BF16;
        let ob = cfg.heads * cfg.head_dim * BF16; // post-prep q/gate and ao row bytes
        let kvb = cfg.kv_heads * cfg.head_dim * BF16; // k and v row bytes
        let ldb = w.gdn_in; // gdn_in row, elements
        let ldbb = ldb * BF16;
        let ab = w.attn_in * BF16; // attn_in row bytes
        let kb = kd * BF16; // linear q/k row bytes
        let vb = vd * BF16; // linear v row bytes
        let scale = (cfg.head_dim as f32).powf(-0.5);
        let mut fa_i = 0usize;
        let mut la_i = 0usize;
        // SAFETY (every kernel call below): pointers are weights in the arena or
        // scratch buffers laid out for tend tokens at the widths used here; prefix
        // buffers hold at least `len` rows (a capture's `tend`) of the shapes
        // allocated by alloc_prefix.
        unsafe {
            check(
                (self
                    .reference
                    .map_or(cuda::api().cs1_rms_norm, |r| r.rms_norm))(
                    p(s.res + qb * hb),
                    self.layers[0].input_norm.ptr,
                    p(s.x + qb * hb),
                    ti,
                    hi,
                    eps,
                    st,
                ),
                "input norm",
            )?;
        }
        let convolution = self.convolution();
        for (i, layer) in self.layers.iter().enumerate() {
            match &layer.mixer {
                Mixer::Linear(la) => {
                    project(
                        s.x + qb * hb,
                        &la.in_proj,
                        s.gdn_in + qb * ldbb,
                        Some(Group::LinearInput),
                    )?;
                    self.apply_lora(
                        i,
                        Group::LinearInput,
                        s,
                        s.x + qb * hb,
                        s.gdn_in + qb * ldbb,
                        &[rows],
                    )?;
                    let ld = ldb as i32;
                    let z = s.gdn_in + w.conv * BF16;
                    let b = z + vd * BF16;
                    let a = b + hv * BF16;
                    let s_in: *const c_void =
                        prefix.map_or(std::ptr::null(), |pre| pre.gdn_state[la_i].at(0));
                    let s_out: *mut c_void = capture
                        .as_deref()
                        .map_or(std::ptr::null_mut(), |cap| cap.gdn_state[la_i].at(0));
                    unsafe {
                        if qb > 0 {
                            cuda::copy_dd(
                                p(s.gdn_in + (qb - 3) * ldbb),
                                prefix.unwrap().conv_tail[la_i].at(0),
                                3 * ldbb,
                                st,
                            )?;
                        }
                        if let Some(cap) = capture.as_deref() {
                            // conv window tail: the last three pre-conv columns, which
                            // include restored prefix columns for a window under three rows.
                            cuda::copy_dd(
                                cap.conv_tail[la_i].at(0),
                                p(s.gdn_in + (tend - 3) * ldbb),
                                3 * ldbb,
                                st,
                            )?;
                        }
                        // The conv kernel zero-pads its local first three rows.
                        // Include the restored history in its input and discard
                        // those three outputs so continuation rows see all taps.
                        let conv_start = qb.saturating_sub(3);
                        check(
                            (convolution)(
                                p(s.gdn_in + conv_start * ldbb),
                                ld,
                                la.conv.ptr,
                                p(s.lq + conv_start * kb),
                                p(s.lk + conv_start * kb),
                                p(s.lv + conv_start * vb),
                                (tend - conv_start) as i32,
                                kd as i32,
                                vd as i32,
                                st,
                            ),
                            "gdn conv",
                        )?;
                        check(
                            (cuda::api().cs1_gdn_gates)(
                                p(b + qb * ldbb),
                                p(a + qb * ldbb),
                                ld,
                                la.a_log.ptr,
                                la.dt_bias.ptr,
                                p(s.beta + qb * hv * BF16),
                                p(s.g + qb * hv * F32).cast(),
                                ti,
                                hv as i32,
                                st,
                            ),
                            "gdn gates",
                        )?;
                        check(
                            (cuda::api().cs1_gdn_prefill_x)(
                                p(s.lq + qb * kb),
                                p(s.lk + qb * kb),
                                p(s.lv + qb * vb),
                                p(s.g + qb * hv * F32).cast(),
                                p(s.beta + qb * hv * BF16),
                                p(s.lo + qb * vb),
                                p(s.workspace).cast(),
                                ti,
                                hv as i32,
                                cfg.lin_k_heads as i32,
                                (cfg.lin_k_dim as f32).powf(-0.5),
                                s_in,
                                s_out,
                                st,
                            ),
                            "gdn prefill",
                        )?;
                        check(
                            (cuda::api().cs1_gated_rms_norm)(
                                p(s.lo + qb * vb),
                                p(z + qb * ldbb),
                                ld,
                                la.norm.ptr,
                                p(s.ln + qb * vb),
                                ti,
                                hv as i32,
                                cfg.lin_v_dim as i32,
                                eps,
                                st,
                            ),
                            "gated norm",
                        )?;
                    }
                    project(s.ln + qb * vb, &la.out, s.delta + qb * hb, None)?;
                    self.apply_lora(
                        i,
                        Group::LinearOutput,
                        s,
                        s.ln + qb * vb,
                        s.delta + qb * hb,
                        &[rows],
                    )?;
                    la_i += 1;
                }
                Mixer::Full(fa) => {
                    project(
                        s.x + qb * hb,
                        &fa.qkv,
                        s.attn_in + qb * ab,
                        Some(Group::AttentionInput),
                    )?;
                    self.apply_lora(
                        i,
                        Group::AttentionInput,
                        s,
                        s.x + qb * hb,
                        s.attn_in + qb * ab,
                        &[rows],
                    )?;
                    let k = s.attn_in + w.attn_q * BF16;
                    let v = k + kvb;
                    let (q_base, t_flash) = (qb as i32, tend as i32);
                    unsafe {
                        if qb > 0 {
                            let pre = prefix.unwrap();
                            // Restore the prefix V columns and post-prep K rows in place;
                            // the windowed flash then sees the identical k/v layout.
                            cuda::copy2d(p(v), ab, pre.attn_kv[fa_i].1.at(0), kvb, kvb, qb, st)?;
                            cuda::copy_dd(p(s.ak), pre.attn_kv[fa_i].0.at(0), qb * kvb, st)?;
                        }
                        check(
                            (self
                                .reference
                                .map_or(cuda::api().cs1_attn_prep, |r| r.attn_prep))(
                                p(s.attn_in + qb * ab),
                                p(k + qb * ab),
                                w.attn_in as i32,
                                fa.q_norm.ptr,
                                fa.k_norm.ptr,
                                p(cos),
                                p(sin),
                                p(s.aq + qb * ob),
                                p(s.agate + qb * ob),
                                p(s.ak + qb * kvb),
                                ti,
                                hq,
                                hk,
                                hd,
                                cfg.rotary_half as i32,
                                eps,
                                st,
                            ),
                            "attention prep",
                        )?;
                        if let Some(cap) = capture.as_deref() {
                            // K must be post-prep (normed + rotary); V is the raw column.
                            cuda::copy_dd(cap.attn_kv[fa_i].0.at(0), p(s.ak), tend * kvb, st)?;
                            cuda::copy2d(cap.attn_kv[fa_i].1.at(0), kvb, p(v), ab, kvb, tend, st)?;
                        }
                        check(
                            (cuda::api().cs1_attention_gated_prefix)(
                                p(s.aq),
                                p(s.ak),
                                p(v),
                                w.attn_in as i32,
                                p(s.agate),
                                p(s.ao),
                                t_flash,
                                hq,
                                hk,
                                hd,
                                scale,
                                q_base,
                                st,
                            ),
                            "gated attention",
                        )?;
                    }
                    project(s.ao + qb * ob, &fa.o, s.delta + qb * hb, None)?;
                    self.apply_lora(
                        i,
                        Group::AttentionOutput,
                        s,
                        s.ao + qb * ob,
                        s.delta + qb * hb,
                        &[rows],
                    )?;
                    fa_i += 1;
                }
            }
            unsafe {
                check(
                    (self
                        .reference
                        .map_or(cuda::api().cs1_add_rms_norm, |r| r.add_rms_norm))(
                        p(s.res + qb * hb),
                        p(s.delta + qb * hb),
                        layer.post_norm.ptr,
                        p(s.x + qb * hb),
                        ti,
                        hi,
                        eps,
                        st,
                    ),
                    "post-attention norm",
                )?;
            }
            project(
                s.x + qb * hb,
                &layer.gate_up,
                s.gate_up + qb * 2 * cfg.intermediate * BF16,
                Some(Group::GateUp),
            )?;
            self.apply_lora(
                i,
                Group::GateUp,
                s,
                s.x + qb * hb,
                s.gate_up + qb * 2 * cfg.intermediate * BF16,
                &[rows],
            )?;
            unsafe {
                check(
                    (cuda::api().cs1_silu_mul)(
                        p(s.gate_up + qb * 2 * cfg.intermediate * BF16),
                        (2 * cfg.intermediate) as i32,
                        p(s.act + qb * cfg.intermediate * BF16),
                        ti,
                        cfg.intermediate as i32,
                        st,
                    ),
                    "silu mul",
                )?;
            }
            project(
                s.act + qb * cfg.intermediate * BF16,
                &layer.down,
                s.delta + qb * hb,
                None,
            )?;
            self.apply_lora(
                i,
                Group::Down,
                s,
                s.act + qb * cfg.intermediate * BF16,
                s.delta + qb * hb,
                &[rows],
            )?;
            let next = self
                .layers
                .get(i + 1)
                .map_or(&self.final_norm, |l| &l.input_norm);
            unsafe {
                check(
                    (self
                        .reference
                        .map_or(cuda::api().cs1_add_rms_norm, |r| r.add_rms_norm))(
                        p(s.res + qb * hb),
                        p(s.delta + qb * hb),
                        next.ptr,
                        p(s.x + qb * hb),
                        ti,
                        hi,
                        eps,
                        st,
                    ),
                    "input norm",
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../../../tests/qwen3_5/multimodal_graph.rs"]
mod graph_tests;
#[cfg(test)]
#[path = "../../../../../tests/qwen3_5/model.rs"]
mod tests;

#[cfg(test)]
#[path = "../../../../../tests/qwen3_5/prefix_groups.rs"]
mod prefix_group_tests;

#[cfg(test)]
#[path = "../../../../../tests/qwen3_5/last_mlp.rs"]
mod last_mlp_tests;
