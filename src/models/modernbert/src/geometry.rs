use crate::structure::ModernBertStructure;

/// Every tensor shape below is derived, never read from a checkpoint:
/// the structure is the only geometry input.
pub fn embedding_shape(s: &ModernBertStructure) -> Vec<usize> {
    vec![s.vocab_size, s.hidden_size]
}

pub fn qkv_shape(s: &ModernBertStructure) -> Vec<usize> {
    vec![3 * s.hidden_size, s.hidden_size]
}

pub fn attn_out_shape(s: &ModernBertStructure) -> Vec<usize> {
    vec![s.hidden_size, s.hidden_size]
}

/// The fused gated-MLP up projection; `2 * intermediate` is derived
/// geometry, not a semantic field of its own.
pub fn mlp_in_shape(s: &ModernBertStructure) -> Vec<usize> {
    vec![2 * s.intermediate_size, s.hidden_size]
}

pub fn mlp_out_shape(s: &ModernBertStructure) -> Vec<usize> {
    vec![s.hidden_size, s.intermediate_size]
}

/// Rotary dimensions per head (rotate-half pairing needs the even half).
pub fn rotary_half(s: &ModernBertStructure) -> usize {
    s.head_dim / 2
}

/// One cos/sin table covers `max_positions` rows of `rotary_half` f32
/// columns; a bundle serving shorter requests sizes tables to its own
/// length bound instead.
pub fn rope_table_bytes(s: &ModernBertStructure) -> usize {
    s.max_positions * rotary_half(s) * 4
}

/// The canonical encoder tensor inventory under family naming. Layer
/// zero's attention norm is the embedding norm (the embedding kernel
/// applies it); every layer owns its post-attention norm, and the last
/// layer's post-MLP norm is the final norm.
pub fn encoder_tensors(s: &ModernBertStructure) -> Vec<(String, Vec<usize>)> {
    let h = s.hidden_size;
    let mut tensors = vec![
        (
            "encoder.embeddings.tok_embeddings.weight".to_string(),
            embedding_shape(s),
        ),
        ("encoder.embeddings.norm.weight".to_string(), vec![h]),
        ("encoder.final_norm.weight".to_string(), vec![h]),
    ];
    for i in 0..s.num_hidden_layers {
        let p = format!("encoder.layers.{i}");
        if i > 0 {
            tensors.push((format!("{p}.attn_norm.weight"), vec![h]));
        }
        tensors.push((format!("{p}.mlp_norm.weight"), vec![h]));
        tensors.push((format!("{p}.attn.Wqkv.weight"), qkv_shape(s)));
        tensors.push((format!("{p}.attn.Wo.weight"), attn_out_shape(s)));
        tensors.push((format!("{p}.mlp.Wi.weight"), mlp_in_shape(s)));
        tensors.push((format!("{p}.mlp.Wo.weight"), mlp_out_shape(s)));
    }
    if let Some(rows) = s.type_embedding_rows {
        tensors.push(("type_emb.weight".to_string(), vec![rows, h]));
    }
    tensors
}

/// Encoder scratch for one request of `batch` rows packed to `len`
/// tokens, in canonical allocation order. Task-side buffers are sized
/// by the consuming model crate and appended after these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncoderScratch {
    pub ids: usize,
    pub lens: usize,
    pub x: usize,
    pub y: usize,
    pub qkv: usize,
    pub o: usize,
    pub gate: usize,
}

impl EncoderScratch {
    pub fn new(batch: usize, len: usize, s: &ModernBertStructure) -> Self {
        let m = batch * len;
        Self {
            ids: m * 8,
            lens: batch * 4,
            x: m * s.hidden_size * 4,
            y: m * s.hidden_size * 2,
            qkv: m * 3 * s.hidden_size * 2,
            o: m * s.hidden_size * 2,
            gate: m * s.intermediate_size * 2,
        }
    }

    pub fn sizes(&self) -> Vec<usize> {
        [self.ids, self.lens, self.x, self.y, self.qkv, self.o, self.gate].into()
    }

    pub fn total(&self) -> usize {
        self.sizes().iter().sum()
    }
}
