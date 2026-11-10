//! A request-local prefix tree shares all layer work, retaining original GEMM plans.
use super::*;

pub(super) struct Tree {
    rows: usize,
    prefix: usize,
    sequences: Vec<(usize, usize)>,
    input_rows: usize,
    output_rows: usize,
    gemm: cuda::ReuseGemm,
    ops: cuda::SharedSuffix,
    gdn: cuda::SharedBatchedGdn,
    attention_batch: Option<cuda::SharedBatchedAttention>,
    state: *mut c_void,
    values: *mut c_void,
}

impl Tree {
    pub(super) fn project(
        &self,
        model: &Model,
        s: &Scratch,
        x: usize,
        weight: &Tensor,
        y: usize,
        role: Projection,
    ) -> Result<()> {
        let original = match role {
            Projection::MixerInput | Projection::GateUp => self.input_rows,
            Projection::MixerOutput | Projection::Down => self.output_rows,
        };
        // SAFETY: every weight shape and original group was checked before embedding.
        // The compact rows and output dimensions fit distinct scratch regions.
        unsafe {
            check(
                (self.gemm.run)(
                    model.gemm,
                    s.at(x),
                    weight.ptr,
                    s.at(y),
                    self.rows as i32,
                    weight.shape[0] as i32,
                    weight.shape[1] as i32,
                    original as i32,
                    model.stream,
                ),
                "tree retained projection",
            )
        }
    }

    pub(super) fn linear(
        &self,
        model: &Model,
        s: &Scratch,
        la: &LinearAttention,
        lengths: &[usize],
    ) -> Result<()> {
        let cfg = &model.cfg;
        let w = Widths::of(cfg);
        let (kd, vd, hv) = (cfg.key_dim(), cfg.value_dim(), cfg.lin_v_heads);
        let (ld, st) = (w.gdn_in as i32, model.stream);
        let p = |off| s.at(off);
        let z = s.gdn_in + w.conv * BF16;
        let b = z + vd * BF16;
        let a = b + hv * BF16;
        // SAFETY: scratch holds one prefix then disjoint suffixes. Convolution
        // history reads exactly three prefix rows; the model-owned FP32 state is
        // written before every layer's suffixes and reused only after they finish.
        unsafe {
            for &(offset, length) in &self.sequences {
                check(
                    if offset == 0 {
                        (model.convolution())(
                            p(s.gdn_in),
                            ld,
                            la.conv.ptr,
                            p(s.lq),
                            p(s.lk),
                            p(s.lv),
                            length as i32,
                            kd as i32,
                            vd as i32,
                            st,
                        )
                    } else {
                        (self.ops.conv)(
                            p(s.gdn_in + offset * w.gdn_in * BF16),
                            p(s.gdn_in + (self.prefix - 3) * w.gdn_in * BF16),
                            ld,
                            la.conv.ptr,
                            p(s.lq + offset * kd * BF16),
                            p(s.lk + offset * kd * BF16),
                            p(s.lv + offset * vd * BF16),
                            length as i32,
                            kd as i32,
                            vd as i32,
                            st,
                        )
                    },
                    "tree convolution",
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
                    self.rows as i32,
                    hv as i32,
                    st,
                ),
                "tree gates",
            )?;
            let scale = (cfg.lin_k_dim as f32).powf(-0.5);
            check(
                (cuda::api().cs1_gdn_prefill_x)(
                    p(s.lq),
                    p(s.lk),
                    p(s.lv),
                    p(s.g).cast(),
                    p(s.beta),
                    p(s.lo),
                    p(s.workspace).cast(),
                    self.prefix as i32,
                    hv as i32,
                    cfg.lin_k_heads as i32,
                    scale,
                    std::ptr::null(),
                    self.state,
                    st,
                ),
                "tree prefix state",
            )?;
            let offset = self.prefix;
            check(
                (self.gdn)(
                    p(s.lq + offset * kd * BF16),
                    p(s.lk + offset * kd * BF16),
                    p(s.lv + offset * vd * BF16),
                    p(s.g + offset * hv * F32).cast(),
                    p(s.beta + offset * hv * BF16),
                    p(s.lo + offset * vd * BF16),
                    p(s.workspace).cast(),
                    s.workspace_bytes,
                    lengths[1..].as_ptr(),
                    lengths.len() - 1,
                    hv as i32,
                    cfg.lin_k_heads as i32,
                    scale,
                    self.state,
                    st,
                ),
                "tree suffix states",
            )?;
            check(
                (cuda::api().cs1_gated_rms_norm)(
                    p(s.lo),
                    p(z),
                    ld,
                    la.norm.ptr,
                    p(s.ln),
                    self.rows as i32,
                    hv as i32,
                    cfg.lin_v_dim as i32,
                    cfg.eps,
                    st,
                ),
                "tree gated norm",
            )
        }
    }

    pub(super) fn attention(
        &self,
        model: &Model,
        s: &Scratch,
        fa: &FullAttention,
        lengths: &[usize],
    ) -> Result<()> {
        let cfg = &model.cfg;
        let w = Widths::of(cfg);
        let (hq, hk, hd) = (cfg.heads as i32, cfg.kv_heads as i32, cfg.head_dim as i32);
        let (ld, st) = (w.attn_in as i32, model.stream);
        let qb = cfg.heads * cfg.head_dim * BF16;
        let kb = cfg.kv_heads * cfg.head_dim * BF16;
        let p = |off| s.at(off);
        let k = s.attn_in + w.attn_q * BF16;
        let v = k + kb;
        let scale = (cfg.head_dim as f32).powf(-0.5);
        // SAFETY: custom rotary rows encode each branch's original position.
        // Prefix K is contiguous in scratch; V is copied out of strided QKV once.
        // Every suffix sees only that prefix and its own K/V, never another branch.
        unsafe {
            check(
                (cuda::api().cs1_attn_prep)(
                    p(s.attn_in),
                    p(k),
                    ld,
                    fa.q_norm.ptr,
                    fa.k_norm.ptr,
                    p(s.custom_cos),
                    p(s.custom_sin),
                    p(s.aq),
                    p(s.agate),
                    p(s.ak),
                    self.rows as i32,
                    hq,
                    hk,
                    hd,
                    cfg.rotary_half as i32,
                    cfg.eps,
                    st,
                ),
                "tree attention prep",
            )?;
            cuda::copy2d(self.values, kb, p(v), w.attn_in * BF16, kb, self.prefix, st)?;
            for &(offset, length) in &self.sequences {
                if offset != 0 && self.attention_batch.is_some() {
                    break;
                }
                check(
                    if offset == 0 {
                        (cuda::api().cs1_attention_gated)(
                            p(s.aq),
                            p(s.ak),
                            p(v),
                            ld,
                            p(s.agate),
                            p(s.ao),
                            length as i32,
                            hq,
                            hk,
                            hd,
                            scale,
                            st,
                        )
                    } else {
                        (self.ops.attention)(
                            p(s.aq + offset * qb),
                            p(s.ak + offset * kb),
                            p(v + offset * w.attn_in * BF16),
                            ld,
                            p(s.agate + offset * qb),
                            p(s.ao + offset * qb),
                            p(s.ak),
                            self.values,
                            length as i32,
                            self.prefix as i32,
                            hq,
                            hk,
                            hd,
                            scale,
                            st,
                        )
                    },
                    "tree attention",
                )?;
            }
            if let Some(batch) = self.attention_batch {
                let offset = self.prefix;
                check(
                    batch(
                        p(s.aq + offset * qb),
                        p(s.ak + offset * kb),
                        p(v + offset * w.attn_in * BF16),
                        ld,
                        p(s.agate + offset * qb),
                        p(s.ao + offset * qb),
                        p(s.ak),
                        self.values,
                        lengths[1..].as_ptr(),
                        lengths.len() - 1,
                        self.prefix as i32,
                        hq,
                        hk,
                        hd,
                        scale,
                        st,
                    ),
                    "batched tree suffix attention",
                )?;
            }
        }
        Ok(())
    }
}

impl Model {
    /// Compute a 64-aligned common prefix once in each layer, together with its
    /// candidate suffixes. Original input groups and per-candidate output plans
    /// are checked separately. Unsupported shapes decline before request work.
    pub fn try_forward_prefix_tree(
        &mut self,
        inputs: &[&[u32]],
        prefix: usize,
        input_rows: &[usize],
    ) -> Result<Option<SharedPrefixOutput>> {
        if (
            self.cfg.hidden,
            self.cfg.intermediate,
            self.layers.len(),
            self.cfg.heads,
            self.cfg.kv_heads,
            self.cfg.head_dim,
            self.cfg.lin_k_heads,
            self.cfg.lin_v_heads,
            self.cfg.lin_k_dim,
            self.cfg.lin_v_dim,
        ) != (4096, 12288, 32, 16, 4, 256, 16, 32, 128, 128)
            || self.packed_gemm.is_none()
            || self.batched_gdn.is_none()
            || self.reference.is_some()
            || self.lora.is_some()
            || self.fp32_gdn.is_some()
        {
            return Ok(None);
        }
        ensure!(
            (2..=16).contains(&inputs.len())
                && input_rows.len() == inputs.len()
                && prefix >= GDN_CHUNK
                && prefix.is_multiple_of(GDN_CHUNK),
            "invalid prefix tree layout"
        );
        ensure!(
            inputs.iter().all(|ids| ids.len() > prefix
                && ids.len() <= self.cfg.max_positions
                && ids[..prefix] == inputs[0][..prefix]),
            "prefix tree input mismatch"
        );
        ensure!(
            inputs
                .iter()
                .flat_map(|ids| ids.iter())
                .all(|&id| (id as usize) < self.embed.shape[0]),
            "token outside vocabulary"
        );
        ensure!(
            inputs
                .iter()
                .zip(input_rows)
                .all(|(ids, &rows)| rows >= ids.len() && rows <= i32::MAX as usize),
            "invalid original input projection rows"
        );
        let output_rows: Vec<_> = inputs.iter().map(|ids| ids.len()).collect();
        let lengths: Vec<_> = std::iter::once(prefix)
            .chain(inputs.iter().map(|ids| ids.len() - prefix))
            .collect();
        let rows: usize = lengths.iter().sum();
        // Bound scratch independently of candidate count; oversized requests keep
        // the existing segmented shared-prefix path.
        if rows > 8192 {
            return Ok(None);
        }
        let (Some(gemm), Some(ops), Some(gdn)) = (
            cuda::api().reuse_gemm(),
            cuda::api().shared_suffix(),
            cuda::api().shared_batched_gdn(),
        ) else {
            return Ok(None);
        };
        cuda::set_device(0)?;
        let mut shapes = std::collections::BTreeSet::new();
        for layer in &self.layers {
            let (input, output) = match &layer.mixer {
                Mixer::Linear(la) => (&la.in_proj, &la.out),
                Mixer::Full(fa) => (&fa.qkv, &fa.o),
            };
            for (weight, packed) in [
                (input, true),
                (output, false),
                (&layer.gate_up, true),
                (&layer.down, false),
            ] {
                shapes.insert((weight.shape[0], weight.shape[1], packed));
            }
        }
        for (n, k, packed) in shapes {
            let originals = if packed { input_rows } else { &output_rows };
            // SAFETY: shape-only checks consume live host arrays; no request data
            // is executed unless every changed projection is compatible.
            let rc = unsafe {
                (gemm.check)(
                    self.gemm,
                    originals.as_ptr(),
                    originals.len(),
                    &rows,
                    1,
                    n as i32,
                    k as i32,
                )
            };
            if rc == -1 {
                return Ok(None);
            }
            check(rc, "prefix tree projection compatibility")?;
        }
        let state_bytes = self.cfg.lin_v_heads * self.cfg.lin_k_dim * self.cfg.lin_v_dim * F32;
        let bytes = state_bytes + prefix * self.cfg.kv_heads * self.cfg.head_dim * BF16;
        if self
            .tree_workspace
            .as_ref()
            .is_none_or(|(cap, _)| *cap < bytes)
        {
            self.synchronize()?;
            self.tree_workspace = Some((bytes, DeviceBuffer::new(bytes)?));
        }
        self.prepare_scratch(rows)?;
        let buffer = &self.tree_workspace.as_ref().unwrap().1;
        let mut offset = 0;
        let sequences: Vec<_> = lengths
            .iter()
            .map(|&length| {
                let start = offset;
                offset += length;
                (start, length)
            })
            .collect();
        let tree = Tree {
            rows,
            prefix,
            sequences,
            input_rows: input_rows[0],
            output_rows: output_rows[0],
            gemm,
            ops,
            gdn,
            attention_batch: cuda::api().shared_batched_attention(),
            state: buffer.at(0),
            values: buffer.at(state_bytes),
        };
        let ids: Vec<_> = inputs[0][..prefix]
            .iter()
            .copied()
            .chain(inputs.iter().flat_map(|ids| ids[prefix..].iter().copied()))
            .collect();
        let result = (|| {
            let s = self.scratch.as_ref().unwrap();
            let rb = self.cfg.rotary_half * BF16;
            for &(offset, length) in &tree.sequences {
                let position = if offset == 0 { 0 } else { prefix };
                // SAFETY: static tables cover every original position. Compact
                // destination rows are disjoint and their total is scratch capacity.
                unsafe {
                    cuda::copy_dd(
                        s.at(s.custom_cos + offset * rb),
                        s.at(s.cos + position * rb),
                        length * rb,
                        self.stream,
                    )?;
                    cuda::copy_dd(
                        s.at(s.custom_sin + offset * rb),
                        s.at(s.sin + position * rb),
                        length * rb,
                        self.stream,
                    )?;
                }
            }
            self.embed_tokens(s, &ids)?;
            self.run_with_layout(s, &lengths, true, false, None, None, None, Some(&tree))?;
            let hidden = tree.sequences[1..]
                .iter()
                .map(|&(offset, length)| self.last_hidden(s, offset + length))
                .collect::<Result<_>>()?;
            Ok(SharedPrefixOutput {
                hidden,
                saved_tokens: (inputs.len() - 1) * prefix,
            })
        })();
        // Error paths must retire all consumers before scratch/workspace is reused.
        self.synchronize()?;
        result.map(Some)
    }
}
