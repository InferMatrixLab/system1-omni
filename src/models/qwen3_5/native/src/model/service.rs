//! Eager segmented text readout with the pinned Torch 2.8 arithmetic.
//! Each call owns temporary state; no prefix survives across calls.
use super::*;

pub(super) struct ServiceWindow {
    pub ops: cuda::ServiceReadout,
    pub leading: usize,
    pub real_end: usize,
    pub key_left: usize,
    pub workspace: *mut c_void,
    pub projection: Option<ProjectionShape>,
}

pub(super) struct ProjectionShape {
    pub rows: usize,
    pub index: usize,
    pub input: *mut c_void,
    pub output: *mut c_void,
}

impl Model {
    /// Select Kev's pinned default-service arithmetic before any execution.
    /// The optional CUDA symbols leave other Qwen consumers on their old profile.
    pub fn enable_service_readout_numerics(&mut self) -> Result<()> {
        ensure!(
            self.cfg.hidden == 2560
                && self.cfg.head_dim == 256
                && self.cfg.lin_k_dim == 128
                && self.cfg.lin_v_dim == 128
                && self.cfg.lin_k_heads == 16
                && self.cfg.lin_v_heads == 32,
            "service readout requires the pinned 4B layout"
        );
        ensure!(
            self.scratch.is_none()
                && self.lora.is_none()
                && self.reference.is_none()
                && self.fp32_gdn.is_none()
                && !self.graph_enabled
                && self.batched_gdn.is_none()
                && self.packed_gemm.is_none(),
            "select service readout on a merged eager model before its first forward"
        );
        cuda::set_device(0)?;
        let ops = cuda::api()
            .service_readout()
            .context("Kev requires recipe/kev/build_cuda.sh")?;
        ensure!(
            cuda::api().shared_suffix().is_some(),
            "missing prefix convolution CUDA symbol"
        );
        let gdn = cuda::api().torch28_gdn()?;
        let reference = cuda::api().torch28_reference()?;
        self.synchronize()?;
        // SAFETY: no queued work/capture uses the old model-owned handle.
        let gemm = unsafe { (reference.gemm_create)(GEMM_WORKSPACE) };
        ensure!(!gemm.is_null(), "service cuBLASLt setup failed");
        unsafe { (cuda::api().cs1_gemm_destroy)(self.gemm) };
        self.gemm = gemm;
        self.reference = Some(reference);
        self.fp32_gdn = Some(gdn);
        self.service_ops = Some(ops);
        Ok(())
    }

    /// Compute state then question with explicit padding, returning selected
    /// question positions in caller order. All temporary allocations are drained
    /// on success and failure before their ownership ends.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_segmented_readout(
        &mut self,
        input: &MultimodalInput<'_>,
        split: usize,
        state_left_pad: usize,
        branch_right_pad: usize,
        prefix_left_pad: usize,
        pad_id: u32,
        batch_rows: usize,
        batch_index: usize,
        readouts: &[usize],
    ) -> Result<Readout> {
        let ops = self
            .service_ops
            .context("select service readout numerics first")?;
        input.validate(
            self.cfg.hidden,
            self.embed.shape[0],
            self.cfg.image_token_id.unwrap_or(u32::MAX),
            self.cfg.max_positions,
        )?;
        ensure!(
            input
                .position_ids
                .iter()
                .all(|p| *p == input.position_ids[0]),
            "segmented readout requires text positions"
        );
        ensure!(
            (pad_id as usize) < self.embed.shape[0],
            "padding token outside vocabulary"
        );
        ensure!(
            split > 0 && split < input.token_ids.len(),
            "invalid state split"
        );
        ensure!(
            input.image_token_indices.is_empty() && input.image_embeddings.is_empty(),
            "text readout only"
        );
        ensure!(
            readouts
                .iter()
                .all(|&r| r >= split && r < input.token_ids.len()),
            "readout outside question"
        );
        let state_len = split
            .checked_add(state_left_pad)
            .context("state length overflow")?;
        let question_len = input.token_ids.len() - split;
        let rows = question_len
            .checked_add(branch_right_pad)
            .context("question length overflow")?;
        let prefix_len = split
            .checked_add(prefix_left_pad)
            .context("prefix length overflow")?;
        let total = prefix_len
            .checked_add(rows)
            .context("total length overflow")?;
        ensure!(
            state_len >= 3
                && total <= self.cfg.max_positions
                && state_len <= self.cfg.max_positions,
            "segment capacity"
        );
        ensure!(state_len <= 65536 && total <= 65536, "service shape limit");
        self.prepare_scratch(total.max(state_len))?;
        let mut state = self.alloc_prefix_storage(state_len.max(prefix_len))?;
        state.len = state_len;
        // SAFETY: pure size queries for the checked nonempty extents.
        let size = unsafe {
            (ops.bytes)(state_len as i32, state_len as i32, self.cfg.heads as i32).max((ops.bytes)(
                rows as i32,
                total as i32,
                self.cfg.heads as i32,
            ))
        };
        ensure!(size > 0, "invalid service workspace size");
        let workspace = DeviceBuffer::new(size)?;
        ensure!(
            (1..=32).contains(&batch_rows) && batch_index < batch_rows,
            "invalid projection batch"
        );
        let projection_rows = rows
            .checked_mul(batch_rows)
            .context("projection shape overflow")?;
        ensure!(projection_rows <= 65536, "projection shape limit");
        let width = self
            .layers
            .iter()
            .flat_map(|l| [&l.gate_up, &l.down])
            .flat_map(|w| w.shape.iter().copied())
            .max()
            .unwrap()
            .max(Widths::of(&self.cfg).gdn_in)
            .max(Widths::of(&self.cfg).attn_in);
        let projection = if batch_rows > 1 {
            Some((
                DeviceBuffer::new(projection_rows * width * BF16)?,
                DeviceBuffer::new(projection_rows * width * BF16)?,
            ))
        } else {
            None
        };
        let result = (|| {
            let s = self.scratch.as_ref().unwrap();
            let mut ids = vec![pad_id; state_left_pad];
            ids.extend_from_slice(&input.token_ids[..split]);
            let mut pos = vec![0; state_left_pad];
            pos.extend_from_slice(&input.position_ids[0][..split]);
            self.upload_positions(s, [&pos, &pos, &pos])?;
            self.embed_tokens_at(s, &ids, 0)?;
            let plan = ServiceWindow {
                ops,
                leading: state_left_pad,
                real_end: state_len,
                key_left: state_left_pad,
                workspace: workspace.at(0),
                projection: None,
            };
            self.run_window(
                s,
                0,
                state_len,
                true,
                Some(&mut state),
                None,
                self.gemm,
                None,
                Some(&plan),
            )?;
            // Keep the real prefix KV, right-aligned in the continuation's key extent.
            let kb = self.cfg.kv_heads * self.cfg.head_dim * BF16;
            for (k, v) in &state.attn_kv {
                for buffer in [k, v] {
                    // SAFETY: state buffers hold cap * kv width; scratch holds split rows.
                    unsafe {
                        cuda::copy_dd(
                            s.at(s.ak),
                            buffer.at(state_left_pad * kb),
                            split * kb,
                            self.stream,
                        )?;
                        check(
                            (ops.zero)(buffer.at(0), state.cap * kb, self.stream),
                            "clear padding keys",
                        )?;
                        cuda::copy_dd(
                            buffer.at(prefix_left_pad * kb),
                            s.at(s.ak),
                            split * kb,
                            self.stream,
                        )?;
                    }
                }
            }
            state.len = prefix_len;
            state.initialized = true;
            let mut ids = input.token_ids[split..].to_vec();
            ids.resize(rows, pad_id);
            let mut pos = input.position_ids[0][split..].to_vec();
            pos.resize(rows, 0);
            self.upload_positions(s, [&pos, &pos, &pos])?;
            self.embed_tokens_at(s, &ids, prefix_len)?;
            let plan = ServiceWindow {
                ops,
                leading: 0,
                real_end: question_len,
                key_left: prefix_left_pad,
                workspace: workspace.at(0),
                projection: projection.as_ref().map(|(x, y)| ProjectionShape {
                    rows: projection_rows,
                    index: batch_index,
                    input: x.at(0),
                    output: y.at(0),
                }),
            };
            self.run_window(
                s,
                prefix_len,
                total,
                true,
                None,
                Some(&state),
                self.gemm,
                None,
                Some(&plan),
            )?;
            let hidden = readouts
                .iter()
                .map(|&r| self.hidden_row(s, r - split + prefix_len))
                .collect::<Result<Vec<_>>>()?;
            Ok(Readout {
                hidden,
                logprobs: vec![],
            })
        })();
        let drained = self.synchronize();
        match result {
            Ok(r) => {
                drained?;
                Ok(r)
            }
            Err(e) => {
                let _ = drained;
                Err(e)
            }
        }
    }
}
