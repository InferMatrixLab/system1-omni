//! Independent multimodal prompts, sharing weights and launches within one call.
use super::*;

/// Maximum scratch rows or tokens in one unshared batch, including image placeholders.
pub const MAX_MULTIMODAL_BATCH_TOKENS: usize = 8192;

fn batch_lengths(inputs: &[MultimodalInput<'_>], prefix: usize) -> Result<Vec<usize>> {
    ensure!(
        (1..=16).contains(&inputs.len()),
        "multimodal batch requires 1..=16 prompts"
    );
    let mut total = 0usize;
    let lengths = inputs
        .iter()
        .map(|input| {
            let n = input.token_ids.len();
            ensure!(
                (1..=MAX_MULTIMODAL_BATCH_TOKENS).contains(&n),
                "empty or oversized multimodal batch prompt"
            );
            total = total
                .checked_add(n)
                .context("multimodal batch token overflow")?;
            ensure!(
                prefix > 0 || total <= MAX_MULTIMODAL_BATCH_TOKENS,
                "multimodal batch exceeds token budget"
            );
            Ok(n)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(lengths)
}

impl Model {
    /// Enable bounded multimodal packing before the first forward. Other callers
    /// keep their existing projection/conv dispatch; only independent multimodal
    /// batches use the packed GEMM path and the existing batched GDN operator.
    pub fn enable_multimodal_batching(&mut self) -> Result<()> {
        ensure!(
            self.scratch.is_none() && self.packed_gemm.is_none(),
            "select multimodal batching before the first forward, without text packing"
        );
        self.ensure_fixed_profile()?;
        let packed = cuda::api().packed_gemm()?;
        let gdn = cuda::api().batched_gdn()?;
        self.multimodal_packed = Some(packed);
        self.batched_gdn = Some(gdn);
        Ok(())
    }

    /// Independent causal multimodal prompts, in input order. Each prompt retains
    /// its original GEMM algorithms, T/H/W positions and attention/GDN boundaries.
    /// Supports at most 16 prompts and 8192 combined tokens on merged/default
    /// numerics. Call `enable_multimodal_batching` before any execution.
    pub fn forward_multimodal_batch(
        &mut self,
        inputs: &[MultimodalInput<'_>],
    ) -> Result<Vec<Vec<f32>>> {
        self.forward_multimodal_batch_with_prefix(inputs, 0)
    }

    /// Share an exact chunk-aligned prefix within compatible cohorts, then pack
    /// suffixes retaining the complete prompts' algorithms. Unsupported cohorts
    /// retain serial execution. Prefix image values and all three rotary axes
    /// must agree; every request computes fresh prefix state. Each of at most
    /// 16 prompts is bounded to 8192 tokens. Shared execution allocates at most
    /// 8192 scratch rows; raw duplicate prefix rows do not consume that budget.
    pub fn forward_multimodal_batch_with_prefix(
        &mut self,
        inputs: &[MultimodalInput<'_>],
        prefix_len: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.run_multimodal_batch(inputs, None, prefix_len)
        }));
        self.finish_multimodal(result)
    }

    /// Independent prompts using borrowed adapted device rows. The producer
    /// finishes writes before entry; all consumer work drains on return or unwind.
    pub fn forward_multimodal_device_batch(
        &mut self,
        inputs: &[DeviceMultimodalInput<'_>],
    ) -> Result<Vec<Vec<f32>>> {
        self.forward_multimodal_device_batch_with_prefix(inputs, 0)
    }

    /// Device counterpart of `forward_multimodal_batch_with_prefix`. Sharing
    /// image rows requires the same borrowed allocation for the shared prefix.
    /// Captures retain only model-owned scratch/prefix addresses.
    pub fn forward_multimodal_device_batch_with_prefix(
        &mut self,
        inputs: &[DeviceMultimodalInput<'_>],
        prefix_len: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let image_token = self
                .cfg
                .image_token_id
                .context("checkpoint has no image_token_id")?;
            for input in inputs {
                input.validate(
                    self.cfg.hidden,
                    self.embed.shape[0],
                    image_token,
                    self.cfg.max_positions,
                    0,
                )?;
            }
            let host: Vec<_> = inputs
                .iter()
                .map(|input| MultimodalInput {
                    token_ids: input.token_ids,
                    image_token_indices: input.image_token_indices,
                    image_embeddings: &[],
                    position_ids: input.position_ids,
                })
                .collect();
            let devices: Vec<_> = inputs.iter().map(|input| input.image_embeddings).collect();
            self.run_multimodal_batch(&host, Some(&devices), prefix_len)
        }));
        self.finish_multimodal(result)
    }

    fn finish_multimodal<T>(&self, result: std::thread::Result<Result<T>>) -> Result<T> {
        let synced = self.synchronize();
        match result {
            Ok(value) => {
                synced?;
                value
            }
            Err(panic) => std::panic::resume_unwind(panic),
        }
    }

    pub(super) fn run_multimodal_batch(
        &mut self,
        inputs: &[MultimodalInput<'_>],
        devices: Option<&[cuda::DeviceBf16Matrix<'_>]>,
        prefix_len: usize,
    ) -> Result<Vec<Vec<f32>>> {
        ensure!(
            self.multimodal_packed.is_some(),
            "enable multimodal batching before execution"
        );
        self.ensure_fixed_profile()?;
        ensure!(
            self.matched_ref == 0,
            "multimodal batching does not use a matched-prefix handle"
        );
        let lengths = batch_lengths(inputs, prefix_len)?;
        let image_token = self
            .cfg
            .image_token_id
            .context("checkpoint has no image_token_id")?;
        if devices.is_none() {
            for input in inputs {
                input.validate(
                    self.cfg.hidden,
                    self.embed.shape[0],
                    image_token,
                    self.cfg.max_positions,
                )?;
            }
        }
        if prefix_len > 0 {
            super::multimodal_shared::common_header(inputs, devices, prefix_len, self.cfg.hidden)?;
        }
        cuda::set_device(0)?;
        if let Some(devices) = devices {
            ensure!(devices.len() == inputs.len(), "device batch count mismatch");
            for (index, features) in devices.iter().enumerate() {
                // Immutable producer views can repeat across questions. Scan each
                // allocation/device/extent once in this call, never across calls.
                if !devices[..index]
                    .iter()
                    .any(|previous| features.same_storage(previous))
                {
                    features.validate_finite(self.stream)?;
                }
            }
        }
        if prefix_len > 0 {
            if let Some(rows) = self.try_multimodal_shared(inputs, devices, &lengths, prefix_len)? {
                return Ok(rows);
            }
            return self.serial_multimodal(inputs, devices);
        }
        let t = lengths.iter().sum();
        self.prepare_scratch(t)?;
        let s = self.scratch.as_ref().unwrap();
        let positions: [Vec<i64>; 3] = std::array::from_fn(|axis| {
            inputs
                .iter()
                .flat_map(|input| input.position_ids[axis].iter().copied())
                .collect()
        });
        self.upload_positions(s, [&positions[0], &positions[1], &positions[2]])?;
        let ids: Vec<_> = inputs
            .iter()
            .flat_map(|input| input.token_ids.iter().copied())
            .collect();
        self.embed_tokens(s, &ids)?;
        let mut offset = 0;
        for (index, input) in inputs.iter().enumerate() {
            self.stage_multimodal_rows(
                s,
                input,
                devices.map(|d| d[index]),
                0..input.token_ids.len(),
                offset,
            )?;
            offset += input.token_ids.len();
        }
        if self.graph_enabled {
            if let Some((_, graph)) = self.mm_graphs.iter().find(|(shape, _)| *shape == lengths) {
                graph.launch(self.stream)?;
                self.graph_stats.replays += 1;
                if std::env::var("CUA_S1_GRAPH_TRACE").as_deref() == Ok("1") {
                    eprintln!(
                        "CUA_S1_GRAPH mode=multimodal event=replay tokens={t} sequences={}",
                        lengths.len()
                    );
                }
            } else {
                self.run(s, &lengths, true, false)?;
                cuda::synchronize(self.stream)?;
                let captured =
                    cuda::Graph::capture(self.stream, || self.run(s, &lengths, true, false));
                self.cache_graph(&lengths, true, captured)?;
            }
        } else {
            self.run(s, &lengths, true, false)?;
        }
        let s = self.scratch.as_ref().unwrap();
        let mut end = 0;
        lengths
            .iter()
            .map(|&n| {
                end += n;
                self.last_hidden(s, end)
            })
            .collect()
    }

    pub(super) fn stage_multimodal_rows(
        &self,
        s: &Scratch,
        input: &MultimodalInput<'_>,
        device: Option<cuda::DeviceBf16Matrix<'_>>,
        range: std::ops::Range<usize>,
        offset: usize,
    ) -> Result<()> {
        let indices = input.image_token_indices;
        let first = indices.partition_point(|&i| i < range.start);
        let limit = indices.partition_point(|&i| i < range.end);
        let row_bytes = self.cfg.hidden * BF16;
        if let Some(device) = device {
            let mut begin = first;
            while begin < limit {
                let mut end = begin + 1;
                while end < limit && indices[end] == indices[end - 1] + 1 {
                    end += 1;
                }
                // SAFETY: complete input metadata and source extents were checked;
                // the validated window lies in owned scratch. Borrow lasts to drain.
                unsafe {
                    cuda::copy_dd(
                        s.at(s.res + (offset + indices[begin] - range.start) * row_bytes),
                        device.at(begin * row_bytes),
                        (end - begin) * row_bytes,
                        self.stream,
                    )?;
                }
                begin = end;
            }
            Ok(())
        } else {
            let indices: Vec<_> = indices[first..limit]
                .iter()
                .map(|&i| i - range.start)
                .collect();
            let window = MultimodalInput {
                token_ids: &input.token_ids[range.clone()],
                image_token_indices: &indices,
                image_embeddings: &input.image_embeddings
                    [first * self.cfg.hidden..limit * self.cfg.hidden],
                position_ids: input.position_ids.map(|axis| &axis[range.clone()]),
            };
            self.overwrite_image_rows(s, &window, offset)
        }
    }

    pub(super) fn serial_multimodal(
        &mut self,
        inputs: &[MultimodalInput<'_>],
        devices: Option<&[cuda::DeviceBf16Matrix<'_>]>,
    ) -> Result<Vec<Vec<f32>>> {
        (0..inputs.len())
            .map(|i| {
                self.run_multimodal_batch(&inputs[i..i + 1], devices.map(|d| &d[i..i + 1]), 0)
                    .map(|mut rows| rows.pop().unwrap())
            })
            .collect()
    }
}

#[cfg(test)]
#[path = "../../../../../../tests/qwen3_5/multimodal_batch.rs"]
mod tests;
