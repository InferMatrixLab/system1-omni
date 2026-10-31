//! Share pointwise prefix rows while retaining each complete sequence's mixers.
use super::*;

#[derive(Clone, Copy)]
pub(super) enum Projection {
    MixerInput,
    MixerOutput,
    GateUp,
    Down,
}

pub(super) struct Compaction {
    pub rows: usize,
    full_rows: usize,
    original_output_rows: usize,
    expand: *const i32,
    gather: *const i32,
    gemm: cuda::ReuseGemm,
}

struct RowMaps {
    expand: Vec<i32>,
    gather: Vec<i32>,
    ends: Vec<usize>,
}

fn row_maps(lengths: &[usize], prefix: usize) -> RowMaps {
    let mut expand = Vec::new();
    let mut gather: Vec<i32> = (0..prefix as i32).collect();
    let mut ends = Vec::new();
    let mut full_start = 0;
    let mut compact_start = prefix;
    for &length in lengths {
        expand.extend(0..prefix as i32);
        expand.extend((compact_start..compact_start + length - prefix).map(|n| n as i32));
        gather.extend((full_start + prefix..full_start + length).map(|n| n as i32));
        compact_start += length - prefix;
        full_start += length;
        ends.push(compact_start);
    }
    RowMaps {
        expand,
        gather,
        ends,
    }
}

impl Compaction {
    /// Scratch has full sequence capacity; gate_up is temporary until the MLP.
    pub(super) fn project(
        &self,
        model: &Model,
        s: &Scratch,
        x: usize,
        weight: &Tensor,
        y: usize,
        role: Projection,
    ) -> Result<()> {
        let (n, k) = (weight.shape[0], weight.shape[1]);
        let (source, target, original_rows) = match role {
            Projection::MixerInput => (x, s.gate_up, self.full_rows),
            Projection::MixerOutput => (s.x, y, self.original_output_rows),
            Projection::GateUp => (x, y, self.full_rows),
            Projection::Down => (x, y, self.original_output_rows),
        };
        // SAFETY: checked maps index the full mixer rows or compact pointwise rows.
        // All buffers are distinct scratch regions; the stream orders their reuse.
        unsafe {
            if matches!(role, Projection::MixerOutput) {
                check(
                    (cuda::api().cs1_embed)(
                        self.gather,
                        s.at(x),
                        s.at(s.x),
                        self.rows as i32,
                        k as i32,
                        model.stream,
                    ),
                    "compact mixer output",
                )?;
            }
            check(
                (self.gemm.run)(
                    model.gemm,
                    s.at(source),
                    weight.ptr,
                    s.at(target),
                    self.rows as i32,
                    n as i32,
                    k as i32,
                    original_rows as i32,
                    model.stream,
                ),
                "compact retained projection",
            )?;
            if matches!(role, Projection::MixerInput) {
                check(
                    (cuda::api().cs1_embed)(
                        self.expand,
                        s.at(target),
                        s.at(y),
                        self.full_rows as i32,
                        n as i32,
                        model.stream,
                    ),
                    "expand mixer input",
                )?;
            }
        }
        Ok(())
    }
}

impl Model {
    /// Capture only the pointwise-compacted layout. Model-owned maps and scratch
    /// remain live; every request replaces values before replay. Prefix trees use
    /// their eager path and never reference these graph keys.
    fn run_compaction(
        &mut self,
        lengths: &[usize],
        prefix: usize,
        layout: &Compaction,
    ) -> Result<()> {
        let run = |model: &Self| {
            model.run_with_layout(
                model.scratch.as_ref().unwrap(),
                lengths,
                false,
                false,
                None,
                None,
                Some(layout),
                None,
            )
        };
        if !self.graph_enabled {
            return run(self);
        }
        // Normal text keys contain positive lengths or zero-delimited original
        // groups. The sentinel separates compact layouts even at equal row counts.
        let mut key = vec![usize::MAX, 0, prefix];
        key.extend_from_slice(lengths);
        if let Some((_, graph)) = self.graphs.iter().find(|(shape, _)| *shape == key) {
            graph.launch(self.stream)?;
            self.graph_stats.replays += 1;
            return Ok(());
        }
        run(self)?;
        // Record while eager kernels execute; first capture must keep the eager
        // result, since replay here would advance the residual twice.
        let captured = cuda::Graph::capture(self.stream, || run(self));
        self.cache_graph(&key, false, layout.full_rows, captured)
    }

    /// Share pointwise prefix rows, keeping original full-length causal mixers.
    /// Unsupported algorithms decline before any request values are computed.
    /// The pinned9B widths guarantee mixer inputs fit the gate/up temporary and
    /// gathered mixer outputs fit the compact hidden buffer.
    pub fn try_forward_prefix_compaction(
        &mut self,
        inputs: &[&[u32]],
        prefix: usize,
    ) -> Result<Option<Vec<Vec<f32>>>> {
        if self.cfg.hidden != 4096
            || self.cfg.intermediate != 12288
            || self.layers.len() != 32
            || (
                self.cfg.heads,
                self.cfg.kv_heads,
                self.cfg.head_dim,
                self.cfg.lin_k_heads,
                self.cfg.lin_v_heads,
                self.cfg.lin_k_dim,
                self.cfg.lin_v_dim,
            ) != (16, 4, 256, 16, 32, 128, 128)
            || self.packed_gemm.is_none()
            || self.reference.is_some()
            || self.lora.is_some()
            || self.fp32_gdn.is_some()
        {
            return Ok(None);
        }
        ensure!(
            (2..=16).contains(&inputs.len()) && (16..512).contains(&prefix),
            "invalid projection prefix"
        );
        ensure!(
            inputs
                .iter()
                .all(|ids| ids.len() > prefix && ids.len() <= self.cfg.max_positions)
                && inputs
                    .iter()
                    .all(|ids| ids[..prefix] == inputs[0][..prefix]),
            "projection prefix differs or leaves an empty suffix"
        );
        ensure!(
            inputs
                .iter()
                .flat_map(|ids| ids.iter())
                .all(|&id| (id as usize) < self.embed.shape[0]),
            "token id outside the vocabulary"
        );
        let lengths: Vec<usize> = inputs.iter().map(|ids| ids.len()).collect();
        let full_rows = lengths
            .iter()
            .try_fold(0usize, |n, &m| n.checked_add(m))
            .context("projection row overflow")?;
        if full_rows > 4096 {
            return Ok(None);
        }
        let rows = full_rows - (inputs.len() - 1) * prefix;
        let Some(gemm) = cuda::api().reuse_gemm() else {
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
            let originals: &[usize] = if packed {
                std::slice::from_ref(&full_rows)
            } else {
                &lengths
            };
            // SAFETY: shape-only check reads live arrays; all projections are checked
            // before embedding or modifying residuals, including original split-K.
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
            check(rc, "projection compaction compatibility")?;
        }
        let maps = row_maps(&lengths, prefix);
        let capacity = if self.graph_enabled {
            8192
        } else {
            full_rows + rows
        };
        if self
            .compaction_maps
            .as_ref()
            .is_none_or(|(n, _)| *n < capacity)
        {
            // First allocation cannot invalidate any captured map address. Only
            // replacement of an existing map allocation requires graph retirement.
            if self.compaction_maps.is_some() {
                let retired = self.synchronize();
                if !self.graphs.is_empty() {
                    self.graph_stats.invalidations += 1;
                }
                self.graphs.clear();
                retired?;
            }
            self.compaction_maps = Some((capacity, DeviceBuffer::new(capacity * 4)?));
        }
        self.prepare_scratch(full_rows)?;
        let map_start = self.compaction_maps.as_ref().unwrap().1.at(0);
        let map_gather = self.compaction_maps.as_ref().unwrap().1.at(full_rows * 4);
        let bytes: Vec<u8> = maps
            .expand
            .iter()
            .chain(&maps.gather)
            .flat_map(|n| n.to_le_bytes())
            .collect();
        let layout = Compaction {
            rows,
            full_rows,
            original_output_rows: lengths[0],
            expand: map_start.cast(),
            gather: map_gather.cast(),
            gemm,
        };
        let ids: Vec<u32> = inputs[0][..prefix]
            .iter()
            .copied()
            .chain(inputs.iter().flat_map(|ids| ids[prefix..].iter().copied()))
            .collect();
        let result = (|| {
            // SAFETY: persistent model-owned buffer holds exactly these live row maps.
            unsafe {
                cuda::upload(map_start, &bytes, self.stream)?;
            }
            let s = self.scratch.as_ref().unwrap();
            self.embed_tokens(s, &ids)?;
            self.run_compaction(&lengths, prefix, &layout)?;
            let s = self.scratch.as_ref().unwrap();
            maps.ends
                .iter()
                .map(|&end| self.last_hidden(s, end))
                .collect::<Result<Vec<_>>>()
        })();
        // No caller may reuse maps/scratch while a failed operation still owns them.
        self.synchronize()?;
        result.map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn maps_preserve_every_suffix_and_restore_original_row_order() {
        let maps = row_maps(&[5, 6, 4], 2);
        assert_eq!(maps.expand, [0, 1, 2, 3, 4, 0, 1, 5, 6, 7, 8, 0, 1, 9, 10]);
        assert_eq!(maps.gather, [0, 1, 2, 3, 4, 7, 8, 9, 10, 13, 14]);
        assert_eq!(maps.ends, [5, 9, 11]);
        let full = [10, 11, 12, 13, 14, 10, 11, 22, 23, 24, 25, 10, 11, 32, 33];
        let compact: Vec<_> = maps.gather.iter().map(|&i| full[i as usize]).collect();
        assert_eq!(
            maps.expand
                .iter()
                .map(|&i| compact[i as usize])
                .collect::<Vec<_>>(),
            full
        );
    }
}
