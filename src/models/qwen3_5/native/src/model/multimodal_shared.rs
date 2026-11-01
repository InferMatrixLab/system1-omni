//! Fresh request-local multimodal prefixes with independent compact suffixes.
use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum PassKey {
    Prefix {
        prefix: usize,
        reference: usize,
    },
    Window {
        prefix: usize,
        suffix: usize,
        reference: usize,
    },
    Compact {
        prefix: usize,
        lengths: Vec<usize>,
        reference: usize,
    },
    Batch {
        prefix: usize,
        lengths: Vec<usize>,
        reference: usize,
    },
}

pub(super) struct Entry {
    pub(super) key: PassKey,
    graph: cuda::Graph,
}

fn compact_rows(lengths: &[usize], prefix: usize) -> Option<usize> {
    if !(2..=16).contains(&lengths.len())
        || prefix < 64
        || !prefix.is_multiple_of(64)
        || lengths.iter().any(|&n| n <= prefix)
    {
        return None;
    }
    let rows = lengths
        .iter()
        .try_fold(prefix, |rows, &n| rows.checked_add(n.checked_sub(prefix)?))?;
    (rows <= super::multimodal_batch::MAX_MULTIMODAL_BATCH_TOKENS
        && rows - prefix <= MAX_SUFFIX_BATCH_ROWS)
        .then_some(rows)
}

pub(super) fn common_header(
    inputs: &[MultimodalInput<'_>],
    devices: Option<&[cuda::DeviceBf16Matrix<'_>]>,
    prefix: usize,
    hidden: usize,
) -> Result<()> {
    ensure!(
        inputs.len() >= 2 && prefix >= 64 && prefix.is_multiple_of(64),
        "shared multimodal prefix requires multiple prompts and a positive chunk multiple"
    );
    ensure!(
        inputs.iter().all(|p| p.token_ids.len() > prefix
            && p.position_ids
                .iter()
                .all(|axis| axis.len() == p.token_ids.len())),
        "invalid shared prefix boundary"
    );
    let first = &inputs[0];
    let rows = first.image_token_indices.partition_point(|&i| i < prefix);
    let elements = rows
        .checked_mul(hidden)
        .context("shared image extent overflow")?;
    for (index, input) in inputs.iter().enumerate().skip(1) {
        ensure!(
            input.token_ids[..prefix] == first.token_ids[..prefix]
                && (0..3)
                    .all(|axis| input.position_ids[axis][..prefix]
                        == first.position_ids[axis][..prefix]),
            "multimodal prefix tokens or positions differ"
        );
        let count = input.image_token_indices.partition_point(|&i| i < prefix);
        ensure!(
            input.image_token_indices[..count] == first.image_token_indices[..rows],
            "multimodal prefix image indices differ"
        );
        if let Some(devices) = devices {
            ensure!(devices.len() == inputs.len(), "device batch count mismatch");
            ensure!(
                rows == 0 || devices[index].at(0) == devices[0].at(0),
                "shared image prefix requires the same resident allocation"
            );
        } else {
            let a = first
                .image_embeddings
                .get(..elements)
                .context("shared image extent")?;
            let b = input
                .image_embeddings
                .get(..elements)
                .context("shared image extent")?;
            ensure!(
                a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()),
                "multimodal prefix image values differ"
            );
        }
    }
    Ok(())
}

impl Model {
    pub(super) fn try_multimodal_shared(
        &mut self,
        inputs: &[MultimodalInput<'_>],
        devices: Option<&[cuda::DeviceBf16Matrix<'_>]>,
        lengths: &[usize],
        prefix: usize,
    ) -> Result<Option<Vec<Vec<f32>>>> {
        let Some(reuse) = cuda::api().reuse_gemm() else {
            return Ok(None);
        };
        let mut shapes = std::collections::BTreeSet::new();
        for layer in &self.layers {
            let mixer = match &layer.mixer {
                Mixer::Linear(l) => [&l.in_proj, &l.out],
                Mixer::Full(l) => [&l.qkv, &l.o],
            };
            for weight in mixer.into_iter().chain([&layer.gate_up, &layer.down]) {
                shapes.insert((weight.shape[0], weight.shape[1]));
            }
        }
        let compatible = |originals: &[usize], targets: &[usize]| -> Result<bool> {
            for &(n, k) in &shapes {
                // SAFETY: bounded positive host row arrays and validated weights;
                // capability checks do not produce device outputs or prefix state.
                let rc = unsafe {
                    (reuse.check)(
                        self.gemm,
                        originals.as_ptr(),
                        originals.len(),
                        targets.as_ptr(),
                        targets.len(),
                        n as i32,
                        k as i32,
                    )
                };
                if rc == -1 {
                    return Ok(false);
                }
                check(rc, "multimodal suffix compatibility")?;
            }
            Ok(true)
        };
        let ops = cuda::api().shared_suffix();
        let Some(cohorts) = compatible_prefix_groups(inputs.len(), |indices| {
            let originals: Vec<_> = indices.iter().map(|&i| lengths[i]).collect();
            if ops.is_some()
                && let Some(rows) = compact_rows(&originals, prefix)
                && compatible(&originals, &[rows])?
            {
                return Ok(true);
            }
            let mut targets: Vec<_> = originals.iter().map(|&n| n - prefix).collect();
            targets.push(prefix);
            compatible(&originals, &targets)
        })?
        else {
            return Ok(None);
        };
        let mut plans = Vec::new();
        let mut required = *lengths.iter().max().unwrap();
        for cohort in cohorts {
            let mut segments = Vec::new();
            let originals: Vec<_> = cohort.iter().map(|&i| lengths[i]).collect();
            let compact = if ops.is_some()
                && let Some(rows) = compact_rows(&originals, prefix)
                && compatible(&originals, &[rows])?
            {
                required = required.max(rows);
                true
            } else {
                false
            };
            if cohort.len() > 1 && !compact {
                for indices in suffix_segments(&cohort, lengths, prefix) {
                    let rows: usize = indices.iter().map(|&i| lengths[i] - prefix).sum();
                    let packed = indices.len() > 1
                        && ops.is_some()
                        && compatible(&[lengths[indices[0]]], &[rows])?;
                    if packed {
                        required = required.max(rows);
                    }
                    segments.push((indices, packed));
                }
            }
            plans.push((cohort, segments, compact));
        }
        // Window Graphs retain this model-owned prefix allocation. Retire those
        // Graphs before replacing it, including after failures or capacity growth.
        if self
            .mm_batch_prefix
            .as_ref()
            .is_none_or(|(cap, _)| *cap < prefix)
        {
            self.synchronize()?;
            if !self.mm_batch_windows.is_empty() {
                self.graph_stats.invalidations += 1;
            }
            self.mm_batch_windows.clear();
            self.mm_batch_prefix = None;
            self.mm_batch_prefix = Some((prefix, self.alloc_prefix(prefix)?));
        }
        self.prepare_scratch(required)?;
        let (capacity, mut state) = self.mm_batch_prefix.take().unwrap();
        state.len = prefix;
        state.initialized = false;
        // Restore ownership even on unwind: worker panic recovery may retain the
        // model, so cached Graphs must never outlive recorded prefix allocations.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut hidden = vec![Vec::new(); inputs.len()];
            for (cohort, segments, compact) in plans {
                if cohort.len() == 1 {
                    let index = cohort[0];
                    hidden[index] = self
                        .run_multimodal_batch(
                            &inputs[index..index + 1],
                            devices.map(|d| &d[index..index + 1]),
                            0,
                        )?
                        .pop()
                        .unwrap();
                    continue;
                }
                let first = cohort[0];
                let reference = lengths[first];
                state.initialized = false;
                if compact {
                    let batch = SuffixBatch {
                        prefix: &state,
                        ops: ops.unwrap(),
                        gemm: reuse,
                        original_rows: reference,
                        seeded_gdn: cuda::api().seeded_gdn_batch(),
                        include_prefix: true,
                    };
                    let rows = self.multimodal_compact_pass(inputs, devices, &cohort, &batch)?;
                    for (index, row) in cohort.into_iter().zip(rows) {
                        hidden[index] = row;
                    }
                    continue;
                }
                let s = self.scratch.as_ref().unwrap();
                self.upload_positions(s, inputs[first].position_ids.map(|axis| &axis[..prefix]))?;
                self.embed_tokens(s, &inputs[first].token_ids[..prefix])?;
                self.stage_multimodal_rows(
                    s,
                    &inputs[first],
                    devices.map(|d| d[first]),
                    0..prefix,
                    0,
                )?;
                self.multimodal_prefix_pass(&mut state, reference, reuse)?;
                state.initialized = true;
                for (indices, packed) in segments {
                    if packed {
                        let lens: Vec<_> = indices.iter().map(|&i| lengths[i] - prefix).collect();
                        let ids: Vec<_> = indices
                            .iter()
                            .flat_map(|&i| inputs[i].token_ids[prefix..].iter().copied())
                            .collect();
                        let positions: [Vec<i64>; 3] = std::array::from_fn(|axis| {
                            indices
                                .iter()
                                .flat_map(|&i| {
                                    inputs[i].position_ids[axis][prefix..].iter().copied()
                                })
                                .collect()
                        });
                        let s = self.scratch.as_ref().unwrap();
                        self.upload_positions(s, [&positions[0], &positions[1], &positions[2]])?;
                        self.embed_tokens(s, &ids)?;
                        let mut offset = 0;
                        for (&index, &length) in indices.iter().zip(&lens) {
                            self.stage_multimodal_rows(
                                s,
                                &inputs[index],
                                devices.map(|d| d[index]),
                                prefix..lengths[index],
                                offset,
                            )?;
                            offset += length;
                        }
                        self.multimodal_suffix_pass(
                            &state,
                            reference,
                            &lens,
                            Some(ops.unwrap()),
                            reuse,
                        )?;
                        let s = self.scratch.as_ref().unwrap();
                        let mut end = 0;
                        for (&index, &length) in indices.iter().zip(&lens) {
                            end += length;
                            hidden[index] = self.last_hidden(s, end)?;
                        }
                    } else {
                        for index in indices {
                            let s = self.scratch.as_ref().unwrap();
                            self.upload_positions(
                                s,
                                inputs[index].position_ids.map(|axis| &axis[prefix..]),
                            )?;
                            self.embed_tokens_at(s, &inputs[index].token_ids[prefix..], prefix)?;
                            self.stage_multimodal_rows(
                                s,
                                &inputs[index],
                                devices.map(|d| d[index]),
                                prefix..lengths[index],
                                prefix,
                            )?;
                            self.multimodal_suffix_pass(
                                &state,
                                reference,
                                &[lengths[index] - prefix],
                                None,
                                reuse,
                            )?;
                            hidden[index] =
                                self.last_hidden(self.scratch.as_ref().unwrap(), lengths[index])?;
                        }
                    }
                }
            }
            Ok(hidden)
        }));
        let synced = self.synchronize();
        state.initialized = false;
        self.mm_batch_prefix = Some((capacity, state));
        if synced.is_err() || !matches!(&result, Ok(Ok(_))) {
            self.mm_batch_windows.clear();
        }
        match result {
            Ok(value) => {
                synced?;
                value.map(Some)
            }
            Err(panic) => std::panic::resume_unwind(panic),
        }
    }

    /// Project each layer's prefix and suffix rows together, then seed that
    /// layer's independent mixers from freshly produced prefix state.
    fn multimodal_compact_pass(
        &mut self,
        inputs: &[MultimodalInput<'_>],
        devices: Option<&[cuda::DeviceBf16Matrix<'_>]>,
        indices: &[usize],
        batch: &SuffixBatch<'_>,
    ) -> Result<Vec<Vec<f32>>> {
        let prefix = batch.prefix.len;
        let originals: Vec<_> = indices.iter().map(|&i| inputs[i].token_ids.len()).collect();
        compact_rows(&originals, prefix).context("invalid compact cohort")?;
        ensure!(batch.include_prefix, "compact pass must produce its prefix");
        let lengths: Vec<_> = originals.iter().map(|&n| n - prefix).collect();
        let first = indices[0];
        let ids: Vec<_> = inputs[first].token_ids[..prefix]
            .iter()
            .copied()
            .chain(
                indices
                    .iter()
                    .flat_map(|&i| inputs[i].token_ids[prefix..].iter().copied()),
            )
            .collect();
        let positions: [Vec<i64>; 3] = std::array::from_fn(|axis| {
            inputs[first].position_ids[axis][..prefix]
                .iter()
                .copied()
                .chain(
                    indices
                        .iter()
                        .flat_map(|&i| inputs[i].position_ids[axis][prefix..].iter().copied()),
                )
                .collect()
        });
        {
            let s = self.scratch.as_ref().unwrap();
            self.upload_positions(s, [&positions[0], &positions[1], &positions[2]])?;
            self.embed_tokens(s, &ids)?;
            self.stage_multimodal_rows(s, &inputs[first], devices.map(|d| d[first]), 0..prefix, 0)?;
            let mut offset = prefix;
            for (&index, &length) in indices.iter().zip(&lengths) {
                self.stage_multimodal_rows(
                    s,
                    &inputs[index],
                    devices.map(|d| d[index]),
                    prefix..inputs[index].token_ids.len(),
                    offset,
                )?;
                offset += length;
            }
        }
        let key = PassKey::Compact {
            prefix,
            lengths: lengths.clone(),
            reference: batch.original_rows,
        };
        if !self.replay_multimodal_pass(&key)? {
            // Only suffixes enter the 16-sequence GDN descriptor. The leading
            // prefix is processed first on the same stream, before they read it.
            let segments: Vec<_> = std::iter::once(prefix)
                .chain(lengths.iter().copied())
                .collect();
            let s = self.scratch.as_ref().unwrap();
            self.run_with_suffix(s, &segments, true, false, Some(batch))?;
            if self.graph_enabled {
                self.synchronize()?;
                let captured = cuda::Graph::capture(self.stream, || {
                    self.run_with_suffix(s, &segments, true, false, Some(batch))
                });
                self.remember_multimodal_pass(key, captured)?;
            }
        }
        let s = self.scratch.as_ref().unwrap();
        let mut end = prefix;
        lengths
            .iter()
            .map(|&n| {
                end += n;
                self.last_hidden(s, end)
            })
            .collect()
    }

    fn replay_multimodal_pass(&mut self, key: &PassKey) -> Result<bool> {
        if !self.graph_enabled {
            return Ok(false);
        }
        let Some(entry) = self.mm_batch_windows.iter().find(|entry| &entry.key == key) else {
            return Ok(false);
        };
        entry.graph.launch(self.stream)?;
        self.graph_stats.replays += 1;
        if std::env::var("CUA_S1_GRAPH_TRACE").as_deref() == Ok("1") {
            eprintln!("CUA_S1_GRAPH mode=multimodal-batch event=replay key={key:?}");
        }
        Ok(true)
    }

    fn remember_multimodal_pass(
        &mut self,
        key: PassKey,
        captured: Result<cuda::Graph>,
    ) -> Result<()> {
        match captured {
            Ok(graph) => {
                if self.mm_batch_windows.len() == 64 {
                    self.synchronize()?;
                    self.mm_batch_windows.pop_front();
                }
                self.graph_stats.captures += 1;
                if std::env::var("CUA_S1_GRAPH_TRACE").as_deref() == Ok("1") {
                    eprintln!("CUA_S1_GRAPH mode=multimodal-batch event=capture key={key:?}");
                }
                self.mm_batch_windows.push_back(Entry { key, graph });
            }
            Err(error) => {
                self.cache_graph(&[], true, Err(error))?;
            }
        }
        Ok(())
    }

    fn multimodal_prefix_pass(
        &mut self,
        state: &mut PrefixState,
        reference: usize,
        reuse: cuda::ReuseGemm,
    ) -> Result<()> {
        let key = PassKey::Prefix {
            prefix: state.len,
            reference,
        };
        if self.replay_multimodal_pass(&key)? {
            return Ok(());
        }
        let s = self.scratch.as_ref().unwrap();
        self.run_window(
            s,
            0,
            state.len,
            true,
            Some(&mut *state),
            None,
            self.gemm,
            Some((reuse, reference)),
        )?;
        if self.graph_enabled {
            self.synchronize()?;
            let captured = cuda::Graph::capture(self.stream, || {
                self.run_window(
                    s,
                    0,
                    state.len,
                    true,
                    Some(&mut *state),
                    None,
                    self.gemm,
                    Some((reuse, reference)),
                )
            });
            self.remember_multimodal_pass(key, captured)?;
        }
        Ok(())
    }

    fn multimodal_suffix_pass(
        &mut self,
        state: &PrefixState,
        reference: usize,
        lengths: &[usize],
        ops: Option<cuda::SharedSuffix>,
        reuse: cuda::ReuseGemm,
    ) -> Result<()> {
        let key = if ops.is_some() {
            PassKey::Batch {
                prefix: state.len,
                lengths: lengths.to_vec(),
                reference,
            }
        } else {
            PassKey::Window {
                prefix: state.len,
                suffix: lengths[0],
                reference,
            }
        };
        if self.replay_multimodal_pass(&key)? {
            return Ok(());
        }
        let s = self.scratch.as_ref().unwrap();
        let run = || {
            if let Some(ops) = ops {
                self.run_with_suffix(
                    s,
                    lengths,
                    true,
                    false,
                    Some(&SuffixBatch {
                        prefix: state,
                        ops,
                        gemm: reuse,
                        original_rows: reference,
                        seeded_gdn: cuda::api().seeded_gdn_batch(),
                        include_prefix: false,
                    }),
                )
            } else {
                self.run_window(
                    s,
                    state.len,
                    state.len + lengths[0],
                    true,
                    None,
                    Some(state),
                    self.gemm,
                    Some((reuse, reference)),
                )
            }
        };
        run()?;
        if self.graph_enabled {
            self.synchronize()?;
            let captured = cuda::Graph::capture(self.stream, run);
            self.remember_multimodal_pass(key, captured)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../../../../tests/qwen3_5/multimodal_shared.rs"]
mod tests;
