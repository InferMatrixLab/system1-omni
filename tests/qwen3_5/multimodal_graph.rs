use super::*;
use half::bf16;

fn load_model() -> Result<Model> {
    let dir = std::env::var("CUA_S1_TEST_MODEL").context("CUA_S1_TEST_MODEL")?;
    let lib = std::env::var("CUA_S1_CUDA_LIB").context("CUA_S1_CUDA_LIB")?;
    let mut model = Model::load(Path::new(&dir), Path::new(&lib))?;
    model.graph_enabled = false;
    Ok(model)
}

fn text_tokens(model: &Model, count: usize) -> Result<Vec<u32>> {
    let vocabulary: Vec<u32> = (0..model.embed.shape[0])
        .filter_map(|id| u32::try_from(id).ok())
        .filter(|id| Some(*id) != model.cfg.image_token_id)
        .take(4)
        .collect();
    ensure!(vocabulary.len() == 4, "test needs four non-image token IDs");
    Ok(vocabulary.into_iter().cycle().take(count).collect())
}

fn assert_hidden_eq(actual: &[f32], expected: &[f32], context: &str) {
    assert!(
        expected.iter().all(|x| x.is_finite()),
        "{context}: eager hidden state must be finite"
    );
    assert_eq!(actual, expected, "{context}");
}

#[test]
#[ignore = "requires the Cua-S1 language checkpoint and real CUDA library; run serially"]
fn multimodal_graph_replays_current_inputs_and_clears_on_growth() -> Result<()> {
    let mut model = load_model()?;
    ensure!(model.cfg.max_positions >= 2048, "test needs 2048 positions");
    let image_token = model.cfg.image_token_id.context("image_token_id")?;
    let text = text_tokens(&model, 4)?;
    let ids = [text[0], image_token, image_token, text[1]];
    let indices = [1, 2];
    let positions = [0, 1, 1, 2];
    let zeros = vec![bf16::ZERO; 2 * model.cfg.hidden];
    let input = MultimodalInput {
        token_ids: &ids,
        image_token_indices: &indices,
        image_embeddings: &zeros,
        position_ids: [&positions; 3],
    };
    let eager = model.forward_multimodal(&input)?;
    model.graph_enabled = true;
    assert_hidden_eq(
        &model.forward_multimodal(&input)?,
        &eager,
        "capture miss must retain the eager result",
    );
    assert_eq!(model.mm_graphs.len(), 1, "miss must cache a graph");
    assert_eq!(model.graph_stats().captures, 1);
    assert_eq!(model.graph_stats().cached_shapes, 1);
    assert!(model.graphs.is_empty(), "text cache must remain separate");
    assert!(model.graph_enabled, "capture must remain enabled");
    assert_hidden_eq(
        &model.forward_multimodal(&input)?,
        &eager,
        "replay must equal eager exactly",
    );

    assert_eq!(model.graph_stats().replays, 1);

    let changed_ids = [text[2], image_token, image_token, text[3]];
    let changed_features: Vec<bf16> = (0..2 * model.cfg.hidden)
        .map(|i| bf16::from_f32((i % 17) as f32 / 16.0 - 0.5))
        .collect();
    let temporal = [0, 3, 3, 5];
    let height = [0, 2, 4, 5];
    let width = [0, 4, 2, 5];
    let changed = MultimodalInput {
        token_ids: &changed_ids,
        image_token_indices: &indices,
        image_embeddings: &changed_features,
        position_ids: [&temporal, &height, &width],
    };
    model.graph_enabled = false;
    let changed_eager = model.forward_multimodal(&changed)?;
    assert_ne!(
        changed_eager, eager,
        "changed inputs must exercise new data"
    );
    model.graph_enabled = true;
    assert_hidden_eq(
        &model.forward_multimodal(&changed)?,
        &changed_eager,
        "replay must upload current IDs, image rows and T/H/W positions",
    );
    assert_eq!(model.mm_graphs.len(), 1, "same length must reuse its graph");

    model.graph_enabled = false;
    let text_eager = model.forward(&text)?;
    model.graph_enabled = true;
    assert_hidden_eq(&model.forward(&text)?, &text_eager, "text capture miss");
    assert_hidden_eq(&model.forward(&text)?, &text_eager, "text replay");
    assert_eq!(model.graphs.len(), 1);
    assert_eq!(model.mm_graphs.len(), 1);
    assert_hidden_eq(
        &model.forward_multimodal(&changed)?,
        &changed_eager,
        "multimodal replay after text interleave",
    );
    // Equal packed totals must not alias singleton or differently ordered
    // sequence boundaries. Reuse every shape again with changed token IDs.
    let changed_text = [text[3], text[2], text[1], text[0]];
    let shapes = [vec![4], vec![1, 3], vec![3, 1]];
    for tokens in [text.as_slice(), changed_text.as_slice()] {
        for lengths in &shapes {
            let mut begin = 0;
            let inputs: Vec<&[u32]> = lengths
                .iter()
                .map(|&length| {
                    let end = begin + length;
                    let ids = &tokens[begin..end];
                    begin = end;
                    ids
                })
                .collect();
            model.graph_enabled = false;
            let eager = model.forward_batch(&inputs)?;
            model.graph_enabled = true;
            for context in ["packed text capture/hit", "packed text replay"] {
                let actual = model.forward_batch(&inputs)?;
                assert_eq!(actual.len(), eager.len(), "{context}: sequence count");
                for (actual, expected) in actual.iter().zip(&eager) {
                    assert_hidden_eq(actual, expected, context);
                }
            }
            assert!(model.graph_enabled, "packed shape capture must succeed");
            assert!(model.graphs.iter().any(|(shape, _)| shape == lengths));
            assert_eq!(model.mm_graphs.len(), 1, "text must retain MM graph");
            assert_hidden_eq(
                &model.forward_multimodal(&changed)?,
                &changed_eager,
                "multimodal replay after packed text interleave",
            );
        }
    }
    assert_eq!(
        model
            .graphs
            .iter()
            .map(|(shape, _)| shape.clone())
            .collect::<Vec<_>>(),
        shapes.to_vec(),
        "singleton, split and reversed splits must keep distinct FIFO keys",
    );

    let next_cap = model.scratch.as_ref().unwrap().cap + 1;
    model.prepare_scratch(next_cap)?;
    assert!(model.graphs.is_empty(), "growth must clear text graphs");
    assert!(model.mm_graphs.is_empty(), "growth must clear MM graphs");
    assert!(model.graph_enabled, "growth must allow fresh captures");
    assert_hidden_eq(&model.forward(&text)?, &text_eager, "text after growth");
    assert_hidden_eq(
        &model.forward_multimodal(&changed)?,
        &changed_eager,
        "multimodal after growth",
    );
    assert_eq!(model.graphs.len(), 1);
    assert_eq!(model.mm_graphs.len(), 1);
    model.synchronize()?;
    Ok(())
}

#[test]
#[ignore = "requires the Cua-S1 language checkpoint and real CUDA library; run serially"]
fn multimodal_graph_fifo_evicts_and_recaptures_without_touching_text() -> Result<()> {
    let mut model = load_model()?;
    ensure!(model.cfg.max_positions >= 1024, "test needs 1024 positions");
    let text = text_tokens(&model, 4)?;
    let text_eager = model.forward(&text)?;
    model.graph_enabled = true;
    assert_hidden_eq(&model.forward(&text)?, &text_eager, "text capture miss");
    assert_eq!(model.graphs.len(), 1);
    let scratch_address = model.scratch.as_ref().unwrap().buf.at(0);
    assert_eq!(model.scratch.as_ref().unwrap().cap, 1024);

    let mut first_eager = Vec::new();
    for length in 1..=65 {
        let ids = text_tokens(&model, length)?;
        let positions: Vec<i64> = (0..length as i64).collect();
        let input = MultimodalInput {
            token_ids: &ids,
            image_token_indices: &[],
            image_embeddings: &[],
            position_ids: [&positions; 3],
        };
        model.graph_enabled = false;
        let eager = model.forward_multimodal(&input)?;
        if length == 1 {
            first_eager = eager.clone();
        }
        model.graph_enabled = true;
        assert_hidden_eq(
            &model.forward_multimodal(&input)?,
            &eager,
            &format!("multimodal capture miss at length {length}"),
        );
        assert!(model.graph_enabled, "capture failed at length {length}");
        assert_eq!(model.mm_graphs.len(), length.min(64));
        let expected: Vec<usize> = (length.saturating_sub(63).max(1)..=length).collect();
        assert_eq!(
            model
                .mm_graphs
                .iter()
                .map(|(shape, _)| shape[0])
                .collect::<Vec<_>>(),
            expected,
            "multimodal cache must evict in insertion order"
        );
        assert_eq!(model.scratch.as_ref().unwrap().cap, 1024);
        assert_eq!(model.scratch.as_ref().unwrap().buf.at(0), scratch_address);
        assert_eq!(model.graphs.len(), 1, "MM capture must keep text graph");
        assert_eq!(model.graphs.front().unwrap().0, vec![text.len()]);
    }
    assert!(
        !model
            .mm_graphs
            .iter()
            .any(|(shape, _)| shape.as_slice() == [1])
    );

    // A hit must not refresh FIFO order. Length 2 remains the oldest capture.
    let ids = text_tokens(&model, 2)?;
    let positions = [0, 1];
    let input = MultimodalInput {
        token_ids: &ids,
        image_token_indices: &[],
        image_embeddings: &[],
        position_ids: [&positions; 3],
    };
    model.graph_enabled = false;
    let eager = model.forward_multimodal(&input)?;
    model.graph_enabled = true;
    assert_hidden_eq(&model.forward_multimodal(&input)?, &eager, "oldest hit");
    assert_eq!(model.mm_graphs.front().unwrap().0, vec![2]);
    assert_eq!(model.mm_graphs.back().unwrap().0, vec![65]);

    let ids = text_tokens(&model, 1)?;
    let positions = [0];
    let input = MultimodalInput {
        token_ids: &ids,
        image_token_indices: &[],
        image_embeddings: &[],
        position_ids: [&positions; 3],
    };
    assert_hidden_eq(
        &model.forward_multimodal(&input)?,
        &first_eager,
        "recapturing evicted length must retain its eager result",
    );
    assert_eq!(model.mm_graphs.len(), 64);
    let expected: Vec<usize> = (3..=65).chain(std::iter::once(1)).collect();
    assert_eq!(
        model
            .mm_graphs
            .iter()
            .map(|(shape, _)| shape[0])
            .collect::<Vec<_>>(),
        expected
    );
    assert_hidden_eq(
        &model.forward_multimodal(&input)?,
        &first_eager,
        "recaptured length replay",
    );
    assert_eq!(model.graphs.len(), 1, "text graph must survive eviction");
    assert_eq!(model.graphs.front().unwrap().0, vec![text.len()]);
    assert_hidden_eq(&model.forward(&text)?, &text_eager, "surviving text replay");
    assert!(model.graph_enabled);
    assert_eq!(model.scratch.as_ref().unwrap().cap, 1024);
    assert_eq!(model.scratch.as_ref().unwrap().buf.at(0), scratch_address);
    model.synchronize()?;
    Ok(())
}

#[test]
#[ignore = "requires the Cua-S1 language checkpoint and real CUDA library; run serially"]
fn graph_recording_failure_keeps_eager_result_and_disables_both_modes() -> Result<()> {
    for multimodal in [false, true] {
        let mut model = load_model()?;
        let image_token = model.cfg.image_token_id.context("image_token_id")?;
        let text = text_tokens(&model, 4)?;
        let ids = [text[0], image_token, image_token, text[1]];
        let indices = [1, 2];
        let temporal = [0, 1, 1, 3];
        let height = [0, 1, 2, 3];
        let width = [0, 2, 1, 3];
        let features: Vec<bf16> = (0..2 * model.cfg.hidden)
            .map(|i| bf16::from_f32((i % 11) as f32 / 10.0 - 0.5))
            .collect();
        let input = MultimodalInput {
            token_ids: &ids,
            image_token_indices: &indices,
            image_embeddings: &features,
            position_ids: [&temporal, &height, &width],
        };
        let text_eager = model.forward(&text)?;
        let mm_eager = model.forward_multimodal(&input)?;
        model.graph_enabled = true;
        assert_hidden_eq(&model.forward(&text)?, &text_eager, "seed text graph");
        assert_hidden_eq(
            &model.forward_multimodal(&input)?,
            &mm_eager,
            "seed multimodal graph",
        );
        assert!(model.graph_enabled, "both seed captures must succeed");
        assert_eq!(model.graphs.len(), 1);
        assert_eq!(model.mm_graphs.len(), 1);

        // Leave a completed eager hidden state in scratch before a real CUDA
        // capture begins and its recording closure returns an error.
        model.graph_enabled = false;
        let prior_eager = if multimodal {
            model.forward_multimodal(&input)?
        } else {
            model.forward(&text)?
        };
        model.graph_enabled = true;
        let captured =
            cuda::Graph::capture(model.stream, || anyhow::bail!("injected record failure"));
        assert!(
            captured
                .as_ref()
                .err()
                .context("recording failure unexpectedly succeeded")?
                .to_string()
                .contains("injected record failure"),
            "must exercise the actual recording-error path"
        );
        model.cache_graph(&[text.len()], multimodal, captured)?;
        assert_hidden_eq(
            &model.last_hidden(model.scratch.as_ref().unwrap(), text.len())?,
            &prior_eager,
            "recording failure must preserve the earlier eager hidden state",
        );
        assert!(
            !model.graph_enabled,
            "recording failure must disable replay"
        );
        assert!(model.graphs.is_empty(), "failure must clear text graphs");
        assert!(model.mm_graphs.is_empty(), "failure must clear MM graphs");

        for _ in 0..2 {
            assert_hidden_eq(
                &model.forward(&text)?,
                &text_eager,
                "public text forward after capture failure",
            );
            assert_hidden_eq(
                &model.forward_multimodal(&input)?,
                &mm_eager,
                "public multimodal forward after capture failure",
            );
            assert!(!model.graph_enabled, "eager fallback must stay enabled");
            assert!(model.graphs.is_empty(), "text must not recapture");
            assert!(model.mm_graphs.is_empty(), "multimodal must not recapture");
        }
        model.synchronize()?;
    }
    Ok(())
}

#[test]
#[ignore = "requires the Cua-S1 language checkpoint and real CUDA library; run serially"]
fn device_multimodal_graph_uses_current_rows_and_releases_external_storage() -> Result<()> {
    let mut model = load_model()?;
    let image_token = model.cfg.image_token_id.context("image_token_id")?;
    let text = text_tokens(&model, 4)?;
    let mut last = Vec::new();
    for seed in 0..4 {
        let ids = [text[seed], image_token, image_token, text[(seed + 1) % 4]];
        let indices = [1, 2];
        let temporal = [0, 1 + seed as i64, 1 + seed as i64, 5];
        let height = [0, 1, 2 + seed as i64, 5];
        let width = [0, 2 + seed as i64, 1, 5];
        let features: Vec<bf16> = (0..2 * model.cfg.hidden)
            .map(|i| bf16::from_f32(((i + seed * 7) % 19) as f32 / 18.0 - 0.5))
            .collect();
        let input = MultimodalInput {
            token_ids: &ids,
            image_token_indices: &indices,
            image_embeddings: &features,
            position_ids: [&temporal, &height, &width],
        };
        model.graph_enabled = false;
        let expected = model.forward_multimodal(&input)?;
        if !last.is_empty() {
            assert_ne!(expected, last, "changed input must exercise new data");
        }
        let bytes: Vec<_> = features.iter().flat_map(|v| v.to_le_bytes()).collect();
        let source = DeviceBuffer::new(bytes.len())?;
        unsafe { cuda::upload(source.at(0), &bytes, model.stream)? };
        model.synchronize()?;
        let device = DeviceMultimodalInput {
            token_ids: &ids,
            image_token_indices: &indices,
            image_embeddings: source.bf16_matrix(2, model.cfg.hidden)?,
            position_ids: input.position_ids,
        };
        model.graph_enabled = true;
        assert_hidden_eq(
            &model.forward_multimodal_device(&device)?,
            &expected,
            "device capture/replay",
        );
        assert_eq!(
            model.mm_graphs.len(),
            1,
            "device path must cache the shared MM graph"
        );
        assert_eq!(
            model.graph_stats().captures,
            1,
            "same shape must not recapture"
        );
        assert_eq!(model.graph_stats().replays, 2 * seed as u64);
        // A capture must not retain the external producer address. Releasing it
        // and using host input must replay the same graph safely.
        drop(source);
        assert_hidden_eq(
            &model.forward_multimodal(&input)?,
            &expected,
            "host replay after producer release",
        );
        assert_eq!(model.graph_stats().replays, 2 * seed as u64 + 1);
        last = expected;
    }
    model.prepare_scratch(model.scratch.as_ref().unwrap().cap + 1)?;
    assert!(model.mm_graphs.is_empty(), "growth must retire captures");
    let ids = [text[0], image_token, image_token, text[1]];
    let positions = [0, 1, 1, 2];
    let source = DeviceBuffer::new(2 * model.cfg.hidden * BF16)?;
    let bytes = vec![0; 2 * model.cfg.hidden * BF16];
    unsafe { cuda::upload(source.at(0), &bytes, model.stream)? };
    model.synchronize()?;
    let device = DeviceMultimodalInput {
        token_ids: &ids,
        image_token_indices: &[1, 2],
        image_embeddings: source.bf16_matrix(2, model.cfg.hidden)?,
        position_ids: [&positions; 3],
    };
    let captured = model.forward_multimodal_device(&device)?;
    assert_eq!(model.mm_graphs.len(), 1);
    assert_eq!(model.graph_stats().captures, 2);
    let failed = cuda::Graph::capture(model.stream, || {
        anyhow::bail!("injected device record failure")
    });
    assert!(failed.is_err());
    model.cache_graph(&[4], true, failed)?;
    assert!(!model.graph_enabled);
    assert!(model.mm_graphs.is_empty());
    assert_hidden_eq(
        &model.forward_multimodal_device(&device)?,
        &captured,
        "device eager recovery",
    );
    assert!(!model.graph_enabled);
    Ok(())
}

#[test]
#[ignore = "requires pinned Cua-S1 language weights and CUDA; run serially"]
fn device_window_graph_stages_current_rows_and_owns_prefix_storage_until_retirement() -> Result<()>
{
    let mut model = load_model()?;
    let image = model.cfg.image_token_id.context("image token")?;
    let text = text_tokens(&model, 132)?;
    let mut ids = text.clone();
    ids[32] = image;
    ids[33] = image;
    let indices = [32, 33];
    let mut positions: [Vec<i64>; 3] = std::array::from_fn(|_| (0..132).collect());
    let mut state = model.alloc_prefix(128)?;
    let mut prior = Vec::new();
    for seed in 0..3 {
        ids[0] = text[seed];
        positions[seed][30] += 1;
        let features: Vec<_> = (0..2 * model.cfg.hidden)
            .map(|i| bf16::from_f32(((i + seed * 7) % 19) as f32 / 18. - 0.5))
            .collect();
        let full = MultimodalInput {
            token_ids: &ids,
            image_token_indices: &indices,
            image_embeddings: &features,
            position_ids: std::array::from_fn(|a| positions[a].as_slice()),
        };
        model.graph_enabled = false;
        let expected = model
            .readout_window(&full, None, None, &[131], &[])?
            .hidden
            .remove(0);
        if !prior.is_empty() {
            assert_ne!(expected, prior);
        }
        let buffer = DeviceBuffer::new(features.len() * BF16)?;
        let bytes: Vec<_> = features.iter().flat_map(|v| v.to_le_bytes()).collect();
        unsafe {
            cuda::upload(buffer.at(0), &bytes, model.stream)?;
        }
        model.synchronize()?;
        let prefix = DeviceMultimodalInput {
            token_ids: &ids[..128],
            image_token_indices: &indices,
            image_embeddings: buffer.bf16_matrix(2, model.cfg.hidden)?,
            position_ids: std::array::from_fn(|a| &positions[a][..128]),
        };
        model.graph_enabled = true;
        model.readout_window_device(&prefix, None, Some(&mut state), &[], &[])?;
        // The window Graph cannot retain the producer allocation.
        drop(buffer);
        let suffix = MultimodalInput {
            token_ids: &ids[128..],
            image_token_indices: &[],
            image_embeddings: &[],
            position_ids: std::array::from_fn(|a| &positions[a][128..]),
        };
        let actual = model
            .readout_window(&suffix, Some(&state), None, &[131], &[])?
            .hidden
            .remove(0);
        assert_hidden_eq(&actual, &expected, "current device prefix + host suffix");
        assert_eq!(model.window_graphs.len(), 2);
        prior = expected;
    }
    assert_eq!(model.graph_stats().captures, 2);
    assert_eq!(model.graph_stats().replays, 4);
    let retired = Arc::downgrade(&state.buffers);
    drop(state);
    assert!(
        retired.upgrade().is_some(),
        "live graphs must own recorded prefix storage"
    );
    // New same-sized storage must not alias the retired prefix's cache key.
    let mut replacement = model.alloc_prefix(128)?;
    let prefix = MultimodalInput {
        token_ids: &text[..128],
        image_token_indices: &[],
        image_embeddings: &[],
        position_ids: std::array::from_fn(|a| &positions[a][..128]),
    };
    model.readout_window(&prefix, None, Some(&mut replacement), &[], &[])?;
    assert_eq!(model.window_graphs.len(), 3);
    assert_eq!(model.graph_stats().captures, 3);
    model.prepare_scratch(1025)?;
    assert!(model.window_graphs.is_empty());
    assert!(
        retired.upgrade().is_none(),
        "growth must retire Graph-owned storage"
    );
    // Exercise the production retirement helper with a real failed recording.
    model.readout_window(&prefix, None, Some(&mut replacement), &[], &[])?;
    let failed = cuda::Graph::capture(model.stream, || {
        anyhow::bail!("injected window recording failure")
    });
    assert!(failed.is_err());
    model.cache_window_graph([0, 128, 0, 0, 0], [None, None], failed)?;
    assert!(!model.graph_enabled);
    assert!(model.window_graphs.is_empty());
    model.readout_window(&prefix, None, Some(&mut replacement), &[], &[])?;
    Ok(())
}

#[test]
#[ignore = "requires pinned Cua-S1 language weights and CUDA; run serially"]
fn window_graph_retains_and_keys_both_prefix_and_capture_allocations() -> Result<()> {
    let mut model = load_model()?;
    let ids = text_tokens(&model, 132)?;
    let positions: Vec<i64> = (0..132).collect();
    let input = |range: std::ops::Range<usize>| MultimodalInput {
        token_ids: &ids[range.clone()],
        image_token_indices: &[],
        image_embeddings: &[],
        position_ids: [&positions[range.clone()]; 3],
    };
    let expected = model
        .readout_window(&input(0..132), None, None, &[131], &[])?
        .hidden
        .remove(0);
    let mut source = model.alloc_prefix(64)?;
    let mut first = model.alloc_prefix(128)?;
    let mut second = model.alloc_prefix(128)?;
    model.graph_enabled = true;
    model.readout_window(&input(0..64), None, Some(&mut source), &[], &[])?;
    model.readout_window(&input(64..128), Some(&source), Some(&mut first), &[], &[])?;
    let retired = Arc::downgrade(&first.buffers);
    drop(first);
    assert!(
        retired.upgrade().is_some(),
        "Graph must retain destination as well as source"
    );
    model.readout_window(&input(64..128), Some(&source), Some(&mut second), &[], &[])?;
    assert_eq!(
        model.window_graphs.len(),
        3,
        "a different destination requires a new Graph"
    );
    let actual = model
        .readout_window(&input(128..132), Some(&second), None, &[131], &[])?
        .hidden
        .remove(0);
    assert_hidden_eq(
        &actual,
        &expected,
        "continuation from second Graph destination",
    );
    model.prepare_scratch(1025)?;
    assert!(retired.upgrade().is_none());
    Ok(())
}

#[test]
#[ignore = "requires pinned Cua-S1 language weights and CUDA; run serially"]
fn window_graph_fifo_and_matched_workspace_retirement_release_owned_storage() -> Result<()> {
    let mut model = load_model()?;
    model.graph_enabled = true;
    let state = model.alloc_prefix(64)?;
    let retired = Arc::downgrade(&state.buffers);
    // Real empty CUDA Graphs isolate cache ownership from arithmetic. They are
    // never replayed; the subsequent matched section uses real window kernels.
    for i in 0..128 {
        let graph = cuda::Graph::capture(model.stream, || Ok(()))?;
        let held = if i == 0 {
            Some(Arc::clone(&state.buffers))
        } else {
            None
        };
        model.cache_window_graph([i, 64, 0, 0, 0], [None, held], Ok(graph))?;
    }
    drop(state);
    assert_eq!(model.window_graphs.len(), 128);
    assert!(retired.upgrade().is_some());
    let graph = cuda::Graph::capture(model.stream, || Ok(()))?;
    model.cache_window_graph([128, 64, 0, 0, 0], [None, None], Ok(graph))?;
    assert_eq!(model.window_graphs.len(), 128);
    assert!(
        retired.upgrade().is_none(),
        "129th insertion must release the oldest entry"
    );
    model.prepare_scratch(1025)?;
    assert!(model.window_graphs.is_empty());
    let ids = text_tokens(&model, 64)?;
    let positions: Vec<i64> = (0..64).collect();
    let input = MultimodalInput {
        token_ids: &ids,
        image_token_indices: &[],
        image_embeddings: &[],
        position_ids: [&positions; 3],
    };
    let mut state = model.alloc_prefix(64)?;
    let retired = Arc::downgrade(&state.buffers);
    assert!(
        model
            .with_matched_prefix(&[64, 64], &[64], |m| m.readout_window(
                &input,
                None,
                Some(&mut state),
                &[],
                &[]
            ))?
            .is_some()
    );
    assert_eq!(model.window_graphs.len(), 1);
    drop(state);
    assert!(retired.upgrade().is_some());
    for m in 1..=17 {
        let _ = model.with_matched_prefix(&[m, m], &[m], |_| Ok(()))?;
    }
    assert!(
        model.window_graphs.is_empty(),
        "matched workspace retirement must clear dependent Graphs"
    );
    assert!(retired.upgrade().is_none());
    let captures = model.graph_stats().captures;
    let mut state = model.alloc_prefix(64)?;
    assert!(
        model
            .with_matched_prefix(&[64, 64], &[64], |m| m.readout_window(
                &input,
                None,
                Some(&mut state),
                &[],
                &[]
            ))?
            .is_some()
    );
    assert_eq!(
        model.graph_stats().captures,
        captures + 1,
        "retired shape must recapture"
    );
    let replays = model.graph_stats().replays;
    assert!(
        model
            .with_matched_prefix(&[64, 64], &[64], |m| {
                m.readout_window(&input, None, Some(&mut state), &[], &[])
            })?
            .is_some()
    );
    assert_eq!(model.graph_stats().replays, replays + 1);
    assert_eq!(
        model.graph_stats().captures,
        captures + 1,
        "same shape must replay after recapture"
    );
    Ok(())
}
