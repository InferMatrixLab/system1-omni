use super::*;

fn prompt(tokens: usize) -> ImagePrompt {
    ImagePrompt {
        token_ids: vec![7; tokens],
        image_token_indices: (16..tokens.min(144)).collect(),
        position_ids: std::array::from_fn(|_| (0..tokens as i64).collect()),
    }
}

#[test]
fn shared_prefix_requires_multiple_prompts_and_aligns_tokens_and_all_positions() {
    assert_eq!(prefix::common_prefix(&[]), 0);
    assert_eq!(prefix::common_prefix(&[prompt(193)]), 0);
    let mut prompts = vec![prompt(193), prompt(193)];
    assert_eq!(prefix::common_prefix(&prompts), 192);
    prompts[1].token_ids[130] = 9;
    assert_eq!(prefix::common_prefix(&prompts), 128);
    for axis in 0..3 {
        let mut prompts = vec![prompt(193), prompt(193)];
        prompts[1].position_ids[axis][70] += 1;
        assert_eq!(prefix::common_prefix(&prompts), 64);
    }
    assert_eq!(prefix::common_prefix(&[prompt(64), prompt(64)]), 0);
    assert_eq!(prefix::common_prefix(&[prompt(65), prompt(193)]), 64);
}

#[test]
#[ignore = "requires CUDA and pinned CUA_S1_* weights; run serially"]
fn screenshot_prefix_matches_fixed_and_does_not_retain_previous_request_state() {
    let mut engine = device_feature_tests::load();
    for (width, height, seed, count) in [
        (256, 256, 1, 1),
        (256, 256, 2, 2),
        (512, 256, 3, 4),
        (256, 512, 4, 8),
        (256, 256, 1, 2),
        (256, 256, 8, 1),
    ] {
        let rgb: Vec<_> = (0..width * height * 3)
            .map(|i| ((i * 17 + seed * 31) % 251) as u8)
            .collect();
        let questions: Vec<_> = (0..count)
            .map(|i| {
                let mut q = device_feature_tests::questions(seed + i).remove(i % 2);
                q.name = format!("q{i}");
                q
            })
            .collect();
        let prepared = engine.prepare(width, height, &rgb, &questions).unwrap();
        if count > 1 {
            assert!(prefix::common_prefix(&prepared.prompts) >= 64);
        }
        let features = engine.vision.forward(&prepared.image).unwrap();
        let fixed = prefix::execute(
            &mut engine.language,
            &engine.letters,
            &prepared,
            &features,
            prefix::Mode::Fixed,
            &mut None,
        )
        .unwrap();
        let shared = prefix::execute(
            &mut engine.language,
            &engine.letters,
            &prepared,
            &features,
            prefix::Mode::Shared,
            &mut engine.prefix_storage,
        )
        .unwrap();
        assert_eq!(
            shared, fixed,
            "shared letter logits must equal complete fixed passes"
        );
        engine.prefix_mode = prefix::Mode::Shared;
        let response = engine.predict_prepared(&prepared).unwrap();
        engine.prefix_mode = prefix::Mode::Fixed;
        assert_eq!(engine.predict_prepared(&prepared).unwrap(), response);
        assert_eq!(response, prepared.context.finish(fixed).unwrap());
    }
}

#[test]
#[ignore = "requires CUDA and pinned CUA_S1_* weights; run serially"]
fn prefix_rejects_malformed_prepared_inputs_and_recovers() {
    let mut engine = device_feature_tests::load();
    engine.prefix_mode = prefix::Mode::Shared;
    let rgb = vec![80; 256 * 256 * 3];
    let mut prepared = engine
        .prepare(256, 256, &rgb, &device_feature_tests::questions(1))
        .unwrap();
    let last = prepared.prompts[0].position_ids[0].pop().unwrap();
    assert!(engine.predict_prepared(&prepared).is_err());
    prepared.prompts[0].position_ids[0].push(last);
    let index = prepared.prompts[0].image_token_indices[0];
    prepared.prompts[0].image_token_indices[0] = usize::MAX;
    assert!(engine.predict_prepared(&prepared).is_err());
    prepared.prompts[0].image_token_indices[0] = index;
    let other = prepared.prompts[1].image_token_indices[0];
    prepared.prompts[1].image_token_indices[0] = other - 1;
    assert!(engine.predict_prepared(&prepared).is_err());
    prepared.prompts[1].image_token_indices[0] = other;
    // Reject a suffix token after the common prefix has already been captured,
    // then recover without retaining that request's state.
    let last = prepared.prompts[0].token_ids.last_mut().unwrap();
    let token = *last;
    *last = u32::MAX;
    assert!(engine.predict_prepared(&prepared).is_err());
    *prepared.prompts[0].token_ids.last_mut().unwrap() = token;
    assert!(engine.predict_prepared(&prepared).is_ok());
}

#[test]
fn matched_windows_validate_all_axes_and_leave_nonempty_aligned_suffixes() {
    let mut prompts = vec![prompt(193), prompt(219)];
    assert_eq!(
        prefix::matched_windows(&prompts).unwrap(),
        (vec![193, 219], vec![192, 1, 27])
    );
    prompts[1].token_ids[70] = 9;
    assert_eq!(
        prefix::matched_windows(&prompts).unwrap().1,
        vec![64, 129, 155]
    );
    for axis in 0..3 {
        let mut invalid = vec![prompt(193), prompt(219)];
        invalid[1].position_ids[axis].pop();
        assert!(prefix::matched_windows(&invalid).is_err());
    }
    assert!(prefix::matched_windows(&[prompt(0), prompt(193)]).is_err());
    assert!(
        prefix::matched_windows(&[prompt(64), prompt(64)])
            .unwrap()
            .1
            .is_empty()
    );
    assert!(
        prefix::matched_windows(&[prompt(193)])
            .unwrap()
            .1
            .is_empty()
    );
}

#[test]
#[ignore = "requires CUDA and pinned CUA_S1_* weights; run serially with vision Graph0/1"]
fn matched_prefix_preserves_default_and_original_m64_after_errors() {
    let mut engine = device_feature_tests::load();
    let mut admitted = 0;
    for (width, height, seed, count) in [
        (256, 256, 1, 2),
        (512, 256, 2, 4),
        (256, 512, 3, 8),
        (256, 256, 1, 1),
        (256, 256, 4, 2),
    ] {
        let rgb: Vec<_> = (0..width * height * 3)
            .map(|i| ((i * 17 + seed * 31) % 251) as u8)
            .collect();
        let questions: Vec<_> = (0..count)
            .map(|i| {
                let mut q = device_feature_tests::questions(seed + i).remove(i % 2);
                q.name = format!("q{i}");
                q
            })
            .collect();
        let mut prepared = engine.prepare(width, height, &rgb, &questions).unwrap();
        engine.prefix_mode = prefix::Mode::Off;
        let ordinary = engine.execute_prepared(&prepared).unwrap();
        engine.prefix_mode = prefix::Mode::Fixed;
        let m64 = engine.execute_prepared(&prepared).unwrap();
        let bits = |rows: Vec<Vec<f32>>| {
            rows.into_iter()
                .map(|r| r.into_iter().map(f32::to_bits).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        };
        let (lengths, windows) = prefix::matched_windows(&prepared.prompts).unwrap();
        let features = engine.vision.forward(&prepared.image).unwrap();
        if let Some(logits) = engine
            .language
            .with_matched_prefix(&lengths, &windows, |model| {
                prefix::execute(
                    model,
                    &engine.letters,
                    &prepared,
                    &features,
                    prefix::Mode::Shared,
                    &mut engine.prefix_storage,
                )
            })
            .unwrap()
        {
            admitted += 1;
            assert_eq!(bits(logits), bits(ordinary.clone()));
        }
        engine.prefix_mode = prefix::Mode::Matched;
        for _ in 0..2 {
            assert_eq!(
                bits(engine.execute_prepared(&prepared).unwrap()),
                bits(ordinary.clone())
            );
        }
        let token = *prepared.prompts[0].token_ids.last().unwrap();
        *prepared.prompts[0].token_ids.last_mut().unwrap() = u32::MAX;
        assert!(engine.execute_prepared(&prepared).is_err());
        *prepared.prompts[0].token_ids.last_mut().unwrap() = token;
        assert_eq!(
            bits(engine.execute_prepared(&prepared).unwrap()),
            bits(ordinary.clone())
        );
        let pos = prepared.prompts[0].position_ids[2].pop().unwrap();
        assert!(engine.execute_prepared(&prepared).is_err());
        prepared.prompts[0].position_ids[2].push(pos);
        let index = prepared.prompts[0].image_token_indices[0];
        prepared.prompts[0].image_token_indices[0] = usize::MAX;
        assert!(engine.execute_prepared(&prepared).is_err());
        prepared.prompts[0].image_token_indices[0] = index;
        assert_eq!(
            bits(engine.execute_prepared(&prepared).unwrap()),
            bits(ordinary.clone())
        );
        engine.prefix_mode = prefix::Mode::Shared;
        assert_eq!(bits(engine.execute_prepared(&prepared).unwrap()), bits(m64));
        engine.prefix_mode = prefix::Mode::Off;
        assert_eq!(
            bits(engine.execute_prepared(&prepared).unwrap()),
            bits(ordinary)
        );
    }
    assert!(admitted > 0, "all shared-window probes fell back");
}
