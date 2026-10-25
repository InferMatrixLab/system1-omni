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
        )
        .unwrap();
        let shared = prefix::execute(
            &mut engine.language,
            &engine.letters,
            &prepared,
            &features,
            prefix::Mode::Shared,
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
