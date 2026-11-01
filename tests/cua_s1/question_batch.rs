use super::*;

#[test]
fn question_batch_opt_in_is_strict_and_does_not_mix_prefix_modes() {
    assert!(!enabled(None, prefix::Mode::Off).unwrap());
    assert!(!enabled(Some("0"), prefix::Mode::Matched).unwrap());
    assert!(enabled(Some("1"), prefix::Mode::Off).unwrap());
    assert!(enabled(Some("2"), prefix::Mode::Off).is_err());
    assert!(enabled(Some("1"), prefix::Mode::Matched).is_err());
}

#[test]
fn question_batch_groups_preserve_order_and_bound_total_tokens() {
    assert_eq!(groups(&[4096, 4096, 4096, 1], false).unwrap(), [0..2, 2..4]);
    let all = groups(&[100; 8], false).unwrap();
    assert_eq!(all.len(), 1);
    assert_eq!(all[0], 0..8);
    for bad in [vec![], vec![0], vec![4097], vec![1; 9]] {
        assert!(groups(&bad, false).is_err());
    }
}

#[test]
#[ignore = "requires CUDA and pinned CUA_S1_* weights; run serially with MM_PREFIX=off/MM_BATCH=0"]
fn question_batch_keeps_logits_response_order_and_changed_pixels() {
    use crate::vision_engine::device_feature_tests::{load, questions};
    let mut engine = load();
    engine.language.enable_multimodal_batching().unwrap();
    engine.batch_questions = true;
    for (width, height, count, seed) in [
        (256, 256, 1, 1),
        (256, 256, 4, 1),
        (256, 256, 4, 2),
        (512, 256, 8, 3),
        (1024, 1024, 8, 4),
        (256, 256, 4, 1),
    ] {
        let rgb: Vec<_> = (0..width * height * 3)
            .map(|j| ((j * 17 + seed * 31 + j / 251) % 256) as u8)
            .collect();
        let qs: Vec<_> = (0..count)
            .map(|i| {
                let mut q = questions(seed).remove(i % 2);
                q.name = format!("q{i}");
                let options = [1, 2, 26][i % 3];
                q.keys = (0..options).map(|j| format!("k{j}")).collect();
                q.labels = (0..options)
                    .map(|j| format!("Decision number {j}"))
                    .collect();
                q
            })
            .collect();
        let mut prepared = engine.prepare(width, height, &rgb, &qs).unwrap();
        let host = engine.vision.forward(&prepared.image).unwrap();
        let expected: Vec<_> = prepared
            .prompts
            .iter()
            .zip(prepared.context.option_counts())
            .map(|(p, n)| engine.score(p, &host, n).unwrap().logits)
            .collect();
        let expected_response = prepared.context.finish(expected.clone()).unwrap();
        for device in [false, true] {
            engine.device_features = device;
            for _ in 0..2 {
                let actual = engine.execute_prepared(&prepared).unwrap();
                for (a, b) in actual.iter().zip(&expected) {
                    assert_eq!(
                        a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        b.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        "{width}x{height} count={count} seed={seed} device={device}"
                    );
                }
                assert_eq!(prepared.context.finish(actual).unwrap(), expected_response);
            }
        }
        // A late malformed question must not poison captures or request recovery.
        let last = prepared.prompts.last_mut().unwrap();
        let old = last.position_ids[2].pop().unwrap();
        assert!(engine.execute_prepared(&prepared).is_err());
        prepared.prompts.last_mut().unwrap().position_ids[2].push(old);
        assert_eq!(
            engine.predict_prepared(&prepared).unwrap(),
            expected_response
        );
    }
}

#[test]
fn shared_prefix_questions_keep_one_group_when_compact_work_is_bounded() {
    assert_eq!(groups(&[1030; 8], false).unwrap(), vec![0..7, 7..8]);
    let shared = groups(&[1030; 8], true).unwrap();
    assert_eq!(shared.len(), 1);
    assert_eq!(shared[0], 0..8);
}

#[test]
fn short_position_axes_return_an_error_before_prefix_discovery() {
    use crate::contract::Question;
    use tokenizers::{AddedToken, Tokenizer, models::wordlevel::WordLevel};
    let vocab = [("[UNK]".to_owned(), 0), ("<|image_pad|>".to_owned(), 1)]
        .into_iter()
        .collect();
    let mut tokenizer = Tokenizer::new(
        WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .unwrap(),
    );
    tokenizer.add_special_tokens(&[AddedToken::from("<|image_pad|>", true)]);
    let processor = VisionProcessor::new(tokenizer, 1);
    let questions: Vec<_> = (0..2)
        .map(|i| Question {
            name: format!("q{i}"),
            goal: String::new(),
            keys: vec!["only".into()],
            labels: vec!["Only".into()],
        })
        .collect();
    let mut prepared = processor
        .prepare(256, 256, &vec![0; 256 * 256 * 3], &questions)
        .unwrap();
    assert!(plan(&prepared).is_ok());
    for axis in 0..3 {
        let saved = std::mem::take(&mut prepared.prompts[1].position_ids[axis]);
        assert!(plan(&prepared).is_err());
        prepared.prompts[1].position_ids[axis] = saved;
    }
    assert!(plan(&prepared).is_ok());
}
