//! CPU-only worker tests (spec 03, N4 + N5): the finishing layer's byte
//! parity with the Python reference postprocessor for all three question
//! types (recorded once from `models.valen.postprocess` in the pinned venv
//! and pinned here), the head readout's formula on a deterministic synthetic
//! head, and the HTTP surface over real sockets against a stub inference —
//! the same shape as the reference suite's `test_worker_server.py`
//! (FakeProcessor/FakeExecutor), with the real wire and finishing layers in
//! the loop.

use std::sync::Arc;
use std::time::Duration;

use omni_valen_native::finish::{
    answer, build_response, choice_answer, json_string, noul_answer, python_float, score_answer,
};
use omni_valen_native::protocol::{self, Kind};
use omni_valen_native::server::{Inference, Worker};
use omni_valen_native::wire::RequestError;

// ---------------------------------------------------------------------------
// Finishing parity — the bytes the reference worker sends for these logits.
// ---------------------------------------------------------------------------

/// Recorded from the pinned venv's `repr` (CPython 3.12): shortest round-trip
/// digits, the fixed/scientific switch at 1e-4 and 1e16, two-digit exponents.
#[test]
fn float_formatting_matches_cpython_repr() {
    let vectors: &[(&str, &str)] = &[
        ("0", "0.0"),
        ("-0", "-0.0"),
        ("1", "1.0"),
        ("0.5", "0.5"),
        ("0.2689414213699951", "0.2689414213699951"),
        ("0.7310585786300049", "0.7310585786300049"),
        ("0.0001", "0.0001"),
        ("0.00001", "1e-05"),
        ("0.000061", "6.1e-05"),
        ("1e15", "1000000000000000.0"),
        ("1e16", "1e+16"),
        ("1e-300", "1e-300"),
        ("5e-324", "5e-324"),
        ("2.2250738585072014e-308", "2.2250738585072014e-308"),
        ("1.7976931348623157e308", "1.7976931348623157e+308"),
        ("123.456", "123.456"),
        ("100", "100.0"),
        ("0.1", "0.1"),
        ("0.3333333333333333", "0.3333333333333333"),
        ("3", "3.0"),
        ("9.5e-7", "9.5e-07"),
        ("0.9999999999999999", "0.9999999999999999"),
        ("2", "2.0"),
        ("4", "4.0"),
    ];
    for (literal, expected) in vectors {
        let value: f64 = literal.parse().unwrap();
        assert_eq!(&python_float(value), expected, "repr of {literal}");
    }
}

#[test]
fn json_strings_match_the_reference_escaping() {
    assert_eq!(json_string("plain"), "\"plain\"");
    assert_eq!(json_string("say \"hi\""), "\"say \\\"hi\\\"\"");
    assert_eq!(json_string("back\\slash"), "\"back\\\\slash\"");
    assert_eq!(json_string("a\nb\tc"), "\"a\\nb\\tc\"");
    assert_eq!(json_string("\u{1}\u{8}\u{c}\u{1f}"), "\"\\u0001\\b\\f\\u001f\"");
    // ensure_ascii=False leaves non-ASCII verbatim, including the noul wording.
    assert_eq!(json_string("True / 是：满足问题中的条件。"), "\"True / 是：满足问题中的条件。\"");
}

/// Recorded from the pinned venv's `models.valen.postprocess` + `json.dumps`:
/// the exact response body bytes for these keys and logits.
#[test]
fn choice_responses_are_byte_identical_to_the_reference() {
    struct Case {
        keys: &'static [&'static str],
        logits: &'static [f32],
        body: &'static str,
    }
    let cases = [
        Case {
            keys: &["ok"],
            logits: &[0.0],
            body: "{\"model\": \"valen-preview-0923\", \"answers\": {\"q\": {\"type\": \"choice\", \"choice\": \"ok\", \"probabilities\": {\"ok\": 1.0}, \"confidence\": 1.0}}, \"usage\": {\"input_tokens\": 17, \"output_tokens\": 0}, \"internal_usage\": {\"compute_tokens\": 42}}",
        },
        Case {
            keys: &["down", "up"],
            logits: &[0.0, 1.0],
            body: "{\"model\": \"valen-preview-0923\", \"answers\": {\"q\": {\"type\": \"choice\", \"choice\": \"up\", \"probabilities\": {\"down\": 0.2689414213699951, \"up\": 0.7310585786300049}, \"confidence\": 0.4621171572600098}}, \"usage\": {\"input_tokens\": 17, \"output_tokens\": 0}, \"internal_usage\": {\"compute_tokens\": 42}}",
        },
        Case {
            keys: &["a", "b", "c"],
            logits: &[0.5, 0.2, 0.3],
            body: "{\"model\": \"valen-preview-0923\", \"answers\": {\"q\": {\"type\": \"choice\", \"choice\": \"a\", \"probabilities\": {\"a\": 0.3906938314430244, \"b\": 0.2894331099035222, \"c\": 0.31987305865345345}, \"confidence\": 0.0860407471645366}}, \"usage\": {\"input_tokens\": 17, \"output_tokens\": 0}, \"internal_usage\": {\"compute_tokens\": 42}}",
        },
        Case {
            // A tie picks the first candidate, as Python's max(range) does.
            keys: &["a", "b"],
            logits: &[1.0, 1.0],
            body: "{\"model\": \"valen-preview-0923\", \"answers\": {\"q\": {\"type\": \"choice\", \"choice\": \"a\", \"probabilities\": {\"a\": 0.5, \"b\": 0.5}, \"confidence\": 0.0}}, \"usage\": {\"input_tokens\": 17, \"output_tokens\": 0}, \"internal_usage\": {\"compute_tokens\": 42}}",
        },
        Case {
            // Candidate keys pass through as UTF-8, never \u-escaped.
            keys: &["白色", "黑色"],
            logits: &[0.3, 0.7],
            body: "{\"model\": \"valen-preview-0923\", \"answers\": {\"q\": {\"type\": \"choice\", \"choice\": \"黑色\", \"probabilities\": {\"白色\": 0.4013123456158106, \"黑色\": 0.5986876543841895}, \"confidence\": 0.19737530876837894}}, \"usage\": {\"input_tokens\": 17, \"output_tokens\": 0}, \"internal_usage\": {\"compute_tokens\": 42}}",
        },
        Case {
            keys: &["x"],
            logits: &[2.5],
            body: "{\"model\": \"valen-preview-0923\", \"answers\": {\"q\": {\"type\": \"choice\", \"choice\": \"x\", \"probabilities\": {\"x\": 1.0}, \"confidence\": 1.0}}, \"usage\": {\"input_tokens\": 17, \"output_tokens\": 0}, \"internal_usage\": {\"compute_tokens\": 42}}",
        },
        Case {
            keys: &["a", "b", "c", "d"],
            logits: &[-1.0, -0.5, -2.0, 0.0],
            body: "{\"model\": \"valen-preview-0923\", \"answers\": {\"q\": {\"type\": \"choice\", \"choice\": \"d\", \"probabilities\": {\"a\": 0.17437148764032928, \"b\": 0.2874899806762354, \"c\": 0.06414768542935739, \"d\": 0.47399084625407806}, \"confidence\": 0.2986544616721041}}, \"usage\": {\"input_tokens\": 17, \"output_tokens\": 0}, \"internal_usage\": {\"compute_tokens\": 42}}",
        },
    ];
    for case in cases {
        let answer = choice_answer(
            &case.keys.iter().map(|k| k.to_string()).collect::<Vec<_>>(),
            case.logits,
        )
        .unwrap();
        let body = build_response(&[("q".to_string(), answer)], 17, 42).unwrap();
        assert_eq!(body, case.body.as_bytes(), "keys {:?}", case.keys);
    }
    // Several questions keep the request's insertion order through the map.
    let a = choice_answer(&["p".into(), "q".into()], &[0.0, 1.0]).unwrap();
    let b = choice_answer(&["z".into()], &[0.0]).unwrap();
    let body = build_response(&[("second".into(), a), ("first".into(), b)], 5, 6).unwrap();
    let text = String::from_utf8(body).unwrap();
    assert!(text.contains("\"answers\": {\"second\":"));
    assert!(text.contains(", \"first\":"));
}

/// Recorded from the pinned venv's `models.valen.postprocess._noul_answer` +
/// `json.dumps`: the exact answer bytes for `true`/`false` keys and these
/// logits. P(true) plus the full pair, choice confidence on the two
/// candidates, field order type/noul/probabilities/confidence.
#[test]
fn noul_responses_are_byte_identical_to_the_reference() {
    struct Case {
        logits: &'static [f32],
        body: &'static str,
    }
    let cases = [
        // A tie: P(true) 0.5, confidence 0.
        Case {
            logits: &[0.0, 0.0],
            body: "{\"type\": \"noul\", \"noul\": 0.5, \"probabilities\": {\"true\": 0.5, \"false\": 0.5}, \"confidence\": 0.0}",
        },
        Case {
            logits: &[0.0, 1.0],
            body: "{\"type\": \"noul\", \"noul\": 0.2689414213699951, \"probabilities\": {\"true\": 0.2689414213699951, \"false\": 0.7310585786300049}, \"confidence\": 0.4621171572600098}",
        },
        Case {
            logits: &[2.5, -3.0],
            body: "{\"type\": \"noul\", \"noul\": 0.995929862284104, \"probabilities\": {\"true\": 0.995929862284104, \"false\": 0.004070137715896127}, \"confidence\": 0.9918597245682079}",
        },
        // Negative logits pick the same softmax shape as [0,1].
        Case {
            logits: &[-1.0, -2.0],
            body: "{\"type\": \"noul\", \"noul\": 0.7310585786300049, \"probabilities\": {\"true\": 0.7310585786300049, \"false\": 0.2689414213699951}, \"confidence\": 0.4621171572600098}",
        },
    ];
    let keys: Vec<String> = ["true", "false"].iter().map(|k| k.to_string()).collect();
    for case in cases {
        let produced = noul_answer(&keys, case.logits).unwrap();
        assert_eq!(produced, case.body, "logits {:?}", case.logits);
    }
    // A logit-count mismatch is the worker's inference failure (500), as the
    // reference's ValueError on `answer` is.
    assert!(noul_answer(&keys, &[0.0]).is_err());
}

/// Recorded from the pinned venv's `models.valen.postprocess._score_answer` +
/// `json.dumps`: expected level `sum(i*p)`, the static legend keyed "0"…, and
/// the concentration confidence `max(0, 1 - distance/uniform_mad)`, field
/// order type/probabilities/score/legend/confidence.
#[test]
fn score_responses_are_byte_identical_to_the_reference() {
    struct Case {
        levels: &'static [&'static str],
        logits: &'static [f32],
        body: &'static str,
    }
    const THREE: &[&str] = &["No error", "A minor error", "A major error"];
    let cases = [
        // Uniform: score 1.0, no concentration.
        Case {
            levels: THREE,
            logits: &[0.0, 0.0, 0.0],
            body: "{\"type\": \"score\", \"probabilities\": {\"0\": 0.3333333333333333, \"1\": 0.3333333333333333, \"2\": 0.3333333333333333}, \"score\": 1.0, \"legend\": {\"0\": \"No error\", \"1\": \"A minor error\", \"2\": \"A major error\"}, \"confidence\": 0.0}",
        },
        // Fully peaked on level 1.
        Case {
            levels: THREE,
            logits: &[-100.0, 3.0, -100.0],
            body: "{\"type\": \"score\", \"probabilities\": {\"0\": 1.8521167695179754e-45, \"1\": 1.0, \"2\": 1.8521167695179754e-45}, \"score\": 1.0, \"legend\": {\"0\": \"No error\", \"1\": \"A minor error\", \"2\": \"A major error\"}, \"confidence\": 1.0}",
        },
        // Tilted toward p = [0.7, 0.2, 0.1]: mode 0, distance 0.4, MAD 2/3 →
        // 0.4-class. Logits are the f32 head outputs of ln(7)/ln(2)/0 (the
        // finisher's input contract), so the expected bytes were recorded
        // from the f32-rounded values.
        Case {
            levels: THREE,
            logits: &[1.9459101490553132, 0.6931471805599453, 0.0],
            body: "{\"type\": \"score\", \"probabilities\": {\"0\": 0.6999999886271213, \"1\": 0.20000000770889612, \"2\": 0.10000000366398262}, \"score\": 0.40000001503686133, \"legend\": {\"0\": \"No error\", \"1\": \"A minor error\", \"2\": \"A major error\"}, \"confidence\": 0.39999997744470794}",
        },
        // Mass split across the endpoints clamps confidence to 0.
        Case {
            levels: THREE,
            logits: &[0.4054651081081644, -50.0, 0.0],
            body: "{\"type\": \"score\", \"probabilities\": {\"0\": 0.5999999971505067, \"1\": 7.714999446815271e-23, \"2\": 0.4000000028494934}, \"score\": 0.8000000056989868, \"legend\": {\"0\": \"No error\", \"1\": \"A minor error\", \"2\": \"A major error\"}, \"confidence\": 0.0}",
        },
        // Two levels are the noul pair in score clothing: expected level is
        // P(1), confidence the choice confidence.
        Case {
            levels: &["lo", "hi"],
            logits: &[0.0, 1.0],
            body: "{\"type\": \"score\", \"probabilities\": {\"0\": 0.2689414213699951, \"1\": 0.7310585786300049}, \"score\": 0.7310585786300049, \"legend\": {\"0\": \"lo\", \"1\": \"hi\"}, \"confidence\": 0.4621171572600098}",
        },
        // Ten levels pin the legend's verbatim UTF-8 (never \u-escaped) and a
        // spread distribution that clamps confidence to 0.
        Case {
            levels: &[
                "等级 0：描述“0”", "等级 1：描述“1”", "等级 2：描述“2”", "等级 3：描述“3”",
                "等级 4：描述“4”", "等级 5：描述“5”", "等级 6：描述“6”", "等级 7：描述“7”",
                "等级 8：描述“8”", "等级 9：描述“9”",
            ],
            logits: &[0.5, 0.2, 0.3, -1.0, -2.0, 0.0, 1.0, 0.1, -0.5, 0.0],
            body: "{\"type\": \"score\", \"probabilities\": {\"0\": 0.1478252054193804, \"1\": 0.10951160597706486, \"2\": 0.12102904319969529, \"3\": 0.03298426175920089, \"4\": 0.012134231783427398, \"5\": 0.08966051936517236, \"6\": 0.24372256052054833, \"7\": 0.0990901986496028, \"8\": 0.054381853960735325, \"9\": 0.08966051936517236}, \"score\": 4.345328261256573, \"legend\": {\"0\": \"等级 0：描述“0”\", \"1\": \"等级 1：描述“1”\", \"2\": \"等级 2：描述“2”\", \"3\": \"等级 3：描述“3”\", \"4\": \"等级 4：描述“4”\", \"5\": \"等级 5：描述“5”\", \"6\": \"等级 6：描述“6”\", \"7\": \"等级 7：描述“7”\", \"8\": \"等级 8：描述“8”\", \"9\": \"等级 9：描述“9”\"}, \"confidence\": 0.0}",
        },
        // A three-way tie: mode is level 0 by candidate order, uniform score 1.
        Case {
            levels: &["a", "b", "c"],
            logits: &[1.0, 1.0, 1.0],
            body: "{\"type\": \"score\", \"probabilities\": {\"0\": 0.3333333333333333, \"1\": 0.3333333333333333, \"2\": 0.3333333333333333}, \"score\": 1.0, \"legend\": {\"0\": \"a\", \"1\": \"b\", \"2\": \"c\"}, \"confidence\": 0.0}",
        },
    ];
    for case in cases {
        let keys: Vec<String> = (0..case.levels.len()).map(|i| i.to_string()).collect();
        let levels: Vec<String> = case.levels.iter().map(|l| l.to_string()).collect();
        let produced = score_answer(&keys, &levels, case.logits).unwrap();
        assert_eq!(produced, case.body, "levels {:?}", case.levels);
    }
    let keys: Vec<String> = (0..3).map(|i| i.to_string()).collect();
    let levels: Vec<String> = THREE.iter().map(|l| l.to_string()).collect();
    assert!(score_answer(&keys, &levels, &[0.0, 1.0]).is_err());
}

/// The finisher's per-kind dispatch, `postprocess.answer`'s shape: one mixed
/// body finishes choice + noul + score in request order with each type's
/// exact bytes. Expected body recorded from the pinned venv (stub logits:
/// candidate k scores k).
#[test]
fn mixed_type_bodies_finish_every_question_in_order() {
    let body = vec![
        (
            "move".to_string(),
            answer(
                Kind::Choice,
                &["down".to_string(), "wait".to_string()],
                &["Close".to_string(), "Wait".to_string()],
                &[0.0, 1.0],
            )
            .unwrap(),
        ),
        (
            "dup".to_string(),
            answer(
                Kind::Noul,
                &["true".to_string(), "false".to_string()],
                &["True / 是：满足问题中的条件。".to_string(), "False / 否：不满足问题中的条件。".to_string()],
                &[0.0, 1.0],
            )
            .unwrap(),
        ),
        (
            "sev".to_string(),
            answer(
                Kind::Score,
                &["0".to_string(), "1".to_string(), "2".to_string()],
                &["No error".to_string(), "A minor error".to_string(), "A major error".to_string()],
                &[0.0, 1.0, 2.0],
            )
            .unwrap(),
        ),
    ];
    let bytes = build_response(&body, 17, 42).unwrap();
    assert_eq!(
        String::from_utf8(bytes).unwrap(),
        "{\"model\": \"valen-preview-0923\", \"answers\": {\"move\": {\"type\": \"choice\", \"choice\": \"wait\", \"probabilities\": {\"down\": 0.2689414213699951, \"wait\": 0.7310585786300049}, \"confidence\": 0.4621171572600098}, \"dup\": {\"type\": \"noul\", \"noul\": 0.2689414213699951, \"probabilities\": {\"true\": 0.2689414213699951, \"false\": 0.7310585786300049}, \"confidence\": 0.4621171572600098}, \"sev\": {\"type\": \"score\", \"probabilities\": {\"0\": 0.09003057317038046, \"1\": 0.24472847105479764, \"2\": 0.6652409557748218}, \"score\": 1.5752103826044412, \"legend\": {\"0\": \"No error\", \"1\": \"A minor error\", \"2\": \"A major error\"}, \"confidence\": 0.3628155739066622}}, \"usage\": {\"input_tokens\": 17, \"output_tokens\": 0}, \"internal_usage\": {\"compute_tokens\": 42}}",
    );
}

// ---------------------------------------------------------------------------
// Head readout on the deterministic synthetic head (tests/valen/data).
// ---------------------------------------------------------------------------

fn lcg(i: usize, j: usize) -> f32 {
    ((((i.wrapping_mul(73_856_093)) ^ j.wrapping_mul(19_349_663)) & 0xFFFF_FFFF) % 1_000_003)
        as f32
        / 1_000_003.0
        - 0.5
}

/// The fixture's logits were recorded from numpy over the same deterministic
/// weights; the head's sequential FP32 reduction may differ from BLAS only at
/// the FP32 rounding scale.
#[test]
fn head_readout_matches_the_recorded_numpy_logits() {
    let fixture: serde_json::Value = serde_json::from_slice(&std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../../tests/valen/data/head-readout.json"),
    ).unwrap()).unwrap();
    assert_eq!(fixture["format"], "valen-head-readout/1");
    let (projection, hidden) = (256usize, 2048usize);
    let candidates = fixture["candidates"].as_u64().unwrap() as usize;
    let candidate: Vec<f32> = (0..projection)
        .flat_map(|i| (0..hidden).map(move |j| lcg(i, j)))
        .collect();
    let decision: Vec<f32> = (0..projection)
        .flat_map(|i| (0..hidden).map(move |j| lcg(i + 7919, j + 104_729)))
        .collect();
    let rows: Vec<Vec<f32>> = (0..candidates)
        .map(|k| (0..hidden).map(|j| lcg(3 + k, 31 + j)).collect())
        .collect();
    let hidden_decision: Vec<f32> = (0..hidden).map(|j| lcg(5, 7 + j)).collect();
    let head = omni_valen_native::head::Head::from_parts(candidate, decision, projection, hidden);
    let logits = head.logits(&rows, &hidden_decision).unwrap();
    let recorded_f32: Vec<f64> = fixture["logits_f32"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let recorded_f64: Vec<f64> = fixture["logits_f64"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    assert_eq!(logits.len(), recorded_f32.len());
    for (index, value) in logits.iter().enumerate() {
        let delta = (*value as f64 - recorded_f32[index]).abs();
        assert!(
            delta <= 1e-3,
            "logit {index}: native {value} vs numpy f32 {} (delta {delta})",
            recorded_f32[index]
        );
        assert!(
            (*value as f64 - recorded_f64[index]).abs() <= 1e-3,
            "logit {index} vs the float64 oracle"
        );
    }
}

// -----------------------------------------------------------------------
// HTTP contract over real sockets, stub inference (no tokenizer, no CUDA).
// -----------------------------------------------------------------------

mod http {
    use super::*;

    /// The reference suite's FakeProcessor/FakeExecutor shape: the real wire
    /// layer and the real finishing over fixed logits.
    struct Stub {
        delay: Duration,
        panic: bool,
    }

    impl Inference for Stub {
        fn decide(&self, raw: &[u8]) -> Result<Vec<u8>, RequestError> {
            if self.panic {
                panic!("executor boom");
            }
            if self.delay > Duration::ZERO {
                std::thread::sleep(self.delay);
            }
            let body = omni_valen_native::wire::parse_body(raw)?;
            let request = protocol::parse_request(&body, &protocol::default_special_tokens())?;
            let mut answers = Vec::new();
            for question in &request.questions {
                let logits: Vec<f32> =
                    question.keys.iter().enumerate().map(|(i, _)| i as f32).collect();
                let descriptions: Vec<String> = question
                    .descriptions
                    .iter()
                    .map(|d| d.clone().unwrap_or_default())
                    .collect();
                let produced = answer(question.kind, &question.keys, &descriptions, &logits)
                    .map_err(|_| omni_valen_native::wire::error(500, "inference failed"))?;
                answers.push((question.name.clone(), produced));
            }
            // Token accounting values are the stub's; the compiled accounting
            // is pinned by the compiler tests and the GPU gate.
            build_response(&answers, 17, 17)
                .map_err(|_| omni_valen_native::wire::error(500, "inference failed"))
        }
    }

    fn body() -> Vec<u8> {
        br#"{"model": "valen-preview-0923", "state": "The card was charged twice.", "questions": {"move": {"type": "choice", "instructions": "Close it?", "criteria": {"down": "Close", "wait": "Wait"}}}}"#.to_vec()
    }

    /// A choice + noul + score body (the N5 all-types round trip): the same
    /// stub-logits convention, so the response is the recorded reference body.
    fn mixed_body() -> Vec<u8> {
        br#"{"model": "valen-preview-0923", "state": "The card was charged twice.", "questions": {"move": {"type": "choice", "instructions": "Close it?", "criteria": {"down": "Close", "wait": "Wait"}}, "dup": {"type": "noul", "instructions": "Was it duplicated?"}, "sev": {"type": "score", "instructions": "How severe?", "criteria": ["No error", "A minor error", "A major error"]}}}"#.to_vec()
    }

    async fn start(worker: Worker) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            axum::serve(listener, omni_valen_native::server::router(Arc::new(worker)))
                .await
                .unwrap();
        });
        url
    }

    fn client() -> reqwest::Client {
        reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn health_and_unknown_routes_match_the_reference_bodies() {
        let url = start(Worker::new(Arc::new(Stub {
            delay: Duration::ZERO,
            panic: false,
        })))
        .await;
        let health = client().get(format!("{url}/health")).send().await.unwrap();
        assert_eq!(health.status(), 200);
        assert_eq!(health.headers()["content-type"], "application/json");
        assert_eq!(
            health.bytes().await.unwrap(),
            b"{\"status\": \"ready\", \"model\": \"valen-preview-0923\", \"modality\": \"multimodal\"}"[..]
        );
        let get = client().get(format!("{url}/nope")).send().await.unwrap();
        assert_eq!(get.status(), 404);
        assert_eq!(get.bytes().await.unwrap(), b"{\"detail\": \"unknown route\"}"[..]);
        let post = client()
            .post(format!("{url}/other"))
            .header("content-type", "application/json")
            .body("{}")
            .send()
            .await
            .unwrap();
        assert_eq!(post.status(), 404);
        assert_eq!(post.bytes().await.unwrap(), b"{\"detail\": \"unknown route\"}"[..]);
    }

    #[tokio::test]
    async fn a_valid_choice_request_answers_200() {
        let url = start(Worker::new(Arc::new(Stub {
            delay: Duration::ZERO,
            panic: false,
        })))
        .await;
        let response = client()
            .post(format!("{url}/v1/systemone"))
            .header("content-type", "application/json")
            .body(body())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        assert_eq!(response.headers()["content-type"], "application/json");
        let text = String::from_utf8(response.bytes().await.unwrap().to_vec()).unwrap();
        assert_eq!(
            text,
            "{\"model\": \"valen-preview-0923\", \"answers\": {\"move\": {\"type\": \"choice\", \"choice\": \"wait\", \"probabilities\": {\"down\": 0.2689414213699951, \"wait\": 0.7310585786300049}, \"confidence\": 0.4621171572600098}}, \"usage\": {\"input_tokens\": 17, \"output_tokens\": 0}, \"internal_usage\": {\"compute_tokens\": 17}}",
        );
    }

    #[tokio::test]
    async fn client_errors_keep_the_reference_status_and_detail() {
        let url = start(Worker::new(Arc::new(Stub {
            delay: Duration::ZERO,
            panic: false,
        })))
        .await;
        let post = |payload: Vec<u8>| {
            let client = client();
            let url = url.clone();
            async move {
                client
                    .post(format!("{url}/v1/systemone"))
                    .header("content-type", "application/json")
                    .body(payload)
                    .send()
                    .await
                    .unwrap()
            }
        };
        let response = post(b"{not json".to_vec()).await;
        assert_eq!(response.status(), 400);
        assert_eq!(
            response.bytes().await.unwrap(),
            b"{\"detail\": \"request body must contain valid JSON and UTF-8 text\"}"[..]
        );

        let wrong_model = br#"{"model": "Valen", "state": "x", "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {"a": "A"}}}}"#.to_vec();
        let response = post(wrong_model).await;
        assert_eq!(response.status(), 422);
        assert_eq!(
            response.bytes().await.unwrap(),
            b"{\"detail\": \"model must be valen-preview-0923\"}"[..]
        );

        // N5 removed the N4 milestone guard: all three question types serve.
        // A malformed noul still gets the wire layer's 422, byte-identical to
        // the reference's validation order.
        let bad_noul = br#"{"model": "valen-preview-0923", "state": "x", "questions": {"n": {"type": "noul", "instructions": "i", "criteria": {"maybe": "Neither"}}}}"#.to_vec();
        let response = post(bad_noul).await;
        assert_eq!(response.status(), 422);
        assert_eq!(
            response.bytes().await.unwrap(),
            b"{\"detail\": \"question 'n': noul criteria must describe true and/or false\"}"[..]
        );

        let bad_score = br#"{"model": "valen-preview-0923", "state": "x", "questions": {"s": {"type": "score", "instructions": "i", "criteria": ["only one"]}}}"#.to_vec();
        let response = post(bad_score).await;
        assert_eq!(response.status(), 422);
        assert_eq!(
            response.bytes().await.unwrap(),
            b"{\"detail\": \"question 's': score requires an ordered list of 2 to 10 level descriptions\"}"[..]
        );
    }

    /// All three question types in one body answer 200 through the real wire
    /// and finishing layers, each type's bytes the reference's own.
    #[tokio::test]
    async fn a_mixed_type_request_answers_200_for_every_kind() {
        let url = start(Worker::new(Arc::new(Stub {
            delay: Duration::ZERO,
            panic: false,
        })))
        .await;
        let response = client()
            .post(format!("{url}/v1/systemone"))
            .header("content-type", "application/json")
            .body(mixed_body())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let text = String::from_utf8(response.bytes().await.unwrap().to_vec()).unwrap();
        assert_eq!(
            text,
            "{\"model\": \"valen-preview-0923\", \"answers\": {\"move\": {\"type\": \"choice\", \"choice\": \"wait\", \"probabilities\": {\"down\": 0.2689414213699951, \"wait\": 0.7310585786300049}, \"confidence\": 0.4621171572600098}, \"dup\": {\"type\": \"noul\", \"noul\": 0.2689414213699951, \"probabilities\": {\"true\": 0.2689414213699951, \"false\": 0.7310585786300049}, \"confidence\": 0.4621171572600098}, \"sev\": {\"type\": \"score\", \"probabilities\": {\"0\": 0.09003057317038046, \"1\": 0.24472847105479764, \"2\": 0.6652409557748218}, \"score\": 1.5752103826044412, \"legend\": {\"0\": \"No error\", \"1\": \"A minor error\", \"2\": \"A major error\"}, \"confidence\": 0.3628155739066622}}, \"usage\": {\"input_tokens\": 17, \"output_tokens\": 0}, \"internal_usage\": {\"compute_tokens\": 17}}",
        );
    }

    #[tokio::test]
    async fn header_gates_match_the_reference_order_and_bodies() {
        let url = start(Worker::new(Arc::new(Stub {
            delay: Duration::ZERO,
            panic: false,
        })))
        .await;
        let client = client();
        // 415 before the body is read, whatever the body says.
        let response = client
            .post(format!("{url}/v1/systemone"))
            .header("content-type", "text/plain")
            .body(body())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 415);
        assert_eq!(
            response.bytes().await.unwrap(),
            b"{\"detail\": \"Content-Type must be application/json\"}"[..]
        );
        // Parameters do not change the media type.
        let response = client
            .post(format!("{url}/v1/systemone"))
            .header("content-type", "application/json; charset=utf-8")
            .body(body())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
    }

    /// Raw-socket cases axum's own client cannot produce: chunked upload,
    /// duplicate Content-Length, oversized declared length, truncated body
    /// and the slow-body timeout.
    #[tokio::test]
    async fn raw_sockets_reproduce_the_reference_transport_rejections() {
        let url = start(
            Worker::new(Arc::new(Stub {
                delay: Duration::ZERO,
                panic: false,
            }))
            .with_body_timeout(Duration::from_millis(300)),
        )
        .await;
        let address = Arc::new(url.trim_start_matches("http://").to_owned());
        let raw = |request: String| {
            let address = address.clone();
            async move {
                use tokio::io::{AsyncReadExt, AsyncWriteExt};
                let mut stream = tokio::net::TcpStream::connect(&*address).await.unwrap();
                stream.write_all(request.as_bytes()).await.unwrap();
                let mut buffer = Vec::new();
                let _ = stream.read_to_end(&mut buffer).await;
                buffer
            }
        };
        let payload = body();

        // Chunked: the reference rejects Transfer-Encoding with 411.
        let response = raw(format!(
            "POST /v1/systemone HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\nContent-Type: application/json\r\n\r\n{:x}\r\n",
            payload.len()
        ))
        .await;
        let head = String::from_utf8_lossy(&response).to_string();
        assert!(
            head.starts_with("HTTP/1.1 411")
                && head.contains("Content-Length is required; chunked requests are unsupported"),
            "chunked: {head}"
        );

        // Duplicate Content-Length is unreachable through hyper: it normalizes
        // equal duplicates (and rejects conflicting ones) before the handler
        // runs, where the reference answers 411 "one Content-Length is
        // required" — a recorded transport deviation, see native.md.

        // Declared length beyond the limit: 413 before reading the body.
        let response = raw(
            "POST /v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: 9000000\r\n\r\n"
                .to_owned(),
        )
        .await;
        assert!(response.starts_with(b"HTTP/1.1 413"), "{}", String::from_utf8_lossy(&response));
        assert!(response.ends_with(b"{\"detail\": \"request exceeds body limit\"}"));

        // Truncated body with the write side closed: 400 incomplete body
        // (a stall without closing is the 408 case below, as in Python).
        {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut stream = tokio::net::TcpStream::connect(&*address).await.unwrap();
            stream
                .write_all(
                    format!(
                        "POST /v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{{",
                        payload.len()
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
            stream.shutdown().await.unwrap();
            let mut buffer = Vec::new();
            let _ = stream.read_to_end(&mut buffer).await;
            let head = String::from_utf8_lossy(&buffer).to_string();
            assert!(
                head.starts_with("HTTP/1.1 400") && head.contains("incomplete body"),
                "truncated body: {head}"
            );
        }

        // A body that stalls past the timeout: 408.
        use tokio::io::AsyncWriteExt;
        let mut stream = tokio::net::TcpStream::connect(&*address).await.unwrap();
        stream
            .write_all(
                format!(
                    "POST /v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                    payload.len()
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        stream.flush().await.unwrap();
        tokio::time::sleep(Duration::from_millis(600)).await;
        let mut buffer = Vec::new();
        use tokio::io::AsyncReadExt;
        let _ = stream.read_to_end(&mut buffer).await;
        let head = String::from_utf8_lossy(&buffer).to_string();
        assert!(
            head.starts_with("HTTP/1.1 408") && head.contains("request body timed out"),
            "slow body: {head}"
        );
    }

    #[tokio::test]
    async fn busy_worker_returns_503_and_keeps_serving() {
        let url = start(Worker::new(Arc::new(Stub {
            delay: Duration::from_millis(600),
            panic: false,
        })))
        .await;
        let first = {
            let client = client();
            let url = url.clone();
            tokio::spawn(async move {
                client
                    .post(format!("{url}/v1/systemone"))
                    .header("content-type", "application/json")
                    .body(body())
                    .send()
                    .await
                    .unwrap()
                    .status()
            })
        };
        tokio::time::sleep(Duration::from_millis(150)).await;
        let busy = client()
            .post(format!("{url}/v1/systemone"))
            .header("content-type", "application/json")
            .body(body())
            .send()
            .await
            .unwrap();
        assert_eq!(busy.status(), 503);
        assert_eq!(busy.bytes().await.unwrap(), b"{\"detail\": \"worker busy\"}"[..]);
        assert_eq!(first.await.unwrap(), 200);
        // Admission is released after completion.
        let after = client()
            .post(format!("{url}/v1/systemone"))
            .header("content-type", "application/json")
            .body(body())
            .send()
            .await
            .unwrap();
        assert_eq!(after.status(), 200);
    }

    #[tokio::test]
    async fn a_panicking_inference_maps_to_500_inference_failed() {
        let url = start(Worker::new(Arc::new(Stub {
            delay: Duration::ZERO,
            panic: true,
        })))
        .await;
        let response = client()
            .post(format!("{url}/v1/systemone"))
            .header("content-type", "application/json")
            .body(body())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 500);
        assert_eq!(
            response.bytes().await.unwrap(),
            b"{\"detail\": \"inference failed\"}"[..]
        );
    }
}
