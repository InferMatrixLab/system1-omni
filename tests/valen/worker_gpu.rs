//! Device-gated native worker checks (spec 03, N4 choice + N5 noul/score).
//!
//! The parity fixture is recorded by `recipe/valen/dump_choice_parity.py`
//! from the pinned reference stack (Valen @ 750bfcf, Preview checkpoint
//! 81b9c63 sha256-pinned, Qwen3.5-2B base 15852e8c, peft 0.18.1 unmerged
//! FP32 adapters, eager attention, TF32 off, BF16) over five fixtures: the
//! N2 hidden fixture's text choice state, the example image choice request,
//! the example noul request (plain + custom wording), the example score
//! request (3 levels, one branch per level), and one mixed-type body —
//! committed as `recipe/valen/fixtures/choice-parity-reference.json`.
//!
//! Gate semantics (measured first, then declared in the fixture): exact
//! agreement on every discrete field (the chosen key per choice question,
//! the static score legend, the full answer shape, the usage accounting)
//! and an absolute tolerance on the probability and confidence scalars; the
//! score questions' expected level carries its own declared tolerance
//! (an aggregate of the gated probabilities). The native stack runs its own
//! kernels over the same numeric profile — cross-kernel BF16 drift
//! accumulates over 24 language layers (N2 measured 0.5 max-abs on the
//! final hidden at scale 50) and, on the image path, the vision tower adds
//! its own declared drift (N1 measured 1.03 max-abs on a 2.5-scale output)
//! — so the probability-level tolerance quantifies end-to-end divergence
//! instead of demanding bit equality. A `null` tolerance refuses to pass:
//! measure the native deltas first, then declare them in the fixture.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;

use omni_valen_native::engine::Engine;
use omni_valen_native::head::Head;
use omni_valen_native::server::Worker;

fn fixture() -> (Value, PathBuf) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../recipe/valen/fixtures/choice-parity-reference.json");
    let fixture: Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert!(
        fixture["format"] == "valen-choice-parity/2",
        "unexpected fixture format"
    );
    (fixture, path)
}

fn env_path(name: &str) -> PathBuf {
    PathBuf::from(std::env::var_os(name).expect(name))
}

fn request_bytes(body: &Value) -> Vec<u8> {
    // The fixture bodies are plain JSON; re-serialize compactly exactly as
    // the recording script served them (json.dumps separators ", "/": ").
    serde_json::to_vec(body).unwrap()
}

fn decode_bf16(hex: &str) -> Vec<f32> {
    let bytes = (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex digit"))
        .collect::<Vec<_>>();
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
        .collect()
}

fn load() -> Engine {
    Engine::load(
        &env_path("VALEN_EXPORT"),
        &env_path("VALEN_BASE"),
        &env_path("VALEN_CUDA_LIB"),
    )
    .unwrap()
}

/// Compare one case's native outcome against the recorded reference.
/// A `None` tolerance measures and reports without asserting, so the first
/// run on new hardware produces the declaration numbers.
fn compare_case(name: &str, case_name: &str, fixture: &Value, native: &Value, recorded: &Value, tolerance: Option<f64>, score_tolerance: Option<f64>) {
    let reference_answers = &recorded["answers"];
    let native_answers = &native["answers"];
    let mut problems = Vec::new();
    let mut max_probability = 0.0f64;
    let mut max_confidence = 0.0f64;
    let mut max_score = 0.0f64;
    let mut checked = 0usize;
    for (qid, expected) in reference_answers.as_object().unwrap() {
        let answer = &native_answers[qid];
        if answer["type"] != expected["type"] {
            problems.push(format!("{qid}: type {} != {}", answer["type"], expected["type"]));
            continue;
        }
        if answer.is_object() && answer.as_object().unwrap().contains_key("choice")
            && answer["choice"] != expected["choice"]
        {
            // An answer flip is a discrete failure unless the fixture carries
            // an explicit per-case declaration that the reference's own margin
            // was near a tie inside the declared component tolerances.
            if fixture["declared_flips"][case_name].is_null() {
                problems.push(format!(
                    "{qid}: chose {:?}, the reference chose {:?}",
                    answer["choice"], expected["choice"]
                ));
            } else {
                eprintln!(
                    "{qid}: answer flip (native {:?} vs reference {:?}) is the \
                     fixture's declared near-tie case",
                    answer["choice"], expected["choice"]
                );
            }
        }
        if expected.is_object() && expected.as_object().unwrap().contains_key("noul") {
            // The noul scalar is P(true): gated by the probability tolerance
            // and folded into the same reported figure.
            let delta = (answer["noul"].as_f64().unwrap()
                - expected["noul"].as_f64().unwrap())
                .abs();
            max_probability = max_probability.max(delta);
            if tolerance.is_some_and(|t| delta > t) {
                problems.push(format!(
                    "{qid}: noul {} vs {} (delta {delta})",
                    answer["noul"], expected["noul"]
                ));
            }
        }
        if expected.is_object() && expected.as_object().unwrap().contains_key("score") {
            let delta = (answer["score"].as_f64().unwrap()
                - expected["score"].as_f64().unwrap())
                .abs();
            max_score = max_score.max(delta);
            if score_tolerance.is_some_and(|t| delta > t) {
                problems.push(format!(
                    "{qid}: score {} vs {} (delta {delta})",
                    answer["score"], expected["score"]
                ));
            }
            // The legend is static text from the request: exact or a bug.
            if answer["legend"] != expected["legend"] {
                problems.push(format!(
                    "{qid}: legend {:?} != {:?}",
                    answer["legend"], expected["legend"]
                ));
            }
        }
        for (key, value) in expected["probabilities"].as_object().unwrap() {
            let native_value = answer["probabilities"][key].as_f64().unwrap();
            let reference_value = value.as_f64().unwrap();
            let delta = (native_value - reference_value).abs();
            max_probability = max_probability.max(delta);
            checked += 1;
            if tolerance.is_some_and(|t| delta > t) {
                problems.push(format!(
                    "{qid}/{key}: probability {native_value} vs {reference_value} (delta {delta})"
                ));
            }
        }
        let delta = (answer["confidence"].as_f64().unwrap()
            - expected["confidence"].as_f64().unwrap())
            .abs();
        max_confidence = max_confidence.max(delta);
        if tolerance.is_some_and(|t| delta > t) {
            problems.push(format!(
                "{qid}: confidence {} vs {} (delta {delta})",
                answer["confidence"], expected["confidence"]
            ));
        }
    }
    if native["usage"] != recorded["usage"] {
        problems.push(format!(
            "usage {:?} != {:?}",
            native["usage"], recorded["usage"]
        ));
    }
    if native["internal_usage"] != recorded["internal_usage"] {
        problems.push(format!(
            "internal_usage {:?} != {:?}",
            native["internal_usage"], recorded["internal_usage"]
        ));
    }
    eprintln!(
        "valen parity ({name}): {checked} probabilities compared, \
         max probability delta {max_probability:.6e}, max confidence delta \
         {max_confidence:.6e}, max score delta {max_score:.6e}, tolerance \
         {tolerance:?} (score {score_tolerance:?})"
    );
    let discrete: Vec<String> = problems
        .iter()
        .filter(|p| !p.contains("delta"))
        .cloned()
        .collect();
    assert!(
        discrete.is_empty(),
        "native parity failed for {name}: {}",
        discrete.join("; ")
    );
    if tolerance.is_some() {
        let numeric: Vec<String> = problems
            .iter()
            .filter(|p| p.contains("delta"))
            .cloned()
            .collect();
        assert!(
            numeric.is_empty(),
            "native parity failed for {name}: {}",
            numeric.join("; ")
        );
    }
}

fn tolerance(fixture: &Value, kind: &str) -> Option<f64> {
    fixture["tolerance"][kind].as_f64()
}

fn case<'a>(fixture: &'a Value, name: &str) -> &'a Value {
    fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == name)
        .unwrap_or_else(|| panic!("the fixture carries no {name} case"))
}

fn native_response(engine: &mut Engine, body: &Value) -> Value {
    let outcome = engine.decide(&request_bytes(body)).unwrap();
    let bytes =
        omni_valen_native::finish::build_response(&outcome.answers, outcome.logical_tokens, outcome.compute_tokens)
            .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

/// GPU parity for the text fixture: exact answers and declared probability
/// tolerance against the recorded reference.
#[test]
#[ignore = "requires CUDA, VALEN_EXPORT, VALEN_BASE and VALEN_CUDA_LIB"]
fn native_text_choice_matches_the_recorded_reference() {
    let (fixture, _) = fixture();
    let case = case(&fixture, "text");
    let mut engine = load();
    let native = native_response(&mut engine, &case["request"]);
    let tolerance = tolerance(&fixture, "probability");
    compare_case("text", "text", &fixture, &native, &case["response"], tolerance, None);
    assert!(
        tolerance.is_some(),
        "the fixture declares no probability tolerance; the measured deltas above \
         are the declaration input (dump_choice_parity.py --tolerance-probability)"
    );
}

/// GPU parity for the image fixture: the full native path (decode → resize →
/// vision tower → multimodal forward with mrope → head) against the same
/// recorded reference.
#[test]
#[ignore = "requires CUDA, VALEN_EXPORT, VALEN_BASE and VALEN_CUDA_LIB"]
fn native_image_choice_matches_the_recorded_reference() {
    let (fixture, _) = fixture();
    let case = case(&fixture, "image");
    let mut engine = load();
    let native = native_response(&mut engine, &case["request"]);
    let tolerance = tolerance(&fixture, "probability");
    compare_case("image", "image", &fixture, &native, &case["response"], tolerance, None);
    assert!(
        tolerance.is_some(),
        "the fixture declares no probability tolerance; the measured deltas above \
         are the declaration input (dump_choice_parity.py --tolerance-probability)"
    );
}

/// GPU parity for the noul fixture (N5): one plain noul question (the
/// reference's native noul record) and one with custom true/false wording
/// (compiled as a choice record, finished as noul) — both answer `type: noul`
/// with the reference's field semantics.
#[test]
#[ignore = "requires CUDA, VALEN_EXPORT, VALEN_BASE and VALEN_CUDA_LIB"]
fn native_noul_matches_the_recorded_reference() {
    let (fixture, _) = fixture();
    let case = case(&fixture, "noul");
    let mut engine = load();
    let native = native_response(&mut engine, &case["request"]);
    let tolerance = tolerance(&fixture, "probability");
    compare_case("noul", "noul", &fixture, &native, &case["response"], tolerance, None);
    assert!(
        tolerance.is_some(),
        "the fixture declares no probability tolerance; the measured deltas above \
         are the declaration input (dump_choice_parity.py --tolerance-probability)"
    );
}

/// GPU parity for the score fixture (N5): three levels, one branch per level,
/// cross-branch softmax → expected level + legend + concentration confidence.
/// The expected-level scalar carries its own declared tolerance.
#[test]
#[ignore = "requires CUDA, VALEN_EXPORT, VALEN_BASE and VALEN_CUDA_LIB"]
fn native_score_matches_the_recorded_reference() {
    let (fixture, _) = fixture();
    let case = case(&fixture, "score");
    let mut engine = load();
    let native = native_response(&mut engine, &case["request"]);
    let probability = tolerance(&fixture, "probability");
    let score = tolerance(&fixture, "score");
    compare_case("score", "score", &fixture, &native, &case["response"], probability, score);
    assert!(
        probability.is_some(),
        "the fixture declares no probability tolerance; the measured deltas above \
         are the declaration input (dump_choice_parity.py --tolerance-probability)"
    );
    assert!(
        score.is_some(),
        "the fixture declares no score tolerance; the measured max score delta above \
         is the declaration input (dump_choice_parity.py --tolerance-score)"
    );
}

/// GPU parity for the mixed fixture (N5): choice + noul + score in one body,
/// one admission, per-kind execution and finishing in request order.
#[test]
#[ignore = "requires CUDA, VALEN_EXPORT, VALEN_BASE and VALEN_CUDA_LIB"]
fn native_mixed_types_match_the_recorded_reference() {
    let (fixture, _) = fixture();
    let case = case(&fixture, "mixed");
    let mut engine = load();
    let native = native_response(&mut engine, &case["request"]);
    let probability = tolerance(&fixture, "probability");
    let score = tolerance(&fixture, "score");
    compare_case("mixed", "mixed", &fixture, &native, &case["response"], probability, score);
    assert!(
        probability.is_some() && score.is_some(),
        "the fixture declares no probability/score tolerance; the measured deltas \
         above are the declaration input (dump_choice_parity.py)"
    );
}

/// The confidence tolerance may differ from the probability tolerance; it is
/// declared separately in the fixture and checked inside `compare_case`
/// against the probability figure only for reporting. Re-run the discrete
/// checks with the confidence tolerance applied.
#[test]
#[ignore = "requires CUDA, VALEN_EXPORT, VALEN_BASE and VALEN_CUDA_LIB"]
fn native_confidence_deltas_are_inside_the_declared_tolerance() {
    let (fixture, _) = fixture();
    let confidence = tolerance(&fixture, "confidence");
    assert!(
        confidence.is_some(),
        "the fixture declares no confidence tolerance; measure first, then declare"
    );
    let confidence = confidence.unwrap();
    let mut engine = load();
    for case in fixture["cases"].as_array().unwrap() {
        let native = native_response(&mut engine, &case["request"]);
        for (qid, expected) in case["response"]["answers"].as_object().unwrap() {
            let delta = (native["answers"][qid]["confidence"].as_f64().unwrap()
                - expected["confidence"].as_f64().unwrap())
            .abs();
            assert!(
                delta <= confidence,
                "{qid} ({}) confidence delta {delta} exceeds {confidence}",
                case["name"]
            );
        }
    }
}

/// The CPU head on the reference's own captured readout rows (BF16 bits),
/// against the FP32 logits the reference head computed on exactly those rows.
#[test]
#[ignore = "requires VALEN_EXPORT (the head file); no CUDA forward"]
fn native_head_matches_the_reference_probe() {
    let (fixture, _) = fixture();
    let probe = &fixture["head_probe"];
    let head = Head::load(&env_path("VALEN_EXPORT")).unwrap();
    let rows = decode_bf16(probe["rows_bf16_hex"].as_str().unwrap());
    let hidden = probe["hidden_size"].as_u64().unwrap() as usize;
    let candidates = probe["candidate_count"].as_u64().unwrap() as usize;
    assert_eq!(rows.len() % hidden, 0);
    let all: Vec<Vec<f32>> = rows.chunks(hidden).map(<[f32]>::to_vec).collect();
    assert_eq!(all.len(), candidates + 1);
    let decision = all[candidates].clone();
    let logits = head.logits(&all[..candidates], &decision).unwrap();
    let expected: Vec<f64> = probe["logits_f32"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let head_tolerance = tolerance(&fixture, "head_probe");
    let mut max_delta = 0.0f64;
    for (index, (native, reference)) in logits.iter().zip(&expected).enumerate() {
        let delta = (*native as f64 - reference).abs();
        max_delta = max_delta.max(delta);
        if let Some(t) = head_tolerance {
            assert!(
                delta <= t,
                "head probe logit {index}: native {native} vs reference {reference} \
                 (delta {delta} exceeds {t})"
            );
        }
    }
    if head_tolerance.is_none() {
        panic!(
            "the fixture declares no head-probe tolerance; measured max logit delta \
             {max_delta:.6e} over {} logits — declare with dump_choice_parity.py \
             --head-tolerance",
            expected.len()
        );
    }
    eprintln!("valen head probe: max logit delta {max_delta:.6e} over {} logits", expected.len());
}

/// The serving surface over a real socket with the real engine: health after
/// warmup is the main binary's job; here one admitted mixed-type request
/// round-trips (N5: all three finishing variants over one HTTP response).
#[tokio::test]
#[ignore = "requires CUDA, VALEN_EXPORT, VALEN_BASE and VALEN_CUDA_LIB"]
async fn native_worker_serves_one_request_over_http() {
    let (fixture, _) = fixture();
    let case = case(&fixture, "mixed");
    let engine = load();
    let worker = Arc::new(Worker::with_engine(engine));
    let router = omni_valen_native::server::router(worker);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    let health = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap()
        .get(format!("{url}/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(health.status(), 200);
    let decision = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(120))
        .build()
        .unwrap()
        .post(format!("{url}/v1/systemone"))
        .header("content-type", "application/json")
        .body(request_bytes(&case["request"]))
        .send()
        .await
        .unwrap();
    assert_eq!(decision.status(), 200);
    let body: Value = serde_json::from_slice(&decision.bytes().await.unwrap()).unwrap();
    for qid in ["move", "duplicate", "severity"] {
        assert_eq!(
            body["answers"][qid]["type"],
            case["response"]["answers"][qid]["type"],
            "{qid}: the served answer type must be the reference's"
        );
    }
    assert_eq!(
        body["answers"]["move"]["choice"],
        case["response"]["answers"]["move"]["choice"]
    );
}

/// Decomposition diagnostic for the image path, on the N1 vision fixture's
/// image (the recipe example): (1) the native pixel preprocessing vs the
/// recorded HF-processed BF16 pixel bits, (2) the native tower rows vs the
/// recorded reference tower rows, (3) the native language+head readout fed
/// the *reference* tower rows vs the reference's recorded logits (isolating
/// the mrope positions, the row splice and the head readout positions), and
/// (4) the fully native logits. Failure modes are printed, not asserted.
#[test]
#[ignore = "requires CUDA, VALEN_EXPORT, VALEN_BASE and VALEN_CUDA_LIB"]
fn image_path_decomposition_diagnostic() {
    let vision: Value = serde_json::from_slice(&std::fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../../recipe/valen/fixtures/vision-tower-example-image.json"),
    ).unwrap()).unwrap();
    let (fixture, _) = fixture();
    let case = case(&fixture, "image");
    let expected_logits: Vec<f64> = case["logits"][0]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let decode = |hex: &str| -> Vec<half::bf16> {
        let bytes = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex digit"))
            .collect::<Vec<_>>();
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])))
            .collect()
    };
    let grid: [usize; 3] = vision["image_grid_thw"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let reference_pixels = decode(vision["pixel_values_bf16_hex"].as_str().unwrap());
    let reference_rows = decode(vision["tower_output_bf16_hex"].as_str().unwrap());

    // The request: the example image, parsed through the real wire layer.
    let request_bytes = serde_json::to_vec(&case["request"]).unwrap();
    let mut engine = load();
    let request = omni_valen_native::protocol::parse_request(
        &omni_valen_native::wire::parse_body(&request_bytes).unwrap(),
        &engine.processor.special_tokens,
    )
    .unwrap();
    let prepared = omni_valen_native::preprocess::prepare(&request);
    let compiled = engine.processor.compiler().compile(&prepared).unwrap();
    let branch = &compiled.questions[0].branches[0];

    // (1) pixel bits
    use omni_valen_native::engine::VALEN_IMAGE_LIMITS;
    let omni_valen_native::protocol::State::Image(image) = &request.state else {
        panic!("fixture is an image request");
    };
    let decoded = image::load_from_memory(&image.data).unwrap().into_rgb8();
    let processed = omni_qwen3_5_native::image_preprocess::preprocess_rgb8_with_limits(
        decoded.width() as usize,
        decoded.height() as usize,
        decoded.as_raw(),
        &VALEN_IMAGE_LIMITS,
    )
    .unwrap();
    let native_pixel_bits: Vec<u16> = processed
        .pixel_values
        .iter()
        .map(|v| half::bf16::from_f32(*v).to_bits())
        .collect();
    let reference_pixel_bits: Vec<u16> = reference_pixels.iter().map(|v| v.to_bits()).collect();
    let pixel_exact = native_pixel_bits == reference_pixel_bits;
    let pixel_diff = native_pixel_bits
        .iter()
        .zip(&reference_pixel_bits)
        .filter(|(a, b)| a != b)
        .count();
    eprintln!(
        "(1) pixel preprocessing: {} exact, {pixel_diff}/{} BF16 elements differ, grid {:?}",
        if pixel_exact { "BIT" } else { "NOT" },
        native_pixel_bits.len(),
        processed.image_grid_thw,
    );

    // (2) native tower rows vs recorded
    let native_rows = engine.image_rows_probe(&request, &prepared, &compiled).unwrap();
    let mut tower_max = 0.0f32;
    for (a, b) in native_rows.iter().zip(&reference_rows) {
        tower_max = tower_max.max((a.to_f32() - b.to_f32()).abs());
    }
    eprintln!(
        "(2) tower output: max abs delta {tower_max:.6} over {} rows (N1 declared 1.25)",
        native_rows.len() / 2048
    );

    // (3)+(4) language + head over either row source
    let mut readout = |rows: &[half::bf16], label: &str| {
        let mrope = omni_valen_native::engine::mrope_positions(
            &branch.tokens,
            omni_valen_native::compile::IMAGE_PAD,
            &grid,
        )
        .unwrap();
        let mut positions = branch.candidate_positions.clone();
        positions.push(branch.decision_position);
        let mut hidden = engine
            .text()
            .language()
            .forward_multimodal_positions(
                &omni_qwen3_5_native::inputs::MultimodalInput {
                    token_ids: &branch.tokens,
                    image_token_indices: &mrope.1,
                    image_embeddings: rows,
                    position_ids: [
                        mrope.0[0].as_slice(),
                        mrope.0[1].as_slice(),
                        mrope.0[2].as_slice(),
                    ],
                },
                &positions,
            )
            .unwrap();
        let decision = hidden.pop().unwrap();
        let logits = engine.head().logits(&hidden, &decision).unwrap();
        let report: Vec<String> = logits
            .iter()
            .zip(&expected_logits)
            .map(|(native, reference)| {
                format!("{native:.6} (ref {reference:.6}, delta {:.6})", (*native as f64 - reference).abs())
            })
            .collect();
        eprintln!("({label}) logits: {}", report.join(", "));
    };
    readout(&reference_rows, "3 language+head on REFERENCE tower rows");
    readout(&native_rows, "4 language+head on NATIVE tower rows");
}
