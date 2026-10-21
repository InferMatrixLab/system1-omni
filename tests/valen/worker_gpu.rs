//! Device-gated native choice-worker checks (spec 03, N4).
//!
//! The parity fixture is recorded by `recipe/valen/dump_choice_parity.py`
//! from the pinned reference stack (Valen @ 750bfcf, Preview checkpoint
//! 81b9c63 sha256-pinned, Qwen3.5-2B base 15852e8c, peft 0.18.1 unmerged
//! FP32 adapters, eager attention, TF32 off, BF16) over the recipe's two
//! choice fixtures — the N2 hidden fixture's text state and the example
//! image request — and committed as
//! `recipe/valen/fixtures/choice-parity-reference.json`.
//!
//! Gate semantics (measured first, then declared in the fixture): exact
//! agreement on every discrete field (the chosen key per question, the full
//! answer shape, the usage accounting) and an absolute tolerance on the
//! probability and confidence scalars. The native stack runs its own
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
        .join("../../recipe/valen/fixtures/choice-parity-reference.json");
    let fixture: Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert!(
        fixture["format"] == "valen-choice-parity/1",
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
fn compare_case(name: &str, fixture: &Value, native: &Value, recorded: &Value, tolerance: f64) {
    let reference_answers = &recorded["answers"];
    let native_answers = &native["answers"];
    let mut problems = Vec::new();
    let mut max_probability = 0.0f64;
    let mut max_confidence = 0.0f64;
    let mut checked = 0usize;
    for (qid, expected) in reference_answers.as_object().unwrap() {
        let answer = &native_answers[qid];
        if answer["type"] != expected["type"] {
            problems.push(format!("{qid}: type {} != {}", answer["type"], expected["type"]));
            continue;
        }
        if answer["choice"] != expected["choice"] {
            problems.push(format!(
                "{qid}: chose {:?}, the reference chose {:?}",
                answer["choice"], expected["choice"]
            ));
        }
        for (key, value) in expected["probabilities"].as_object().unwrap() {
            let native_value = answer["probabilities"][key].as_f64().unwrap();
            let reference_value = value.as_f64().unwrap();
            let delta = (native_value - reference_value).abs();
            max_probability = max_probability.max(delta);
            checked += 1;
            if delta > tolerance {
                problems.push(format!(
                    "{qid}/{key}: probability {native_value} vs {reference_value} (delta {delta})"
                ));
            }
        }
        let delta = (answer["confidence"].as_f64().unwrap()
            - expected["confidence"].as_f64().unwrap())
        .abs();
        max_confidence = max_confidence.max(delta);
        if delta > tolerance {
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
        "valen choice parity ({name}): {checked} probabilities compared, \
         max probability delta {max_probability:.6e}, max confidence delta \
         {max_confidence:.6e}, tolerance {tolerance}"
    );
    assert!(
        problems.is_empty(),
        "native choice parity failed for {name}: {}",
        problems.join("; ")
    );
}

fn tolerance(fixture: &Value, kind: &str) -> f64 {
    let value = fixture["tolerance"][kind].as_f64();
    value.unwrap_or_else(|| {
        panic!(
            "the fixture declares no {kind} tolerance; measure the native deltas \
             first, then declare them with dump_choice_parity.py"
        )
    })
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
    let case = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "text")
        .unwrap();
    let mut engine = load();
    let native = native_response(&mut engine, &case["request"]);
    compare_case(
        "text",
        &fixture,
        &native,
        &case["response"],
        tolerance(&fixture, "probability"),
    );
}

/// GPU parity for the image fixture: the full native path (decode → resize →
/// vision tower → multimodal forward with mrope → head) against the same
/// recorded reference.
#[test]
#[ignore = "requires CUDA, VALEN_EXPORT, VALEN_BASE and VALEN_CUDA_LIB"]
fn native_image_choice_matches_the_recorded_reference() {
    let (fixture, _) = fixture();
    let case = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "image")
        .unwrap();
    let mut engine = load();
    let native = native_response(&mut engine, &case["request"]);
    compare_case(
        "image",
        &fixture,
        &native,
        &case["response"],
        tolerance(&fixture, "probability"),
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
        assert!(
            delta <= head_tolerance,
            "head probe logit {index}: native {native} vs reference {reference} \
             (delta {delta} exceeds {head_tolerance})"
        );
    }
    eprintln!("valen head probe: max logit delta {max_delta:.6e} over {} logits", expected.len());
}

/// The serving surface over a real socket with the real engine: health after
/// warmup is the main binary's job; here one admitted request round-trips.
#[tokio::test]
#[ignore = "requires CUDA, VALEN_EXPORT, VALEN_BASE and VALEN_CUDA_LIB"]
async fn native_worker_serves_one_choice_request_over_http() {
    let (fixture, _) = fixture();
    let case = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "text")
        .unwrap();
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
    assert_eq!(
        body["answers"]["which_passed"]["choice"],
        case["response"]["answers"]["which_passed"]["choice"]
    );
}
