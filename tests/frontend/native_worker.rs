//! The native Valen worker behind the frontend, over real sockets:
//! direct-to-worker and frontend-forwarded responses must be byte-identical,
//! and the worker's contract rejections must pass through the proxy
//! unchanged. The worker runs its real wire and finishing layers over a stub
//! inference (the reference suite's FakeProcessor/FakeExecutor pattern), so
//! none of this needs weights or a GPU.

use std::sync::Arc;

use omni_jev::Config;
use omni_valen_native::finish::{answer, build_response};
use omni_valen_native::server::{Inference, Worker};
use omni_valen_native::wire::{RequestError, error};

mod common;

use common::{client, listen};

/// Fixed per-request logits: candidate k scores k, so the last candidate
/// always wins and the probabilities are the softmax of [0, 1, ..., k).
struct Stub;

impl Inference for Stub {
    fn decide(&self, raw: &[u8]) -> Result<Vec<u8>, RequestError> {
        let body = omni_valen_native::wire::parse_body(raw)?;
        let request = omni_valen_native::protocol::parse_request(
            &body,
            &omni_valen_native::protocol::default_special_tokens(),
        )?;
        let mut answers = Vec::new();
        for question in &request.questions {
            let logits: Vec<f32> = question
                .keys
                .iter()
                .enumerate()
                .map(|(i, _)| i as f32)
                .collect();
            let descriptions: Vec<String> = question
                .descriptions
                .iter()
                .map(|d| d.clone().unwrap_or_default())
                .collect();
            let produced = answer(question.kind, &question.keys, &descriptions, &logits)
                .map_err(|_| error(500, "inference failed"))?;
            answers.push((question.name.clone(), produced));
        }
        build_response(&answers, 17, 17).map_err(|_| error(500, "inference failed"))
    }
}

async fn start_worker() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let router = omni_valen_native::server::router(Arc::new(Worker::new(Arc::new(Stub))));
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    url
}

async fn start_frontend(worker: &str) -> String {
    let config = Config::new("127.0.0.1:0", worker).unwrap();
    listen(omni_jev::app(&config).unwrap()).await
}

fn text_choice(state: &str) -> Vec<u8> {
    format!(
        "{{\"model\": \"valen-preview-0923\", \"state\": \"{state}\", \"questions\": \
         {{\"move\": {{\"type\": \"choice\", \"instructions\": \"Close it?\", \
         \"criteria\": {{\"down\": \"Close\", \"wait\": \"Wait\"}}}}}}}}"
    )
    .into_bytes()
}

/// All three question types in one body (N5): the forwarded bytes must stay
/// byte-identical whichever finishing variant answered.
fn mixed_types() -> Vec<u8> {
    br#"{"model": "valen-preview-0923", "state": "The card was charged twice.", "questions": {"move": {"type": "choice", "instructions": "Close it?", "criteria": {"down": "Close", "wait": "Wait"}}, "dup": {"type": "noul", "instructions": "Was it duplicated?"}, "sev": {"type": "score", "instructions": "How severe?", "criteria": ["No error", "A minor error", "A major error"]}}}"#.to_vec()
}

#[tokio::test]
async fn native_worker_answers_choice_directly() {
    let worker = start_worker().await;
    let response = client()
        .post(format!("{worker}/v1/systemone"))
        .header("content-type", "application/json")
        .body(text_choice("Dialog: Update installed."))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["content-type"], "application/json");
    assert_eq!(
        response.bytes().await.unwrap(),
        b"{\"model\": \"valen-preview-0923\", \"answers\": {\"move\": {\"type\": \"choice\", \"choice\": \"wait\", \"probabilities\": {\"down\": 0.2689414213699951, \"wait\": 0.7310585786300049}, \"confidence\": 0.4621171572600098}}, \"usage\": {\"input_tokens\": 17, \"output_tokens\": 0}, \"internal_usage\": {\"compute_tokens\": 17}}"[..]
    );
}

#[tokio::test]
async fn frontend_forwards_the_native_worker_bytes_unchanged() {
    let worker = start_worker().await;
    let frontend = start_frontend(&worker).await;
    for request in [
        text_choice("The card was charged twice for one order. 请处理。"),
        mixed_types(),
    ] {
        let direct = client()
            .post(format!("{worker}/v1/systemone"))
            .header("content-type", "application/json")
            .body(request.clone())
            .send()
            .await
            .unwrap();
        assert_eq!(direct.status(), 200);
        let direct_bytes = direct.bytes().await.unwrap();

        let forwarded = client()
            .post(format!("{frontend}/v1/systemone"))
            .header("content-type", "application/json")
            .body(request)
            .send()
            .await
            .unwrap();
        assert_eq!(forwarded.status(), 200);
        assert_eq!(forwarded.bytes().await.unwrap(), direct_bytes);
    }
}

#[tokio::test]
async fn native_contract_rejections_pass_through_the_frontend_unchanged() {
    let worker = start_worker().await;
    let frontend = start_frontend(&worker).await;
    let cases: Vec<(&str, u16, Vec<u8>)> = vec![
        (
            "malformed json",
            400,
            b"{not json".to_vec(),
        ),
        (
            "wrong model",
            422,
            br#"{"model": "other", "state": "x", "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {"a": "A"}}}}"#.to_vec(),
        ),
        (
            "invalid noul criteria",
            422,
            br#"{"model": "valen-preview-0923", "state": "x", "questions": {"n": {"type": "noul", "instructions": "i", "criteria": {"maybe": "Neither"}}}}"#.to_vec(),
        ),
        (
            "invalid score levels",
            422,
            br#"{"model": "valen-preview-0923", "state": "x", "questions": {"s": {"type": "score", "instructions": "i", "criteria": ["only one"]}}}"#.to_vec(),
        ),
    ];
    for (name, status, request) in cases {
        let direct = client()
            .post(format!("{worker}/v1/systemone"))
            .header("content-type", "application/json")
            .body(request.clone())
            .send()
            .await
            .unwrap();
        assert_eq!(direct.status().as_u16(), status, "{name}: status");
        let direct_bytes = direct.bytes().await.unwrap();
        let forwarded = client()
            .post(format!("{frontend}/v1/systemone"))
            .header("content-type", "application/json")
            .body(request)
            .send()
            .await
            .unwrap();
        assert_eq!(
            forwarded.status().as_u16(),
            status,
            "{name}: forwarded status"
        );
        assert_eq!(
            forwarded.bytes().await.unwrap(),
            direct_bytes,
            "{name}: forwarded body must be byte-identical to the direct body"
        );
    }
}

#[tokio::test]
async fn health_flows_through_the_frontend() {
    let worker = start_worker().await;
    let frontend = start_frontend(&worker).await;
    let health = client()
        .get(format!("{frontend}/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(health.status(), 200);
    assert_eq!(
        health.bytes().await.unwrap(),
        b"{\"status\": \"ready\", \"model\": \"valen-preview-0923\", \"modality\": \"multimodal\"}"
            [..]
    );
    // A busy worker's 503 is observable only while one request is in flight;
    // the stub here is instant, so that mapping is pinned in the worker's own
    // suite (tests/valen/worker.rs) rather than through the proxy.
}
