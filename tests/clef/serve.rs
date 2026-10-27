//! The default Clef HTTP path is `omni-clef`.
//!
//! A fake executor process stands in for the checkpoint. Concurrent posts
//! must share its pid and its main thread. That is the opposite of
//! `ThreadingHTTPServer`, which scored each request on a new thread.

use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    path::PathBuf,
    process::{Command, Stdio},
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use omni_clef::{
    Cli, Engine, MAX_BODY, ServeArgs, locate_pythonpath_from, parse_cli, python_worker_command,
    serve_on,
};
use serde_json::Value;
use tokio::net::TcpListener;

const BODY: &[u8] = br#"{"model":"clef-flash","state":"up","questions":{"urgent":{"type":"noul","instructions":"now?"}}}"#;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../..")
        .canonicalize()
        .expect("repository root")
}

fn spawn_fake(env: &[(&str, &str)]) -> anyhow::Result<Engine> {
    let root = repo_root();
    let mut command = Command::new("python3");
    command.arg(root.join("tests/clef/fake_worker.py"));
    command.env("PYTHONPATH", root.join("src"));
    command.stderr(Stdio::null());
    for (key, value) in env {
        command.env(key, value);
    }
    let mut engine = Engine::spawn(command)?;
    engine.body_timeout = Duration::from_secs(15);
    Ok(engine)
}

struct Running {
    addr: SocketAddr,
    engine: Arc<Engine>,
}

async fn boot(env: &[(&str, &str)], timeout: Duration) -> Running {
    let mut engine = spawn_fake(env).expect("fake executor");
    engine.body_timeout = timeout;
    let engine = Arc::new(engine);
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local addr");
    let served = Arc::clone(&engine);
    tokio::spawn(async move {
        serve_on(listener, served).await.expect("serve");
    });
    Running { addr, engine }
}

fn exchange(addr: SocketAddr, head: &str, body: &[u8]) -> (u16, Vec<u8>) {
    let mut stream = TcpStream::connect(addr).expect("connect");
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .expect("read timeout");
    stream
        .set_write_timeout(Some(Duration::from_secs(5)))
        .expect("write timeout");
    stream.write_all(head.as_bytes()).expect("write head");
    stream.write_all(body).expect("write body");
    read_http(&mut stream)
}

fn post(addr: SocketAddr, body: &[u8]) -> (u16, Vec<u8>) {
    let head = format!(
        "POST /v1/systemone HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        body.len()
    );
    exchange(addr, &head, body)
}

fn read_http(stream: &mut TcpStream) -> (u16, Vec<u8>) {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 2048];
    let header_end = loop {
        let count = stream.read(&mut chunk).expect("read");
        assert!(count > 0, "connection closed before headers");
        buffer.extend_from_slice(&chunk[..count]);
        if let Some(index) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
            break index;
        }
        assert!(buffer.len() < 65_536, "headers too large");
    };
    let header = std::str::from_utf8(&buffer[..header_end]).expect("header utf-8");
    let status: u16 = header
        .lines()
        .next()
        .expect("status line")
        .split_whitespace()
        .nth(1)
        .expect("status code")
        .parse()
        .expect("status integer");
    let mut length = None;
    for line in header.lines() {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if name.eq_ignore_ascii_case("content-length") {
            length = Some(value.trim().parse::<usize>().expect("length"));
        }
    }
    let length = length.expect("content-length");
    let mut body = buffer[header_end + 4..].to_vec();
    while body.len() < length {
        let count = stream.read(&mut chunk).expect("body");
        assert!(count > 0, "truncated body");
        body.extend_from_slice(&chunk[..count]);
    }
    body.truncate(length);
    (status, body)
}

fn json_body(bytes: &[u8]) -> Value {
    serde_json::from_slice(bytes).expect("json")
}

#[test]
fn worker_command_is_the_persistent_python_executor() {
    let args = ServeArgs {
        model: "snap".into(),
        variant: "clef-flash".into(),
        device: "cuda".into(),
        gdn_prefill: "fla".into(),
        host: "127.0.0.1".into(),
        port: 8000,
    };
    let command = python_worker_command(&args, "python3".as_ref(), "src".as_ref());
    assert_eq!(command.get_program(), "python3");
    let rendered: Vec<_> = command
        .get_args()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        rendered,
        [
            "-u",
            "-m",
            "models.clef.worker",
            "--model",
            "snap",
            "--variant",
            "clef-flash",
            "--device",
            "cuda",
            "--gdn-prefill",
            "fla",
        ]
    );
    let path = command
        .get_envs()
        .find(|(key, _)| *key == "PYTHONPATH")
        .and_then(|(_, value)| value)
        .expect("PYTHONPATH");
    let mut expected = std::ffi::OsString::from("src");
    if let Some(existing) = std::env::var_os("PYTHONPATH") {
        expected.push(":");
        expected.push(existing);
    }
    assert_eq!(path, expected);
    assert!(
        !rendered
            .iter()
            .any(|arg| arg.contains("ThreadingHTTPServer")),
        "the default executor command must not be the Python HTTP server"
    );
}

#[test]
fn parse_cli_defaults_and_rejects_a_missing_model() {
    let Cli::Serve(args) = parse_cli(["--model", "snap"]).expect("parse") else {
        panic!("expected serve");
    };
    assert_eq!(args.variant, "clef-flash");
    assert_eq!(args.device, "cuda");
    assert_eq!(args.gdn_prefill, "fla");
    assert_eq!(args.host, "127.0.0.1");
    assert_eq!(args.port, 8000);
    assert!(parse_cli(["--variant", "clef"]).is_err());
    assert!(parse_cli(["--model", "snap", "--variant", "other"]).is_err());
    assert!(matches!(parse_cli(["--help"]), Ok(Cli::Help)));
}

#[test]
fn locates_src_from_the_crate_directory() {
    let src = locate_pythonpath_from(env!("CARGO_MANIFEST_DIR").as_ref()).expect("src");
    assert!(src.join("models/clef/worker.py").is_file());
}

#[test]
fn spawn_waits_for_executor_warmup_before_it_returns() {
    let started = Instant::now();
    let engine = spawn_fake(&[("CLEF_FAKE_WARMUP", "0.3")]).expect("warm");
    assert!(started.elapsed() >= Duration::from_millis(250));
    assert!(engine.ready());
    drop(engine);
}

#[test]
fn failed_warmup_does_not_open_a_port() {
    let Err(error) = spawn_fake(&[("CLEF_FAKE_WARMUP_FAIL", "1")]) else {
        panic!("warmup failure must not return an engine");
    };
    let message = format!("{error:#}");
    assert!(message.contains("port was not opened"), "{message}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn health_and_post_keep_the_python_envelope() {
    let running = boot(&[], Duration::from_secs(2)).await;
    let (status, body) = get(running.addr, "/health");
    assert_eq!(status, 200);
    let health = json_body(&body);
    assert_eq!(health["variant"], "clef-flash");
    assert_eq!(health["prompt_mode"], "non-thinking");
    assert_eq!(health["modalities"], serde_json::json!(["text"]));
    assert_eq!(health["max_input_tokens"], 16384);
    assert_eq!(health["linear_attention"], "fla+causal_conv1d");
    assert_eq!(health["cuda_main_thread"], true);
    assert!(
        body.windows(b"\"prompt_mode\": \"non-thinking\"".len())
            .any(|window| window == b"\"prompt_mode\": \"non-thinking\""),
        "health bytes should be the executor's JSON, not a re-encoded document"
    );
    let (status, body) = post(running.addr, BODY);
    assert_eq!(status, 200);
    let answer = json_body(&body);
    assert_eq!(answer["answers"]["urgent"]["noul"], 0.25);
    assert_eq!(answer["usage"]["output_tokens"], 0);
    assert_eq!(answer["cuda_main_thread"], true);
    assert_eq!(answer["pid"], health["pid"]);
    assert_eq!(answer["cuda_thread_id"], health["cuda_thread_id"]);
    let _ = running.engine;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn transport_errors_match_the_clef_handler() {
    let running = boot(&[], Duration::from_secs(2)).await;
    let addr = running.addr;

    let (status, body) = post(addr, b"{");
    assert_eq!(status, 400);
    assert!(json_body(&body).get("detail").is_some());

    let head = "POST /v1/systemone HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: text/plain\r\nContent-Length: 2\r\n\r\n";
    let (status, _) = exchange(addr, head, b"{}");
    assert_eq!(status, 415);

    let head = format!(
        "POST /v1/systemone HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        MAX_BODY + 1
    );
    let (status, body) = exchange(addr, &head, b"");
    assert_eq!(status, 413, "{}", String::from_utf8_lossy(&body));
    assert!(
        body.windows(b"8 MiB".len())
            .any(|window| window == b"8 MiB")
    );

    let head = "POST /v1/systemone HTTP/1.1\r\nHost: 127.0.0.1\r\nTransfer-Encoding: chunked\r\nContent-Type: application/json\r\n\r\n0\r\n\r\n";
    let (status, body) = exchange(addr, head, b"");
    assert_eq!(status, 411, "{}", String::from_utf8_lossy(&body));

    let head = "POST /v1/systemone HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: abc\r\n\r\n";
    let (status, _) = exchange(addr, head, b"");
    assert_eq!(status, 400);

    // Hyper rejects a negative Content-Length before the handler, so this is
    // 400 rather than the Python server's 413. A length above 8 MiB still
    // reaches the handler and keeps that detail.
    let head = "POST /v1/systemone HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: -1\r\n\r\n";
    let (status, body) = exchange(addr, head, b"");
    assert_eq!(status, 400);
    assert!(body.is_empty());

    let (status, body) = post_path(addr, "/missing", b"{}");
    assert_eq!(status, 404);
    assert_eq!(body, br#"{"detail": "unknown route"}"#);

    let (status, _) = get(addr, "/v1/systemone");
    assert_eq!(status, 404);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_requests_share_the_executor_main_thread() {
    let running = boot(&[("CLEF_FAKE_DELAY", "0.35")], Duration::from_secs(5)).await;
    let addr = running.addr;
    let started = Instant::now();
    let first = thread::spawn(move || post(addr, BODY));
    let second = thread::spawn(move || post(addr, BODY));
    let (status_a, body_a) = first.join().expect("first");
    let (status_b, body_b) = second.join().expect("second");
    let elapsed = started.elapsed();
    assert_eq!(status_a, 200, "{}", String::from_utf8_lossy(&body_a));
    assert_eq!(status_b, 200, "{}", String::from_utf8_lossy(&body_b));
    assert!(
        elapsed >= Duration::from_millis(650),
        "requests overlapped on different threads or processes: {elapsed:?}"
    );
    let left = json_body(&body_a);
    let right = json_body(&body_b);
    assert_eq!(left["cuda_main_thread"], true);
    assert_eq!(right["cuda_main_thread"], true);
    assert_eq!(left["active_threads"], 1);
    assert_eq!(right["active_threads"], 1);
    assert_eq!(left["cuda_thread_id"], right["cuda_thread_id"]);
    assert_eq!(left["pid"], right["pid"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stalled_body_does_not_hold_admission() {
    let running = boot(&[], Duration::from_millis(700)).await;
    let addr = running.addr;
    let stalled = thread::spawn(move || {
        let mut stream = TcpStream::connect(addr).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_secs(3)))
            .expect("timeout");
        let head = "POST /v1/systemone HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n";
        stream.write_all(head.as_bytes()).expect("write");
        read_http(&mut stream)
    });
    thread::sleep(Duration::from_millis(40));
    let started = Instant::now();
    let (status, body) = post(addr, BODY);
    assert_eq!(status, 200, "{}", String::from_utf8_lossy(&body));
    assert!(
        started.elapsed() < Duration::from_millis(400),
        "a partial body held the executor"
    );
    let (stalled_status, stalled_body) = stalled.join().expect("stalled");
    assert_eq!(
        stalled_status,
        408,
        "{}",
        String::from_utf8_lossy(&stalled_body)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn inference_failure_stays_on_the_same_executor() {
    let running = boot(&[("CLEF_FAKE_FAIL", "1")], Duration::from_secs(2)).await;
    let (status, body) = post(running.addr, BODY);
    assert_eq!(status, 500);
    assert_eq!(body, br#"{"detail": "inference failed"}"#);
    let (status, _) = post(running.addr, BODY);
    assert_eq!(status, 500);
    let (status, health) = get(running.addr, "/health");
    assert_eq!(status, 200);
    assert_eq!(json_body(&health)["status"], "ready");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_dead_executor_does_not_hang_the_next_request() {
    let running = boot(&[("CLEF_FAKE_EXIT", "1")], Duration::from_secs(2)).await;
    let (status, _) = post(running.addr, BODY);
    assert_eq!(status, 200);
    let (status, body) = post(running.addr, BODY);
    assert_eq!(status, 500, "{}", String::from_utf8_lossy(&body));
    assert_eq!(body, br#"{"detail": "inference failed"}"#);
    let (status, body) = post(running.addr, BODY);
    assert_eq!(status, 503, "{}", String::from_utf8_lossy(&body));
    assert_eq!(body, br#"{"detail": "worker unavailable"}"#);
}

fn get(addr: SocketAddr, path: &str) -> (u16, Vec<u8>) {
    let head = format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");
    exchange(addr, &head, b"")
}

fn post_path(addr: SocketAddr, path: &str, body: &[u8]) -> (u16, Vec<u8>) {
    let head = format!(
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        body.len()
    );
    exchange(addr, &head, body)
}
