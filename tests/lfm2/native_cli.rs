use serde_json::{Value, json};
use std::{
    io::Write,
    process::{Command, Output, Stdio},
};

const BINARY: &str = env!("CARGO_BIN_EXE_omni-lfm2-native");

fn call_with_stdin(args: &[&str], body: &[u8]) -> Output {
    let mut child = Command::new(BINARY)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(body).unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn cli_renders_request_without_weights() {
    let output = call_with_stdin(
        &["render", "-"],
        br#"{"model":"lfm2.5-350m","state":"hello","questions":{"q":{
            "type":"choice","instructions":"","criteria":{"z":null,"a":[]}
        }}}"#,
    );
    assert!(output.status.success());
    let output: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(output["questions"][0]["id"], "q");
    assert_eq!(
        output["questions"][0]["schema"]["fields"][0]["candidates"][0]["value_text"],
        "\"z\"\n"
    );
}

#[test]
fn cli_validates_all_questions_before_opening_tokenizer() {
    let output = call_with_stdin(
        &["prepare", "unavailable-tokenizer.json", "-"],
        br#"{"model":"lfm2.5-350m","state":"ok","questions":{
            "first":{"type":"choice","instructions":"","criteria":{"x":null}},
            "second":{"type":"choice","instructions":"","criteria":{}}
        }}"#,
    );
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let error: Value = serde_json::from_slice(&output.stderr).unwrap();
    assert_eq!(
        error,
        json!({"status":422,"detail":"question 'second' criteria must map 1..255 keys"})
    );
}

#[test]
#[ignore = "requires the exported artifact via LFM2_ARTIFACT_DIR"]
fn cli_inspects_real_f32_artifact() {
    let directory = std::env::var_os("LFM2_ARTIFACT_DIR").expect("LFM2_ARTIFACT_DIR");
    let output = Command::new(BINARY)
        .arg("inspect")
        .arg(directory)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let inspection: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(inspection["manifest"]["format"], "omni-lfm2-f32-v1");
    assert_eq!(inspection["tensors"], 148);
    assert_eq!(inspection["parameters"], 354_483_968);
    assert_eq!(inspection["tensor_data_bytes"], 1_417_935_872u64);
}
