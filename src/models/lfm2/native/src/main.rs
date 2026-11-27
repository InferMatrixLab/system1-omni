use anyhow::{Context, Result};
use omni_lfm2_native::{artifact, processing};
use serde_json::Value;
use std::{
    env, fs,
    io::{self, Read, Write},
    path::Path,
    process::ExitCode,
};

const HELP: &str = "\
LFM2.5-350M Choice preparation and score replay\n\
Usage: omni-lfm2-native <command> ...\n\
  validate REQUEST\n\
  render REQUEST\n\
  prepare TOKENIZER_JSON REQUEST\n\
  finish TOKENIZER_JSON REQUEST SCORES_JSON\n\
  inspect ARTIFACT_DIR\n\
Use '-' to read one request or score file from stdin. Scores are one array of\n\
full-candidate FP32 scores per question, in token-plan order.";

fn read(path: &str) -> Result<Vec<u8>> {
    if path == "-" {
        let mut bytes = Vec::new();
        io::stdin().read_to_end(&mut bytes)?;
        Ok(bytes)
    } else {
        fs::read(path).with_context(|| format!("read {path}"))
    }
}

fn run(args: &[String]) -> Result<Value> {
    match args {
        [command, request] if command == "validate" => {
            processing::validate_request(&read(request)?)?;
            Ok(serde_json::json!({"valid": true}))
        }
        [command, request] if command == "render" => Ok(serde_json::to_value(
            processing::render_request(&read(request)?)?,
        )?),
        [command, tokenizer, request] if command == "prepare" => {
            let rendered = processing::render_request(&read(request)?)?;
            let tokenizer = artifact::load_tokenizer(Path::new(tokenizer))?;
            Ok(serde_json::to_value(rendered.tokenize(&tokenizer)?)?)
        }
        [command, tokenizer, request, scores]
            if command == "finish" && (request != "-" || scores != "-") =>
        {
            let rendered = processing::render_request(&read(request)?)?;
            let tokenizer = artifact::load_tokenizer(Path::new(tokenizer))?;
            let scores: Vec<Vec<f32>> =
                serde_json::from_slice(&read(scores)?).context("invalid score arrays")?;
            processing::finish_request(&rendered.tokenize(&tokenizer)?, &scores)
        }
        [command, artifact] if command == "inspect" => Ok(serde_json::to_value(
            artifact::inspect(Path::new(artifact))?,
        )?),
        _ => anyhow::bail!("{HELP}"),
    }
}

fn main() -> ExitCode {
    let args: Vec<_> = env::args().skip(1).collect();
    if matches!(args.as_slice(), [flag] if flag == "--help" || flag == "-h") {
        println!("{HELP}");
        return ExitCode::SUCCESS;
    }
    let result = run(&args).and_then(|output| {
        let mut out = io::stdout().lock();
        serde_json::to_writer_pretty(&mut out, &output)?;
        writeln!(out)?;
        Ok(())
    });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            if let Some(request) = error.downcast_ref::<processing::RequestError>() {
                eprintln!(
                    "{}",
                    serde_json::to_string(request).expect("request error serialization")
                );
            } else {
                eprintln!("{error:#}");
            }
            ExitCode::FAILURE
        }
    }
}
