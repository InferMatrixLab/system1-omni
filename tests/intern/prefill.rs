use std::{fs, path::PathBuf};

use anyhow::{Result, ensure};
use omni_intern_native::{executor::Executor, finishing, processing::Processor};
use serde_json::{Value, json};

#[test]
#[ignore = "requires the frozen Intern oracle inputs and tokenizer"]
fn frozen_processing() -> Result<()> {
    let (_, cases) = frozen_inputs()?;
    ensure!(cases.len() == 6, "expected all six text fixtures");
    Ok(())
}

#[test]
#[ignore = "requires the frozen Intern checkpoint and sm_89 device library"]
fn frozen_prefill() -> Result<()> {
    let (processor, cases) = frozen_inputs()?;
    let checkpoint = PathBuf::from(std::env::var("INTERN_CHECKPOINT")?);
    let library = PathBuf::from(std::env::var("INTERN_CUDA_LIB")?);
    let mut executor = Executor::load(&checkpoint, &library)?;
    let mut outputs = Vec::with_capacity(cases.len());
    for (id, prepared) in cases {
        let hidden = executor.hidden(&prepared.ids, &prepared.readout)?;
        let logits = executor.project(&hidden, &processor.symbol_ids)?;
        outputs.push(json!({"id": id, "logits": logits, "hidden": hidden}));
    }
    fs::write(
        std::env::var("INTERN_PREFILL_OUT")?,
        serde_json::to_vec(&outputs)?,
    )?;
    Ok(())
}

#[test]
#[ignore = "replays saved native logits with the frozen tokenizer and requests"]
fn frozen_finishing_replay() -> Result<()> {
    let (_, cases) = frozen_inputs()?;
    let logits = fs::read(std::env::var("INTERN_PREFILL_OUT")?)?;
    let logits: Vec<Value> = serde_json::from_slice(&logits)?;
    assert_eq!(cases.len(), logits.len());
    let mut outputs = Vec::with_capacity(cases.len());
    for ((id, prepared), recorded) in cases.into_iter().zip(logits) {
        assert_eq!(json!(id), recorded["id"]);
        let rows: Vec<Vec<f32>> = serde_json::from_value(recorded["logits"].clone())?;
        assert_eq!(rows.len(), prepared.fields.len());
        let raw = rows
            .iter()
            .zip(&prepared.fields)
            .map(|(row, field)| {
                let (raw, _) = finishing::probabilities(&row[..field.options.len()])?;
                Ok((field.name.clone(), json!(raw)))
            })
            .collect::<Result<serde_json::Map<_, _>>>()?;
        let response = finishing::finish(&prepared.fields, &rows, prepared.ids.len(), 0.0)?;
        outputs.push(json!({"id": id, "uncalibrated": raw, "response": response}));
    }
    fs::write(
        std::env::var("INTERN_FINISHING_OUT")?,
        serde_json::to_vec(&outputs)?,
    )?;
    Ok(())
}

fn frozen_inputs() -> Result<(
    Processor,
    Vec<(String, omni_intern_native::processing::Prepared)>,
)> {
    let input = PathBuf::from(std::env::var("INTERN_FROZEN_INPUTS")?);
    let checkpoint = PathBuf::from(std::env::var("INTERN_CHECKPOINT")?);
    let processor = Processor::load(&checkpoint.join("tokenizer.json"))?;
    let fixtures: Vec<Value> = serde_json::from_slice(&fs::read(input)?)?;
    let mut cases = Vec::new();
    for fixture in fixtures {
        let id = fixture["id"].as_str().unwrap();
        let prepared = processor.prepare(&serde_json::to_vec(&fixture["request"])?)?;
        assert_eq!(json!(prepared.prompt), fixture["prompt"], "{id} prompt");
        assert_eq!(json!(prepared.ids), fixture["ids"], "{id} tokens");
        assert_eq!(json!(prepared.readout), fixture["readout"], "{id} readout");
        for (field, expected) in prepared
            .fields
            .iter()
            .zip(fixture["fields"].as_array().unwrap())
        {
            assert_eq!(json!(field.name), expected["field"]);
            assert_eq!(
                json!(
                    field
                        .options
                        .iter()
                        .map(|(name, _)| name)
                        .collect::<Vec<_>>()
                ),
                expected["values"]
            );
            assert_eq!(
                json!(&processor.symbol_ids[..field.options.len()]),
                expected["symbol_ids"]
            );
        }
        cases.push((id.to_owned(), prepared));
    }
    Ok((processor, cases))
}
