use anyhow::Result;
use omni_intern_native::{
    finishing::{finish, probabilities},
    processing::{Kind, compile},
};
use serde_json::{Value, json};

#[test]
fn ordered_schema_and_typed_answers() -> Result<()> {
    let body = json!({
        "state": {"路线": ["重複", null], "flags": {"on": true}},
        "questions": {
            "route": {"type": "choice", "criteria": {"z": 1.0, "a": null}},
            "refund": {"type": "noul", "criteria": {"TRUE": "Affirmative", "false": "Negative"}},
            "severity": {"type": "score", "criteria": {"2.0": "low", "1e1": "high"}},
            "single": {"type": "score", "criteria": ["zero"]}
        }
    });
    let (prompt, fields) = compile(body.as_object().unwrap())?;
    assert!(prompt.contains(
        "{\n  \"路线\": [\n    \"重複\",\n    null\n  ],\n  \"flags\": {\n    \"on\": true\n  }\n}"
    ));
    assert!(prompt.contains("    A = z: 1.0\n    B = a: None"));
    assert!(prompt.contains("    A = no: Negative\n    B = yes: Affirmative"));
    assert!(prompt.contains("{\n    \"route\": \"<decision>\",\n    \"refund\": \"<decision>\",\n    \"severity\": \"<decision>\",\n    \"single\": \"<decision>\"\n}"));
    assert_eq!(
        fields.iter().map(|field| field.kind).collect::<Vec<_>>(),
        [Kind::Choice, Kind::Noul, Kind::Score, Kind::Score]
    );
    let output = finish(
        &fields,
        &[vec![0.0, 0.0], vec![2.0, -2.0], vec![0.0, 0.0], vec![8.0]],
        183,
        1.23,
    )?;
    let answers = &output["answers"];
    assert_eq!(answers["route"]["decision"], "a");
    assert_eq!(answers["route"]["choice"], "a");
    assert_eq!(
        answers["route"]["probabilities"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["z", "a"]
    );
    assert_eq!(answers["refund"]["decision"], "no");
    assert_eq!(
        answers["refund"]["noul"],
        answers["refund"]["probabilities"]["yes"]
    );
    assert_eq!(
        answers["refund"]["confidence"],
        answers["refund"]["probabilities"]["no"]
    );
    assert_eq!(answers["severity"]["score"], 6.0);
    assert_eq!(answers["severity"]["decision"], "1e1");
    assert_eq!(answers["single"]["score"], 0.0);
    assert_eq!(
        output["usage"],
        json!({"input_tokens": 183, "output_tokens": 4, "decision_count": 4})
    );
    Ok(())
}

#[test]
fn rejects_semantic_errors_before_tokenization() {
    let question = json!({"type": "choice", "criteria": {"a": "yes"}});
    for body in [
        json!({"questions": {"q": question}}),
        json!({"state": "hello", "images": ["image.jpg"], "questions": {"q": question}}),
        json!({"state": {"text": "<decision>"}, "questions": {"q": question}}),
        json!({"state": "hello", "questions": {"": question}}),
        json!({"state": "hello", "questions": {"q": {"type":"choice","criteria":{}}}}),
        json!({"state": "hello", "questions": {"q": {"type":"score","criteria":{"NaN":"no"}}}}),
        json!({"state": "hello", "questions": {}}),
    ] {
        assert_eq!(compile(body.as_object().unwrap()).unwrap_err().status, 422);
    }
}

#[test]
fn probabilities_and_extreme_scores_remain_finite() -> Result<()> {
    let (raw, scaled) = probabilities(&[1000.0, 999.0, -1000.0])?;
    assert_eq!(raw[2], 0.0);
    assert_eq!(scaled[2], 0.0);
    assert!((scaled.iter().sum::<f64>() - 1.0).abs() <= 1e-6);
    let request = json!({"state": "", "questions": {
        "extreme": {"type": "score", "criteria": {"-1.7976931348623157e308": "low", "1.7976931348623157e308": "high"}},
        "aliases": {"type": "score", "criteria": {"1": "low", "1.0": "middle", "1e0": "high"}}
    }});
    let (_, fields) = compile(request.as_object().unwrap())?;
    let response = finish(&fields, &[vec![0.0, 0.0], vec![1.0, 2.0, 3.0]], 100, 0.0)?;
    assert_eq!(response["answers"]["extreme"]["score"], 0.0);
    assert_eq!(response["answers"]["aliases"]["score"], 1.0);
    assert!(matches!(
        response["answers"]["aliases"]["score"],
        Value::Number(..)
    ));
    Ok(())
}
