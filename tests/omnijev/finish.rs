//! Response finishing against `MSO1._finish` (tests/omnijev/data/finish.json).
mod common;

use omni_omnijev_native::contract::{self, Calibration, Kind, Question};
use serde_json::Value;

fn calibration(fixture: &Value) -> Calibration {
    let t = &fixture["temperatures"];
    Calibration {
        temperatures: ["noul", "choice", "score"].map(|k| t[k].as_f64().unwrap()),
        noul_bias: fixture["biases"]["noul"].as_f64().unwrap(),
    }
}

fn question(kind: Kind, keys: &[String]) -> Question {
    Question {
        id: "q".into(),
        kind,
        instructions: "x".into(),
        options: keys.to_vec(),
        keys: keys.to_vec(),
    }
}

/// Values within one unit of the fourth decimal, and the same answer key unless the
/// reference's top two are that close.
fn near(actual: &Value, expected: &Value, case: &Value) {
    let probs = &expected["probabilities"];
    let top: Vec<f64> = {
        let mut p: Vec<f64> = probs
            .as_object()
            .map(|m| m.values().map(|v| v.as_f64().unwrap()).collect())
            .unwrap_or_default();
        p.sort_by(|a, b| b.partial_cmp(a).unwrap());
        p
    };
    for (key, e) in expected.as_object().unwrap() {
        let a = &actual[key];
        match e {
            Value::Number(_) => {
                assert!(
                    (a.as_f64().unwrap() - e.as_f64().unwrap()).abs() <= 1.0001e-4,
                    "{key}: {case}"
                )
            }
            Value::Object(m) => {
                for (k, v) in m {
                    assert!(
                        (a[k].as_f64().unwrap() - v.as_f64().unwrap()).abs() <= 1.0001e-4,
                        "{k}: {case}"
                    );
                }
            }
            _ if top.len() > 1 && top[0] - top[1] <= 2e-4 => {}
            _ => assert_eq!(a, e, "{key}: {case}"),
        }
    }
}

#[test]
fn answers_match_reference() {
    let fixture = common::fixture("finish.json");
    let calibration = calibration(&fixture);
    for case in fixture["cases"].as_array().unwrap() {
        let kind = match case["type"].as_str().unwrap() {
            "noul" => Kind::Noul,
            "choice" => Kind::Choice,
            _ => Kind::Score,
        };
        let keys: Vec<String> = case["keys"]
            .as_array()
            .unwrap()
            .iter()
            .map(|k| k.as_str().unwrap().into())
            .collect();
        let mu: Vec<f32> = case["mu"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let answer = contract::answer(&question(kind, &keys), &mu, &calibration, 0.0).unwrap();
        // Up to four options the float32 sums add in the same order as PyTorch's; with
        // more, PyTorch's vectorized order can move the fourth decimal by one.
        if keys.len() <= 4 {
            assert_eq!(answer, case["answer"], "{case}");
        } else {
            near(&answer, &case["answer"], case);
        }
    }
}

#[test]
fn invalid_calibration_is_an_error() {
    let good = calibration(&common::fixture("finish.json"));
    let mut bad = vec![Calibration {
        noul_bias: f64::NAN,
        ..good
    }];
    for t in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        for i in 0..3 {
            let mut c = good;
            c.temperatures[i] = t;
            bad.push(c);
        }
    }
    let keys = ["a".to_string(), "b".to_string()];
    for c in &bad {
        for kind in [Kind::Noul, Kind::Choice, Kind::Score] {
            let err = contract::answer(&question(kind, &keys), &[0.3, 0.6], c, 0.0).unwrap_err();
            assert_eq!(err.to_string(), "invalid calibration");
        }
    }
}

#[test]
fn round4_matches_python() {
    // round(x, 4) on exactly representable ties goes to the even digit.
    for (x, expected) in [
        (0.03125, 0.0312),
        (0.09375, 0.0938),
        (0.5, 0.5),
        (0.99995, 1.0),
        (1e-9, 0.0),
    ] {
        assert_eq!(contract::round4(x), expected, "{x}");
    }
}
