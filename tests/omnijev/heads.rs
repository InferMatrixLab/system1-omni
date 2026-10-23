//! Heads and LM features against the pinned reference (tests/omnijev/data).
mod common;

use omni_omnijev_native::heads::{self, Heads};

/// The fixture generator's inputs in [-2, 2): integer arithmetic, then one rounding.
fn sequence(seed: u64, n: usize) -> Vec<f32> {
    (0..n as u64)
        .map(|i| {
            let x = (i * 2_654_435_761 + seed * 40_503) % (1 << 32);
            (x as f64 / 4_294_967_296.0 * 4.0 - 2.0) as f32
        })
        .collect()
}

fn close(actual: &[f32], expected: &serde_json::Value, tolerance: f32, what: &str) {
    let expected: Vec<f32> = expected
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    assert_eq!(actual.len(), expected.len(), "{what}");
    for (a, e) in actual.iter().zip(&expected) {
        assert!(
            (a - e).abs() <= tolerance * e.abs().max(1.0),
            "{what}: {actual:?} vs {expected:?}"
        );
    }
}

#[test]
fn lm_features_match_reference() {
    let fixture = common::fixture("features.json");
    let rows = fixture["rows"].as_array().unwrap();
    for group in fixture["groups"].as_array().unwrap() {
        let members: Vec<usize> = group
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let picked: Vec<Vec<f32>> = members
            .iter()
            .map(|&i| {
                rows[i]["picked"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect()
            })
            .collect();
        let first: Vec<Option<f32>> = members
            .iter()
            .map(|&i| rows[i]["first"].as_f64().map(|v| v as f32))
            .collect();
        let features = heads::lm_features(&picked, &first);
        for (f, &i) in features.iter().zip(&members) {
            close(f, &rows[i]["feats"], 1e-6, &format!("row {i}"));
        }
    }
}

/// Needs an export: `OMNIJEV_EXPORT=<recipe/omnijev/export.py output>`.
#[test]
#[ignore]
fn heads_match_reference() {
    let dir = std::env::var("OMNIJEV_EXPORT").expect("set OMNIJEV_EXPORT");
    let heads = Heads::load(&std::path::Path::new(&dir).join("heads.safetensors")).unwrap();
    let fixture = common::fixture("heads.json");
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let seeds: Vec<u64> = case["seeds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        let k = case["options"].as_u64().unwrap() as usize;
        let type_id = case["type_id"].as_u64().unwrap() as usize;
        let u: Vec<Vec<f32>> = sequence(seeds[0], k * heads::HIDDEN)
            .chunks(heads::HIDDEN)
            .map(<[f32]>::to_vec)
            .collect();
        let zq = sequence(seeds[1], heads::HIDDEN);
        let features: Vec<[f32; 6]> = if type_id == 0 {
            vec![[0.0; 6]; k]
        } else {
            sequence(seeds[2], k * 6)
                .chunks(6)
                .map(|c| c.try_into().unwrap())
                .collect()
        };
        close(
            &heads.option_logits(&u, &zq, type_id, &features).unwrap(),
            &case["logits"],
            1e-5,
            name,
        );
        close(
            &heads
                .option_probabilities(&u, &zq, type_id, &features)
                .unwrap(),
            &case["mu"],
            1e-5,
            name,
        );
        close(
            &heads.ordinal(&zq, &u).unwrap(),
            &case["ordinal"],
            1e-5,
            name,
        );
    }
}
