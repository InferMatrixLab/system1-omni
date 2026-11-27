use super::*;

#[test]
fn geometry_matches_pinned_official_length_groups() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("data/planning.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let lengths: Vec<_> = case["lengths"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let actual = geometry(case["split"].as_u64().unwrap() as usize, &lengths);
        for (a, e) in actual.iter().zip(case["plans"].as_array().unwrap()) {
            assert_eq!(
                [a.split, a.state, a.question, a.prefix, a.count, a.index],
                ["split", "state", "question", "prefix", "count", "index"]
                    .map(|k| e[k].as_u64().unwrap() as usize),
                "{case}"
            );
        }
    }
}

#[test]
fn state_boundary_and_common_prefix_are_validated() {
    let row = |ids: Vec<u32>| Row {
        positions: (0..ids.len() as i64).collect(),
        ids,
        readouts: vec![],
    };
    assert!(plan(&[]).is_err());
    assert!(
        plan(&[
            row(vec![248060, 1, 248061, 2]),
            row(vec![248060, 3, 248061, 2])
        ])
        .is_err()
    );
    assert!(plan(&[row(vec![248061, 2])]).is_err());
    assert!(plan(&[row(vec![248060, 248061, 2, 248061])]).is_err());
    assert!(plan(&[row(vec![248060, 248061, 2])]).is_ok());
}
