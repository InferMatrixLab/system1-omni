//! Debug helper: compare the rendered rows against one fixture and
//! print the first differing field. Run with:
//!
//!     VELA_NATIVE_BUNDLE=/abs/path cargo run -p omni-vela-native --example parity -- example-request

use std::path::PathBuf;

use serde_json::Value;

fn main() {
    let case = std::env::args().nth(1).unwrap_or("example-request".into());
    let bundle = PathBuf::from(
        std::env::var("VELA_NATIVE_BUNDLE").expect("VELA_NATIVE_BUNDLE points at the export"),
    );
    let fixture: Value = serde_json::from_str(
        &std::fs::read_to_string(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../../../tests/vela/fixtures")
                .join(format!("native-plan-{case}.json")),
        )
        .expect("fixture"),
    )
    .expect("fixture json");

    let processor = omni_vela_native::processing::Processor::load(&bundle).expect("load");
    let request = fixture["request"].as_object().expect("request");
    match processor.plan(request) {
        Ok(planned) => {
            let got = serde_json::json!({
                "rows": serde_json::to_value(&planned.rows).unwrap(),
                "window_rows": serde_json::to_value(&planned.window_rows).unwrap(),
            });
            compare(
                "rows",
                got["rows"].as_array().unwrap(),
                fixture["rows"].as_array().unwrap(),
            );
            if !fixture["window_rows"].is_null() {
                compare(
                    "window_rows",
                    got["window_rows"].as_array().unwrap(),
                    fixture["window_rows"].as_array().unwrap(),
                );
            }
            println!("parity: no field differences found");
        }
        Err(error) => println!(
            "plan error: {}",
            serde_json::to_string(&error.body).unwrap()
        ),
    }
}

fn compare(name: &str, got: &[Value], want: &[Value]) {
    println!("{name}: {} vs {} row(s)", got.len(), want.len());
    for (index, (a, b)) in got.iter().zip(want).enumerate() {
        let (Some(a), Some(b)) = (a.as_object(), b.as_object()) else {
            continue;
        };
        for key in b.keys() {
            if a.get(key) != b.get(key) {
                println!("row {index} field {key}: differs");
                println!(
                    "  want: {}",
                    serde_json::to_string(&b[key])
                        .unwrap()
                        .chars()
                        .take(300)
                        .collect::<String>()
                );
                println!(
                    "  got:  {}",
                    serde_json::to_string(&a[key])
                        .unwrap()
                        .chars()
                        .take(300)
                        .collect::<String>()
                );
            }
        }
        for key in a.keys() {
            if !b.contains_key(key) {
                println!("row {index} field {key}: extra in ours");
            }
        }
    }
}
