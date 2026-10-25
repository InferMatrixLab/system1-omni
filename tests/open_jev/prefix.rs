//! Prefix-reuse switches, and (with a GPU) shared prefixes against fixed-algorithm rows.

use std::path::PathBuf;

use omni_open_jev_native::{
    engine::{Engine, check_graph, prefix_values},
    executor::PrefixUse,
};
use omni_qwen3_5_native::prefix::PrefixMode;
use serde_json::json;

#[test]
fn switches_follow_decider() {
    assert_eq!(prefix_values(None, None).unwrap(), PrefixMode::Off);
    assert_eq!(
        prefix_values(Some("0"), Some("0")).unwrap(),
        PrefixMode::Off
    );
    assert_eq!(prefix_values(Some("1"), None).unwrap(), PrefixMode::Shared);
    assert_eq!(prefix_values(Some("auto"), None).unwrap(), PrefixMode::Auto);
    assert_eq!(prefix_values(None, Some("1")).unwrap(), PrefixMode::Fixed);
    for (prefix, fixed) in [
        (Some("1"), Some("1")),
        (Some("auto"), Some("1")),
        (Some("yes"), None),
        (None, Some("auto")),
    ] {
        assert!(
            prefix_values(prefix, fixed).is_err(),
            "{prefix:?} {fixed:?}"
        );
    }
    check_graph(PrefixMode::Off, true).unwrap();
    for mode in [PrefixMode::Fixed, PrefixMode::Shared, PrefixMode::Auto] {
        check_graph(mode, false).unwrap();
        assert!(check_graph(mode, true).is_err(), "{mode:?}");
    }
}

/// Two questions over a long shared state: a five-option Choice, whose instructions are
/// long enough for its candidates to share a question prefix past the next 64-token
/// boundary, and a Noul.
fn request() -> Vec<u8> {
    let state =
        "Policy: refunds need a receipt, a purchase within 30 days and an unused item. ".repeat(40);
    let route = format!(
        "Which team should handle this? {}",
        "Weigh the order history, the product, the payment method and any security signal. "
            .repeat(6)
    );
    serde_json::to_vec(&json!({
        "state": state,
        "questions": {
            "route": {"type": "choice", "instructions": route,
                      "criteria": {"billing": "Billing", "returns": "Returns", "security": "Security",
                                   "technical": "Technical", "sales": "Sales"}},
            "refund": {"type": "noul", "instructions": "Is a refund permitted?"}
        }
    }))
    .unwrap()
}

async fn run(engine: &Engine, raw: &[u8]) -> (Vec<Vec<f32>>, PrefixUse) {
    let inputs = engine.processor.prepare(raw).unwrap().inputs;
    engine
        .executor
        .execute_with_prefix(&engine.scheduler, inputs)
        .await
        .unwrap()
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "needs the pinned Open-Jev export, CUDA library and a GPU reservation"]
async fn shared_prefixes_match_fixed_rows_and_auto_keeps_small_requests_on_the_default_path() {
    let model = PathBuf::from(std::env::var_os("OPEN_JEV_MODEL").unwrap());
    let library = PathBuf::from(std::env::var_os("OPEN_JEV_CUDA_LIB").unwrap());
    let small = br#"{"state":"Dialog: Update installed.","questions":{"q":{"type":"choice","instructions":"Close it.","criteria":{"ok":"OK","wait":"Wait"}}}}"#;
    let (fixed, off_small) = {
        let engine = Engine::load_with_prefix(&model, &library, PrefixMode::Fixed)
            .await
            .unwrap();
        let fixed = run(&engine, &request()).await;
        assert!(!fixed.1.shared);
        drop(engine);
        let off = Engine::load_with_prefix(&model, &library, PrefixMode::Off)
            .await
            .unwrap();
        (fixed.0, run(&off, small).await.0)
    };
    let engine = Engine::load_with_prefix(&model, &library, PrefixMode::Shared)
        .await
        .unwrap();
    // What sharing only the request prefix would save; the Choice's question prefix
    // must add to it.
    let inputs = engine
        .processor
        .prepare(&request())
        .unwrap()
        .inputs
        .concat();
    let request_prefix = (0..inputs.iter().map(Vec::len).min().unwrap() - 1)
        .take_while(|&i| inputs.iter().all(|ids| ids[i] == inputs[0][i]))
        .count()
        / 64
        * 64;
    for _ in 0..2 {
        let (rows, used) = run(&engine, &request()).await;
        assert!(
            used.shared && used.saved_tokens > (inputs.len() - 1) * request_prefix,
            "{used:?}"
        );
        assert_eq!(rows, fixed, "shared prefixes against fixed rows");
    }
    drop(engine);
    let engine = Engine::load_with_prefix(&model, &library, PrefixMode::Auto)
        .await
        .unwrap();
    let (rows, used) = run(&engine, small).await;
    assert!(!used.shared, "{used:?}");
    assert_eq!(
        rows, off_small,
        "auto keeps small requests on the default path"
    );
}
