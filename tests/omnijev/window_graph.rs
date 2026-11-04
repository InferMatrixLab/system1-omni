//! Exercise OmniJev's production prefix/option routing with language Graphs.
use super::*;

#[path = "common.rs"]
mod common;

// Replace the model only after dropping its prefix states and the old model.
// This selects both modes without mutating process-wide environment variables.
fn language_mode(executor: Executor, dir: &Path, library: &Path, graph: bool) -> Executor {
    let Executor {
        vision,
        language,
        heads,
        calibration,
        request,
        question,
    } = executor;
    drop((request, question, language));
    Executor {
        vision,
        language: Model::load_with_graph(dir, library, graph).unwrap(),
        heads,
        calibration,
        request: None,
        question: None,
    }
}

fn request(
    processor: &Processor,
    width: u32,
    height: u32,
    image_changed: bool,
    question_changed: bool,
    growth: bool,
) -> PreparedRequest {
    let colour = if image_changed {
        [30, 180, 90]
    } else {
        [200, 40, 40]
    };
    let instruction = if growth {
        format!(
            "Look{} at the image. Which colour dominates?",
            " again".repeat(1200)
        )
    } else if question_changed {
        "Which colour occupies most of this picture?".into()
    } else {
        "Which colour dominates the image?".into()
    };
    let options = if question_changed {
        json!([{"key": "green", "text": "green"}, {"key": "red", "text": "red"},
               {"key": "blue", "text": "blue"}])
    } else {
        json!([{"key": "red", "text": "red"}, {"key": "green", "text": "green"}])
    };
    let raw = serde_json::to_vec(&json!({
        "model": contract::MODEL_ID,
        "state": {"images": [common::png_url(width, height, colour)]},
        "questions": {
            "colour": {"type": "choice", "instructions": instruction, "options": options},
            "red": {"type": "noul", "instructions": "The image is mostly red."},
            "warm": {"type": "score", "instructions": "How warm is the colour?",
                     "levels": ["cold", "neutral", "warm", "hot"]}
        }
    }))
    .unwrap();
    processor.prepare(&raw).unwrap()
}

fn bits(outputs: &[Vec<f32>]) -> Vec<u32> {
    outputs.iter().flatten().map(|v| v.to_bits()).collect()
}

fn decisions(mut answers: Vec<Value>) -> Vec<Value> {
    for answer in &mut answers {
        let answer = answer.as_object_mut().unwrap();
        answer.remove("latency_s");
        answer.remove("latency_total_s");
    }
    answers
}

#[test]
#[ignore = "requires CUDA, OMNIJEV_EXPORT and OMNIJEV_CUDA_LIB; run serially"]
fn window_graph_matches_eager_across_requests_growth_and_recovery() {
    let dir =
        std::path::PathBuf::from(std::env::var_os("OMNIJEV_EXPORT").expect("set OMNIJEV_EXPORT"));
    let library = std::path::PathBuf::from(
        std::env::var_os("OMNIJEV_CUDA_LIB").expect("set OMNIJEV_CUDA_LIB"),
    );
    let (executor, processor) = Executor::load(&dir, &library).unwrap();
    let cases = [
        request(&processor, 320, 200, false, false, false),
        request(&processor, 320, 200, true, false, false),
        request(&processor, 320, 200, true, true, false),
        request(&processor, 1280, 720, false, false, true),
    ];
    let mut eager = language_mode(executor, &dir, &library, false);
    let expected: Vec<_> = cases
        .iter()
        .map(|prepared| {
            let shared = eager.head_outputs(prepared, true).unwrap();
            let plain = eager.head_outputs(prepared, false).unwrap();
            assert_eq!(bits(&shared), bits(&plain), "eager prefix/row parity");
            (bits(&shared), decisions(eager.execute(prepared).unwrap()))
        })
        .collect();
    assert_ne!(
        expected[0].0, expected[1].0,
        "changed request must change head outputs"
    );
    assert_eq!(eager.language.graph_stats().captures, 0);
    let mut graph = language_mode(eager, &dir, &library, true);
    assert_eq!(cases[0].inputs[0].token_ids, cases[1].inputs[0].token_ids);
    let mut previous_captures = 0;
    for (visit, case) in [0, 1, 2, 0, 3, 0].into_iter().enumerate() {
        let prepared = &cases[case];
        let before = graph.language.graph_stats();
        let actual = bits(&graph.head_outputs(prepared, true).unwrap());
        assert_eq!(
            actual, expected[case].0,
            "Graph/eager head parity, visit {visit}"
        );
        let captured = graph.language.graph_stats();
        assert!(captured.enabled, "Graph unexpectedly disabled");
        assert!(
            captured.captures > 0,
            "OmniJev windows must capture a Graph"
        );
        assert_eq!(captured.fallbacks, 0, "unexpected capture failure");
        if case == 1 {
            assert_eq!(
                captured.captures, previous_captures,
                "same-layout new pixels must reuse Graphs"
            );
        }
        // Growth can retire the request-prefix Graph midway through the first
        // pass. Stabilize once before asserting warm-cache reuse.
        assert_eq!(bits(&graph.head_outputs(prepared, true).unwrap()), actual);
        let stable = graph.language.graph_stats();
        assert_eq!(bits(&graph.head_outputs(prepared, true).unwrap()), actual);
        let replayed = graph.language.graph_stats();
        assert!(
            replayed.replays > stable.replays,
            "warm OmniJev windows must replay"
        );
        assert_eq!(
            replayed.captures, stable.captures,
            "warm windows must reuse captures"
        );
        let answers = decisions(graph.execute(prepared).unwrap());
        assert_eq!(
            answers, expected[case].1,
            "Graph/eager decisions, visit {visit}"
        );
        if case == 3 {
            assert!(
                replayed.invalidations > before.invalidations,
                "large request must retire scratch Graphs"
            );
        }
        previous_captures = graph.language.graph_stats().captures;
        eprintln!(
            "OMNIJEV_WINDOW_GRAPH {}",
            json!({
                "visit": visit, "case": case, "head_bits": actual, "answers": answers,
                "captures": replayed.captures, "replays": replayed.replays,
                "invalidations": replayed.invalidations, "fallbacks": replayed.fallbacks
            })
        );
    }
    let mut malformed = request(&processor, 320, 200, false, false, false);
    malformed.inputs[0].token_ids[0] = u32::MAX;
    assert!(graph.head_outputs(&malformed, true).is_err());
    assert_eq!(
        bits(&graph.head_outputs(&cases[0], true).unwrap()),
        expected[0].0
    );
    assert_eq!(decisions(graph.execute(&cases[0]).unwrap()), expected[0].1);
    assert!(graph.language.graph_stats().enabled);
    assert_eq!(graph.language.graph_stats().fallbacks, 0);
}
