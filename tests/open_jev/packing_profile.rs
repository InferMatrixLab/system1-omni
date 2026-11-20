//! Exercise the real executor startup with host allocations, without a GPU.
use super::*;
use crate::contract::CHECKPOINTS;

#[path = "../qwen3_5/cpu_support.rs"]
mod cpu_support;

#[tokio::test(flavor = "current_thread")]
async fn unvalidated_packing_uses_serial_eager_execution() {
    let Ok(case) = std::env::var("JEV_PROFILE_TEST_CHILD") else {
        let name = format!(
            "{}::unvalidated_packing_uses_serial_eager_execution",
            module_path!().split_once("::").unwrap().1
        );
        // Backend loading is process-wide; each capability gets a fresh process.
        for case in [
            "unsupported",
            "missing",
            "supported",
            "error",
            "rollback",
            "27b",
        ] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", &name, "--nocapture"])
                .env("JEV_PROFILE_TEST_CHILD", case)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{case}:\n{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        return;
    };
    let dir = std::env::temp_dir().join(format!("jev-packing-profile-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    cpu_support::write_tiny_checkpoint(&dir);
    let result = match case.as_str() {
        "supported" => 1,
        "error" => -1,
        _ => 0,
    };
    let symbol = "cs1_gemm_enable_equivalent_tiles";
    let extra = if case == "missing" {
        String::new()
    } else {
        format!("int {symbol}(void* gemm) {{ (void)gemm; return {result}; }}\n")
    };
    let library = cpu_support::build_fake_library(&dir, &extra, &[symbol]);
    let checkpoint = &CHECKPOINTS[usize::from(case != "27b")];
    let head = || DecisionHead {
        weights: vec![0.; 128],
        bias: 0.,
    };
    let executor = Executor::load(
        &dir,
        &library,
        head(),
        checkpoint,
        PrefixMode::Off,
        case != "rollback",
        case != "rollback",
    )
    .await;
    if case == "error" {
        assert!(
            executor
                .err()
                .unwrap()
                .to_string()
                .contains("projection tile setup failed")
        );
    } else {
        let mut executor = executor.unwrap();
        let packed = matches!(case.as_str(), "supported" | "27b");
        assert_eq!(executor.pack, packed, "effective packing");
        assert_eq!(executor.model.lock().unwrap().graph_stats().enabled, packed);
        let scheduler = executor.scheduler(checkpoint).unwrap();
        assert_eq!(
            matches!(scheduler.0, SchedulerKind::Batched(_)),
            case == "supported"
        );
        assert_eq!(executor.set_exact_prefix(true).is_ok(), packed);
        assert_eq!(
            executor.share_packed_prefix, packed,
            "exact prefix must not bypass fallback"
        );
        assert_eq!(executor.share_packed_inputs, case == "supported");
        if !packed && case != "rollback" {
            for prefix in [PrefixMode::Fixed, PrefixMode::Shared, PrefixMode::Auto] {
                let error = Executor::load(&dir, &library, head(), checkpoint, prefix, false, true)
                    .await
                    .err()
                    .expect("unsupported prefix mode must be rejected");
                assert!(
                    error.to_string().contains("validated packing profile"),
                    "{error:#}"
                );
            }
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}
