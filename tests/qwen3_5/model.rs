use super::*;

fn seq(n: usize, base: u32) -> Vec<u32> {
    (0..n as u32).map(|i| base + i).collect()
}

#[test]
fn runs_keep_every_prompt_and_end_prefixes_on_chunks() {
    for p in [0usize, 1, 63, 64, 65, 130] {
        let prefix = seq(p, 0);
        let groups = [
            (seq(0, 1000), vec![seq(1, 2000), seq(70, 3000)]),
            (seq(1, 4000), vec![seq(2, 5000)]),
            (seq(64, 6000), vec![seq(5, 7000), seq(1, 8000)]),
        ];
        let prompts = SharedPrompts {
            prefix: &prefix,
            groups: groups
                .iter()
                .map(|(prefix, branches)| PromptGroup {
                    prefix,
                    branches: branches.iter().map(Vec::as_slice).collect(),
                })
                .collect(),
        };
        let r = runs(&prompts);
        let chunk = |n: usize| n - n % GDN_CHUNK;
        assert_eq!(r.prefix.len(), chunk(p), "p = {p}");
        for ((group, branches), run) in groups.iter().zip(&r.groups) {
            assert_eq!(r.prefix.len() + run.prefix.len(), chunk(p + group.len()));
            assert_eq!(branches.len(), run.branches.len());
            for (branch, run_branch) in branches.iter().zip(&run.branches) {
                assert!(!run_branch.is_empty());
                assert_eq!(
                    [&r.prefix[..], &run.prefix, run_branch].concat(),
                    [&prefix[..], group, branch].concat(),
                    "p = {p}"
                );
            }
        }
    }
}

#[test]
fn token_logprob_is_a_log_softmax_over_the_vocabulary_only() {
    let vocab = 1001;
    // logits in [-20, 20) and, past the vocabulary, padding that must not count
    let mut x = 7u64;
    let mut values: Vec<f32> = (0..vocab)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 40) as f32 / (1u64 << 24) as f32) * 40.0 - 20.0
        })
        .collect();
    values.extend([1e4f32; 7]);
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
        .collect();
    let rounded: Vec<f64> = values[..vocab]
        .iter()
        .map(|&v| half::bf16::from_f32(v).to_f64())
        .collect();
    let lse = rounded.iter().map(|v| v.exp()).sum::<f64>().ln();
    for token in [0u32, 1, 500, 1000] {
        let want = rounded[token as usize] - lse;
        let got = token_logprob(&bytes, vocab, token) as f64;
        assert!((got - want).abs() <= 1e-5, "token {token}: {got} vs {want}");
    }
    let total: f64 = (0..vocab as u32)
        .map(|t| (token_logprob(&bytes, vocab, t) as f64).exp())
        .sum();
    assert!((total - 1.0).abs() <= 1e-5, "{total}");
}

#[cfg(unix)]
#[path = "cpu_support.rs"]
mod cpu_support;
#[cfg(unix)]
use cpu_support::write_tiny_checkpoint;

#[cfg(unix)]
#[test]
fn cpu_fixed_and_shared_refuse_other_profiles_before_device_work() {
    // The library loads once per process, so this runs in a child process.
    if std::env::var_os("QWEN_TEST_PROFILE_CHILD").is_none() {
        let name = format!(
            "{}::cpu_fixed_and_shared_refuse_other_profiles_before_device_work",
            module_path!().split_once("::").unwrap().1
        );
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", &name, "--nocapture"])
            .env("QWEN_TEST_PROFILE_CHILD", "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let dir = std::env::temp_dir().join(format!("qwen-profile-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    write_tiny_checkpoint(&dir);
    let library = cpu_support::build_fake_library(&dir, "", &[]);
    let mut model =
        Model::load_with_lora(&dir, &library, &dir.join("adapter.safetensors"), 2.).unwrap();
    // SAFETY: the test library defines this counter and stays loaded in the process.
    let metrics = unsafe { libloading::Library::new(&library).unwrap() };
    let calls = unsafe {
        *metrics
            .get::<unsafe extern "C" fn() -> i32>(b"profile_fake_calls\0")
            .unwrap()
    };
    let prompts = SharedPrompts {
        prefix: &[1, 2],
        groups: vec![PromptGroup {
            prefix: &[3],
            branches: vec![&[4], &[5, 6]],
        }],
    };
    let positions = [0, 1, 2];
    let input = MultimodalInput {
        token_ids: &[1, 2, 3],
        image_token_indices: &[],
        image_embeddings: &[],
        position_ids: [&positions; 3],
    };
    let refuse = |model: &mut Model, profile: &str| {
        let before = unsafe { calls() };
        for error in [
            model.forward_fixed(&[1, 2, 3]).unwrap_err(),
            model.forward_shared(&prompts).unwrap_err(),
            model
                .forward_multimodal_readout(&input, &[2], &[(2, 4)])
                .map(drop)
                .unwrap_err(),
        ] {
            assert!(
                error.to_string().contains("only merged weights"),
                "{profile}: {error:#}"
            );
        }
        assert_eq!(unsafe { calls() }, before, "{profile}: CUDA calls");
        assert!(
            model.scratch.is_none()
                && model.vocab_buffers.is_none()
                && model.fixed_gemm.is_null()
                && model.shared.iter().all(Option::is_none),
            "{profile}: buffers"
        );
    };
    refuse(&mut model, "LoRA");
    // Each of the other profiles alone is refused too; without any, the check passes.
    let adapter = model.lora.take();
    model.ensure_fixed_profile().unwrap();
    model.fp32_gdn = Some(cuda::api().fp32_gdn().unwrap());
    refuse(&mut model, "FP32 GDN");
    model.fp32_gdn = None;
    model.reference = Some(cuda::api().language_reference().unwrap());
    refuse(&mut model, "reference numerics");
    model.reference = None;
    model.lora = adapter;
    drop(model);
    let _ = std::fs::remove_dir_all(&dir);
}
