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

/// A two-layer checkpoint with hidden size 128, the smallest whose stacked weights stay
/// contiguous in the arena, and an FP32 adapter on layer 0's down projection.
#[cfg(unix)]
fn write_tiny_checkpoint(dir: &Path) {
    use safetensors::{
        Dtype,
        tensor::{TensorView, serialize},
    };
    let config = serde_json::json!({
        "text_config": {
            "hidden_size": 128, "intermediate_size": 8, "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 256,
            "linear_num_key_heads": 1, "linear_num_value_heads": 1,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4, "hidden_act": "silu", "rms_norm_eps": 1e-6,
            "max_position_embeddings": 4096,
            "rope_parameters": {
                "rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                "mrope_section": [11, 11, 10], "mrope_interleaved": true
            }
        }
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let write = |file: &str, dtype: Dtype, tensors: &[(String, Vec<usize>)]| {
        let size = if dtype == Dtype::F32 { 4 } else { 2 };
        let data: Vec<Vec<u8>> = tensors
            .iter()
            .map(|(_, shape)| vec![0; shape.iter().product::<usize>() * size])
            .collect();
        let views: Vec<_> = tensors
            .iter()
            .zip(&data)
            .map(|((name, shape), bytes)| {
                (
                    name.clone(),
                    TensorView::new(dtype, shape.clone(), bytes).unwrap(),
                )
            })
            .collect();
        std::fs::write(dir.join(file), serialize(views, None).unwrap()).unwrap();
    };
    let layer = |index: usize, mixer: &[(&str, Vec<usize>)]| {
        [
            ("input_layernorm.weight", vec![128]),
            ("post_attention_layernorm.weight", vec![128]),
            ("mlp.gate_proj.weight", vec![8, 128]),
            ("mlp.up_proj.weight", vec![8, 128]),
            ("mlp.down_proj.weight", vec![128, 8]),
        ]
        .into_iter()
        .chain(mixer.iter().cloned())
        .map(|(name, shape)| (format!("layers.{index}.{name}"), shape))
        .collect::<Vec<_>>()
    };
    let weights: Vec<_> = [
        ("embed_tokens.weight".to_owned(), vec![16, 128]),
        ("norm.weight".to_owned(), vec![128]),
    ]
    .into_iter()
    .chain(layer(
        0,
        &[
            ("linear_attn.in_proj_qkv.weight", vec![384, 128]),
            ("linear_attn.in_proj_z.weight", vec![128, 128]),
            ("linear_attn.in_proj_b.weight", vec![1, 128]),
            ("linear_attn.in_proj_a.weight", vec![1, 128]),
            ("linear_attn.conv1d.weight", vec![384, 1, 4]),
            ("linear_attn.A_log", vec![1]),
            ("linear_attn.dt_bias", vec![1]),
            ("linear_attn.norm.weight", vec![128]),
            ("linear_attn.out_proj.weight", vec![128, 128]),
        ],
    ))
    .chain(layer(
        1,
        &[
            ("self_attn.q_proj.weight", vec![512, 128]),
            ("self_attn.k_proj.weight", vec![256, 128]),
            ("self_attn.v_proj.weight", vec![256, 128]),
            ("self_attn.o_proj.weight", vec![128, 256]),
            ("self_attn.q_norm.weight", vec![256]),
            ("self_attn.k_norm.weight", vec![256]),
        ],
    ))
    .map(|(name, shape)| (format!("model.language_model.{name}"), shape))
    .collect();
    write("model.safetensors", Dtype::BF16, &weights);
    let target = "base_model.model.model.language_model.layers.0.mlp.down_proj";
    write(
        "adapter.safetensors",
        Dtype::F32,
        &[
            (format!("{target}.lora_A.weight"), vec![1, 8]),
            (format!("{target}.lora_B.weight"), vec![128, 1]),
        ],
    );
}

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
    let library = dir.join(if cfg!(target_os = "macos") {
        "fake.dylib"
    } else {
        "fake.so"
    });
    let cuda_source = include_str!("../../src/models/qwen3_5/native/src/cuda.rs");
    let reference_source = include_str!("../../src/models/qwen3_5/native/src/cuda/reference.rs");
    let abi = cuda_source
        .lines()
        .find_map(|line| line.strip_prefix("const ABI_VERSION: u32 ="))
        .unwrap()
        .trim()
        .trim_end_matches(';');
    // Every other required and optional symbol counts the call and fails.
    let mut code = include_str!("profile_fake.c").to_owned();
    let required = cuda_source
        .split("api! {")
        .nth(1)
        .unwrap()
        .split("\n}\n")
        .next()
        .unwrap()
        .lines()
        .filter_map(|line| line.trim().strip_prefix("cs1_"));
    let optional = [cuda_source, reference_source]
        .into_iter()
        .flat_map(|source| source.split("b\"cs1_").skip(1));
    let mut names: Vec<_> = required
        .chain(optional)
        .map(|rest| format!("cs1_{}", rest.split(['(', '\\']).next().unwrap()))
        .filter(|name| !code.contains(&format!(" {name}(")))
        .collect();
    names.sort();
    names.dedup();
    for name in names {
        code.push_str(&format!("int {name}(void) {{ ++calls; return 77; }}\n"));
    }
    let source = dir.join("fake.c");
    std::fs::write(&source, code).unwrap();
    let output = std::process::Command::new("cc")
        .args(["-shared", "-fPIC"])
        .arg(format!("-DPROFILE_FAKE_ABI={abi}"))
        .arg(&source)
        .arg("-o")
        .arg(&library)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
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
