//! Real model constructors through a host boundary: no checkpoint download or GPU.
#![cfg(unix)]
use omni_qwen3_5_native::model::Model;
use safetensors::{Dtype, tensor::TensorView};
use std::{path::Path, process::Command};

fn write_tensors(path: &Path, specs: &[(&str, Vec<usize>, Dtype)]) {
    let storage: Vec<Vec<u8>> = specs
        .iter()
        .map(|(_, shape, dtype)| vec![0; shape.iter().product::<usize>() * dtype.bitsize() / 8])
        .collect();
    let views: Vec<_> = specs
        .iter()
        .zip(&storage)
        .map(|((name, shape, dtype), data)| {
            (*name, TensorView::new(*dtype, shape.clone(), data).unwrap())
        })
        .collect();
    std::fs::write(path, safetensors::tensor::serialize(views, None).unwrap()).unwrap();
}

#[test]
fn explicit_graph_mode_and_adapter_models_retire_the_stream_once_after_allocations() {
    let directory = std::env::temp_dir().join(format!("qwen-loading-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let config = serde_json::json!({
        "hidden_size":8,"intermediate_size":16,"rms_norm_eps":1e-6,
        "num_attention_heads":1,"num_key_value_heads":1,"head_dim":256,
        "max_position_embeddings":64,"linear_num_key_heads":1,
        "linear_num_value_heads":1,"linear_key_head_dim":128,"linear_value_head_dim":128,
        "layer_types":["full_attention"],"num_hidden_layers":1,"hidden_act":"silu",
        "linear_conv_kernel_dim":4,"partial_rotary_factor":0.25,"rope_theta":10000.0
    });
    std::fs::write(directory.join("config.json"), config.to_string()).unwrap();
    let bf16 = Dtype::BF16;
    write_tensors(
        &directory.join("model.safetensors"),
        &[
            ("embed_tokens.weight", vec![4, 8], bf16),
            ("norm.weight", vec![8], bf16),
            ("layers.0.input_layernorm.weight", vec![8], bf16),
            ("layers.0.post_attention_layernorm.weight", vec![8], bf16),
            ("layers.0.self_attn.q_proj.weight", vec![512, 8], bf16),
            ("layers.0.self_attn.k_proj.weight", vec![256, 8], bf16),
            ("layers.0.self_attn.v_proj.weight", vec![256, 8], bf16),
            ("layers.0.self_attn.o_proj.weight", vec![8, 256], bf16),
            ("layers.0.self_attn.q_norm.weight", vec![256], bf16),
            ("layers.0.self_attn.k_norm.weight", vec![256], bf16),
            ("layers.0.mlp.gate_proj.weight", vec![16, 8], bf16),
            ("layers.0.mlp.up_proj.weight", vec![16, 8], bf16),
            ("layers.0.mlp.down_proj.weight", vec![8, 16], bf16),
        ],
    );
    let adapter = directory.join("adapter.safetensors");
    write_tensors(
        &adapter,
        &[
            (
                "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight",
                vec![1, 16],
                Dtype::F32,
            ),
            (
                "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_B.weight",
                vec![8, 1],
                Dtype::F32,
            ),
        ],
    );
    let cuda = include_str!("../../src/models/qwen3_5/native/src/cuda.rs");
    let abi = cuda
        .lines()
        .find(|l| l.starts_with("const ABI_VERSION:"))
        .unwrap()
        .split('=')
        .nth(1)
        .unwrap()
        .trim()
        .trim_end_matches(';');
    let mut fake = include_str!("graph_loading_fake.c").to_owned();
    let implemented = [
        "cs1_abi_version",
        "cs1_error_string",
        "cs1_get_device",
        "cs1_set_device",
        "cs1_malloc",
        "cs1_free",
        "cs1_stream_create",
        "cs1_stream_sync",
        "cs1_stream_destroy",
        "cs1_upload",
        "cs1_gemm_create",
        "cs1_gemm_destroy",
    ];
    let api = cuda
        .split("api! {")
        .nth(1)
        .unwrap()
        .split("\n}\n")
        .next()
        .unwrap();
    for line in api.lines().map(str::trim).filter(|l| l.starts_with("cs1_")) {
        let name = line.split('(').next().unwrap();
        if !implemented.contains(&name) {
            fake.push_str(&format!("\nint {name}(void) {{ return 77; }}\n"));
        }
    }
    let source = directory.join("fake.c");
    let library = directory.join(if cfg!(target_os = "macos") {
        "fake.dylib"
    } else {
        "fake.so"
    });
    std::fs::write(&source, fake).unwrap();
    let output = Command::new("cc")
        .args(["-shared", "-fPIC"])
        .arg(format!("-DLOADING_ABI={abi}"))
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
    // SAFETY: the library is retained until all model values have been destroyed.
    let metrics = unsafe { libloading::Library::new(&library).unwrap() };
    let metric =
        |name: &[u8]| unsafe { metrics.get::<unsafe extern "C" fn() -> i32>(name).unwrap()() };
    for graph in [false, true] {
        let model = Model::load_with_graph(&directory, &library, graph).unwrap();
        assert_eq!(model.graph_stats().requested, graph);
        assert_eq!(model.graph_stats().enabled, graph);
        drop(model);
        assert_eq!(metric(b"loading_streams\0"), metric(b"loading_destroyed\0"));
        assert_eq!(metric(b"loading_live\0"), 0);
        assert_eq!(metric(b"loading_gemms\0"), 0);
        assert_eq!(
            metric(b"loading_early\0"),
            0,
            "stream destroyed before model allocations"
        );
    }
    let model = Model::load_with_lora(&directory, &library, &adapter, 2.0).unwrap();
    assert_eq!(
        model.graph_stats().requested,
        std::env::var("CUA_S1_GRAPH").as_deref() == Ok("1")
    );
    drop(model);
    assert_eq!(metric(b"loading_streams\0"), 3);
    assert_eq!(
        metric(b"loading_destroyed\0"),
        3,
        "each owned stream retires exactly once"
    );
    assert_eq!(metric(b"loading_live\0"), 0);
    assert_eq!(metric(b"loading_early\0"), 0);
    // This failure occurs after the base arena and FP32 adapter have been uploaded.
    write_tensors(
        &directory.join("model.safetensors"),
        &[("embed_tokens.weight", vec![4, 8], bf16)],
    );
    assert!(Model::load_with_lora(&directory, &library, &adapter, 2.0).is_err());
    assert_eq!(metric(b"loading_streams\0"), 4);
    assert_eq!(metric(b"loading_destroyed\0"), 4);
    assert_eq!(metric(b"loading_live\0"), 0);
    assert_eq!(metric(b"loading_early\0"), 0);
    std::fs::remove_file(directory.join("model.safetensors")).unwrap();
    assert!(Model::load_with_graph(&directory, &library, true).is_err());
    assert_eq!(metric(b"loading_streams\0"), 5);
    assert_eq!(
        metric(b"loading_destroyed\0"),
        5,
        "failed construction retires its stream"
    );
    std::fs::remove_dir_all(directory).unwrap();
}
