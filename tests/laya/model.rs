use super::storage_dtype;
#[cfg(unix)]
use super::*;

#[test]
fn resident_precision_matches_cuda_consumers() {
    for (name, expected) in [
        ("encoder.embeddings.tok_embeddings.weight", "f16"),
        ("encoder.embeddings.norm.weight", "f32"),
        ("encoder.layers.1.attn_norm.weight", "f32"),
        ("encoder.layers.0.attn.Wqkv.weight", "bf16"),
        ("head.layers.0.norm1.bias", "f32"),
        ("head.layers.0.self_attn.in_proj_bias", "f32"),
        ("head.layers.0.self_attn.in_proj_weight", "bf16"),
        ("scorer.0.bias", "f32"),
        ("scorer.1.bias", "bf16"),
        ("act_head.2.weight", "bf16"),
        ("temperature", "f32"),
    ] {
        assert_eq!(storage_dtype(name), expected, "{name}");
    }
    let inventory = crate::weights::checkpoint_tensors();
    assert_eq!(inventory.len(), 206);
    assert_eq!(
        inventory
            .iter()
            .filter(|t| storage_dtype(&t.name) == "f16")
            .count(),
        1
    );
}

// Private model regression helpers remain in the repository-level tests tree.
#[cfg(unix)]
#[path = "../cuda/laya/fixture.rs"]
mod fixture;
#[cfg(unix)]
#[path = "model_fixture.rs"]
mod model_fixture;
#[cfg(unix)]
use model_fixture::fixture_model_at;
#[cfg(unix)]
#[path = "model_cache.rs"]
mod cache;
#[cfg(unix)]
#[path = "model_dispatch.rs"]
mod dispatch;
#[cfg(unix)]
#[path = "model_grouped.rs"]
mod grouped;
#[cfg(unix)]
#[path = "original_encode.rs"]
mod original_encode;

#[test]
fn workspace_sizes_match_the_recorded_profile_layout() {
    let structure = gate_structure();
    for (b, l) in [(1usize, 16usize), (1, 512), (16, 512), (4, 128)] {
        let m = b * l;
        let sizes = Workspace::sizes(b, l, &structure).unwrap();
        assert_eq!(
            sizes,
            [
                m * 8,
                b * 4,
                b * 8,
                m * 1024 * 4,
                m * 1024 * 2,
                m * 1024 * 6,
                m * 1024 * 2,
                m * 2624 * 2,
                m * 4096 * 2,
                2048 * 4,
                17 * 4,
                2048 * 1024 * 2,
                2048 * 1024 * 2,
                2048 * 2,
                16 * 1028 * 2,
                16 * 256 * 2,
                16 * 2 * 2,
            ],
            "b={b} l={l}"
        );
    }
}

fn gate_structure() -> omni_modernbert::structure::ModernBertStructure {
    omni_modernbert::gate::laya_0_3_20().structure
}
