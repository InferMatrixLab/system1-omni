//! CPU contract for the Instant 4B GDN `in_proj` pack. No CUDA library and no GPU.
//!
//! After `in_proj`, prompt-bidirectional Instant prefill copies QKV, z, and the
//! contiguous b|a strip out of the fat `gdn_in` row with `cs1_copy2d`. Conv,
//! gates, and gated RMSNorm then use those planes and their tight leading
//! dimensions. `cs1_gdn_prefill` is not part of this pack: it already reads
//! contiguous `lq` / `lk` / `lv`.
//!
//! This is not tip1 (PR #178), which packed attention V (`ldv = Hk * Dh = 1024`,
//! source column 9216, QKV stride 10240) before bidirectional attention. That
//! path was a different tensor and a different leading dimension. Qwen3.5-9B
//! shares both the GDN row width 12352 and that attention V geometry, and it
//! still does not take this pack.

use omni_qwen3_5_native::model::{
    Config, InstantGdnInProjPack, InstantGdnPackShape, instant_gdn_in_proj_pack,
};
use std::path::Path;

/// Tip1 attention V pack on Instant 4B / Qwen3.5-9B. Not a GDN plane.
const TIP1_ATTENTION_V_LDV: usize = 1024;
const TIP1_ATTENTION_V_SOURCE_STRIDE: usize = 10240;
const TIP1_ATTENTION_V_SOURCE_COLUMN: usize = 9216;

fn instant_4b() -> InstantGdnPackShape {
    InstantGdnPackShape {
        hidden: 2560,
        intermediate: 9216,
        layers: 32,
        full_attention_every_fourth: true,
        heads: 16,
        kv_heads: 4,
        head_dim: 256,
        lin_k_heads: 16,
        lin_v_heads: 32,
        lin_k_dim: 128,
        lin_v_dim: 128,
    }
}

fn instant_pack() -> InstantGdnInProjPack {
    instant_gdn_in_proj_pack(instant_4b()).expect("Instant 4B is the pack")
}

fn pack_columns(
    src: &[u32],
    tokens: usize,
    stride: usize,
    column: usize,
    width: usize,
) -> Vec<u32> {
    let mut dst = vec![0; tokens * width];
    for token in 0..tokens {
        let from = token * stride + column;
        let to = token * width;
        dst[to..to + width].copy_from_slice(&src[from..from + width]);
    }
    dst
}

#[test]
fn instant_4b_gdn_in_proj_pack_widths_are_not_tip1_attention_v_pack() {
    let pack = instant_pack();
    assert_eq!(pack.channels, 8192, "GDN QKV channels, not attention ldv");
    assert_eq!(pack.gdn_in, 12352);
    assert_eq!(pack.z_column, 8192);
    assert_eq!(pack.z_width, 4096);
    assert_eq!(pack.b_column, 12288);
    assert_eq!(pack.a_column, 12320);
    assert_eq!(pack.gate_heads, 32);
    assert_eq!(pack.a_column, pack.b_column + pack.gate_heads);
    assert_eq!(pack.b_column, pack.z_column + pack.z_width);
    assert_eq!(pack.gdn_in, pack.a_column + pack.gate_heads);
    assert_eq!(pack.gates_ld(), 64);
    assert_eq!(pack.qkv_row_bytes(), 16384);
    assert_eq!(pack.z_row_bytes(), 8192);
    assert_eq!(pack.ba_row_bytes(), 128);
    assert_eq!(pack.source_pitch_bytes(), 24704);
    assert_eq!(pack.a_offset(0), pack.gate_heads * 2);
    assert_eq!(pack.qkv_offset(3), 3 * pack.qkv_row_bytes());
    assert_eq!(pack.z_offset(3), 3 * pack.z_row_bytes());
    assert_eq!(pack.ba_offset(3), 3 * pack.ba_row_bytes());

    let dead = pack.gdn_in - pack.channels;
    assert_eq!(dead, 4160);
    // 4160/12352 = 33.67% of the fat row sits past the QKV channels.
    assert_eq!(dead * 10_000 / pack.gdn_in, 3367);

    assert_ne!(pack.channels, TIP1_ATTENTION_V_LDV);
    assert_ne!(pack.gdn_in, TIP1_ATTENTION_V_SOURCE_STRIDE);
    assert_ne!(pack.z_column, TIP1_ATTENTION_V_SOURCE_COLUMN);
    assert_ne!(pack.b_column, TIP1_ATTENTION_V_SOURCE_COLUMN);
    assert_ne!(pack.a_column, TIP1_ATTENTION_V_SOURCE_COLUMN);
    assert!(
        (pack.z_column..pack.z_column + pack.z_width).contains(&TIP1_ATTENTION_V_SOURCE_COLUMN),
        "tip1's attention V column sits inside GDN z; it is not a plane boundary"
    );
    assert!(pack.qkv_row_bytes() < pack.source_pitch_bytes());
    assert!(pack.channels < pack.gdn_in);
}

#[test]
fn packed_gdn_planes_match_strided_qkv_z_and_gate_columns() {
    let pack = instant_pack();
    let tokens = 4usize;
    let mut src = vec![0u32; tokens * pack.gdn_in];
    for token in 0..tokens {
        for col in 0..pack.gdn_in {
            src[token * pack.gdn_in + col] = (token * pack.gdn_in + col) as u32;
        }
    }
    let qkv = pack_columns(&src, tokens, pack.gdn_in, 0, pack.channels);
    let z = pack_columns(&src, tokens, pack.gdn_in, pack.z_column, pack.z_width);
    let ba = pack_columns(
        &src,
        tokens,
        pack.gdn_in,
        pack.b_column,
        pack.gate_heads * 2,
    );
    assert_eq!(qkv.len(), tokens * pack.channels);
    assert_ne!(qkv.len(), tokens * pack.gdn_in);
    assert_eq!(qkv[0], 0);
    assert_ne!(qkv[0], src[pack.z_column], "z must not occupy packed QKV");
    assert_ne!(
        qkv[0], src[TIP1_ATTENTION_V_SOURCE_COLUMN],
        "tip1 attention V is not the GDN QKV plane"
    );
    for token in 0..tokens {
        for col in 0..pack.channels {
            assert_eq!(
                qkv[token * pack.channels + col],
                src[token * pack.gdn_in + col]
            );
        }
        for col in 0..pack.z_width {
            assert_eq!(
                z[token * pack.z_width + col],
                src[token * pack.gdn_in + pack.z_column + col]
            );
        }
        for head in 0..pack.gate_heads {
            assert_eq!(
                ba[token * pack.gates_ld() + head],
                src[token * pack.gdn_in + pack.b_column + head]
            );
            assert_eq!(
                ba[token * pack.gates_ld() + pack.gate_heads + head],
                src[token * pack.gdn_in + pack.a_column + head]
            );
        }
        assert_eq!(
            ba[pack.a_offset(token) / 2],
            src[token * pack.gdn_in + pack.a_column]
        );
    }
}

#[test]
fn non_instant_and_tip1_attention_shapes_do_not_take_the_gdn_in_proj_pack_path() {
    let mut nine_b = instant_4b();
    nine_b.hidden = 4096;
    nine_b.intermediate = 12288;
    assert!(
        instant_gdn_in_proj_pack(nine_b).is_none(),
        "Qwen3.5-9B has the same GDN row and the same tip1 attention V geometry"
    );

    let mut twenty_seven_b = instant_4b();
    twenty_seven_b.hidden = 5120;
    twenty_seven_b.intermediate = 17408;
    twenty_seven_b.layers = 64;
    twenty_seven_b.heads = 24;
    twenty_seven_b.lin_v_heads = 48;
    assert!(instant_gdn_in_proj_pack(twenty_seven_b).is_none());

    let mut broken_pattern = instant_4b();
    broken_pattern.full_attention_every_fourth = false;
    assert!(instant_gdn_in_proj_pack(broken_pattern).is_none());

    // Tip1's attention counts without Instant's hidden size are not this path.
    let mut attention_only = instant_4b();
    attention_only.hidden = 2048;
    attention_only.intermediate = 6144;
    assert_eq!(
        (
            TIP1_ATTENTION_V_LDV,
            TIP1_ATTENTION_V_SOURCE_STRIDE,
            TIP1_ATTENTION_V_SOURCE_COLUMN
        ),
        (1024, 10240, 9216)
    );
    assert!(instant_gdn_in_proj_pack(attention_only).is_none());

    let cfg = Config::load(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../../tests/qwen3_5/data"
    )))
    .unwrap();
    assert!(
        cfg.instant_gdn_in_proj_pack().is_none(),
        "the 27B fixture must stay on the fat gdn_in stride"
    );
}
