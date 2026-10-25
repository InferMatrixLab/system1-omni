//! CPU contract for the Instant V pack. No CUDA library and no GPU.
//!
//! Prompt-bidirectional attention copies V out of the QKV row into contiguous
//! `[T, Hk, Dh]` scratch. The device copy is `cs1_copy2d`: `tokens` rows,
//! width = destination pitch = one packed bfloat16 row, source pitch = the
//! QKV row. Each sequence then starts at `packed_offset(token)`.

use omni_qwen3_5_native::model::{PromptBidirectionalVLayout, prompt_bidirectional_v_layout};

fn pack_rows(src: &[u32], layout: PromptBidirectionalVLayout, tokens: usize) -> Vec<u32> {
    assert_eq!(layout.source_column + layout.ldv, layout.source_stride);
    assert!(layout.row_bytes() <= layout.source_pitch_bytes());
    let mut dst = vec![0; tokens * layout.ldv];
    for token in 0..tokens {
        let from = token * layout.source_stride + layout.source_column;
        let to = token * layout.ldv;
        dst[to..to + layout.ldv].copy_from_slice(&src[from..from + layout.ldv]);
    }
    dst
}

#[test]
fn instant_4b_v_ldv_is_hk_dh_not_the_qkv_stride() {
    let (heads, kv, dh) = (16usize, 4usize, 256usize);
    let layout = prompt_bidirectional_v_layout(heads, kv, dh);
    assert_eq!(layout.ldv, kv * dh);
    assert_eq!(layout.ldv, 1024);
    assert_eq!(layout.source_stride, heads * dh * 2 + 2 * kv * dh);
    assert_eq!(layout.source_stride, 10240);
    assert_eq!(layout.source_column, heads * dh * 2 + kv * dh);
    assert_eq!(layout.source_column, 9216);
    assert_eq!(layout.source_column + layout.ldv, layout.source_stride);
    assert_eq!(layout.row_bytes(), 2048);
    assert_eq!(layout.source_pitch_bytes(), 20480);
    assert_eq!(layout.ldv % 8, 0, "kernel requires ldv % 8 == 0");
    assert!(layout.ldv < layout.source_stride);
    assert_ne!(layout.ldv, layout.source_stride);
}

#[test]
fn packed_rows_match_the_strided_v_column_and_skip_q_and_k() {
    let layout = prompt_bidirectional_v_layout(16, 4, 256);
    let tokens = 5usize;
    let mut src = vec![0u32; tokens * layout.source_stride];
    for token in 0..tokens {
        for col in 0..layout.source_stride {
            src[token * layout.source_stride + col] = (token * layout.source_stride + col) as u32;
        }
    }
    let packed = pack_rows(&src, layout, tokens);
    assert_eq!(packed.len(), tokens * layout.ldv);
    assert_eq!(packed[0], layout.source_column as u32);
    assert_ne!(packed[0], src[0], "Q must not occupy packed V");
    for token in 0..tokens {
        for i in 0..layout.ldv {
            assert_eq!(
                packed[token * layout.ldv + i],
                (token * layout.source_stride + layout.source_column + i) as u32
            );
        }
        assert_eq!(
            layout.packed_offset(token) / 2,
            token * layout.ldv,
            "sequence base is a token count times the packed row"
        );
    }
    let seq = 3usize;
    assert_eq!(layout.packed_offset(seq), seq * layout.row_bytes());
    assert_eq!(
        packed[seq * layout.ldv],
        src[seq * layout.source_stride + layout.source_column]
    );
}

#[test]
fn other_head_counts_keep_ldv_at_hk_dh() {
    let layout = prompt_bidirectional_v_layout(24, 4, 256);
    assert_eq!(layout.ldv, 1024);
    assert_eq!(layout.source_stride, 14336);
    assert_eq!(layout.source_column + layout.ldv, layout.source_stride);
    assert_eq!(layout.row_bytes(), layout.ldv * 2);
}
