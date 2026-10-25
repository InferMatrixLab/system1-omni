"""CPU checks for the Hopper GDN prefill choice. No CUDA and no FlashInfer."""

from __future__ import annotations

from models.pplx_decider.gdn import (
    FLA_CHUNK,
    FLA_VERSION,
    FLASHINFER,
    linear_head_dims,
    select_gdn_prefill,
    version_base,
)


def test_hopper_128_selects_flashinfer_when_it_imports():
    assert (
        select_gdn_prefill((9, 0), 128, 128, flashinfer_available=True) == FLASHINFER
    )


def test_missing_flashinfer_stays_on_fla_chunk():
    assert select_gdn_prefill((9, 0), 128, 128, flashinfer_available=False) == FLA_CHUNK


def test_non_hopper_stays_on_fla_chunk():
    assert select_gdn_prefill((8, 9), 128, 128, flashinfer_available=True) == FLA_CHUNK
    assert select_gdn_prefill((10, 0), 128, 128, flashinfer_available=True) == FLA_CHUNK


def test_other_head_dims_stay_on_fla_chunk():
    assert select_gdn_prefill((9, 0), 64, 128, flashinfer_available=True) == FLA_CHUNK
    assert select_gdn_prefill((9, 0), None, None, flashinfer_available=True) == FLA_CHUNK


def test_pinned_text_config_shape():
    key, value = linear_head_dims({"text_config": {"linear_key_head_dim": 128, "linear_value_head_dim": 128}})
    assert (key, value) == (128, 128)
    assert linear_head_dims({}) == (None, None)
    assert linear_head_dims({"text_config": {"linear_key_head_dim": True, "linear_value_head_dim": 128}}) == (
        None,
        None,
    )


def test_fla_version_pin():
    assert FLA_VERSION == "0.5.2"
    assert version_base("0.5.2+cu128") == "0.5.2"
