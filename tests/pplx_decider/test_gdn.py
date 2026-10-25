"""CPU checks for the FLA chunk pin. No CUDA."""

from __future__ import annotations

from models.pplx_decider.gdn import FLA_VERSION, LINEAR_ATTENTION, version_base


def test_linear_attention_name_is_the_fla_chunk():
    assert LINEAR_ATTENTION == "fla-chunk"
    assert FLA_VERSION == "0.5.2"
    assert version_base("0.5.2+cu128") == "0.5.2"
