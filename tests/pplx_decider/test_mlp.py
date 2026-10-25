"""CPU checks for the rejected fused gate-up switch. No weights."""

from __future__ import annotations

import pytest

from models.pplx_decider.mlp import mlp_preference


def test_mlp_stays_on_the_checkpoint_projections():
    assert mlp_preference("cuda", None) == "default"
    assert mlp_preference("cuda", "default") == "default"
    assert mlp_preference("cpu", None) == "default"
    with pytest.raises(RuntimeError, match="not selectable"):
        mlp_preference("cuda", "fused-gate-up")
    with pytest.raises(RuntimeError, match="unset or default"):
        mlp_preference("cpu", "compile")
