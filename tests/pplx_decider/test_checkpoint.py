"""Weight-free checkpoint pin checks. No model load."""

from __future__ import annotations

import hashlib
import json

import pytest

from models.pplx_decider.engine import verify_checkpoint


def test_missing_directory(tmp_path):
    with pytest.raises(ValueError, match="checkpoint directory does not exist"):
        verify_checkpoint(tmp_path / "missing")


def test_missing_pinned_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "models.pplx_decider.engine.PINNED_SHA256",
        {"decision_config.json": "ab"},
    )
    with pytest.raises(ValueError, match="pinned checkpoint files differ: decision_config.json"):
        verify_checkpoint(tmp_path)


def test_hash_mismatch(tmp_path, monkeypatch):
    (tmp_path / "decision_config.json").write_bytes(b"{}")
    monkeypatch.setattr(
        "models.pplx_decider.engine.PINNED_SHA256",
        {"decision_config.json": "0" * 64},
    )
    with pytest.raises(ValueError, match="pinned checkpoint files differ: decision_config.json"):
        verify_checkpoint(tmp_path)


def test_decision_config_field_mismatch(tmp_path, monkeypatch):
    raw = json.dumps({"format_version": 2}).encode()
    (tmp_path / "decision_config.json").write_bytes(raw)
    monkeypatch.setattr(
        "models.pplx_decider.engine.PINNED_SHA256",
        {"decision_config.json": hashlib.sha256(raw).hexdigest()},
    )
    with pytest.raises(
        ValueError, match="decision_config.json field format_version differs from the pin"
    ):
        verify_checkpoint(tmp_path)
