import json
from pathlib import Path

import pytest

from models.clef.engine import (
    DEFAULT_VARIANT,
    LOADER_SHA256,
    TOKENIZER_SHA256,
    VARIANTS,
    require_token_agreement,
    sha256_file,
    untruncated_budget,
    verify_snapshot,
)
from models.clef.protocol import RequestError


def test_pins_name_both_public_revisions_and_the_shared_loader():
    flash = VARIANTS["clef-flash"]
    clef = VARIANTS["clef"]
    assert DEFAULT_VARIANT == "clef-flash"
    assert flash.repo_id == "Cloudflare/clef-flash"
    assert flash.revision == "f0b7d3cd65296abfe459e6c228b618f5d8b26d2f"
    assert flash.base_model == "Qwen/Qwen3.5-9B"
    assert flash.hidden_size == 4096
    assert flash.num_hidden_layers == 32
    assert clef.repo_id == "Cloudflare/clef"
    assert clef.revision == "a30162b99d68a481d8bd75d38df069a0cf83df40"
    assert clef.base_model == "Qwen/Qwen3.8-27B"
    assert clef.hidden_size == 5120
    assert clef.num_hidden_layers == 64
    assert LOADER_SHA256 == "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"
    assert TOKENIZER_SHA256 == "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
    assert flash.config_sha256 != clef.config_sha256
    assert flash.joint_head_config_sha256 != clef.joint_head_config_sha256


def _snapshot(tmp_path: Path, *, extra: str | None = None) -> tuple[Path, object]:
    shared = {"notes.txt": b"loader-bytes"}
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {"hidden_size": 4096, "num_hidden_layers": 32},
    }
    head = {"hidden_size": 4096, "width": 1024}
    (tmp_path / "notes.txt").write_bytes(shared["notes.txt"])
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "joint_head_config.json").write_text(json.dumps(head), encoding="utf-8")
    if extra:
        (tmp_path / extra).write_text("print('no')\n", encoding="utf-8")
    variant = VARIANTS["clef-flash"]
    forged = type(variant)(
        **{
            **variant.__dict__,
            "config_sha256": sha256_file(tmp_path / "config.json"),
            "joint_head_config_sha256": sha256_file(tmp_path / "joint_head_config.json"),
        }
    )
    shared_hashes = {name: sha256_file(tmp_path / name) for name in shared}
    return forged, shared_hashes


def test_verify_snapshot_accepts_a_matching_tree_and_rejects_drift(tmp_path):
    variant, shared = _snapshot(tmp_path)
    verify_snapshot(tmp_path, variant, shared)
    (tmp_path / "notes.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="notes.txt"):
        verify_snapshot(tmp_path, variant, shared)


def test_verify_snapshot_rejects_extra_python(tmp_path):
    variant, shared = _snapshot(tmp_path, extra="extra.py")
    with pytest.raises(ValueError, match="unexpected Python file"):
        verify_snapshot(tmp_path, variant, shared)


def test_budget_refuses_overflow_and_disagreement():
    untruncated_budget(10, 4, 10)
    with pytest.raises(RequestError, match="was not truncated"):
        untruncated_budget(11, 4, 10)
    with pytest.raises(RequestError, match="schema requires"):
        untruncated_budget(11, 0, 10)
    with pytest.raises(RuntimeError, match="token accounting mismatch"):
        untruncated_budget(3, 4, 10)


def test_response_token_count_must_match_admission():
    response = {"usage": {"input_tokens": 12, "output_tokens": 0}}
    assert require_token_agreement(12, response) is response
    with pytest.raises(RuntimeError, match="encoder returned 9"):
        require_token_agreement(12, {"usage": {"input_tokens": 9}})
