import json
import sys
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


def _bound_pair():
    return type(
        "M",
        (),
        {"chunk_gated_delta_rule": object(), "causal_conv1d_fn": object()},
    )()


def test_linear_attention_name_follows_the_bound_callables():
    from models.clef.engine import (
        bind_selected_prefill,
        bound_linear_attention,
        flashinfer_chunk_gated_delta_rule,
        install_flashinfer_gdn_prefill,
        require_bound_linear_attention,
        resolve_gdn_prefill,
        select_gdn_prefill,
    )

    assert bound_linear_attention(None) == "unloaded"
    assert bound_linear_attention(object()) == "torch_chunk"
    fla_only = type("M", (), {"chunk_gated_delta_rule": object()})()
    conv_only = type("M", (), {"causal_conv1d_fn": object()})()
    assert bound_linear_attention(fla_only) == "fla"
    assert bound_linear_attention(conv_only) == "causal_conv1d"
    fla_ready = _bound_pair()
    assert bound_linear_attention(fla_ready) == "fla+causal_conv1d"
    assert require_bound_linear_attention(fla_ready) == "fla+causal_conv1d"
    install_flashinfer_gdn_prefill(fla_ready)
    assert fla_ready.chunk_gated_delta_rule is flashinfer_chunk_gated_delta_rule
    assert fla_ready.clef_gdn_prefill == "flashinfer"
    assert bound_linear_attention(fla_ready) == "flashinfer+fla+causal_conv1d"
    assert require_bound_linear_attention(fla_ready) == "flashinfer+fla+causal_conv1d"
    for modeling in (None, object(), fla_only, conv_only):
        with pytest.raises(RuntimeError, match="not ready"):
            require_bound_linear_attention(modeling)
    with pytest.raises(RuntimeError, match="causal_conv1d_fn"):
        install_flashinfer_gdn_prefill(fla_only)
    with pytest.raises(RuntimeError, match="chunk_gated_delta_rule"):
        install_flashinfer_gdn_prefill(conv_only)
    assert select_gdn_prefill((8, 9)) == "fla"
    assert select_gdn_prefill((9, 0)) == "fla"
    assert select_gdn_prefill((10, 0)) == "fla"
    assert select_gdn_prefill((12, 0)) == "fla"
    assert select_gdn_prefill(None) == "fla"
    assert select_gdn_prefill((9, 0), "flashinfer") == "flashinfer"
    assert select_gdn_prefill((10, 0), "flashinfer") == "flashinfer"
    assert select_gdn_prefill((12, 0), "flashinfer") == "flashinfer"
    assert select_gdn_prefill((8, 9), "flashinfer") == "fla"
    assert select_gdn_prefill(None, "flashinfer") == "fla"
    assert resolve_gdn_prefill((9, 0), "fla", True) == "fla"
    assert resolve_gdn_prefill((9, 0), "flashinfer", True) == "flashinfer"
    with pytest.raises(RuntimeError, match="reports capability 8.9"):
        resolve_gdn_prefill((8, 9), "flashinfer", True)
    with pytest.raises(RuntimeError, match="did not import"):
        resolve_gdn_prefill((9, 0), "flashinfer", False)
    reported_89 = _bound_pair()
    original_chunk = reported_89.chunk_gated_delta_rule
    bind_selected_prefill(reported_89, select_gdn_prefill((8, 9), "flashinfer"), True)
    assert reported_89.chunk_gated_delta_rule is original_chunk
    assert require_bound_linear_attention(reported_89) == "fla+causal_conv1d"
    default_90 = _bound_pair()
    bind_selected_prefill(default_90, select_gdn_prefill((9, 0)), True)
    assert require_bound_linear_attention(default_90) == "fla+causal_conv1d"
    reported_90 = _bound_pair()
    bind_selected_prefill(reported_90, select_gdn_prefill((9, 0), "flashinfer"), True)
    assert require_bound_linear_attention(reported_90) == "flashinfer+fla+causal_conv1d"
    unloaded_fi = _bound_pair()
    bind_selected_prefill(unloaded_fi, select_gdn_prefill((9, 0), "flashinfer"), False)
    assert require_bound_linear_attention(unloaded_fi) == "fla+causal_conv1d"
    with pytest.raises(RuntimeError, match="one sequence"):
        flashinfer_chunk_gated_delta_rule(object(), object(), object())
    with pytest.raises(RuntimeError, match="recurrent state"):
        flashinfer_chunk_gated_delta_rule(object(), object(), object(), output_final_state=True)


def test_linear_attention_pins_and_startup_fail_closed(monkeypatch):
    import importlib.metadata

    from models.clef import engine
    from models.clef.engine import (
        FI_VERSION,
        FLA_VERSION,
        check_distribution_version,
        require_fast_linear_attention,
        version_tuple,
    )

    assert version_tuple("1.5.0.post8") == (1, 5, 0)
    assert version_tuple("0.5.2") == (0, 5, 2)
    assert version_tuple("0.6.16.post3") == (0, 6, 16)
    check_distribution_version("0.5.2", exact=FLA_VERSION)
    check_distribution_version(FI_VERSION, exact=FI_VERSION)
    check_distribution_version("1.5.0.post8", minimum=(1, 5))
    check_distribution_version("1.6", minimum=(1, 5))
    with pytest.raises(RuntimeError, match="required 0.5.2"):
        check_distribution_version("0.5.1", exact=FLA_VERSION)
    with pytest.raises(RuntimeError, match="required 0.6.16.post3"):
        check_distribution_version("0.6.16", exact=FI_VERSION)
    with pytest.raises(RuntimeError, match="required >=1.5"):
        check_distribution_version("1.4.9", minimum=(1, 5))

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.delitem(sys.modules, engine.MODELING, raising=False)
    monkeypatch.setattr(engine.importlib.metadata, "version", missing)
    with pytest.raises(RuntimeError, match="not installed"):
        require_fast_linear_attention()
