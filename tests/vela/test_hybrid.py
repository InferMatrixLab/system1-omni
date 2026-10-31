"""Tests for the Vela 2.0 0.8B hybrid member: gate, worker dispatch, response finishing.

Gate tests mirror test_engine_manifest: temp directories with an
injectable pin table and recomputed hashes — no weights, no torch, no
network. Worker tests drive FakeVela2Hybrid (from test_worker's fake,
plus the hybrid response surface) through decide/detect_member/
models_entry: no weights, no torch, no GPU.

    PYTHONPATH=src python -m pytest tests/vela -q
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from frontend.vela import decide, detect_member, models_entry
from models.vela.engine import Vela2Executor, verify_files
from models.vela.hybrid import (
    EXPECTED_ARCHITECTURE,
    EXPECTED_MAX_LENGTH,
    EXPECTED_MODEL_NAME,
    EXPECTED_MODEL_TYPE,
    EXPECTED_WEIGHTS_BYTES,
    HYBRID_REPO,
    HYBRID_REVISION,
    PINNED_FILES_SHA256,
    WEIGHTS_NAME,
    Vela2HybridExecutor,
    verify_config,
    verify_package,
    verify_weights,
)
from models.vela.postprocess import finish_response

from .test_protocol import body
from .test_worker import FakeVela2, five_kind_body

CONFIG = {
    "model_type": EXPECTED_MODEL_TYPE,
    "architectures": [EXPECTED_ARCHITECTURE],
    "max_length": EXPECTED_MAX_LENGTH,
    "model_name": EXPECTED_MODEL_NAME,
}


def table_of(files: dict[str, bytes]) -> dict[str, str]:
    return {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}


def small_files():
    return {
        "vela2_serve.py": b"upstream server stub\n",
        "vela2_inference.py": b"upstream engine stub\n",
        "modeling_vela2.py": b"upstream model stub\n",
        "configuration_vela2.py": b"upstream config stub\n",
        "calibration.json": b'{"temperature": {}}',
        "tokenizer.json": b'{"version": "stub"}',
        "tokenizer_config.json": b'{"model_type": "stub"}',
        "model.safetensors.index.json": b'{"metadata": {"total_size": 1}}',
        "broad_head.safetensors": b"second span head stub\n",
    }


def build_package(root: Path, files: dict[str, bytes], config: dict | None = None) -> dict[str, str]:
    for name, content in files.items():
        (root / name).write_bytes(content)
    (root / "config.json").write_text(
        json.dumps(config if config is not None else CONFIG), encoding="utf-8"
    )
    with (root / WEIGHTS_NAME).open("wb") as stream:
        stream.truncate(EXPECTED_WEIGHTS_BYTES)
    files = dict(files)
    files["config.json"] = (root / "config.json").read_bytes()
    return table_of(files)


def hybrid_body():
    value = five_kind_body()
    value["model"] = EXPECTED_MODEL_NAME
    return value


class FakeVela2Hybrid(FakeVela2):
    """The pinned hybrid runtime's contract, driven without its code."""

    is_hybrid = True

    def __init__(self, model_name=EXPECTED_MODEL_NAME, too_long=False):
        super().__init__(model_name, too_long)
        self.max_length = 16384

        class _Engine:
            config = {
                "model_name": EXPECTED_MODEL_NAME,
                "description": (
                    "Vela 2.0 0.8B: SystemOne Choice / Noul / Score "
                    "with span and set extensions, 16,384-token input"
                ),
                "release_date": "",
            }

        self.engine = _Engine()

    def system_one(self, body):
        response = super().system_one(body)
        if "spans" in response:
            # Upstream emits span_heads together with spans, keyed by the
            # span question ids; router stands in for both heads here.
            response["span_heads"] = {qid: "router" for qid in response["spans"]}
        return response


# --------------------------------------------------------------------- gate


def test_the_checked_in_hybrid_pin_table_shape_is_locked():
    assert HYBRID_REPO == "vllm-sr/Vela-2.0-0.8B"
    assert HYBRID_REVISION == "326b01d81f61fff7fd4ae570e54b87a49119e490"
    assert set(PINNED_FILES_SHA256) == {
        "vela2_serve.py", "vela2_inference.py", "modeling_vela2.py",
        "configuration_vela2.py", "config.json", "calibration.json", "tokenizer.json",
        "tokenizer_config.json", "model.safetensors.index.json", "broad_head.safetensors",
    }
    for digest in PINNED_FILES_SHA256.values():
        assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_verification_accepts_a_package_matching_its_table(tmp_path):
    table = build_package(tmp_path, small_files())
    assert verify_files(tmp_path, table) == sorted(table)
    verify_config(tmp_path)
    verify_weights(tmp_path)
    verify_package(tmp_path, table)


def test_a_unified_package_is_refused_with_its_own_name(tmp_path):
    build_package(
        tmp_path, small_files(),
        config={**CONFIG, "model_type": "vela2-unified"},
    )
    with pytest.raises(ValueError, match="vela2-unified.*only serves the vela2-decoder member"):
        verify_config(tmp_path)


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"architectures": ["OtherModel"]}, "architectures"),
        ({"max_length": 8192}, "max_length"),
        ({"model_name": "vllm-sr/Vela-2.0-4B"}, "needs its own pinned executor"),
    ],
)
def test_foreign_config_structures_are_refused(tmp_path, override, fragment):
    build_package(tmp_path, small_files(), config={**CONFIG, **override})
    with pytest.raises(ValueError, match=fragment):
        verify_config(tmp_path)


def test_tampered_files_are_rejected(tmp_path):
    table = build_package(tmp_path, small_files())
    victim = tmp_path / "broad_head.safetensors"
    victim.write_bytes(victim.read_bytes() + b"tampered\n")
    # The shared gate must name the hybrid's pin, not the 0.3B's.
    with pytest.raises(
        ValueError, match=f"broad_head.safetensors: SHA-256 differs from the pin for {HYBRID_REVISION}"
    ):
        verify_files(tmp_path, table, HYBRID_REVISION)


def test_a_missing_broad_head_is_rejected(tmp_path):
    table = build_package(tmp_path, small_files())
    (tmp_path / "broad_head.safetensors").unlink()
    with pytest.raises(ValueError, match="missing pinned file broad_head.safetensors"):
        verify_files(tmp_path, table)


def test_a_later_self_consistent_snapshot_is_refused_by_the_checked_in_pin(tmp_path):
    build_package(tmp_path, small_files())
    with pytest.raises(ValueError, match="SHA-256 differs from the pin"):
        verify_files(tmp_path, PINNED_FILES_SHA256)


def test_weights_are_checked_by_exact_pinned_size(tmp_path):
    build_package(tmp_path, small_files())
    verify_weights(tmp_path)

    with (tmp_path / WEIGHTS_NAME).open("wb") as stream:
        stream.truncate(EXPECTED_WEIGHTS_BYTES - 1)
    with pytest.raises(ValueError, match="is 2022189487 bytes"):
        verify_weights(tmp_path)

    (tmp_path / WEIGHTS_NAME).unlink()
    with pytest.raises(ValueError, match=f"missing {WEIGHTS_NAME}"):
        verify_weights(tmp_path)


# ------------------------------------------------------------- worker wiring


def test_detect_member_routes_each_declared_member(tmp_path):
    unified = tmp_path / "unified"
    unified.mkdir()
    (unified / "config.json").write_text(json.dumps({"model_type": "vela2-unified"}))
    decoder = tmp_path / "decoder"
    decoder.mkdir()
    (decoder / "config.json").write_text(json.dumps({"model_type": "vela2-decoder"}))

    cls, name = detect_member(unified)
    assert cls is Vela2Executor and name == "vllm-sr/Vela-2.0-0.3B"
    cls, name = detect_member(decoder)
    assert cls is Vela2HybridExecutor and name == EXPECTED_MODEL_NAME


def test_detect_member_refuses_packages_without_a_usable_config(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="missing config.json"):
        detect_member(empty)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "config.json").write_text(json.dumps({"model_type": "vela2-unified-xl"}))
    with pytest.raises(ValueError, match="vela2-unified-xl"):
        detect_member(foreign)
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "config.json").write_text("{not json")
    with pytest.raises(ValueError, match="not readable JSON"):
        detect_member(broken)
    array = tmp_path / "array"
    array.mkdir()
    (array / "config.json").write_text("[]")
    with pytest.raises(ValueError, match="not a JSON object"):
        detect_member(array)


def test_models_entry_is_verbatim_per_member():
    hybrid = models_entry(FakeVela2Hybrid())
    assert hybrid == {
        "name": EXPECTED_MODEL_NAME,
        "release_date": "",
        "description": (
            "Vela 2.0 0.8B: SystemOne Choice / Noul / Score "
            "with span and set extensions, 16,384-token input"
        ),
    }
    assert models_entry(FakeVela2())["release_date"] == "2026-09-29"


# -------------------------------------------------------- response finishing


def test_decide_answers_all_five_kinds_with_span_heads():
    response = decide(json.dumps(hybrid_body()).encode(), FakeVela2Hybrid())
    assert set(response) == {"model", "answers", "usage", "spans", "sets", "thresholds", "span_heads"}
    assert list(response["span_heads"]) == ["unsupported"]
    assert response["model"] == EXPECTED_MODEL_NAME


def test_decide_accepts_a_hybrid_response_without_span_questions():
    value = body(model=EXPECTED_MODEL_NAME)
    response = decide(json.dumps(value).encode(), FakeVela2Hybrid())
    assert "span_heads" not in response and "spans" not in response


def test_a_hybrid_response_missing_span_heads_is_refused():
    class Headless(FakeVela2Hybrid):
        def system_one(self, body):
            response = super().system_one(body)
            response.pop("span_heads", None)
            return response

    with pytest.raises(RuntimeError, match="span_heads do not match"):
        decide(json.dumps(hybrid_body()).encode(), Headless())


def test_a_hybrid_response_with_stray_span_heads_is_refused():
    class Stray(FakeVela2Hybrid):
        def system_one(self, body):
            response = super().system_one(body)
            response["span_heads"] = {"ghost": "router"}
            return response

    with pytest.raises(RuntimeError, match="span_heads without span questions"):
        decide(json.dumps(body(model=EXPECTED_MODEL_NAME)).encode(), Stray())


def test_the_unified_member_still_refuses_span_heads():
    class UnifiedLeak(FakeVela2):
        def system_one(self, body):
            response = super().system_one(body)
            if "spans" in response:
                response["span_heads"] = {qid: "router" for qid in response["spans"]}
            return response

    with pytest.raises(RuntimeError, match="this worker serves the 0.3B member"):
        decide(json.dumps(five_kind_body()).encode(), UnifiedLeak())


def test_finish_response_keeps_the_hybrid_envelope_tight():
    questions = hybrid_body()["questions"]
    response = FakeVela2Hybrid().system_one(hybrid_body())
    assert finish_response(questions, response, hybrid=True) is response
    del response["answers"]
    with pytest.raises(RuntimeError, match="unexpected response envelope"):
        finish_response(questions, response, hybrid=True)
