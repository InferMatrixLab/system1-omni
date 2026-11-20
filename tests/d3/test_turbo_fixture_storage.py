"""Lossless, CPU-only storage for the shared turbo preparation oracle."""

import copy
import hashlib
import json
from pathlib import Path

import pytest


def storage():
    assert Path(__file__).with_name("turbo_fixture_storage.py").exists(), "fixture storage helper is missing"
    from . import turbo_fixture_storage
    return turbo_fixture_storage


def document(state_once=True):
    return {
        "schema": "d3-turbo-oracle/1",
        "metadata": {"state_once": state_once, "compute_capability": [8, 0]},
        "cases": {"text": {
            "request": {"state": "重复", "questions": {"q": {"type": "noul"}}},
            "preparation": {"keys": ["q"], "state_ends": {"q": 2}, "rows": {"q": {
                "prompt": "a\nb", "token_ids": [1, 2, 3], "state_tokens": [1, 2],
                "question_tokens": [3], "readout_position": 2,
                "tensors": {"input_ids": {"array": "text/q/input_ids", "shape": [1, 3]}}}}},
            "probabilities": {"q": [0.2 if state_once else 0.3, 0.8 if state_once else 0.7]},
            "response": {"answers": {"q": {"type": "noul", "noul": 0.8 if state_once else 0.7}},
                         "usage": {"input_tokens": 3, "output_tokens": 0}},
        }},
        "provenance": {"source_revision": "a" * 40, "generator_sha256": "b" * 64},
        "tensor_file": "old.npz", "tensor_file_sha256": hashlib.sha256(b"tensors").hexdigest(),
    }


def expected_expansion(value):
    result = copy.deepcopy(value)
    result["tensor_file"] = "turbo-inputs.npz"
    return result


def tree_bytes(path):
    return {file.name: file.read_bytes() for file in path.iterdir() if file.is_file()}


def test_roundtrip_expands_exact_original_values_without_mutating_input(tmp_path):
    api = storage()
    original = document()
    before = copy.deepcopy(original)
    path = tmp_path / "state.json"
    api.write_fixture(path, original, b"tensors")
    assert api.load_fixture(path) == expected_expansion(original)
    assert original == before
    stored = json.loads(path.read_text())
    assert stored["schema"] == "d3-turbo-oracle/2"
    assert "preparation" not in stored["cases"]["text"]
    shared = json.loads((tmp_path / "turbo-preparation.json").read_text())
    assert shared["schema"] == "d3-turbo-preparation/1"
    assert "state_tokens" not in shared["cases"]["text"]["preparation"]["rows"]["q"]
    assert "question_tokens" not in shared["cases"]["text"]["preparation"]["rows"]["q"]


def test_modes_share_inputs_but_keep_their_own_outputs_and_provenance(tmp_path):
    api = storage()
    for once, name in [(True, "state"), (False, "full")]:
        api.write_fixture(tmp_path / f"{name}.json", document(once), b"tensors")
    first, second = [json.loads((tmp_path / f"{name}.json").read_text()) for name in ("state", "full")]
    assert first["preparation_file_sha256"] == second["preparation_file_sha256"]
    assert len(list(tmp_path.glob("*.npz"))) == 1
    for once, name in [(True, "state"), (False, "full")]:
        assert api.load_fixture(tmp_path / f"{name}.json") == expected_expansion(document(once))


@pytest.mark.parametrize("change", ["request", "preparation", "tensors", "derived"])
def test_conflicting_write_leaves_existing_shared_and_mode_files_untouched(tmp_path, change):
    api = storage()
    path = tmp_path / "state.json"
    api.write_fixture(path, document(), b"tensors")
    before = tree_bytes(tmp_path)
    changed = document(False)
    tensor_bytes = b"tensors"
    if change == "request":
        changed["cases"]["text"]["request"]["state"] = "different"
    elif change == "preparation":
        changed["cases"]["text"]["preparation"]["rows"]["q"]["prompt"] = "different"
    elif change == "tensors":
        tensor_bytes = b"other tensors"
        changed["tensor_file_sha256"] = hashlib.sha256(tensor_bytes).hexdigest()
    else:
        changed["cases"]["text"]["preparation"]["rows"]["q"]["state_tokens"] = [9, 9]
    with pytest.raises(ValueError):
        api.write_fixture(path, changed, tensor_bytes)
    assert tree_bytes(tmp_path) == before


@pytest.mark.parametrize("change", ["preparation-hash", "tensor-hash", "path", "tensor-path", "case-order"])
def test_load_rejects_corrupt_or_unsafe_references(tmp_path, change):
    api = storage()
    path = tmp_path / "state.json"
    api.write_fixture(path, document(), b"tensors")
    stored = json.loads(path.read_text())
    shared_path = tmp_path / "turbo-preparation.json"
    shared = json.loads(shared_path.read_text())
    if change == "preparation-hash":
        stored["preparation_file_sha256"] = "0" * 64
    elif change == "tensor-hash":
        (tmp_path / "turbo-inputs.npz").write_bytes(b"corrupt")
    elif change == "path":
        stored["preparation_file"] = "../outside.json"
    elif change == "tensor-path":
        shared["tensor_file"] = "../outside.npz"
        shared_path.write_text(json.dumps(shared))
        stored["preparation_file_sha256"] = hashlib.sha256(shared_path.read_bytes()).hexdigest()
    else:
        stored["cases"] = {"other": stored["cases"].pop("text")}
    path.write_text(json.dumps(stored))
    with pytest.raises(ValueError):
        api.load_fixture(path)


def test_scalar_arrays_stay_inline_and_json_roundtrips(tmp_path):
    api = storage()
    path = tmp_path / "state.json"
    api.write_fixture(path, document(), b"tensors")
    assert '"token_ids": [1, 2, 3]' in (tmp_path / "turbo-preparation.json").read_text()
    assert '"compute_capability": [8, 0]' in path.read_text()
    assert api.load_fixture(path) == expected_expansion(document())


def test_load_rejects_reordered_cases_even_when_ids_match(tmp_path):
    api = storage()
    original = document()
    original["cases"]["second"] = copy.deepcopy(original["cases"]["text"])
    path = tmp_path / "state.json"
    api.write_fixture(path, original, b"tensors")
    stored = json.loads(path.read_text())
    stored["cases"] = dict(reversed(list(stored["cases"].items())))
    path.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="IDs/order"):
        api.load_fixture(path)


@pytest.mark.parametrize("existing", [True, False])
def test_failed_mode_replace_preserves_existing_files_and_removes_new_base(tmp_path, monkeypatch, existing):
    api = storage()
    path = tmp_path / "state.json"
    if existing:
        api.write_fixture(path, document(), b"tensors")
    before = tree_bytes(tmp_path)

    def fail_replace(self, target):
        raise OSError("simulated filesystem failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="filesystem failure"):
        api.write_fixture(path, document(False), b"tensors")
    assert tree_bytes(tmp_path) == before


def test_load_rejects_symlink_to_tensor_outside_fixture_directory(tmp_path):
    api = storage()
    root = tmp_path / "fixtures"
    path = root / "state.json"
    api.write_fixture(path, document(), b"tensors")
    tensor_path = root / "turbo-inputs.npz"
    tensor_path.unlink()
    outside = tmp_path / "outside.npz"
    outside.write_bytes(b"tensors")
    tensor_path.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        api.load_fixture(path)
