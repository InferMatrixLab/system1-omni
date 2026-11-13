import hashlib
import json

import pytest

from models.valen_decision_1_5 import pinned
from models.valen_decision_1_5.protocol import MODEL_NAMES


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture()
def release(tmp_path, monkeypatch):
    """A tiny fake release on disk, registered as the only pinned one."""

    files = {
        "config.json": json.dumps({"valen_decision_version": "1.5"}).encode(),
        "chat_template.jinja": b"template",
        "compiler.py": b"compiler = 1\n",
        "tokenizer.json": b"{}",
        "model.safetensors": b"weights",
    }
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    digests = {name: sha(data) for name, data in files.items()}
    entry = pinned.Release(
        "2B",
        "0" * 40,
        {name: digests[name] for name in ("config.json", "chat_template.jinja")},
        {"model.safetensors": digests["model.safetensors"]},
    )
    monkeypatch.setattr(pinned, "RELEASES", (entry,))
    monkeypatch.setattr(
        pinned,
        "SHARED_SHA256",
        {name: digests[name] for name in ("compiler.py", "tokenizer.json")},
    )
    return tmp_path


def test_verify_identifies_the_release_and_returns_its_config(release):
    checkpoint = pinned.verify_checkpoint(release)

    assert checkpoint.path == release.resolve()
    assert checkpoint.release.size == "2B"
    assert checkpoint.release.repo == "Valen-Team/Valen-Decision-1.5-2B"
    assert checkpoint.release.model_name == "valen-decision-1.5-2b"
    assert checkpoint.config == {"valen_decision_version": "1.5"}


def test_verify_refuses_an_unpinned_config(release):
    (release / "config.json").write_text('{"valen_decision_version": "1.0"}')
    with pytest.raises(ValueError, match="does not match any pinned"):
        pinned.verify_checkpoint(release)
    (release / "config.json").unlink()
    with pytest.raises(ValueError, match="must contain config.json"):
        pinned.verify_checkpoint(release)


@pytest.mark.parametrize("name", ["compiler.py", "tokenizer.json", "chat_template.jinja"])
def test_verify_refuses_changed_or_missing_runtime_files(release, name):
    (release / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match=f"{name}: SHA-256 differs"):
        pinned.verify_checkpoint(release, weights=False)
    (release / name).unlink()
    with pytest.raises(ValueError, match=f"missing {name}"):
        pinned.verify_checkpoint(release, weights=False)


def test_weight_hashing_is_the_only_skippable_check(release):
    (release / "model.safetensors").write_bytes(b"other weights")
    with pytest.raises(ValueError, match="model.safetensors: SHA-256 differs"):
        pinned.verify_checkpoint(release)
    assert pinned.verify_checkpoint(release, weights=False).release.size == "2B"


def test_pinned_table_covers_every_served_size_with_distinct_configs():
    assert [entry.size for entry in pinned.RELEASES] == list(MODEL_NAMES)
    configs = {entry.files["config.json"] for entry in pinned.RELEASES}
    assert len(configs) == len(pinned.RELEASES)
    for entry in pinned.RELEASES:
        assert len(entry.revision) == 40
        assert entry.weights
        digests = [*entry.files.values(), *entry.weights.values()]
        assert all(len(digest) == 64 for digest in digests)
    assert {"compiler.py", "modeling_valen.py", "responses.py"} <= set(pinned.SHARED_SHA256)
