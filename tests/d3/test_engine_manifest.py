"""Manifest verification tests for the d3 executor's runtime gate.

Uses the pinned upstream manifest as a fixture and temp directories with
recomputed hashes: no weights, no torch, no network. What is under test is
the worker's own admission — the identity pin that binds the manifest to
REFERENCE_REVISION, and the full-hash gate over the runtime table that runs
before any package code is imported.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from models.d3.engine import (
    MANIFEST_SCHEMA,
    PINNED_RUNTIME_SHA256,
    REFERENCE_REPO,
    load_manifest,
    verify_pinned_package,
    verify_runtime_files,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "MODEL_MANIFEST.json"
PINNED_RUNTIME_SHA = (
    "6c049aa7d3da8da3738602a2812572879d0a47b03c28c3e33f36d7300257de89"
)


def pinned_manifest():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def build_package(root: Path, files: dict[str, bytes]) -> dict:
    """Materialize a same-repo package whose runtime files hash to its own
    manifest — the shape a later self-consistent snapshot would have."""

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "repo_id": REFERENCE_REPO,
        "runtime": {
            "files_sha256": {
                name: hashlib.sha256(content).hexdigest() for name, content in files.items()
            }
        },
    }
    for name, content in files.items():
        (root / name).write_bytes(content)
    (root / "MODEL_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def small_files():
    return {
        "d3_runtime.py": b"verified runtime content\n",
        "requirements.txt": b"torch>=2.8\n",
    }


def test_the_pinned_manifest_fixture_shape_is_locked():
    manifest = pinned_manifest()
    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["repo_id"] == REFERENCE_REPO
    assert manifest["model_name"] == "d3"
    # The upstream runtime this worker was integrated against; the repo may
    # have moved since (drift is why the pin and the hashes exist).
    assert manifest["runtime"]["files_sha256"]["d3_runtime.py"] == PINNED_RUNTIME_SHA
    assert manifest["runtime"]["files_sha256"] == PINNED_RUNTIME_SHA256
    assert len(manifest["files_sha256"]) == 39
    assert len(manifest["runtime"]["files_sha256"]) == 9


def test_the_pinned_manifest_passes_the_identity_gate():
    assert verify_pinned_package(pinned_manifest()) is None


def test_self_consistent_but_unpinned_tables_are_refused(tmp_path):
    # What a later snapshot looks like to the full-hash gate: every file
    # hashes to the package's own manifest, yet the baseline changed.
    manifest = build_package(tmp_path, small_files())

    with pytest.raises(ValueError, match="not the table pinned for"):
        verify_pinned_package(manifest)


def test_foreign_repo_ids_are_refused():
    manifest = pinned_manifest()
    manifest["repo_id"] = "other/d3"

    with pytest.raises(ValueError, match="this worker only serves"):
        verify_pinned_package(manifest)


def test_verification_accepts_a_package_matching_the_manifest(tmp_path):
    manifest = build_package(tmp_path, small_files())
    assert load_manifest(tmp_path) == manifest
    assert verify_runtime_files(tmp_path, manifest) == ["d3_runtime.py", "requirements.txt"]


def test_the_full_nine_file_table_verifies(tmp_path):
    names = list(pinned_manifest()["runtime"]["files_sha256"])
    assert len(names) == 9
    files = {name: f"content of {name}\n".encode() for name in names}
    manifest = build_package(tmp_path, files)

    assert verify_runtime_files(tmp_path, manifest) == sorted(names)


def test_tampered_runtime_content_is_rejected(tmp_path):
    manifest = build_package(tmp_path, small_files())
    victim = tmp_path / "d3_runtime.py"
    victim.write_bytes(victim.read_bytes() + b"tampered\n")

    with pytest.raises(ValueError, match="runtime SHA-256 differs"):
        verify_runtime_files(tmp_path, manifest)


def test_truncated_runtime_file_is_rejected(tmp_path):
    manifest = build_package(tmp_path, small_files())
    victim = tmp_path / "requirements.txt"
    victim.write_bytes(victim.read_bytes()[:5])

    with pytest.raises(ValueError, match="runtime verification failed"):
        verify_runtime_files(tmp_path, manifest)


def test_missing_runtime_file_is_rejected(tmp_path):
    manifest = build_package(tmp_path, small_files())
    (tmp_path / "d3_runtime.py").unlink()

    with pytest.raises(ValueError, match="missing runtime file d3_runtime.py"):
        verify_runtime_files(tmp_path, manifest)


def test_packages_without_a_manifest_or_runtime_table_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="missing MODEL_MANIFEST.json"):
        load_manifest(tmp_path)

    (tmp_path / "MODEL_MANIFEST.json").write_text(json.dumps({"schema": "other"}))
    with pytest.raises(ValueError, match="not a d3-package-manifest/1 manifest"):
        load_manifest(tmp_path)

    manifest = build_package(tmp_path, {})
    del manifest["runtime"]
    (tmp_path / "MODEL_MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="no runtime.files_sha256 table"):
        verify_runtime_files(tmp_path, manifest)


def test_unsafe_runtime_names_are_rejected_before_any_file_access(tmp_path):
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "runtime": {"files_sha256": {"../escape.py": "0" * 64}},
    }
    with pytest.raises(ValueError, match="unsafe runtime file name"):
        verify_runtime_files(tmp_path, manifest)


def test_extra_importable_files_in_the_package_root_are_refused(tmp_path):
    # The package root goes first on sys.path, so an unlisted module there
    # executes in place of the installed dependency it shadows.
    build_package(tmp_path, small_files())
    (tmp_path / "transformers.py").write_text("shadowing module\n")

    with pytest.raises(ValueError, match="unexpected importable file transformers.py"):
        verify_runtime_files(tmp_path, load_manifest(tmp_path))


def test_extra_package_directories_in_the_package_root_are_refused(tmp_path):
    build_package(tmp_path, small_files())
    fla = tmp_path / "fla"
    fla.mkdir()
    (fla / "__init__.py").write_text("shadowing package\n")

    with pytest.raises(ValueError, match="unexpected importable package fla"):
        verify_runtime_files(tmp_path, load_manifest(tmp_path))


def test_unimportable_package_root_entries_still_pass(tmp_path):
    # The pinned package's own layout shape: an assets/ directory without
    # __init__.py, plain data files, and the __pycache__ Python writes next
    # to the imported runtime — none of it can shadow an import.
    manifest = build_package(tmp_path, small_files())
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "banner.png").write_bytes(b"not a python package\n")
    (tmp_path / "LICENSE").write_text("Apache-2.0\n")
    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    (pycache / "d3_runtime.cpython-312.pyc").write_bytes(b"bytecode\n")

    assert verify_runtime_files(tmp_path, manifest) == ["d3_runtime.py", "requirements.txt"]
