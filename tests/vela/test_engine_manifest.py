"""Package-verification tests for the Vela 2.0 0.3B executor's gate.

Uses temp directories with an injectable pin table and recomputed
hashes: no weights, no torch, no network. What is under test is the
worker's own admission — the structural config identity, the full-hash
gate over the pinned files that runs before any package code is
imported, the refusal of unlisted importables in the package root, and
the exact-size weights check.
"""

from __future__ import annotations

import hashlib
import json
import os
import py_compile
import re
import sys
from pathlib import Path

import pytest

from models.vela.engine import (
    EXPECTED_ARCHITECTURE,
    EXPECTED_MARKER_COUNT,
    EXPECTED_MAX_LENGTH,
    EXPECTED_MODEL_TYPE,
    EXPECTED_WEIGHTS_BYTES,
    PINNED_FILES_SHA256,
    REFERENCE_REPO,
    REFERENCE_REVISION,
    put_first_on_sys_path,
    verify_config,
    verify_files,
    verify_no_foreign_runtime,
    verify_package,
    verify_weights,
)

CONFIG = {
    "model_type": EXPECTED_MODEL_TYPE,
    "architectures": [EXPECTED_ARCHITECTURE],
    "max_length": EXPECTED_MAX_LENGTH,
    "marker_ids": {f"[M{i}]": 256000 + i for i in range(EXPECTED_MARKER_COUNT)},
}


def table_of(files: dict[str, bytes]) -> dict[str, str]:
    return {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}


def build_package(root: Path, files: dict[str, bytes], config: dict | None = None) -> dict[str, str]:
    for name, content in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (root / "config.json").write_text(
        json.dumps(config if config is not None else CONFIG), encoding="utf-8"
    )
    (root / "model.safetensors").write_bytes(b"0" * min(EXPECTED_WEIGHTS_BYTES, 64))
    files = dict(files)
    files["config.json"] = (root / "config.json").read_bytes()
    return table_of(files)


def sparse_weights(root: Path, size: int = EXPECTED_WEIGHTS_BYTES) -> None:
    """The gate reads ``stat().st_size`` only, so a sparse hole stands
    in for the 1.2 GB file instead of a full heap allocation."""

    with (root / "model.safetensors").open("wb") as stream:
        stream.truncate(size)


def small_files():
    return {
        "vela2_serve.py": b"upstream server stub\n",
        "vela2_inference.py": b"upstream engine stub\n",
        "modeling_vela2.py": b"upstream model stub\n",
        "configuration_vela2.py": b"upstream config stub\n",
        "calibration.json": b'{"temperature": {}}',
        "tokenizer.json": b'{"version": "stub"}',
    }


def test_the_checked_in_pin_table_shape_is_locked():
    assert REFERENCE_REPO == "vllm-sr/Vela-2.0-0.3B"
    assert REFERENCE_REVISION == "d6f03aa9baca4f9017fa8f8e9c2764d3bed3f409"
    assert len(PINNED_FILES_SHA256) == 7
    assert set(PINNED_FILES_SHA256) == {
        "vela2_serve.py", "vela2_inference.py", "modeling_vela2.py",
        "configuration_vela2.py", "config.json", "calibration.json", "tokenizer.json",
    }
    for digest in PINNED_FILES_SHA256.values():
        assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_verification_accepts_a_package_matching_its_table(tmp_path):
    table = build_package(tmp_path, small_files())
    assert verify_files(tmp_path, table) == sorted(table)
    verify_config(tmp_path)
    # verify_weights needs the exact pinned size; the stub above is short.
    sparse_weights(tmp_path)
    verify_package(tmp_path, table)


def test_config_identity_is_checked_before_any_hashing(tmp_path):
    build_package(tmp_path, small_files(), config={**CONFIG, "model_type": "vela2-decoder"})
    with pytest.raises(ValueError, match="vela2-decoder.*only serves the vela2-unified member"):
        verify_config(tmp_path)
    with pytest.raises(ValueError, match="model_type"):
        verify_package(tmp_path, table_of(small_files()))


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"architectures": ["OtherModel"]}, "architectures"),
        ({"max_length": 16384}, "max_length"),
        ({"marker_ids": {"[Q]": 256000}}, "marker token ids"),
    ],
)
def test_foreign_config_structures_are_refused(tmp_path, override, fragment):
    build_package(tmp_path, small_files(), config={**CONFIG, **override})
    with pytest.raises(ValueError, match=fragment):
        verify_config(tmp_path)


def test_a_missing_config_is_refused(tmp_path):
    with pytest.raises(ValueError, match="missing config.json"):
        verify_config(tmp_path)


def test_tampered_files_are_rejected(tmp_path):
    table = build_package(tmp_path, small_files())
    victim = tmp_path / "vela2_inference.py"
    victim.write_bytes(victim.read_bytes() + b"tampered\n")
    with pytest.raises(ValueError, match="vela2_inference.py: SHA-256 differs"):
        verify_files(tmp_path, table)


def test_missing_pinned_files_are_rejected(tmp_path):
    table = build_package(tmp_path, small_files())
    (tmp_path / "calibration.json").unlink()
    with pytest.raises(ValueError, match="missing pinned file calibration.json"):
        verify_files(tmp_path, table)


def test_a_later_self_consistent_snapshot_is_refused_by_the_checked_in_pin(tmp_path):
    # What a later snapshot looks like: every file hashes to its own table,
    # yet the baseline changed. Only the checked-in table can catch this.
    build_package(tmp_path, small_files())
    with pytest.raises(ValueError, match="SHA-256 differs from the pin"):
        verify_files(tmp_path)


def test_unsafe_pinned_names_are_rejected_before_any_file_access(tmp_path):
    with pytest.raises(ValueError, match="unsafe pinned file name"):
        verify_files(tmp_path, {"../escape.py": "0" * 64})


def test_extra_importable_files_in_the_package_root_are_refused(tmp_path):
    table = build_package(tmp_path, small_files())
    (tmp_path / "tokenizers.py").write_text("shadowing module\n")
    with pytest.raises(ValueError, match="unexpected importable file tokenizers.py"):
        verify_files(tmp_path, table)


@pytest.mark.parametrize(
    "name",
    ["torch.pyc", "torch.abi3.so", "_vela.cpython-312-x86_64-linux-gnu.so"],
)
def test_sourceless_and_extension_module_files_are_refused(tmp_path, name):
    table = build_package(tmp_path, small_files())
    (tmp_path / name).write_bytes(b"module bytes\n")
    with pytest.raises(ValueError, match=f"unexpected importable file {re.escape(name)}"):
        verify_files(tmp_path, table)


def test_extra_package_directories_in_the_package_root_are_refused(tmp_path):
    table = build_package(tmp_path, small_files())
    fla = tmp_path / "fla"
    fla.mkdir()
    (fla / "__init__.py").write_text("shadowing package\n")
    with pytest.raises(ValueError, match="unexpected importable package fla"):
        verify_files(tmp_path, table)


@pytest.mark.parametrize(
    "init_name",
    ["__init__.pyc", "__init__.abi3.so"],
)
def test_sourceless_and_extension_package_inits_are_refused(tmp_path, init_name):
    # A directory whose __init__ is bytecode or an extension — no source at
    # all — still imports as a regular package once the package root goes
    # first on sys.path, executing bytes the pin never hashed: the review's
    # torch/__init__.pyc bypass.
    table = build_package(tmp_path, small_files())
    torch = tmp_path / "torch"
    torch.mkdir()
    (torch / init_name).write_bytes(b"unverified module bytes\n")
    with pytest.raises(ValueError, match="unexpected importable package torch/"):
        verify_files(tmp_path, table)


def test_a_namespace_directory_with_nested_modules_is_refused(tmp_path):
    # Even without any __init__, a directory imports as a namespace package
    # whose nested modules execute on import the same way.
    table = build_package(tmp_path, small_files())
    fla = tmp_path / "fla"
    fla.mkdir()
    (fla / "ops.py").write_text("unverified module\n")
    with pytest.raises(ValueError, match="unexpected importable package fla/"):
        verify_files(tmp_path, table)


def test_a_directory_symlink_hiding_importable_code_is_refused(tmp_path):
    # rglob does not descend into a symlinked subdirectory, but import does:
    # a real directory whose only entry is a symlink to a directory with a
    # module would pass the depth walk, and the module would execute after
    # the package root goes first on sys.path.
    package = tmp_path / "pkg"
    package.mkdir()
    table = build_package(package, small_files())
    target = tmp_path / "target"
    target.mkdir()
    (target / "ops.py").write_text("unverified module\n")
    fla = package / "fla"
    fla.mkdir()
    os.symlink(target, fla / "evil")
    with pytest.raises(ValueError, match="unexpected importable package fla/"):
        verify_files(package, table)


def test_a_directory_symlink_in_the_package_root_is_refused(tmp_path):
    # A symlinked directory at the root is refused outright: walking its
    # target cannot see the target's own nested symlinks either.
    package = tmp_path / "pkg"
    package.mkdir()
    table = build_package(package, small_files())
    target = tmp_path / "target"
    target.mkdir()
    os.symlink(target, package / "torch")
    with pytest.raises(ValueError, match="directory symlink torch"):
        verify_files(package, table)


def test_unimportable_package_root_entries_still_pass(tmp_path):
    # The pinned package's own layout shape: onnx/, assets/, LICENSES/ and
    # figures/ directories without __init__.py, and markdown and license
    # files. An existing __pycache__ is refused separately: the gate below
    # cannot verify cached bytecode against the pin.
    table = build_package(tmp_path, small_files())
    for directory in ("onnx", "assets", "LICENSES", "figures"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "file.bin").write_bytes(b"not a python package\n")
    (tmp_path / "README.md").write_text("# Vela 2.0 0.3B\n")
    (tmp_path / "LICENSE").write_text("Apache-2.0\n")

    assert verify_files(tmp_path, table) == sorted(table)


def test_an_existing_pycache_of_different_code_is_refused(tmp_path):
    # The bypass the gate must close: Python reads a matching unchecked-hash
    # .pyc from __pycache__ in place of the verified source, so a cache
    # compiled from different code would execute unverified bytecode after
    # a clean verify_files pass. Compile the poison, restore the
    # manifest-matching source, and the directory itself must be refused.
    files = small_files()
    table = build_package(tmp_path, files)
    poisoned = tmp_path / "vela2_inference.py"
    poisoned.write_text("POISONED_MARKER = 'not the pinned runtime'\n", encoding="utf-8")
    py_compile.compile(
        poisoned,
        cfile=tmp_path / "__pycache__" / f"vela2_inference.{sys.implementation.cache_tag}.pyc",
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
    )
    poisoned.write_bytes(files["vela2_inference.py"])

    with pytest.raises(ValueError, match="__pycache__"):
        verify_files(tmp_path, table)


def test_no_foreign_runtime_import_is_clean(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    build_package(package, small_files())
    verify_no_foreign_runtime(package)


def test_a_runtime_module_from_another_directory_is_refused(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "modeling_vela2.py").write_text("STUB = 1\n")
    package = tmp_path / "pkg"
    package.mkdir()
    build_package(package, small_files())
    sys.path.insert(0, str(other))
    try:
        import modeling_vela2  # noqa: F401

        with pytest.raises(ValueError, match="modeling_vela2 module from another package"):
            verify_no_foreign_runtime(package)
    finally:
        sys.path.remove(str(other))
        sys.modules.pop("modeling_vela2", None)


def test_the_package_root_is_forced_to_the_front_of_sys_path(tmp_path):
    # A PYTHONPATH copy of the root listed later must not leave a foreign
    # directory ahead of it: verify_no_foreign_runtime cannot catch this,
    # because sys.modules is empty before the first import.
    package = tmp_path / "pkg"
    shadow = tmp_path / "shadow"
    package.mkdir()
    shadow.mkdir()
    resolved = str(package.resolve())
    sys.path.insert(0, str(shadow))
    sys.path.append(str(package))
    try:
        assert put_first_on_sys_path(package) == resolved
        assert sys.path[0] == resolved
        assert resolved not in sys.path[1:]
        assert str(shadow) in sys.path
    finally:
        sys.path[:] = [entry for entry in sys.path if entry not in (str(shadow), resolved)]


def test_weights_are_checked_by_exact_pinned_size(tmp_path):
    build_package(tmp_path, small_files())
    sparse_weights(tmp_path)
    verify_weights(tmp_path)

    sparse_weights(tmp_path, EXPECTED_WEIGHTS_BYTES - 1)
    with pytest.raises(ValueError, match="is 1236473251 bytes"):
        verify_weights(tmp_path)

    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="missing model.safetensors"):
        verify_weights(tmp_path)
