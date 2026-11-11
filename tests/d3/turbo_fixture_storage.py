"""Lossless shared storage for the offline turbo oracle (no NumPy dependency).

Version 2 separates mode outputs from common requests/preparation. Loading
returns the original version-1 shape, including the derived token slices.
Historical generation metadata and provenance are preserved without modification.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile

PREPARATION_FILE = "turbo-preparation.json"
TENSOR_FILE = "turbo-inputs.npz"


def _json(value, level=0):
    """Keep dictionaries readable and scalar arrays on one line."""
    scalar = lambda item: not isinstance(item, (dict, list))
    if isinstance(value, dict) and value:
        entries = ["  " * (level + 1) + json.dumps(key, ensure_ascii=False) + ": " + _json(item, level + 1)
                   for key, item in value.items()]
        return "{\n" + ",\n".join(entries) + "\n" + "  " * level + "}"
    if isinstance(value, list) and not all(scalar(item) for item in value):
        return "[\n" + ",\n".join("  " * (level + 1) + _json(item, level + 1) for item in value) + "\n" + "  " * level + "]"
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _bytes(document):
    return (_json(document) + "\n").encode("utf-8")


def _reference(root, filename, digest):
    if (not isinstance(filename, str) or not filename or filename in (".", "..")
            or Path(filename).name != filename or "\\" in filename):
        raise ValueError("fixture reference must be a sibling filename")
    path = root / filename
    if path.is_symlink():
        raise ValueError("fixture reference must not be a symlink")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("invalid fixture SHA256")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"fixture SHA256 mismatch: {filename}")
    return path


def _token_slices(preparation, *, compact):
    for key, row in preparation["rows"].items():
        ids = row["token_ids"]
        end = preparation["state_ends"][key]
        if type(end) is not int or not 0 <= end <= len(ids):
            raise ValueError(f"invalid state boundary: {key}")
        slices = {"state_tokens": ids[:end], "question_tokens": ids[end:]}
        if compact:
            if any(row.get(name) != value for name, value in slices.items()):
                raise ValueError(f"derived token slices differ: {key}")
            for name in slices:
                del row[name]
        else:
            if any(name in row for name in slices):
                raise ValueError(f"unexpected derived token slices: {key}")
            expanded = {}
            for name, value in row.items():
                expanded[name] = value
                if name == "token_ids":
                    expanded.update(slices)
            row.clear()
            row.update(expanded)


def load_fixture(path):
    """Read and verify sibling references; return the expanded version-1 shape."""
    path = Path(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if document["schema"] == "d3-turbo-oracle/1":
        _reference(path.parent, document["tensor_file"], document["tensor_file_sha256"])
        return document
    if document["schema"] != "d3-turbo-oracle/2":
        raise ValueError("unsupported turbo oracle schema")
    common_path = _reference(path.parent, document.pop("preparation_file"), document.pop("preparation_file_sha256"))
    common = json.loads(common_path.read_text(encoding="utf-8"))
    if common["schema"] != "d3-turbo-preparation/1":
        raise ValueError("unsupported turbo preparation schema")
    _reference(path.parent, common["tensor_file"], common["tensor_file_sha256"])
    if list(document["cases"]) != list(common["cases"]):
        raise ValueError("fixture case IDs/order differ from shared preparation")
    for name, case in common["cases"].items():
        if set(case) != {"request", "preparation"}:
            raise ValueError("shared case must contain request and preparation")
        if set(document["cases"][name]) != {"probabilities", "response"}:
            raise ValueError("mode case must contain probabilities and response")
        _token_slices(case["preparation"], compact=False)
        document["cases"][name] = {**case, **document["cases"][name]}
    document["schema"] = "d3-turbo-oracle/1"
    document["tensor_file"] = common["tensor_file"]
    document["tensor_file_sha256"] = common["tensor_file_sha256"]
    return document


def write_fixture(path, document, tensor_bytes):
    """Write normalized storage, refusing conflicts with any existing shared base.

    All validation and serialization precede writes. Shared files are immutable:
    another mode can reuse identical bytes, but differing inputs require a fresh
    output directory. The mode JSON is replaced only after the base is present.
    """
    path = Path(path)
    if path.name in (PREPARATION_FILE, TENSOR_FILE):
        raise ValueError("mode fixture filename conflicts with shared storage")
    document = copy.deepcopy(document)
    if document["schema"] != "d3-turbo-oracle/1":
        raise ValueError("write_fixture requires an expanded version-1 document")
    tensor_hash = hashlib.sha256(tensor_bytes).hexdigest()
    if document.get("tensor_file_sha256", tensor_hash) != tensor_hash:
        raise ValueError("tensor bytes differ from declared SHA256")
    common = {"schema": "d3-turbo-preparation/1", "cases": {},
              "tensor_file": TENSOR_FILE, "tensor_file_sha256": tensor_hash}
    for name, case in document["cases"].items():
        if set(case) != {"request", "preparation", "probabilities", "response"}:
            raise ValueError("expanded case fields differ from turbo oracle contract")
        common["cases"][name] = {"request": case.pop("request"), "preparation": case.pop("preparation")}
        _token_slices(common["cases"][name]["preparation"], compact=True)
    common_bytes = _bytes(common)
    document["schema"] = "d3-turbo-oracle/2"
    document.pop("tensor_file", None)
    document.pop("tensor_file_sha256", None)
    document["preparation_file"] = PREPARATION_FILE
    document["preparation_file_sha256"] = hashlib.sha256(common_bytes).hexdigest()
    mode_bytes = _bytes(document)
    shared = [(path.parent / PREPARATION_FILE, common_bytes), (path.parent / TENSOR_FILE, tensor_bytes)]
    for target, data in shared:
        if target.is_symlink() or (target.exists() and target.read_bytes() != data):
            raise ValueError(f"conflicting shared fixture {target.name}; use a fresh output directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    created = []
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".turbo-mode-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(mode_bytes)
        for target, data in shared:
            try:
                stream = target.open("xb")
            except FileExistsError:
                if target.is_symlink() or target.read_bytes() != data:
                    raise ValueError(f"conflicting shared fixture {target.name}; use a fresh output directory")
            else:
                created.append(target)
                with stream:
                    stream.write(data)
        temporary.replace(path)
    except BaseException:
        for target in created:
            target.unlink()
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
