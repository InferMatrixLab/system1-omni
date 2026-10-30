"""Integrity and layout checks for small, GPU-generated turbo oracle fixtures.

The NPZ reader only inspects NumPy's public file headers and raw payloads; it
does not require NumPy, torch, a checkpoint or an accelerator on CPU CI.
"""

import ast
import hashlib
import math
from pathlib import Path
import zipfile

import pytest

from .turbo_fixture_storage import load_fixture

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def array_header(raw):
    assert raw[:6] == b"\x93NUMPY"
    major = raw[6]
    assert major in (1, 2, 3)
    length_bytes = 2 if major == 1 else 4
    start = 8 + length_bytes
    end = start + int.from_bytes(raw[8:start], "little")
    return ast.literal_eval(raw[start:end].decode("utf-8").strip()), raw[end:]


@pytest.mark.parametrize("mode,state_once", [("state-once", True), ("full-row", False)])
def test_frozen_preparation_tensor_bytes_layout_and_accounting(mode, state_once):
    path = FIXTURES / f"turbo-oracle-{mode}.json"
    fixture = load_fixture(path)
    metadata = fixture["metadata"]
    assert metadata["package"] == "vllm-sr/d3-turbo"
    assert metadata["revision"] == "a0f3e9e59ebd1202fe1d4de1c4c0f9a986758cbb"
    assert metadata["attention_mode"] == "prefix_bidir"
    assert metadata["state_once"] is state_once
    assert metadata["readout_dtype"] == "float32"
    assert metadata["max_input_tokens"] == 32768
    assert metadata["tolerance"] == 1e-6
    assert len(fixture["provenance"]["source_revision"]) == 40
    tensor_path = path.parent / fixture["tensor_file"]
    assert fixture["tensor_file"] == "turbo-inputs.npz"
    assert hashlib.sha256(tensor_path.read_bytes()).hexdigest() == fixture["tensor_file_sha256"]
    seen = set()
    with zipfile.ZipFile(tensor_path) as archive:
        for name, case in fixture["cases"].items():
            preparation = case["preparation"]
            assert list(case["response"]["answers"]) == preparation["keys"]
            for key, row in preparation["rows"].items():
                assert row["state_tokens"] + row["question_tokens"] == row["token_ids"]
                assert len(row["state_tokens"]) == preparation["state_ends"][key]
                assert row["readout_position"] == len(row["token_ids"]) - 1
                for record in row["tensors"].values():
                    member = record["array"] + ".npy"
                    assert member.startswith(f"{name}/{key}/")
                    header, data = array_header(archive.read(member))
                    assert header["fortran_order"] is False
                    assert list(header["shape"]) == record["shape"]
                    dtype, width = {"float32": ("<f4", 4), "int64": ("<i8", 8)}[record["dtype"]]
                    assert header["descr"] == dtype
                    assert len(data) == math.prod(record["shape"]) * width
                    assert hashlib.sha256(data).hexdigest() == record["sha256"]
                    seen.add(member)
            if name == "over-limit":
                assert not preparation["rows"]
                assert case["response"]["answers"]["over"]["error"] == "max_length_exceeded"
                assert case["response"]["usage"]["input_tokens"] == 0
        assert set(archive.namelist()) == seen
    assert len(seen) == 83
    assert len(fixture["cases"]) == 6
