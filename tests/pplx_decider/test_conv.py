"""CPU checks for the causal conv install gate. No CUDA extension."""

from __future__ import annotations

from pathlib import Path

import pytest

from models.pplx_decider.conv import (
    CAUSAL_CONV1D_VERSION,
    binds_package,
    ensure_prefill_gencode,
    ensure_sm89_gencode,
    ensure_sm90_gencode,
    extension_supports_arch,
    version_matches,
)

UPSTREAM = """
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_87,code=sm_87")
        if bare_metal_version >= Version("11.8"):
"""


def test_sdist_extract_uses_the_data_filter():
    script = Path(__file__).resolve().parents[2] / "recipe/pplx_decider/install_causal_conv1d.py"
    source = script.read_text(encoding="utf-8")
    assert 'extractall(root, filter="data")' in source


def package_fn():
    return None


def torch_fn():
    return None


def _wrap(implementation):
    def wrapped():
        return implementation()

    return wrapped


def test_version_pin_is_1_7_0():
    assert CAUSAL_CONV1D_VERSION == "1.7.0"
    assert version_matches("1.7.0")
    assert version_matches("1.7.0+cu128")
    assert not version_matches("1.6.2")


def test_sm90_gencode_is_inserted_once_when_missing():
    patched = ensure_sm90_gencode(UPSTREAM)
    assert 'arch=compute_90,code=sm_90' in patched
    assert patched.index("compute_87") < patched.index("compute_90")
    assert ensure_sm90_gencode(patched) == patched


def test_upstream_sm90_line_is_left_unchanged():
    upstream = UPSTREAM + '        cc_flag.append("arch=compute_90,code=sm_90")\n'
    assert ensure_sm90_gencode(upstream) == upstream


def test_sm90_gencode_refuses_an_unknown_setup():
    with pytest.raises(RuntimeError, match="sm_87"):
        ensure_sm90_gencode("cc_flag = []\n")


def test_both_cubins_are_inserted_when_the_sdist_has_neither():
    patched = ensure_prefill_gencode(UPSTREAM)
    assert "arch=compute_90,code=sm_90" in patched
    assert "arch=compute_89,code=sm_89" in patched
    assert patched.index("compute_87") < patched.index("compute_89")
    assert patched.index("compute_89") < patched.index("compute_90")
    assert ensure_prefill_gencode(patched) == patched


def test_sm89_is_inserted_beside_an_existing_sm90_line():
    upstream = UPSTREAM + '        cc_flag.append("arch=compute_90,code=sm_90")\n'
    patched = ensure_prefill_gencode(upstream)
    assert "compute_89" in patched
    assert "compute_90" in patched
    assert patched.index("compute_87") < patched.index("compute_89")
    assert ensure_sm89_gencode(patched) == patched
    assert ensure_prefill_gencode(patched) == patched


def test_sm89_gencode_refuses_an_unknown_setup():
    with pytest.raises(RuntimeError, match="sm_87"):
        ensure_sm89_gencode("cc_flag = []\n")


def test_extension_arch_bytes():
    assert extension_supports_arch(b"fatbin sm_89 cubin", 8, 9)
    assert extension_supports_arch(b"compute_90", 9, 0)
    assert not extension_supports_arch(b"sm_80 sm_90", 8, 9)


def test_binds_package_through_a_closure_and_forward():
    wrapped = _wrap(package_fn)

    class Module:
        forward = wrapped

    assert binds_package(Module(), package_fn)
    assert not binds_package(_wrap(torch_fn), package_fn)
    assert binds_package(package_fn, package_fn)
