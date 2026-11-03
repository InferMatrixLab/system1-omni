"""Fixture exporters must leave their verified package reusable (no weights needed)."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from models.vela import hybrid
from .test_hybrid import build_package, small_files


@pytest.mark.parametrize("exporter_name", ["export_token_plan", "export_head_inputs"])
@pytest.mark.parametrize("preload_runtime", [False, True])
def test_exporter_keeps_package_verifiable(tmp_path, monkeypatch, exporter_name, preload_runtime):
    package = tmp_path / "package"
    package.mkdir()
    files = small_files()
    files["vela2_inference.py"] = b"""
from types import SimpleNamespace
class Vela2:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        import configuration_vela2
        return SimpleNamespace(run=SimpleNamespace(model=None))
"""
    files["configuration_vela2.py"] = b"loaded = True\n"
    files["modeling_vela2.py"] = b"loaded = True\n"
    pins = build_package(package, files)
    monkeypatch.setitem(
        hybrid.MEMBERS, hybrid.EXPECTED_MODEL_NAME,
        replace(hybrid.HYBRID_MEMBER, pinned_files=pins),
    )
    # The test covers imports and initialization only; no numerical library
    # is needed by the empty case list.
    monkeypatch.setitem(sys.modules, "numpy", ModuleType("numpy"))
    monkeypatch.setitem(sys.modules, "safetensors", ModuleType("safetensors"))
    monkeypatch.setitem(sys.modules, "safetensors.numpy", SimpleNamespace(save_file=None))
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    for name in ("vela2_inference", "configuration_vela2", "modeling_vela2"):
        monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.delitem(sys.modules, name)
    if preload_runtime:
        # An already imported entry point still imports package modules lazily.
        spec = importlib.util.spec_from_file_location("vela2_inference", package / "vela2_inference.py")
        runtime = importlib.util.module_from_spec(spec)
        exec(files["vela2_inference.py"], runtime.__dict__)
        monkeypatch.setitem(sys.modules, "vela2_inference", runtime)

    path = Path(__file__).resolve().parents[2] / "recipe" / "vela" / f"{exporter_name}.py"
    spec = importlib.util.spec_from_file_location(exporter_name, path)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    monkeypatch.setattr(exporter, "cases", lambda _model: [])
    if exporter_name == "export_head_inputs":
        monkeypatch.setattr(exporter, "Recorder", lambda *_args: None)

    args = ["--model", str(package), "--out", str(tmp_path / "fixtures")]
    for _ in range(2):
        assert exporter.main(args) == 0
        assert not sys.dont_write_bytecode
        hybrid.verify_package(package)
        assert not list(package.rglob("__pycache__"))
