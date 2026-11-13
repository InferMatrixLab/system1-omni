"""The native bundle exporter stages nested BF16 output atomically."""

import importlib.util
from pathlib import Path


def load_exporter():
    path = Path(__file__).resolve().parents[2] / "recipe" / "vela" / "export_native.py"
    spec = importlib.util.spec_from_file_location("vela_export_native", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nested_bf16_output_is_published_with_the_bundle(tmp_path):
    exporter = load_exporter()
    target = tmp_path / "bundle"

    def write_bundle(staging):
        (staging / "bf16").mkdir()
        (staging / "heads.safetensors").write_bytes(b"heads")
        (staging / "bf16" / "model.safetensors").write_bytes(b"bf16")
        (staging / "vela_native_manifest.json").write_text("{}\n", encoding="utf-8")

    exporter.stage_into_place(target, write_bundle)

    assert (target / "heads.safetensors").read_bytes() == b"heads"
    assert (target / "bf16" / "model.safetensors").read_bytes() == b"bf16"
    assert (target / "vela_native_manifest.json").read_text() == "{}\n"
    assert not (tmp_path / "bundle.export-partial").exists()
