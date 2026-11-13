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


@pytest.mark.parametrize("exporter_name", ["export_token_plan", "export_head_inputs"])
def test_exporter_rejects_a_tampered_package(tmp_path, monkeypatch, exporter_name):
    """The recorders import package code only after the gate passes."""

    package = tmp_path / "package"
    package.mkdir()
    files = small_files()
    files["vela2_inference.py"] = b"raise AssertionError('imported before the gate')\n"
    pins = build_package(package, files)
    (package / "vela2_inference.py").write_bytes(
        files["vela2_inference.py"] + b"# tampered after pinning\n"
    )
    monkeypatch.setitem(
        hybrid.MEMBERS, hybrid.EXPECTED_MODEL_NAME,
        replace(hybrid.HYBRID_MEMBER, pinned_files=pins),
    )
    monkeypatch.setitem(sys.modules, "numpy", ModuleType("numpy"))
    monkeypatch.setitem(sys.modules, "safetensors", ModuleType("safetensors"))
    monkeypatch.setitem(sys.modules, "safetensors.numpy", SimpleNamespace(save_file=None))
    monkeypatch.setattr(sys, "path", sys.path.copy())

    path = Path(__file__).resolve().parents[2] / "recipe" / "vela" / f"{exporter_name}.py"
    spec = importlib.util.spec_from_file_location(exporter_name, path)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)

    args = ["--model", str(package), "--out", str(tmp_path / "fixtures")]
    with pytest.raises(ValueError, match="vela2_inference.py"):
        exporter.main(args)


def test_head_fixture_recording_uses_the_predict_execution_once(monkeypatch):
    """Inputs, logits and answers must observe one identical batch composition."""

    numpy = ModuleType("numpy")
    numpy.ndarray = ()
    numpy.integer = ()
    numpy.floating = ()
    numpy.bool_ = ()
    monkeypatch.setitem(sys.modules, "numpy", numpy)
    monkeypatch.setitem(sys.modules, "safetensors", ModuleType("safetensors"))
    monkeypatch.setitem(sys.modules, "safetensors.numpy", SimpleNamespace(save_file=None))
    path = Path(__file__).resolve().parents[2] / "recipe" / "vela" / "export_head_inputs.py"
    spec = importlib.util.spec_from_file_location("export_head_inputs_single_pass", path)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)

    rows = [
        {
            "p_ids": [7],
            "blocks": [
                {"kind": "q", "ids": [7]},
                {"kind": "span", "ids": [8]},
            ],
            "head": "router",
            "n_tokens": 772,
        },
        {
            "p_ids": [3],
            "blocks": [{"kind": "span", "ids": [3]}],
            "head": "router",
            "n_tokens": 269,
        },
    ]
    recorder = SimpleNamespace(calls=[], tensors={}, next_index=0, _on=False)

    class FakeEngine:
        def __init__(self):
            self.predict_count = 0
            self.run_batches = []
            self.run = self._run

        def _run(self, records):
            batch = [record["n_tokens"] for record in records]
            self.run_batches.append(batch)
            batch_total = sum(batch)

            def append_call(record, kind):
                index = len(recorder.calls)
                call = {
                    "kind": kind,
                    "logits": [[batch_total + record["n_tokens"]]],
                }
                if kind == "q":
                    call.update(cand=f"q{index}_cand", query=f"q{index}_query")
                    recorder.tensors[call["cand"]] = [batch_total, record["n_tokens"]]
                    recorder.tensors[call["query"]] = [batch_total]
                else:
                    call.update(
                        head=record["head"],
                        words=f"s{index}_words",
                        labels=f"s{index}_labels",
                    )
                    recorder.tensors[call["words"]] = [batch_total, record["n_tokens"]]
                    recorder.tensors[call["labels"]] = [batch_total]
                recorder.calls.append(call)

            # Pinned run_batch stacks every q block first, then visits
            # non-skipped spans in record order.
            q_records = [
                record
                for record in records
                if any(block["kind"] == "q" for block in record["blocks"])
            ]
            if q_records:
                append_call(q_records[0], "q")
            for record in records:
                for block in record["blocks"]:
                    if block["kind"] == "span" and not block.get("skip"):
                        append_call(record, "span")

        def _predict(self, request):
            self.predict_count += 1
            # The runtime sorts the two logical rows into one batch.
            self.run([request["rows"][1], request["rows"][0]])
            return {"answers": [{"id": "q", "batch_total": sum(self.run_batches[-1])}]}, None

    engine = FakeEngine()
    calls, tensors, predict, batches = exporter.record_case(
        engine, {"rows": rows}, rows, [], recorder
    )

    assert engine.predict_count == 1
    assert engine.run_batches == [[269, 772]]
    assert batches == [[1, 0]]
    assert [call["row"] for call in calls] == [0, 1, 0]
    assert [call["kind"] for call in calls] == ["q", "span", "span"]
    assert {call["logits"][0][0] for call in calls} == {1310, 1813}
    assert predict["answers"][0]["batch_total"] == 1041
    assert tensors == recorder.tensors
    assert all(values[0] == 1041 for values in tensors.values())
