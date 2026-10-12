"""Reject unsupported vision exports before loading weights or dependencies."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "jev_vl_export_vision", REPO / "recipe/jev_vl/export_vision.py"
)
export_vision = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export_vision)


class ExportVisionTests(unittest.TestCase):
    def test_rejects_wrong_dimensions_before_weights_and_output(self):
        valid = json.loads((REPO / "tests/qwen3_5/data/config.json").read_text())
        read_weights = mock.Mock()
        save_weights = mock.Mock()
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(sys.modules, {
            "torch": None,
            "safetensors": mock.Mock(safe_open=read_weights),
            "safetensors.torch": mock.Mock(save_file=save_weights),
        }):
            model = Path(temporary) / "model"
            model.mkdir()
            out = Path(temporary) / "export"
            for section, field, value in (
                ("vision_config", "hidden_size", 1024),
                ("vision_config", "out_hidden_size", 2560),
                ("text_config", "hidden_size", 2560),
            ):
                with self.subTest(section=section, field=field):
                    config = copy.deepcopy(valid)
                    config[section][field] = value
                    (model / "config.json").write_text(json.dumps(config))
                    with self.assertRaisesRegex(ValueError, "expected the JEV-27B-VL"):
                        export_vision.export(model, out)
                    read_weights.assert_not_called()
                    save_weights.assert_not_called()
                    self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
