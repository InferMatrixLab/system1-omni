"""CPU-only exporter gates and LoRA arithmetic; no checkpoint download."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

SPEC = importlib.util.spec_from_file_location(
    "kev_export", Path(__file__).resolve().parents[2] / "recipe/kev/export.py"
)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


class ExportTests(unittest.TestCase):
    def test_all_layers_and_projection_families_are_required(self):
        targets = exporter.expected_targets()
        self.assertEqual(len(targets), 248)
        for name, count in (("gate_proj", 32), ("up_proj", 32), ("down_proj", 32),
                            ("q_proj", 8), ("k_proj", 8), ("v_proj", 8), ("o_proj", 8),
                            ("in_proj_qkv", 24), ("in_proj_a", 24), ("in_proj_z", 24),
                            ("in_proj_b", 24), ("out_proj", 24)):
            self.assertEqual(sum(t.endswith(f".{name}.weight") for t in targets), count)
        exporter.validate_targets(targets)
        missing = "model.language_model.layers.0.linear_attn.in_proj_b.weight"
        # The same count with a different target is also invalid.
        with self.assertRaisesRegex(ValueError, "target mismatch"):
            exporter.validate_targets((targets - {missing}) | {missing.replace(".0.", ".3.")})

    def test_merge_includes_each_gdn_projection_with_one_rounding(self):
        # Includes all five GDN projections omitted by a Cua-S1-only target list.
        for module in (
            "in_proj_qkv",
            "in_proj_a",
            "in_proj_z",
            "in_proj_b",
            "out_proj",
        ):
            name = f"base_model.model.layers.0.linear_attn.{module}"
            a = torch.arange(48, dtype=torch.float32).reshape(16, 3) / 31
            b = torch.arange(32, dtype=torch.float32).reshape(2, 16) / 43
            pair = exporter.adapter_pairs(
                {f"{name}.lora_A.weight": a, f"{name}.lora_B.weight": b}
            )
            target = f"model.language_model.layers.0.linear_attn.{module}.weight"
            self.assertEqual(set(pair), {target})
            weight = torch.tensor(
                [[1.1, -2.2, 3.3], [4.4, 0.1, -0.7]], dtype=torch.bfloat16
            )
            actual = exporter.merge_weight(weight, a, b)
            # Independent elementwise FP64 sums; final BF16 rounding is exact here.
            expected = weight.double() + 2 * torch.stack(
                [
                    torch.stack(
                        [
                            sum(b[i, k].double() * a[k, j].double() for k in range(16))
                            for j in range(3)
                        ]
                    )
                    for i in range(2)
                ]
            )
            self.assertTrue(torch.equal(actual, expected.bfloat16()))

    def test_missing_unknown_nonfinite_and_shape_mismatches_fail(self):
        key = "base_model.model.layers.0.mlp.down_proj.lora_A.weight"
        for tensors in (
            {key: torch.ones(16, 2)},
            {"unknown": torch.ones(2, 2)},
            {key: torch.full((16, 2), float("nan"))},
            {key: torch.ones(16, 2).half()},
        ):
            with self.assertRaises(ValueError):
                exporter.adapter_pairs(tensors)
        with self.assertRaisesRegex(ValueError, "shape"):
            exporter.merge_weight(torch.ones(2, 3), torch.ones(4, 3), torch.ones(2, 4))

    def test_wrong_hash_precedes_pickle_loading_and_output_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "config.json").write_text("wrong revision")
            with patch.object(
                exporter.torch,
                "load",
                side_effect=AssertionError("must not deserialize"),
            ):
                with self.assertRaisesRegex(ValueError, "pinned release"):
                    exporter.export(root, root, root / "out")
            self.assertFalse((root / "out").exists())
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["config.json"])

    def test_never_overwrites_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, "already exists"):
                exporter.export(root, root, root)

    def test_delimiters_use_base_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tokenizer.json"
            tokens = [{"content": k, "id": v} for k, v in exporter.SPECIAL.items()]
            path.write_text(json.dumps({"added_tokens": tokens}))
            exporter.validate_tokenizer(path)
            tokens[0]["id"] = 151659
            path.write_text(json.dumps({"added_tokens": tokens}))
            with self.assertRaisesRegex(ValueError, "delimiter"):
                exporter.validate_tokenizer(path)

    def test_head_metadata_and_calibration_are_required(self):
        meta = dict(
            base=exporter.BASE_ID,
            base_revision=exporter.BASE_REVISION,
            lora=16,
            head_dim=256,
            option_isolation=False,
            special_embeddings=False,
            weights_dtype="fp32",
            temperature=2.406050072164233,
            head={
                "q.weight": torch.zeros(256, 2560),
                "k.weight": torch.ones(256, 2560),
                "q.bias": torch.zeros(256),
                "k.bias": torch.ones(256),
            },
        )
        _, temperature = exporter.pointer_head(meta)
        self.assertEqual(temperature, meta["temperature"])
        for change in (
            dict(temperature=0),
            dict(temperature=float("nan")),
            dict(special_embeddings=True),
            dict(base_revision="wrong"),
            dict(head={}),
            dict(weights_dtype="bf16"),
        ):
            with self.assertRaises(ValueError):
                exporter.pointer_head({**meta, **change})


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
