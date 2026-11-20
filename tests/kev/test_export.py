"""CPU-only exporter gates and LoRA arithmetic; no checkpoint download."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

SPEC = importlib.util.spec_from_file_location(
    "kev_export", Path(__file__).resolve().parents[2] / "recipe/kev/export.py"
)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


class ExportPathTests(unittest.TestCase):
    """Real file I/O and export control flow with tiny synthetic base tensors."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base, self.checkpoint = self.root / "base", self.root / "checkpoint"
        self.base.mkdir()
        self.checkpoint.mkdir()
        self.out = self.root / "export"
        self.shards = [name for name in exporter.BASE_FILES if name.endswith(".safetensors")]
        self.weights = [{}, {}]
        self.expected = {}
        adapter = {}
        self.targets = sorted(exporter.expected_targets())
        for i, target in enumerate(self.targets):
            weight = (torch.arange(6).reshape(2, 3) + i).bfloat16()
            self.weights[i % 2][target] = weight
            # Exact powers of two: sum of 16 * (1/4 * 1/4), times alpha/r=2.
            self.expected[target] = (weight.float() + 2).bfloat16()
            prefix = target.removeprefix("model.language_model.").removesuffix(".weight")
            adapter[f"base_model.model.{prefix}.lora_A.weight"] = torch.full((16, 3), 0.25)
            adapter[f"base_model.model.{prefix}.lora_B.weight"] = torch.full((2, 16), 0.25)
        plain = "model.language_model.norm.weight"
        self.weights[1][plain] = torch.tensor([1.001, 2.003, 3.007])
        self.expected[plain] = self.weights[1][plain].bfloat16()
        self.weights[0]["model.visual.weight"] = torch.ones(2, 3)
        self.weights[1]["mtp.weight"] = torch.ones(2, 3)
        for name in ("config.json", "tokenizer_config.json"):
            (self.base / name).write_text("{}\n")
        (self.base / "tokenizer.json").write_text(json.dumps({"added_tokens": [
            {"content": k, "id": v} for k, v in exporter.SPECIAL.items()
        ]}))
        (self.base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
            key: name for name, weights in zip(self.shards, self.weights) for key in weights
        }}))
        (self.checkpoint / "adapter_config.json").write_text("{}\n")
        save_file(adapter, self.checkpoint / "adapter_model.safetensors")
        self.head = {
            "q.weight": torch.full((256, 2560), 0.1234567),
            "k.weight": torch.full((256, 2560), -0.7654321),
            "q.bias": torch.arange(256).float() / 31,
            "k.bias": -torch.arange(256).float() / 43,
        }
        self.temperature = 2.406050072164233
        torch.save(dict(
            base=exporter.BASE_ID, base_revision=exporter.BASE_REVISION,
            lora=16, head_dim=256, option_isolation=False,
            special_embeddings=False, weights_dtype="fp32",
            temperature=self.temperature, head=self.head,
        ), self.checkpoint / "head.pt")

    @staticmethod
    def hashes(directory):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}

    def run_export(self):
        for name, weights in zip(self.shards, self.weights):
            save_file(weights, self.base / name)
        # Substitute only fixture hashes; keep target, dtype, shape and tokenizer
        # validators, restricted pickle loading, merge, writers and rename real.
        with patch.object(exporter, "BASE_FILES", self.hashes(self.base)), patch.object(
            exporter, "CHECKPOINT_FILES", self.hashes(self.checkpoint)
        ):
            return exporter.export(self.base, self.checkpoint, self.out)

    def test_complete_two_shard_export(self):
        with patch.object(exporter.os, "rename", wraps=exporter.os.rename) as rename:
            manifest = self.run_export()
        rename.assert_called_once()
        staging, destination = rename.call_args.args
        self.assertEqual(destination, self.out)
        self.assertEqual(staging.parent, self.out.parent)
        self.assertFalse(staging.exists())
        self.assertEqual(manifest, json.loads((self.out / "kev_export.json").read_text()))
        self.assertEqual(manifest["merged_targets"], self.targets)
        self.assertEqual(manifest["temperature"], self.temperature)
        self.assertEqual(manifest["special_tokens"], exporter.SPECIAL)
        self.assertEqual(manifest["inputs"], {
            "base": self.hashes(self.base), "checkpoint": self.hashes(self.checkpoint)
        })
        self.assertEqual(manifest["files"], {
            k: v for k, v in self.hashes(self.out).items() if k != "kev_export.json"
        })
        self.assertEqual(set(manifest["files"]), {
            *self.shards, "config.json", "tokenizer.json", "tokenizer_config.json",
            "head.safetensors", "model.safetensors.index.json",
        })
        weight_map = json.loads((self.out / "model.safetensors.index.json").read_text())["weight_map"]
        self.assertEqual(weight_map, {
            key: name for name, weights in zip(self.shards, self.weights)
            for key in weights if key in self.expected
        })
        for name in self.shards:
            tensors = load_file(self.out / name)
            self.assertEqual(set(tensors), {k for k, shard in weight_map.items() if shard == name})
            for key, tensor in tensors.items():
                self.assertEqual(tensor.dtype, torch.bfloat16)
                self.assertTrue(torch.equal(tensor, self.expected[key]), key)
        head = load_file(self.out / "head.safetensors")
        self.assertEqual(set(head), set(self.head))
        for key, tensor in head.items():
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertTrue(torch.equal(tensor, self.head[key]), key)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            self.assertEqual((self.out / name).read_bytes(), (self.base / name).read_bytes())

    def test_late_write_failure_removes_staging_without_publishing(self):
        calls = []

        def fail_second_shard(weights, path):
            calls.append(path)
            if len(calls) == 2:
                self.assertTrue(calls[0].is_file())
                raise OSError("injected second-shard write failure")
            save_file(weights, path)

        with patch.object(exporter, "save_file", side_effect=fail_second_shard):
            with self.assertRaisesRegex(OSError, "second-shard"):
                self.run_export()
        self.assertEqual(len(calls), 2)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["base", "checkpoint"])


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
