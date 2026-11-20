"""Opt-in head-64 reference operators, without checkpoint weights.

CUA_S1_CUDA_LIB=/absolute/libqwen3_5_cuda.so python -m unittest discover \
    -s tests/qwen3_5 -p test_vision_omnijev.py -v
Uses the pinned OmniJev Torch 2.14.0 CUDA reference environment.
"""
import ctypes
import os
import unittest


@unittest.skipUnless(os.environ.get("CUA_S1_CUDA_LIB"), "needs CUDA and CUA_S1_CUDA_LIB")
class OmniJevVisionOperators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch
        if torch.__version__.split("+")[0] != "2.14.0":
            raise unittest.SkipTest("requires pinned torch 2.14.0")
        torch.manual_seed(114)
        cls.library = ctypes.CDLL(os.environ["CUA_S1_CUDA_LIB"])
        cls.stream = torch.cuda.current_stream().cuda_stream

    def bind(self, name, arguments, result=ctypes.c_int):
        operation = getattr(self.library, name)
        operation.argtypes = arguments
        operation.restype = result
        return operation

    def test_attention_matches_sdpa_for_short_tail_and_original_grid(self):
        torch = self.torch
        pointer, integer = ctypes.c_void_p, ctypes.c_int
        operation = self.bind("cs1_omnijev_vision_attention", [pointer] * 4 + [integer] * 3 + [pointer] * 2)
        size = self.bind("cs1_omnijev_vision_attention_workspace_floats", [integer, integer], ctypes.c_size_t)
        for rows in [4, 16, 60, 64, 68, 124, 128, 132, 256, 2304, 4608]:
            with self.subTest(rows=rows):
                qkv = torch.randn(rows, 3, 16, 64, device="cuda", dtype=torch.bfloat16)
                q, k = qkv[:, 0].contiguous(), qkv[:, 1].contiguous()
                actual = torch.empty_like(q)
                workspace = torch.empty(size(rows, 16), device="cuda", dtype=torch.float32)
                expected = torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
                    qkv[:, 2].transpose(0, 1).unsqueeze(0), scale=0.125,
                ).squeeze(0).transpose(0, 1)
                self.assertEqual(operation(q.data_ptr(), k.data_ptr(), qkv.data_ptr() + 4096, actual.data_ptr(), rows, 16, 64, workspace.data_ptr(), self.stream), 0)
                self.assertTrue(torch.equal(actual, expected), f"{rows}: {(actual != expected).sum().item()} BF16 values differ")

    def test_norm_matches_framework_vectorized_welford(self):
        torch = self.torch
        pointer, integer = ctypes.c_void_p, ctypes.c_int
        operation = self.bind("cs1_omnijev_vision_norm", [pointer] * 4 + [integer, integer, pointer])
        for rows in [4, 32, 2304]:
            with self.subTest(rows=rows):
                x = torch.randn(rows, 1024, device="cuda", dtype=torch.bfloat16)
                weight = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
                bias = torch.randn_like(weight)
                actual = torch.empty_like(x)
                expected = torch.nn.functional.layer_norm(x, (1024,), weight, bias, eps=1e-6)
                self.assertEqual(operation(x.data_ptr(), weight.data_ptr(), bias.data_ptr(), actual.data_ptr(), rows, 1024, self.stream), 0)
                self.assertTrue(torch.equal(actual, expected))

    def test_layout_guards_keep_27b_separate(self):
        pointer, integer = ctypes.c_void_p, ctypes.c_int
        for name in ["cs1_omnijev_vision_norm", "cs1_reference_vision_norm"]:
            operation = self.bind(name, [pointer] * 4 + [integer, integer, pointer])
            wrong_width = 1152 if "omnijev" in name else 1024
            self.assertNotEqual(operation(None, None, None, None, 1, wrong_width, self.stream), 0)
        attention = self.bind("cs1_omnijev_vision_attention", [pointer] * 4 + [integer] * 3 + [pointer] * 2)
        for rows, heads, dimension in [(0, 16, 64), (4609, 16, 64), (4, 8, 64), (4, 16, 72)]:
            self.assertNotEqual(attention(None, None, None, None, rows, heads, dimension, None, self.stream), 0)
        create = self.bind("cs1_reference_patch_create_v2", [integer, integer, pointer], pointer)
        self.assertIsNone(create(4, 512, self.stream))

    @unittest.skipUnless(os.environ.get("QWEN_LEGACY_CUDA_LIB"), "needs an immutable pre-change CUDA library")
    def test_existing_4b_and_27b_operators_keep_their_bits(self):
        torch = self.torch
        pointer, integer = ctypes.c_void_p, ctypes.c_int
        old = ctypes.CDLL(os.environ["QWEN_LEGACY_CUDA_LIB"])

        def pair(name, arguments, result=integer):
            new = self.bind(name, arguments, result)
            previous = getattr(old, name)
            previous.argtypes, previous.restype = arguments, result
            return previous, new

        for name, width in [("cs1_vision_norm", 1024), ("cs1_reference_vision_norm", 1152)]:
            x = torch.randn(132, width, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(width, device="cuda", dtype=torch.bfloat16)
            b = torch.randn_like(w)
            outputs = [torch.empty_like(x), torch.empty_like(x)]
            for operation, output in zip(pair(name, [pointer] * 4 + [integer, integer, pointer]), outputs):
                self.assertEqual(operation(x.data_ptr(), w.data_ptr(), b.data_ptr(), output.data_ptr(), 132, width, self.stream), 0)
            self.assertTrue(torch.equal(*outputs), name)

        for dimension in [64, 72]:
            qkv = torch.randn(132, 3, 16, dimension, device="cuda", dtype=torch.bfloat16)
            q, k = qkv[:, 0].contiguous(), qkv[:, 1].contiguous()
            outputs = [torch.empty_like(q), torch.empty_like(q)]
            if dimension == 64:
                operations = pair("cs1_vision_attention", [pointer] * 4 + [integer, pointer])
                for operation, output in zip(operations, outputs):
                    self.assertEqual(operation(q.data_ptr(), k.data_ptr(), qkv[:, 2].data_ptr(), output.data_ptr(), 132, self.stream), 0)
            else:
                operations = pair("cs1_reference_vision_attention", [pointer] * 4 + [integer] * 3 + [pointer] * 2)
                sizes = pair("cs1_reference_vision_attention_workspace_floats", [integer, integer], ctypes.c_size_t)
                for operation, size, output in zip(operations, sizes, outputs):
                    workspace = torch.empty(132 * 16 * 80 * 2 + size(132, 16), device="cuda", dtype=torch.float32)
                    self.assertEqual(operation(q.data_ptr(), k.data_ptr(), qkv[:, 2].data_ptr(), output.data_ptr(), 132, 16, 72, workspace.data_ptr(), self.stream), 0)
            self.assertTrue(torch.equal(*outputs), f"legacy attention head dimension {dimension}")

    def test_original_patch_constructor_keeps_27b_layout(self):
        torch = self.torch
        pointer, integer = ctypes.c_void_p, ctypes.c_int
        old_create = self.bind("cs1_reference_patch_create", [integer, pointer], pointer)
        new_create = self.bind("cs1_reference_patch_create_v2", [integer, integer, pointer], pointer)
        execute = self.bind("cs1_reference_patch", [pointer] * 4)
        destroy = self.bind("cs1_reference_patch_destroy", [pointer], None)
        x = torch.randn(4, 1536, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(1152, 1536, device="cuda", dtype=torch.bfloat16)
        outputs = [torch.empty(4, 1152, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        plans = [old_create(4, self.stream), new_create(4, 1152, self.stream)]
        self.assertTrue(all(plans), "requires the cuDNN 9 build")
        try:
            for plan, output in zip(plans, outputs):
                self.assertEqual(execute(plan, x.data_ptr(), w.data_ptr(), output.data_ptr()), 0)
            self.assertTrue(torch.equal(*outputs))
        finally:
            torch.cuda.synchronize()
            for plan in plans:
                destroy(plan)


if __name__ == "__main__":
    unittest.main()
