"""Compile the production device gate on CPU; no CUDA toolkit or GPU required."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class GemmDeviceProfileTest(unittest.TestCase):
    def test_only_measured_profile_is_allowed(self):
        with tempfile.TemporaryDirectory(prefix="gemm-profile-") as tmp:
            binary = Path(tmp) / "profile"
            subprocess.run([
                "c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT),
                str(ROOT / "tests/cuda/equivalent_tiles_profile.cpp"), "-o", str(binary),
            ], check=True, capture_output=True, text=True)
            subprocess.run([str(binary)], check=True)
