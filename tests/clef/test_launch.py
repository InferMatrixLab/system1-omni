"""The Python module is a launcher, not an HTTP server."""

import ast
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

import frontend.clef as clef
import models.clef.worker as worker


def test_launcher_and_executor_do_not_import_threading_http_server():
    for module in (clef, worker):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "http.server":
                raise AssertionError(f"{module.__name__} imports http.server")
            if isinstance(node, ast.Name) and node.id == "ThreadingHTTPServer":
                raise AssertionError(f"{module.__name__} names ThreadingHTTPServer")


def test_main_execs_the_rust_binary(monkeypatch, tmp_path):
    binary = tmp_path / "omni-clef"
    binary.write_text("")
    monkeypatch.setenv("OMNI_CLEF_BIN", str(binary))
    seen: dict[str, object] = {}

    def fake_exec(path, argv):
        seen["path"] = path
        seen["argv"] = list(argv)
        raise SystemExit(0)

    monkeypatch.setattr(clef.os, "execv", fake_exec)
    with pytest.raises(SystemExit):
        clef.main(
            [
                "--model",
                "snapshot",
                "--variant",
                "clef",
                "--device",
                "cuda",
                "--gdn-prefill",
                "fla",
                "--host",
                "127.0.0.1",
                "--port",
                "9000",
            ]
        )
    assert seen["path"] == str(binary)
    argv = seen["argv"]
    assert isinstance(argv, list)
    assert argv[0] == str(binary)
    assert argv[argv.index("--model") + 1] == "snapshot"
    assert argv[argv.index("--variant") + 1] == "clef"
    assert argv[argv.index("--port") + 1] == "9000"
    assert argv[argv.index("--backbone") + 1] == "eager"
    assert "ThreadingHTTPServer" not in argv


def test_main_refuses_to_serve_http_when_the_binary_is_missing(monkeypatch):
    monkeypatch.delenv("OMNI_CLEF_BIN", raising=False)
    monkeypatch.setattr(clef, "candidate_binaries", lambda: [])
    monkeypatch.setattr(clef.shutil, "which", lambda _name: None)
    with pytest.raises(SystemExit, match="does not start a Python HTTP server"):
        clef.main(["--model", "snapshot"])


def test_executor_module_help_is_not_an_http_server():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "models.clef.worker", "--help"],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--model" in result.stdout
    assert "--backbone" in result.stdout
    assert "Not an HTTP server" in result.stdout
    assert "ThreadingHTTPServer" not in result.stdout
