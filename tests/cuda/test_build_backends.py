"""Check the Tier-2 build sweep without CUDA: the scripts here are fakes.

The point of the sweep is that one backend cannot hide another, so most of these
drive a tree where exactly one backend is wrong and assert that the others still
report. Nothing imports nvcc.
"""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backends.cuda import build_backends


class BuildSweepTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="omni-build-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cuda = self.root / "src" / "backends" / "cuda"
        self.cuda.mkdir(parents=True)
        self.out = self.root / "out"

    # -- fixtures ---------------------------------------------------------

    def script(self, directory, body, executable=True, name="build.sh"):
        path = directory / name
        path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
        path.chmod(0o755 if executable else 0o644)
        return path

    def backend(self, relative, name, output=None, status="experimental",
                build=None, script_body='touch "$1/lib%s.so"\n'):
        """Write one manifest. ``relative`` is "" for the flat layout."""
        directory = self.cuda / relative if relative else self.cuda
        directory.mkdir(parents=True, exist_ok=True)
        output = output or ("lib%s.so" % name)
        if build is None:
            build = {
                "script": "build.sh",
                "output": output,
                "default_arch": 89,
                "architectures": [89],
                "min_capability": 80,
            }
        manifest = {
            "name": name,
            "abi_version": 1,
            "status": status,
            "sources": [],
            "build": build,
        }
        (directory / (name + build_backends.MANIFEST_SUFFIX)).write_text(json.dumps(manifest))
        if script_body is not None:
            self.script(directory, script_body.replace("%s", name))
        return directory

    def sweep(self):
        return build_backends.build_all(str(self.root), str(self.out))

    def labels(self, outcomes):
        return sorted(outcome.label for outcome in outcomes)

    # -- the happy path ---------------------------------------------------

    def test_every_declared_backend_builds(self):
        self.backend("", "flat")
        self.backend("bar", "bar")
        built, skipped, failed = self.sweep()
        self.assertEqual(self.labels(built), ["bar", "flat"])
        self.assertEqual((skipped, failed), ([], []))

    def test_each_build_gets_its_own_directory(self):
        self.backend("", "flat")
        self.backend("bar", "bar")
        self.sweep()
        libraries = sorted(self.out.rglob("lib*.so"), key=lambda path: path.name)
        self.assertEqual([path.name for path in libraries], ["libbar.so", "libflat.so"])
        self.assertNotEqual(libraries[0].parent, libraries[1].parent,
                            "each build must write into its own directory")

    # -- one broken manifest must not silence the rest --------------------

    def test_a_malformed_manifest_is_reported_and_the_rest_still_build(self):
        self.backend("bar", "bar")
        broken = self.cuda / "broken"
        broken.mkdir()
        (broken / ("broken" + build_backends.MANIFEST_SUFFIX)).write_text("{not json")
        built, _, failed = self.sweep()
        self.assertEqual(self.labels(built), ["bar"])
        self.assertEqual(self.labels(failed), ["broken"])
        self.assertIn("cannot be read", failed[0].detail)

    def test_a_manifest_that_is_not_an_object_is_reported(self):
        self.backend("bar", "bar")
        odd = self.cuda / "odd"
        odd.mkdir()
        (odd / ("odd" + build_backends.MANIFEST_SUFFIX)).write_text("[1, 2]")
        built, _, failed = self.sweep()
        self.assertEqual(self.labels(built), ["bar"])
        self.assertEqual(self.labels(failed), ["odd"])
        self.assertIn("JSON object", failed[0].detail)

    def test_a_missing_build_block_is_reported(self):
        self.backend("bar", "bar")
        directory = self.backend("incomplete", "incomplete", build={})
        self.script(directory, 'touch "$1/libincomplete.so"\n')
        built, _, failed = self.sweep()
        self.assertEqual(self.labels(built), ["bar"])
        self.assertEqual(self.labels(failed), ["incomplete"])
        self.assertIn("build.script", failed[0].detail)

    def test_a_non_integer_architecture_is_reported(self):
        self.backend("bar", "bar")
        self.backend("odd", "odd", build={"script": "build.sh", "output": "libodd.so",
                                          "default_arch": "89"})
        built, _, failed = self.sweep()
        self.assertEqual(self.labels(built), ["bar"])
        self.assertEqual(self.labels(failed), ["odd"])
        self.assertIn("default_arch", failed[0].detail)

    def test_a_missing_script_is_reported(self):
        self.backend("bar", "bar")
        self.backend("noscript", "noscript", script_body=None)
        built, _, failed = self.sweep()
        self.assertEqual(self.labels(built), ["bar"])
        self.assertEqual(self.labels(failed), ["noscript"])
        self.assertIn("does not exist", failed[0].detail)

    def test_a_non_executable_script_is_reported(self):
        self.backend("bar", "bar")
        directory = self.backend("dull", "dull", script_body=None)
        self.script(directory, 'touch "$1/libdull.so"\n', executable=False)
        built, _, failed = self.sweep()
        self.assertEqual(self.labels(built), ["bar"])
        self.assertEqual(self.labels(failed), ["dull"])
        self.assertIn("not executable", failed[0].detail)

    def test_a_failing_script_does_not_stop_the_next_backend(self):
        self.backend("aa", "aa", script_body="exit 7\n")
        self.backend("zz", "zz")
        built, _, failed = self.sweep()
        self.assertEqual(self.labels(built), ["zz"])
        self.assertEqual(self.labels(failed), ["aa"])
        self.assertIn("exited 7", failed[0].detail)

    def test_a_script_that_produces_no_library_is_reported(self):
        self.backend("bar", "bar")
        self.backend("hollow", "hollow", script_body='touch "$1/other.so"\n')
        built, _, failed = self.sweep()
        self.assertEqual(self.labels(built), ["bar"])
        self.assertEqual(self.labels(failed), ["hollow"])
        self.assertIn("was not produced", failed[0].detail)

    # -- planned backends -------------------------------------------------

    def test_a_planned_backend_is_skipped_rather_than_built(self):
        self.backend("bar", "bar")
        planned = self.backend("later", "later", status="planned", build={})
        self.script(planned, 'touch "$1/liblater.so"\n')
        built, skipped, failed = self.sweep()
        self.assertEqual(self.labels(built), ["bar"])
        self.assertEqual(self.labels(skipped), ["later"])
        self.assertEqual(failed, [])

    def test_a_tree_of_only_planned_backends_is_not_a_failure(self):
        planned = self.backend("later", "later", status="planned", build={})
        self.script(planned, "true\n")
        built, skipped, failed = self.sweep()
        self.assertEqual((built, self.labels(skipped), failed), ([], ["later"], []))

    # -- identities must not collide --------------------------------------

    def test_reusing_the_output_directory_does_not_reuse_a_library(self):
        # The same --out twice: the first build writes the library, the second
        # writes nothing. Keeping the directory would credit the second with the
        # first one's file and report a build that never happened.
        directory = self.backend("", "flat")
        built, _, failed = self.sweep()
        self.assertEqual(self.labels(built), ["flat"])
        self.script(directory, "true\n")
        built, _, failed = self.sweep()
        self.assertEqual(built, [], "a build that produced nothing must not be reported as built")
        self.assertEqual(self.labels(failed), ["flat"])
        self.assertIn("was not produced", failed[0].detail)

    def test_a_duplicate_name_cannot_be_credited_with_the_other_artifact(self):
        # Same backend name, same output filename. The first produces the
        # library; the second does not. A shared output directory would let the
        # second pass on the first one's file.
        self.backend("aa", "dup", output="libdup.so")
        self.backend("zz", "dup", output="libdup.so", script_body='touch "$1/other.so"\n')
        built, _, failed = self.sweep()
        self.assertEqual(len(built), 1)
        self.assertEqual(len(failed), 1)
        self.assertIn("was not produced", failed[0].detail)

    def test_duplicate_names_keep_separate_artifacts(self):
        self.backend("aa", "dup", output="libdup.so")
        self.backend("zz", "dup", output="libdup.so")
        built, _, failed = self.sweep()
        self.assertEqual(len(built), 2)
        self.assertEqual(failed, [])
        produced = [path for path in self.out.rglob("libdup.so")]
        self.assertEqual(len(produced), 2)

    # -- the command line -------------------------------------------------

    def run_main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = build_backends.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_the_command_line_reports_counts_and_exits_zero(self):
        self.backend("", "flat")
        code, out, _ = self.run_main(["--repo-root", str(self.root), "--out", str(self.out)])
        self.assertEqual(code, 0)
        self.assertIn("1 built, 0 skipped, 0 failed", out)

    def test_the_command_line_fails_once_and_names_every_failure(self):
        self.backend("aa", "aa", script_body="exit 3\n")
        self.backend("bar", "bar")
        self.backend("zz", "zz", script_body='touch "$1/other.so"\n')
        code, out, err = self.run_main(["--repo-root", str(self.root), "--out", str(self.out)])
        self.assertEqual(code, 1)
        self.assertIn("1 built, 0 skipped, 2 failed", out)
        self.assertIn("aa", err)
        self.assertIn("zz", err)
        self.assertIn("bar", out)

    def test_a_tree_with_no_manifest_fails_rather_than_passing_quietly(self):
        code, _, err = self.run_main(["--repo-root", str(self.root), "--out", str(self.out)])
        self.assertEqual(code, 1)
        self.assertIn("no backend manifests", err)


if __name__ == "__main__":
    unittest.main(verbosity=2)
